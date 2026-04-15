import argparse
import csv
import os
import random
from pathlib import Path
from typing import List, Optional, Sequence

import torch
import torchvision.utils as vutils
from PIL import Image
from torchvision import transforms

from dcgan_faces_tutorial import Discriminator, Generator, image_size, ngpu, nz

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def resolve_device(choice: str) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available on this machine")
        return torch.device("cuda:0")
    return torch.device("cuda:0" if torch.cuda.is_available() and ngpu > 0 else "cpu")


def maybe_set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    torch.manual_seed(seed)


def load_generator(weights_path: str, device: torch.device) -> Generator:
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Generator weights not found at {weights_path}")
    model = Generator(ngpu).to(device)
    state = torch.load(weights_path, map_location=device)
    try:
        model.load_state_dict(state)
    except RuntimeError as exc:
        raise RuntimeError(f"Failed to load generator weights from {weights_path}: {exc}") from exc
    model.eval()
    return model


def load_discriminator(weights_path: str, device: torch.device) -> Discriminator:
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Discriminator weights not found at {weights_path}")
    model = Discriminator(ngpu).to(device)
    state = torch.load(weights_path, map_location=device)
    try:
        model.load_state_dict(state)
    except RuntimeError as exc:
        raise RuntimeError(f"Failed to load discriminator weights from {weights_path}: {exc}") from exc
    model.eval()
    return model


def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_tensor_batch(tensors: torch.Tensor, outdir: str, prefix: str, start_index: int) -> List[str]:
    saved = []
    for offset, img in enumerate(tensors):
        filename = f"{prefix}_{start_index + offset:04d}.png"
        full_path = os.path.join(outdir, filename)
        vutils.save_image(img, full_path, normalize=True, range=(-1, 1))
        saved.append(full_path)
    return saved


def collect_for_grid(buffer: List[torch.Tensor], batch: torch.Tensor, limit: int = 64) -> None:
    remaining = limit - sum(t.size(0) for t in buffer)
    if remaining <= 0:
        return
    buffer.append(batch[:remaining])


def list_image_paths(input_dir: str) -> List[Path]:
    directory = Path(input_dir)
    if not directory.exists():
        raise FileNotFoundError(f"Input directory {input_dir} does not exist")
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def load_image_batch(paths: Sequence[Path], transform: transforms.Compose) -> torch.Tensor:
    tensors = []
    for path in paths:
        with Image.open(path) as img:
            tensors.append(transform(img.convert("RGB")))
    return torch.stack(tensors)


def filter_results(results: List[dict], threshold: Optional[float], topk: int) -> List[dict]:
    filtered = [item for item in results if threshold is None or item["score"] >= threshold]
    filtered.sort(key=lambda item: item["score"], reverse=True)
    if topk > 0:
        filtered = filtered[:topk]
    return filtered


def write_scores_csv(path: str, entries: List[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["rank", "score", "source", "saved_path"])
        for idx, entry in enumerate(entries, 1):
            writer.writerow([
                idx,
                f"{entry['score']:.6f}",
                entry["source"],
                entry.get("saved_path", ""),
            ])


def run_generate(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    maybe_set_seed(args.seed)
    netG = load_generator(args.gen_weights, device)
    ensure_outdir(args.outdir)

    total = args.num_images
    batch_size = args.batch_size
    produced = 0
    grid_buffer: List[torch.Tensor] = []
    all_saved = []

    with torch.no_grad():
        while produced < total:
            current = min(batch_size, total - produced)
            noise = torch.randn(current, nz, 1, 1, device=device)
            fake = netG(noise).cpu()
            all_saved.extend(save_tensor_batch(fake, args.outdir, args.prefix, produced))
            collect_for_grid(grid_buffer, fake)
            produced += current

    if args.grid and grid_buffer:
        grid_source = torch.cat(grid_buffer, dim=0)
        grid = vutils.make_grid(grid_source, padding=2, normalize=True)
        grid_path = os.path.join(args.outdir, f"{args.prefix}_grid.png")
        vutils.save_image(grid, grid_path)
        print(f"Saved preview grid to {grid_path}")

    print(f"Generated {len(all_saved)} images to {args.outdir}")


def run_score(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    maybe_set_seed(args.seed)
    netD = load_discriminator(args.disc_weights, device)
    ensure_outdir(args.outdir)

    use_generator = args.input_dir is None
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    results: List[dict] = []

    with torch.no_grad():
        if use_generator:
            netG = load_generator(args.gen_weights, device)
            target = args.num_images
            produced = 0
            while produced < target:
                current = min(args.batch_size, target - produced)
                noise = torch.randn(current, nz, 1, 1, device=device)
                fake = netG(noise)
                scores = netD(fake).view(-1)
                fake_cpu = fake.detach().cpu()
                scores_cpu = scores.detach().cpu()
                for idx in range(current):
                    results.append({
                        "tensor": fake_cpu[idx],
                        "score": float(scores_cpu[idx].item()),
                        "source": f"gen_{produced + idx:04d}",
                    })
                produced += current
        else:
            image_paths = list_image_paths(args.input_dir)
            if not image_paths:
                raise RuntimeError(
                    f"No images with extensions {list(IMAGE_EXTS)} found in {args.input_dir}"
                )
            for start in range(0, len(image_paths), args.batch_size):
                batch_paths = image_paths[start:start + args.batch_size]
                batch_cpu = load_image_batch(batch_paths, transform)
                batch = batch_cpu.to(device)
                scores = netD(batch).view(-1).detach().cpu()
                for tensor, path_obj, score in zip(batch_cpu, batch_paths, scores):
                    results.append({
                        "tensor": tensor,
                        "score": float(score.item()),
                        "source": str(path_obj),
                    })

    if not results:
        print("No samples were available for scoring.")
        return

    filtered = filter_results(results, args.threshold, args.topk)
    if not filtered:
        print("No samples satisfied the provided threshold/top-k filters.")
        return

    saved_entries = []
    grid_candidates: List[torch.Tensor] = []
    for idx, item in enumerate(filtered):
        out_path = os.path.join(args.outdir, f"{args.prefix}_{idx:04d}.png")
        vutils.save_image(item["tensor"], out_path, normalize=True, range=(-1, 1))
        saved_entries.append({
            "score": item["score"],
            "source": item["source"],
            "saved_path": out_path,
        })
        if len(grid_candidates) < 64:
            grid_candidates.append(item["tensor"])

    if args.grid and grid_candidates:
        grid = vutils.make_grid(torch.stack(grid_candidates), padding=2, normalize=True)
        grid_path = os.path.join(args.outdir, f"{args.prefix}_grid.png")
        vutils.save_image(grid, grid_path)
        print(f"Saved preview grid to {grid_path}")

    scores_only = [entry["score"] for entry in saved_entries]
    mean_score = sum(scores_only) / len(scores_only)
    print(
        f"Scored {len(results)} samples; saved {len(saved_entries)} images to {args.outdir}. "
        f"Score range {min(scores_only):.4f}-{max(scores_only):.4f}, mean {mean_score:.4f}."
    )
    preview = min(5, len(saved_entries))
    for rank in range(preview):
        entry = saved_entries[rank]
        print(
            f"[#{rank + 1:02d}] score={entry['score']:.4f} source={entry['source']} -> {entry['saved_path']}"
        )

    if args.scores_file:
        write_scores_csv(args.scores_file, saved_entries)
        print(f"Scores written to {args.scores_file}")


def add_generation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--gen-weights", default="netG.pth", help="Path to generator weights.")
    parser.add_argument("--num-images", type=int, default=16, help="Number of samples to generate.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for inference.")
    parser.add_argument("--outdir", default="outputs", help="Directory to store outputs.")
    parser.add_argument("--prefix", default="sample", help="Filename prefix.")
    parser.add_argument("--grid", action="store_true", help="Save a grid of up to 64 samples.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducibility.")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to run inference on (auto uses CUDA if available).",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DCGAN inference helper")
    subparsers = parser.add_subparsers(dest="command", required=True)

    gen_parser = subparsers.add_parser("generate", help="Only generate images with the trained generator")
    add_generation_args(gen_parser)
    gen_parser.set_defaults(func=run_generate)

    score_parser = subparsers.add_parser(
        "score",
        help="Generate images or load an image folder, then rank with the discriminator",
    )
    add_generation_args(score_parser)
    score_parser.add_argument("--disc-weights", default="netD.pth", help="Path to discriminator weights.")
    score_parser.add_argument(
        "--input-dir",
        default=None,
        help="Optional folder of existing images to score instead of generating new ones.",
    )
    score_parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Minimum discriminator score (0-1) required to keep a sample.",
    )
    score_parser.add_argument(
        "--topk",
        type=int,
        default=0,
        help="Keep only the top-k samples by score after thresholding (0 keeps all).",
    )
    score_parser.add_argument(
        "--scores-file",
        default=None,
        help="Optional CSV path to record ranked scores.",
    )
    score_parser.set_defaults(func=run_score)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
