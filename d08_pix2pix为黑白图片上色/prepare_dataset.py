from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

import random

def split_images(images: list[Path], seed: int = 42) -> dict[str, list[Path]]:
      shuffled = images[:]
      random.Random(seed).shuffle(shuffled)

      total = len(shuffled)
      train_end = total * 8 // 10
      val_end = total * 9 // 10
      return {
          "train": shuffled[:train_end],
          "val": shuffled[train_end:val_end],
          "test": shuffled[val_end:],
      }


ROOT_DIR = Path(__file__).resolve().parent
SOURCE_DIR = ROOT_DIR / "datas"
OUTPUT_DIR = ROOT_DIR / "data"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
JPEG_SUFFIXES = {".jpg", ".jpeg"}
NAME_PREFIX = "cmp_b"
MIN_DIGITS = 3
JPEG_QUALITY = 95


def natural_sort_key(path: Path) -> list[tuple[int, int | str]]:
    parts = re.split(r"(\d+)", path.stem.lower())
    return [
        (0, int(part)) if part.isdigit() else (1, part)
        for part in parts
        if part
    ]


def to_rgb(image):
    from PIL import Image as PillowImage

    if image.mode == "RGB":
        return image
    if image.mode in ("RGBA", "LA"):
        background = PillowImage.new("RGB", image.size, "white")
        alpha = image.getchannel("A")
        background.paste(image.convert("RGBA"), mask=alpha)
        return background
    return image.convert("RGB")


def convert_image_to_jpeg(source_path: Path, target_path: Path) -> None:
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Pillow is required to convert non-JPEG images. "
            "Install it with: pip install pillow"
        ) from exc

    with Image.open(source_path) as image:
        to_rgb(image).save(target_path, format="JPEG", quality=JPEG_QUALITY)


def list_images(directory: Path) -> list[Path]:
    return sorted(
        [
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ],
        key=natural_sort_key,
    )


def rename_images_as_jpeg(directory: Path) -> list[Path]:
    images = list_images(directory)
    if not images:
        return []

    digits = max(MIN_DIGITS, len(str(len(images))))
    temp_dir = directory / ".rename_tmp"
    if temp_dir.exists():
        raise RuntimeError(f"Temporary directory already exists: {temp_dir}")
    temp_dir.mkdir()

    try:
        for index, source_path in enumerate(images, start=1):
            target_name = f"{NAME_PREFIX}{index:0{digits}d}.jpeg"
            temp_target = temp_dir / target_name

            if source_path.suffix.lower() in JPEG_SUFFIXES:
                source_path.replace(temp_target)
                continue

            convert_image_to_jpeg(source_path, temp_target)
            source_path.unlink()

        renamed_paths: list[Path] = []
        for temp_path in sorted(temp_dir.iterdir(), key=natural_sort_key):
            final_path = directory / temp_path.name
            temp_path.replace(final_path)
            renamed_paths.append(final_path)
    finally:
        if temp_dir.exists():
            for leftover in temp_dir.iterdir():
                leftover.replace(directory / leftover.name)
            temp_dir.rmdir()

    return renamed_paths


def clear_split_directory(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            path.unlink()


def split_images(images: list[Path]) -> dict[str, list[Path]]:
    total = len(images)
    train_end = total * 8 // 10
    val_end = total * 9 // 10
    return {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:],
    }


def copy_images_to_splits(images: list[Path], output_dir: Path) -> dict[str, int]:
    split_map = split_images(images)
    split_counts: dict[str, int] = {}

    for split_name, split_images_list in split_map.items():
        split_dir = output_dir / split_name
        clear_split_directory(split_dir)
        for image_path in split_images_list:
            shutil.copy2(image_path, split_dir / image_path.name)
        split_counts[split_name] = len(split_images_list)

    return split_counts


def main() -> int:
    if not SOURCE_DIR.exists():
        print(f"Source directory does not exist: {SOURCE_DIR}")
        return 1

    images = list_images(SOURCE_DIR)
    if not images:
        print(f"No images found in: {SOURCE_DIR}")
        return 1

    print(f"Found {len(images)} images in {SOURCE_DIR}.")

    renamed_images = rename_images_as_jpeg(SOURCE_DIR)
    print(
        f"Renamed {len(renamed_images)} images in {SOURCE_DIR} "
        f"to {NAME_PREFIX}###.jpeg style names."
    )

    split_counts = copy_images_to_splits(renamed_images, OUTPUT_DIR)
    print(f"train: {split_counts['train']}")
    print(f"val: {split_counts['val']}")
    print(f"test: {split_counts['test']}")
    print(f"Split images copied into: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
