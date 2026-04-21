from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image


ROOT_DIR = Path(__file__).resolve().parent  # 获取当前文件名 -> 获取文件绝对路径 -> 获取文件父目录的绝对路径
INPUT_DIR = ROOT_DIR / "CMP_facade_DB_base" / "extended"
OUTPUT_DIR = ROOT_DIR / "datas"


def to_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image
    if image.mode in ("RGBA", "LA"):  # "RGBA" = 4 通道（R, G, B, A）； "LA" = 2 通道（L, A） L 是亮度（灰度） A 是透明度
        background = Image.new("RGB", image.size, "white")  # 创建新的图片，宽度为图片宽度，高度为图片高度，背景为白色
        alpha = image.getchannel("A")  # 获取图片的alpha通道
        background.paste(image.convert("RGBA"), mask=alpha)  # 将图片粘贴到新的图片上，mask为alpha通道
        return background
    return image.convert("RGB")  # 如果图片模式为RGB，则直接返回图片


def collect_pairs(input_dir: Path) -> dict[str, tuple[Path, Path]]:  # 收集jpg和png文件，并返回字典，键为文件名，值为同stem名的jpg和png文件路径元组
    jpg_files = {path.stem: path for path in input_dir.glob("*.jpg")}  # 获取所有的jpg文件，并将其转换为字典，键为文件名，值为文件路径
    jpg_files.update({path.stem: path for path in input_dir.glob("*.jpeg")})  
    png_files = {path.stem: path for path in input_dir.glob("*.png")}  # 获取所有的png文件，并将其转换为字典，键为文件名，值为文件路径

    pairs: dict[str, tuple[Path, Path]] = {}
    for stem in sorted(jpg_files.keys() & png_files.keys()):
        pairs[stem] = (jpg_files[stem], png_files[stem])
    return pairs  # 返回字典，键为文件名，值为同stem名的jpg和png文件路径元组


def merge_pair(jpg_path: Path, png_path: Path, output_path: Path) -> None:
    with Image.open(jpg_path) as jpg_image, Image.open(png_path) as png_image:
        left = to_rgb(jpg_image)
        right = to_rgb(png_image)

        merged_width = left.width + right.width  # 合并后的图片宽度为左右两张图片宽度之和
        merged_height = max(left.height, right.height)  # 合并后的图片高度为左右两张图片高度中的最大值
        merged = Image.new("RGB", (merged_width, merged_height), "white")  # 创建新的图片，宽度为合并后的图片宽度，高度为合并后的图片高度，背景为白色
        merged.paste(left, (0, 0))  # 将左边的图片粘贴到新的图片的左上角
        merged.paste(right, (left.width, 0))  # 将右边的图片粘贴到新的图片的右上角
        merged.save(output_path, format="JPEG", quality=95)  # 将合并后的图片保存为jpeg格式，质量为95


def main() -> int:
    if not INPUT_DIR.exists():
        print(f"输入目录不存在: {INPUT_DIR}")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pairs = collect_pairs(INPUT_DIR)

    if not pairs:
        print(f"未找到可配对的 jpg/png 图片: {INPUT_DIR}")
        return 1

    for index, (stem, (jpg_path, png_path )) in enumerate(pairs.items(), start=379):
        output_path = OUTPUT_DIR / f"{index}.jpeg"
        merge_pair(jpg_path, png_path, output_path)
        print(f"已生成: {output_path}")

    print(f"处理完成，共生成 {len(pairs)} 张图片。")
    return 0


if __name__ == "__main__":
    sys.exit(main())