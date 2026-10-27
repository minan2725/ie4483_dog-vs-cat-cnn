"""Resize dataset images once and cache them as JPEG files."""

from pathlib import Path
from PIL import Image, ImageOps
from tqdm import tqdm
import argparse

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / "datasets"
DEFAULT_CACHE_DIR = ROOT / "cached_256"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def resize_short_side(image, size=256):
    """Resize while preserving aspect ratio."""
    width, height = image.size

    if min(width, height) == size:
        return image

    if width <= height:
        new_width = size
        new_height = round(height * size / width)
    else:
        new_height = size
        new_width = round(width * size / height)

    return image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS,
    )



def cache_image(src, dst, size=256, quality=95):
    """Resize once while preserving the original filename and format."""
    dst.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(src) as image:
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        image = resize_short_side(image, size)

        suffix = dst.suffix.lower()

        if suffix in {".jpg", ".jpeg"}:
            image.save(dst, format="JPEG", quality=quality)
        elif suffix == ".png":
            image.save(dst, format="PNG", optimize=False)
        elif suffix == ".webp":
            image.save(dst, format="WEBP", quality=quality)
        elif suffix == ".bmp":
            image.save(dst, format="BMP")
        else:
            raise ValueError(f"Unsupported image extension: {suffix}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path,
                        default=DEFAULT_DATA_DIR)
    parser.add_argument("--cache-dir", type=Path,
                        default=DEFAULT_CACHE_DIR)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--quality", type=int, default=95)
    parser.add_argument("--force", action="store_true",
                        help="Rebuild existing cached images")
    args = parser.parse_args()

    if args.size <= 0:
        raise ValueError("--size must be positive")

    if not 1 <= args.quality <= 100:
        raise ValueError("--quality must be between 1 and 100")

    # Process every split, including the unlabeled test directory.
    splits = ["train", "val", "test"]

    for split in splits:
        source_split = args.data_dir / split
        if not source_split.is_dir():
            raise FileNotFoundError(
                f"Dataset split not found: {source_split}"
            )

        files = sorted(
            path for path in source_split.rglob("*")
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        print(f"\n{split}: {len(files)} images")

        for src in tqdm(files, desc=f"Caching {split}"):
            relative = src.relative_to(source_split)

            dst = args.cache_dir / split / relative

            if dst.exists() and not args.force:
                continue

            try:
                cache_image(
                    src, dst,
                    size=args.size,
                    quality=args.quality,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to cache image: {src}"
                ) from exc

    print(f"\nCache complete: {args.cache_dir}")


if __name__ == "__main__":
    main()
