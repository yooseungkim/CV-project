"""
Download the MILK10k dataset files from the official ISIC Challenge page.

By default this downloads the files needed by this repository's MILK10KDataset:

    data/MILK10K/
    ├── MILK10k_Training_Input/
    ├── MILK10k_Training_Metadata.csv
    ├── MILK10k_Training_Supplement.csv
    └── MILK10k_Training_GroundTruth.csv

Usage:
    uv run python download_milk10k.py
    uv run python download_milk10k.py --include-test
    uv run python download_milk10k.py --output-dir /path/to/MILK10K --force
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path


SOURCE_PAGE = "https://challenge.isic-archive.com/data/#milk10k"
BASE_URL = "https://isic-archive.s3.amazonaws.com/challenges/milk10k"


@dataclass(frozen=True)
class Asset:
    key: str
    filename: str
    description: str
    extract: bool = False

    @property
    def url(self) -> str:
        return f"{BASE_URL}/{self.filename}"


ASSETS: dict[str, Asset] = {
    "training_images": Asset(
        key="training_images",
        filename="MILK10k_Training_Input.zip",
        description="training JPEG images",
        extract=True,
    ),
    "training_metadata": Asset(
        key="training_metadata",
        filename="MILK10k_Training_Metadata.csv",
        description="training metadata CSV",
    ),
    "training_supplement": Asset(
        key="training_supplement",
        filename="MILK10k_Training_Supplement.csv",
        description="training supplemental metadata CSV",
    ),
    "training_ground_truth": Asset(
        key="training_ground_truth",
        filename="MILK10k_Training_GroundTruth.csv",
        description="training ground-truth CSV",
    ),
    "test_images": Asset(
        key="test_images",
        filename="MILK10k_Test_Input.zip",
        description="test JPEG images",
        extract=True,
    ),
    "test_metadata": Asset(
        key="test_metadata",
        filename="MILK10k_Test_Metadata.csv",
        description="test metadata CSV",
    ),
}


DEFAULT_KEYS = [
    "training_images",
    "training_metadata",
    "training_supplement",
    "training_ground_truth",
]
TEST_KEYS = ["test_images", "test_metadata"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download MILK10k files from the official ISIC Challenge S3 links."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data") / "MILK10K",
        help="Directory where MILK10k files will be stored. Default: data/MILK10K",
    )
    parser.add_argument(
        "--include-test",
        action="store_true",
        help="Also download the test images and test metadata.",
    )
    parser.add_argument(
        "--only",
        choices=sorted(ASSETS),
        nargs="+",
        help="Download only the selected asset keys.",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Download zip archives but do not extract them.",
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Keep zip archives after successful extraction.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if they already exist.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1024 * 1024,
        help="Download chunk size in bytes. Default: 1048576.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Network timeout in seconds. Default: 60.",
    )
    return parser.parse_args()


def human_bytes(num_bytes: int | None) -> str:
    if num_bytes is None:
        return "unknown size"
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.1f}{unit}" if unit != "B" else f"{int(value)}B"
        value /= 1024.0
    return f"{num_bytes}B"


def remote_size(url: str, timeout: int) -> int | None:
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            value = response.headers.get("Content-Length")
            return int(value) if value is not None else None
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
        return None


def should_skip(path: Path, expected_size: int | None, force: bool) -> bool:
    if force or not path.exists():
        return False
    if expected_size is None:
        print(f"  - exists, skipping: {path}")
        return True
    local_size = path.stat().st_size
    if local_size == expected_size:
        print(f"  - exists, size matches, skipping: {path}")
        return True
    print(
        f"  - existing file has different size "
        f"({human_bytes(local_size)} != {human_bytes(expected_size)}), downloading again"
    )
    return False


def download_file(asset: Asset, output_dir: Path, chunk_size: int, timeout: int, force: bool) -> Path:
    output_path = output_dir / asset.filename
    expected_size = remote_size(asset.url, timeout=timeout)

    print(f"\nDownloading {asset.description}")
    print(f"  URL: {asset.url}")
    print(f"  To:  {output_path}")
    print(f"  Expected: {human_bytes(expected_size)}")

    if should_skip(output_path, expected_size, force=force):
        return output_path

    part_path = output_path.with_suffix(output_path.suffix + ".part")
    if part_path.exists():
        part_path.unlink()

    request = urllib.request.Request(asset.url, headers={"User-Agent": "cv-project-milk10k-downloader/1.0"})
    started = time.monotonic()
    downloaded = 0

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            total = expected_size
            with part_path.open("wb") as out_file:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    downloaded += len(chunk)
                    print_progress(downloaded, total, started)
    except Exception:
        print()
        if part_path.exists():
            print(f"  Partial download left at: {part_path}")
        raise

    print()
    if expected_size is not None and downloaded != expected_size:
        raise RuntimeError(
            f"Downloaded size mismatch for {asset.filename}: "
            f"{human_bytes(downloaded)} != {human_bytes(expected_size)}"
        )

    part_path.replace(output_path)
    print(f"  Saved: {output_path}")
    return output_path


def print_progress(downloaded: int, total: int | None, started: float) -> None:
    elapsed = max(time.monotonic() - started, 1e-6)
    rate = downloaded / elapsed
    if total:
        pct = downloaded / total * 100.0
        message = (
            f"  {pct:6.2f}%  {human_bytes(downloaded)} / {human_bytes(total)}  "
            f"{human_bytes(int(rate))}/s"
        )
    else:
        message = f"  {human_bytes(downloaded)}  {human_bytes(int(rate))}/s"
    print("\r" + message, end="", flush=True)


def safe_extract_zip(zip_path: Path, output_dir: Path) -> None:
    print(f"\nExtracting {zip_path.name}")
    output_root = output_dir.resolve()

    with zipfile.ZipFile(zip_path) as archive:
        members = archive.infolist()
        for member in members:
            target = (output_dir / member.filename).resolve()
            if os.path.commonpath([output_root, target]) != str(output_root):
                raise RuntimeError(f"Unsafe zip member path: {member.filename}")

        for index, member in enumerate(members, start=1):
            archive.extract(member, output_dir)
            if index % 500 == 0 or index == len(members):
                print(f"\r  Extracted {index}/{len(members)} files", end="", flush=True)
    print()


def ensure_training_image_layout(output_dir: Path) -> None:
    """Match src.data.milk10k.MILK10KDataset's default image_dir layout."""
    outer_dir = output_dir / "MILK10k_Training_Input"
    expected_dir = outer_dir / "MILK10k_Training_Input"

    if not outer_dir.exists() or expected_dir.exists():
        return

    children = list(outer_dir.iterdir())
    if not children:
        return

    print("\nNormalizing training image layout for this repository")
    expected_dir.mkdir(parents=True, exist_ok=True)
    for child in children:
        if child == expected_dir:
            continue
        child.replace(expected_dir / child.name)
    print(f"  Image directory: {expected_dir}")


def select_assets(args: argparse.Namespace) -> list[Asset]:
    if args.only:
        keys = args.only
    else:
        keys = list(DEFAULT_KEYS)
        if args.include_test:
            keys.extend(TEST_KEYS)
    return [ASSETS[key] for key in keys]


def print_next_steps(output_dir: Path) -> None:
    training_csv = output_dir / "MILK10k_Training_Metadata.csv"
    image_dir = output_dir / "MILK10k_Training_Input" / "MILK10k_Training_Input"
    ground_truth = output_dir / "MILK10k_Training_GroundTruth.csv"

    print("\nDone.")
    print(f"Source page: {SOURCE_PAGE}")
    print("\nRepository defaults now expect:")
    print(f"  csv_path:  {training_csv}")
    print(f"  image_dir: {image_dir}")
    print(f"  labels:    {ground_truth}")
    print("\nYou can train with:")
    print("  uv run python main.py --config_path configs/train_config.yaml")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("MILK10k downloader")
    print(f"Official page: {SOURCE_PAGE}")
    print(f"Output dir:    {output_dir}")

    downloaded_paths: list[tuple[Asset, Path]] = []
    for asset in select_assets(args):
        path = download_file(
            asset=asset,
            output_dir=output_dir,
            chunk_size=args.chunk_size,
            timeout=args.timeout,
            force=args.force,
        )
        downloaded_paths.append((asset, path))

    if not args.no_extract:
        for asset, path in downloaded_paths:
            if asset.extract:
                safe_extract_zip(path, output_dir)
                if asset.key == "training_images":
                    ensure_training_image_layout(output_dir)
                if not args.keep_archives:
                    path.unlink()
                    print(f"  Removed archive: {path}")

    print_next_steps(output_dir)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        raise SystemExit(1)
