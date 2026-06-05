#!/usr/bin/env python3
"""Download and verify the CUB-200-2011 dataset from CaltechDATA.

Reference record:
https://data.caltech.edu/records/65de6-vp158
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import sys
import tarfile
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


RECORD_URL = "https://data.caltech.edu/records/65de6-vp158"
DOWNLOAD_URL = (
    "https://data.caltech.edu/records/65de6-vp158/files/"
    "CUB_200_2011.tgz?download=1"
)
FILENAME = "CUB_200_2011.tgz"
EXPECTED_MD5 = "97eceeb196236b17998738112f37df78"
DEFAULT_DATA_DIR = Path("data")
REQUIRED_EXTRACTED_FILES = (
    "images.txt",
    "classes.txt",
    "image_class_labels.txt",
    "train_test_split.txt",
    "bounding_boxes.txt",
    "attributes/attributes.txt",
    "attributes/image_attribute_labels.txt",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download CUB-200-2011 from CaltechDATA and verify its MD5 hash."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory where the archive and extracted dataset will be stored.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if a verified archive already exists.",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Only download and verify the archive; do not extract it.",
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Extract even if the destination directory already appears complete.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1024 * 1024,
        help="Download/read chunk size in bytes.",
    )
    return parser.parse_args()


def file_md5(path: Path, chunk_size: int) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def is_verified_archive(path: Path, chunk_size: int) -> bool:
    if not path.exists():
        return False
    actual = file_md5(path, chunk_size)
    if actual == EXPECTED_MD5:
        print(f"[verify] OK: {path} md5={actual}")
        return True
    print(f"[verify] MISMATCH: {path} md5={actual}, expected={EXPECTED_MD5}")
    return False


def format_bytes(num_bytes: int | None) -> str:
    if num_bytes is None:
        return "unknown"
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024.0 or unit == "GiB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def download_archive(dest: Path, chunk_size: int) -> None:
    part_path = dest.with_suffix(dest.suffix + ".part")
    if part_path.exists():
        print(f"[download] Removing stale partial file: {part_path}")
        part_path.unlink()

    request = Request(
        DOWNLOAD_URL,
        headers={
            "User-Agent": "CV-project-CUB-downloader/1.0",
            "Accept": "application/octet-stream",
        },
    )

    print(f"[download] Record: {RECORD_URL}")
    print(f"[download] URL: {DOWNLOAD_URL}")
    print(f"[download] Writing: {part_path}")

    try:
        with urlopen(request, timeout=60) as response, part_path.open("wb") as out:
            total = response.headers.get("Content-Length")
            total_bytes = int(total) if total and total.isdigit() else None
            downloaded = 0
            last_report = time.monotonic()
            started = last_report

            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out.write(chunk)
                downloaded += len(chunk)

                now = time.monotonic()
                if now - last_report >= 5.0:
                    elapsed = max(now - started, 1e-6)
                    rate = downloaded / elapsed
                    if total_bytes:
                        pct = downloaded / total_bytes * 100.0
                        print(
                            f"[download] {pct:5.1f}% "
                            f"({format_bytes(downloaded)} / {format_bytes(total_bytes)}) "
                            f"at {format_bytes(int(rate))}/s"
                        )
                    else:
                        print(
                            f"[download] {format_bytes(downloaded)} "
                            f"at {format_bytes(int(rate))}/s"
                        )
                    last_report = now
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"Download failed: {exc}") from exc

    print("[download] Complete; verifying downloaded archive...")
    actual_md5 = file_md5(part_path, chunk_size)
    if actual_md5 != EXPECTED_MD5:
        raise RuntimeError(
            f"MD5 mismatch for {part_path}: got {actual_md5}, expected {EXPECTED_MD5}. "
            "The partial file is kept for inspection."
        )

    if dest.exists():
        dest.unlink()
    part_path.rename(dest)
    print(f"[verify] OK: {dest} md5={actual_md5}")


def extracted_dataset_complete(dataset_dir: Path) -> bool:
    return all((dataset_dir / rel).exists() for rel in REQUIRED_EXTRACTED_FILES)


def extract_archive(archive_path: Path, data_dir: Path, force_extract: bool) -> None:
    dataset_dir = data_dir / "CUB_200_2011"
    normalize_cub_attribute_names_file(data_dir, dataset_dir)
    if extracted_dataset_complete(dataset_dir) and not force_extract:
        print(f"[extract] Existing dataset looks complete: {dataset_dir}")
        print_post_extract_hint(dataset_dir)
        return

    print(f"[extract] Extracting {archive_path} into {data_dir}")
    with tarfile.open(archive_path, "r:gz") as tf:
        safe_extract(tf, data_dir)

    normalize_cub_attribute_names_file(data_dir, dataset_dir)
    missing = [
        rel for rel in REQUIRED_EXTRACTED_FILES if not (dataset_dir / rel).exists()
    ]
    if missing:
        raise RuntimeError(
            f"Extraction finished, but required files are missing: {', '.join(missing)}"
        )
    print(f"[extract] OK: {dataset_dir}")
    print_post_extract_hint(dataset_dir)


def safe_extract(tf: tarfile.TarFile, dest: Path) -> None:
    dest_resolved = dest.resolve()
    for member in tf.getmembers():
        member_path = (dest / member.name).resolve()
        try:
            member_path.relative_to(dest_resolved)
        except ValueError as exc:
            raise RuntimeError(f"Unsafe archive member path: {member.name}") from exc
    tf.extractall(dest)


def normalize_cub_attribute_names_file(data_dir: Path, dataset_dir: Path) -> None:
    """Copy Caltech archive's top-level attributes.txt into the project path.

    The CaltechDATA tarball stores the attribute-name list as `attributes.txt`
    at archive root, while `scratch/convert_cub_attributes.py` expects it at
    `data/CUB_200_2011/attributes/attributes.txt`.
    """
    source = data_dir / "attributes.txt"
    target = dataset_dir / "attributes" / "attributes.txt"

    if target.exists() or not source.exists():
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    print(f"[extract] Copied {source} -> {target}")


def print_post_extract_hint(dataset_dir: Path) -> None:
    concept_config_path = dataset_dir / "concept_config.json"
    if concept_config_path.exists():
        print(f"[hint] Concept config already exists: {concept_config_path}")
        return

    print("[hint] CUB archive extraction is complete, but concept_config.json is not")
    print("[hint] included in the CaltechDATA archive. Generate it with:")
    print("[hint]   uv run python scratch/convert_cub_attributes.py")
    print(f"[hint] Expected output: {concept_config_path}")


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir
    archive_path = data_dir / FILENAME

    data_dir.mkdir(parents=True, exist_ok=True)

    if args.force and archive_path.exists():
        print(f"[force] Removing existing archive: {archive_path}")
        archive_path.unlink()

    if not is_verified_archive(archive_path, args.chunk_size):
        download_archive(archive_path, args.chunk_size)

    if not args.no_extract:
        extract_archive(archive_path, data_dir, args.force_extract)
    else:
        print("[hint] After extracting the archive, generate CUB concept_config.json with:")
        print("[hint]   uv run python scratch/convert_cub_attributes.py")

    print("[done] CUB-200-2011 download and verification complete.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[abort] Interrupted by user.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(1)
