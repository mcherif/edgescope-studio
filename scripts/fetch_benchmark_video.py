from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "EdgeScope/1.0 (+https://github.com/)"},
    )
    with urllib.request.urlopen(req) as resp, out_path.open("wb") as f:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Download benchmark video (or use local path) and record SHA256."
    )
    ap.add_argument("--url", type=str, default=None, help="Direct video URL to download")
    ap.add_argument(
        "--path",
        type=str,
        default=None,
        help="Use an existing local video path instead of downloading",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="benchmarks/6517471-hd_1920_1080_30fps.mp4",
        help="Destination path for the downloaded video",
    )
    ap.add_argument(
        "--sha-out",
        type=str,
        default="benchmarks/6517471-hd_1920_1080_30fps.sha256",
        help="Where to write the SHA256 checksum",
    )
    ap.add_argument("--force", action="store_true", help="Overwrite existing file")
    args = ap.parse_args()

    out_path = Path(args.out)
    sha_path = Path(args.sha_out)

    if args.path:
        video_path = Path(args.path)
        if not video_path.exists():
            print(f"ERROR: file not found: {video_path}", file=sys.stderr)
            return 2
    else:
        if out_path.exists() and not args.force:
            video_path = out_path
            print(f"Using existing file: {video_path}")
        else:
            if not args.url:
                print(
                    "ERROR: --url is required to download, or provide --path to an existing file.",
                    file=sys.stderr,
                )
                return 2
            print(f"Downloading to: {out_path}")
            download(args.url, out_path)
            video_path = out_path

    checksum = sha256_file(video_path)
    sha_path.parent.mkdir(parents=True, exist_ok=True)
    sha_path.write_text(f"{checksum}  {video_path.name}\n", encoding="utf-8")
    print(f"SHA256: {checksum}")
    print(f"Wrote: {sha_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
