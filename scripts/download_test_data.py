#!/usr/bin/env python3
"""Download ground-truth test data from nnAudio repository."""

import urllib.request
from pathlib import Path
import sys

BASE_URL = "https://github.com/KinWaiCheuk/nnAudio/raw/refs/heads/master/Installation/tests/ground-truths/"

GROUND_TRUTH_FILES = [
    "cfp_new.pt",
    "cfp_original.pt",
    "linear-sweep-cqt-1992-complex-ground-truth.npy",
    "linear-sweep-cqt-1992-mag-ground-truth.npy",
    "linear-sweep-cqt-1992-phase-ground-truth.npy",
    "linear-sweep-cqt-2010-complex-ground-truth.npy",
    "linear-sweep-cqt-2010-mag-ground-truth.npy",
    "linear-sweep-cqt-2010-phase-ground-truth.npy",
    "log-sweep-cqt-1992-complex-ground-truth.npy",
    "log-sweep-cqt-1992-mag-ground-truth.npy",
    "log-sweep-cqt-1992-phase-ground-truth.npy",
    "log-sweep-cqt-2010-complex-ground-truth.npy",
    "log-sweep-cqt-2010-mag-ground-truth.npy",
    "log-sweep-cqt-2010-phase-ground-truth.npy",
]


def download_file(url: str, output_path: Path) -> None:
    """Download a file from URL to output path."""
    print(f"Downloading {url} -> {output_path}")
    
    try:
        urllib.request.urlretrieve(url, output_path)
        print(f"✓ Downloaded {output_path.name}")
    except Exception as e:
        print(f"✗ Failed to download {output_path.name}: {e}")
        sys.exit(1)


def main():
    """Download all ground-truth files."""
    # Create output directory
    output_dir = Path("tests/ground_truths")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading ground-truth files to {output_dir}")
    
    # Download each file
    for filename in GROUND_TRUTH_FILES:
        url = BASE_URL + filename
        output_path = output_dir / filename
        
        # Skip if file already exists
        if output_path.exists():
            print(f"⚠ Skipping {filename} (already exists)")
            continue
            
        download_file(url, output_path)
    
    print(f"\n✓ Downloaded {len(GROUND_TRUTH_FILES)} ground-truth files")


if __name__ == "__main__":
    main()