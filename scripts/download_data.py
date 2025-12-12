"""
Download WikiSQL and Spider datasets for NL2SQL training.
"""

import os
import json
import zipfile
import tarfile
import requests
from tqdm import tqdm
from pathlib import Path


# Use project root, not scripts/ directory
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Dataset URLs
WIKISQL_URL = "https://github.com/salesforce/WikiSQL/raw/master/data.tar.bz2"
SPIDER_URL = "https://drive.google.com/uc?export=download&id=1iRDVHLr4mX2wQKSgA9J8Pire73Jahh0m"


def download_file(url: str, dest_path: Path, desc: str = "Downloading") -> None:
    """Download a file with progress bar."""
    response = requests.get(url, stream=True, allow_redirects=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))

    with open(dest_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def download_wikisql() -> Path:
    """Download and extract WikiSQL dataset."""
    wikisql_dir = DATA_DIR / "wikisql"

    if wikisql_dir.exists() and (wikisql_dir / "train.jsonl").exists():
        print("WikiSQL already downloaded.")
        return wikisql_dir

    wikisql_dir.mkdir(parents=True, exist_ok=True)
    archive_path = DATA_DIR / "wikisql.tar.bz2"

    print("Downloading WikiSQL dataset...")
    download_file(WIKISQL_URL, archive_path, "WikiSQL")

    print("Extracting WikiSQL...")
    with tarfile.open(archive_path, "r:bz2") as tar:
        tar.extractall(DATA_DIR)

    # Move files from data/ subdirectory to wikisql/
    extracted_dir = DATA_DIR / "data"
    if extracted_dir.exists():
        for f in extracted_dir.iterdir():
            dest = wikisql_dir / f.name
            if not dest.exists():
                f.rename(dest)
        extracted_dir.rmdir()

    # Clean up archive
    archive_path.unlink()

    print(f"WikiSQL extracted to {wikisql_dir}")
    return wikisql_dir


def download_spider() -> Path:
    """
    Download Spider dataset from Google Drive using gdown.

    The zip extracts to 'spider_data/' but we rename it to 'spider/'.
    """
    spider_dir = DATA_DIR / "spider"

    # Check if Spider is already downloaded
    required_files = ["train_spider.json", "dev.json", "tables.json"]
    if spider_dir.exists():
        existing = [f for f in required_files if (spider_dir / f).exists()]
        if len(existing) == len(required_files):
            print("Spider already downloaded.")
            return spider_dir
        elif existing:
            print(f"Spider partially downloaded. Found: {existing}")

    # Try to download using gdown
    try:
        import gdown
    except ImportError:
        print("Installing gdown...")
        import subprocess
        subprocess.check_call(["pip", "install", "gdown", "-q"])
        import gdown

    print("Downloading Spider dataset from Google Drive...")
    zip_path = DATA_DIR / "spider_data.zip"

    # Google Drive file ID
    file_id = "1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J"
    url = f"https://drive.google.com/uc?id={file_id}"

    try:
        gdown.download(url, str(zip_path), quiet=False)

        print("Extracting Spider dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)

        # Rename spider_data to spider
        spider_data_dir = DATA_DIR / "spider_data"
        if spider_data_dir.exists():
            if spider_dir.exists():
                import shutil
                shutil.rmtree(spider_dir)
            spider_data_dir.rename(spider_dir)
            print(f"Renamed spider_data/ -> spider/")

        # Clean up zip file
        zip_path.unlink()
        print(f"Spider extracted to {spider_dir}")

    except Exception as e:
        print(f"Error downloading Spider: {e}")
        print()
        print("Manual download instructions:")
        print("  1. Go to: https://drive.google.com/file/d/1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J/view")
        print("  2. Download spider_data.zip")
        print("  3. Extract and rename to: data/spider/")

    spider_dir.mkdir(parents=True, exist_ok=True)
    return spider_dir


def verify_datasets() -> dict:
    """Verify downloaded datasets and return statistics."""
    stats = {}

    # Check WikiSQL
    wikisql_dir = DATA_DIR / "wikisql"
    if wikisql_dir.exists():
        wikisql_files = {
            "train": wikisql_dir / "train.jsonl",
            "dev": wikisql_dir / "dev.jsonl",
            "test": wikisql_dir / "test.jsonl",
            "train_tables": wikisql_dir / "train.tables.jsonl",
            "dev_tables": wikisql_dir / "dev.tables.jsonl",
            "test_tables": wikisql_dir / "test.tables.jsonl",
        }

        stats["wikisql"] = {}
        for name, path in wikisql_files.items():
            if path.exists():
                with open(path, 'r') as f:
                    count = sum(1 for _ in f)
                stats["wikisql"][name] = count
                print(f"WikiSQL {name}: {count} entries")

    # Check Spider
    spider_dir = DATA_DIR / "spider"
    if spider_dir.exists():
        spider_files = {
            "train": spider_dir / "train_spider.json",
            "dev": spider_dir / "dev.json",
            "tables": spider_dir / "tables.json",
        }

        stats["spider"] = {}
        for name, path in spider_files.items():
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                count = len(data)
                stats["spider"][name] = count
                print(f"Spider {name}: {count} entries")

    return stats


def main():
    """Download all datasets."""
    print("=" * 60)
    print("NL2SQL Dataset Downloader")
    print("=" * 60)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Download WikiSQL
    print("\n[1/2] WikiSQL Dataset")
    print("-" * 40)
    try:
        download_wikisql()
    except Exception as e:
        print(f"Error downloading WikiSQL: {e}")

    # Download Spider
    print("\n[2/2] Spider Dataset")
    print("-" * 40)
    try:
        download_spider()
    except Exception as e:
        print(f"Error downloading Spider: {e}")

    # Verify
    print("\n" + "=" * 60)
    print("Dataset Verification")
    print("=" * 60)
    verify_datasets()

    print("\nDone!")


if __name__ == "__main__":
    main()
