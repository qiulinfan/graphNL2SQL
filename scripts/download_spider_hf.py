"""
Download Spider dataset with schemas from HuggingFace.
Uses richardr1126/spider-schema which includes table schema information.
"""

import json
from pathlib import Path
from datasets import load_dataset

# Use project root, not scripts/ directory
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "spider"


def download_spider():
    """Download Spider dataset with schema info from HuggingFace."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Spider dataset with schemas from HuggingFace...")

    # Use richardr1126/spider-schema which has schema info embedded
    dataset = load_dataset("richardr1126/spider-schema")

    # Check available features
    print(f"Available splits: {list(dataset.keys())}")
    print(f"Features: {dataset['train'].features.keys()}")

    # Save train split
    train_examples = []
    schemas = {}

    for example in dataset["train"]:
        db_id = example["db_id"]

        train_examples.append({
            "db_id": db_id,
            "query": example["query"],
            "question": example["question"]
        })

        # Extract schema if not already stored
        if db_id not in schemas:
            schemas[db_id] = {
                "db_id": db_id,
                "table_names_original": example.get("table_names_original", []),
                "column_names_original": example.get("column_names_original", []),
                "column_types": example.get("column_types", []),
                "primary_keys": example.get("primary_keys", []),
                "foreign_keys": example.get("foreign_keys", []),
            }

    with open(DATA_DIR / "train_spider.json", 'w', encoding='utf-8') as f:
        json.dump(train_examples, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(train_examples)} training examples")

    # Save validation split as dev
    dev_examples = []

    for example in dataset["validation"]:
        db_id = example["db_id"]

        dev_examples.append({
            "db_id": db_id,
            "query": example["query"],
            "question": example["question"]
        })

        # Extract schema if not already stored
        if db_id not in schemas:
            schemas[db_id] = {
                "db_id": db_id,
                "table_names_original": example.get("table_names_original", []),
                "column_names_original": example.get("column_names_original", []),
                "column_types": example.get("column_types", []),
                "primary_keys": example.get("primary_keys", []),
                "foreign_keys": example.get("foreign_keys", []),
            }

    with open(DATA_DIR / "dev.json", 'w', encoding='utf-8') as f:
        json.dump(dev_examples, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(dev_examples)} dev examples")

    # Save schemas
    tables_list = list(schemas.values())

    # Check if schemas have content
    non_empty_schemas = sum(1 for s in tables_list if s["table_names_original"])
    print(f"Schemas with table info: {non_empty_schemas}/{len(tables_list)}")

    with open(DATA_DIR / "tables.json", 'w', encoding='utf-8') as f:
        json.dump(tables_list, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(tables_list)} database schemas")

    print("Spider dataset downloaded successfully!")


def download_spider_from_repo():
    """
    Download Spider tables.json directly from the spider GitHub archive.
    Falls back to this if HuggingFace doesn't have schema info.
    """
    import requests

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Try downloading tables.json from a known mirror
    urls_to_try = [
        "https://huggingface.co/datasets/spider/resolve/main/tables.json",
        "https://raw.githubusercontent.com/defog-ai/sql-eval/main/data/spider/tables.json",
    ]

    for url in urls_to_try:
        try:
            print(f"Trying to download tables.json from {url}...")
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                data = response.json()
                if data and isinstance(data, list) and len(data) > 0:
                    # Check if it has actual schema content
                    if data[0].get("table_names_original"):
                        with open(DATA_DIR / "tables.json", 'w', encoding='utf-8') as f:
                            json.dump(data, f, indent=2, ensure_ascii=False)
                        print(f"Downloaded {len(data)} schemas from {url}")
                        return True
        except Exception as e:
            print(f"Failed: {e}")
            continue

    return False


if __name__ == "__main__":
    try:
        download_spider()

        # Check if schemas have content
        with open(DATA_DIR / "tables.json", 'r') as f:
            tables = json.load(f)

        has_schema = any(t.get("table_names_original") for t in tables)
        if not has_schema:
            print("\nSchema info is empty, trying alternative download...")
            download_spider_from_repo()
    except Exception as e:
        print(f"Error with HuggingFace: {e}")
        print("Trying alternative download...")
        download_spider_from_repo()
