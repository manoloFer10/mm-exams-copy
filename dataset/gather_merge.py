import os
import json
import argparse
import pandas as pd
from datasets import Dataset, concatenate_datasets, load_from_disk
from huggingface_hub import hf_hub_download
import zipfile
import requests
from typing import Dict, List, Optional


def load_cached_datasets(cache_path: str) -> Dict[str, bool]:
    """
    Load the cached JSON file to track already downloaded datasets.
    If the file doesn't exist, return an empty dictionary.
    """
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            return json.load(f)
    return {}


def save_cached_datasets(cache_path: str, cache: Dict[str, bool]):
    """
    Save the updated cache to the JSON file.
    """
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)


def download_and_extract_images(image_url: str, output_dir: str):
    """
    Download and extract the images.zip file.
    """
    os.makedirs(output_dir, exist_ok=True)
    zip_path = os.path.join(output_dir, "images.zip")

    # Download the zip file
    response = requests.get(image_url)
    with open(zip_path, "wb") as f:
        f.write(response.content)

    # Extract the zip file
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)

    # Remove the zip file
    os.remove(zip_path)


def load_dataset_from_json(json_url: str) -> Dataset:
    """
    Load a dataset from a JSON file hosted online.
    """
    # Download the JSON file
    response = requests.get(json_url)
    data = response.json()

    # Convert to HF Dataset
    return Dataset.from_dict(data)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge datasets from a CSV file into a single Hugging Face dataset."
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="dataset/multimodal_exams_collection_c4ai.csv",
        help="Path to the CSV file containing 'json-link' and 'image-link' columns.",
    )
    parser.add_argument(
        "--cache_path",
        type=str,
        default="dataset/cache.json",
        help="Path to the JSON file to track already downloaded datasets. Default: 'cache.json'.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save the final merged Hugging Face dataset. If not provided, the dataset will not be saved to disk.",
    )
    # Parse arguments
    args = parser.parse_args()
    return args


def merge_datasets(
    csv_path: str,
    cache_path: str,
    output_dataset_path: Optional[str] = None,
) -> Dataset:
    df = pd.read_csv(csv_path, header=1)
    import code

    code.interact(local=locals())


if __name__ == "__main__":
    args = parse_args()
    # Merge datasets
    final_dataset = merge_datasets(
        csv_path=args.csv_path,
        cache_path=args.cache_path,
        output_dataset_path=args.output_path,
    )
    print("Dataset merging completed!")
