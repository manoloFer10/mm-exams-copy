from datasets import Dataset, concatenate_datasets
from pathlib import Path
import os


def merge_datasets(data_dir: str):
    """
    Merge all JSON datasets in the given directory into a single dataset.
    Maps image paths for each dataset before concatenation.

    Args:
        data_dir (str): Path to the directory containing the datasets.

    Returns:
        Dataset: A single concatenated dataset.
    """
    data_dir = Path(data_dir)
    datasets = []

    # Iterate over all subdirectories in the data directory
    for dataset_dir in data_dir.iterdir():
        if dataset_dir.is_dir():
            # Find the JSON file in the dataset directory
            json_files = list(dataset_dir.glob("*.json"))
            if not json_files:
                print(f"No JSON file found in {dataset_dir}. Skipping.")
                continue

            # Load the JSON file as a dataset
            json_file = json_files[
                0
            ]  # Assuming there's only one JSON file per directory
            print(f"Loading dataset from {json_file}")
            dataset = Dataset.from_json(str(json_file))

            # Map image paths
            images_dir = dataset_dir / "images"
            if images_dir.exists():
                print(f"Mapping image paths for {dataset_dir}")
                dataset = dataset.map(
                    lambda x: {
                        "image_path": str(
                            images_dir / x["image_name"]
                        )  # Adjust "image_name" to your JSON field
                    }
                )

            # Add the dataset to the list
            datasets.append(dataset)

    # Concatenate all datasets
    if datasets:
        print("Concatenating datasets...")
        merged_dataset = concatenate_datasets(datasets)
        return merged_dataset
    else:
        raise ValueError("No datasets found to merge.")


# Example usage
if __name__ == "__main__":
    data_dir = "dataset"  # Path to your dataset directory
    merged_dataset = merge_datasets(data_dir)
    print(merged_dataset)
