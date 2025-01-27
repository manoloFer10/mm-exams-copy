from datasets import Dataset, concatenate_datasets, Features, Value, Sequence
from pathlib import Path
import os

DATA_ROOT = Path("data/")

expected_features = Features(
    {
        "language": Value("string"),
        "country": Value("string"),
        "file_name": Value("string"),
        "source": Value("string"),
        "license": Value("string"),
        "level": Value("string"),
        "category_en": Value("string"),
        "category_original_lang": Value("string"),
        "original_question_num": Value("string"),
        "question": Value("string"),
        "options": Sequence(Value("string")),
        "answer": Value("int64"),
        "image_png": Value("string"),
        "image_information": Value("string"),
        "image_type": Value("string"),
        "parallel_question_id": Value("null"),
        "image": Value("string"),
    }
)


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
            json_files = list(dataset_dir.glob("*.json"))
            if len(json_files) != 1:
                print(
                    f"There should be one and only one json file, but {len(json_files)} found: {json_files}."
                )
                continue
            print(f"Loading dataset from {json_files[0]}")
            dataset = Dataset.from_json(str(json_files[0]))

            # Map image paths
            # check that there is an image folder with iamges, if not pass for now.
            images_dir = dataset_dir / "images"
            if images_dir.exists() and any(images_dir.iterdir()):
                print(f"Mapping image paths for {dataset_dir}")
                # add support for non multimodal datasets
                dataset = dataset.map(
                    lambda x: {
                        "image": (
                            str(images_dir / x["image_png"]) if x["image_png"] else None
                        )
                    }
                )
            else:
                print(f"No images on {dataset_dir}")

            # Add the dataset to the list
            datasets.append(dataset)

    # Concatenate all datasets
    if datasets:
        datasets = [dataset.cast(expected_features) for dataset in datasets]
        print("Concatenating datasets...")
        merged_dataset = concatenate_datasets(datasets)
        return merged_dataset
    else:
        raise ValueError("No datasets found to merge.")


# Example usage
if __name__ == "__main__":
    merged_dataset = merge_datasets(DATA_ROOT)
    merge_dataset.save_to_disk("output_dir")
