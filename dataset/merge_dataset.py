from datasets import Dataset, concatenate_datasets, Features, Value, Sequence
from pathlib import Path
import os
import regex as re

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
        "parallel_question_id": Value("string"),
        "image": Value("string"),
    }
)


def check_options_format(question):
    # TODO: should update this in order to consolidate wrapper categories for 'category_en'.
    pattern = r"^\s*[A-Za-z][^\w\s]*\s*"

    question["options"] = [
        (
            re.sub(pattern, "", option, flags=re.IGNORECASE)
            if re.match(pattern, option, flags=re.IGNORECASE)
            else option
        )
        for option in question["options"]
    ]

    return question


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
    datasets_with_problems = []
    # Iterate over all subdirectories in the data directory
    for dataset_dir in data_dir.iterdir():
        if dataset_dir.is_dir():
            json_files = list(dataset_dir.glob("*.json"))
            try:
                for json_dataset in json_files:
                    dataset = Dataset.from_json(str(json_dataset))

                    # Map image paths
                    # check that there is an image folder with iamges, if not pass for now.
                    images_dir = dataset_dir / "images"
                    if images_dir.exists() and any(images_dir.iterdir()):
                        print(f"Mapping image paths for {dataset_dir}")
                        # add support for non multimodal datasets
                        dataset = dataset.map(
                            lambda x: {
                                "image": (
                                    str(images_dir / x["image_png"])
                                    if x["image_png"]
                                    else None
                                )
                            }
                        )
                    else:
                        print(f"No images on {dataset_dir}")

                    # Add the dataset to the list
                    datasets.append(dataset)
            except:
                print(f"PROBLEM READING THE FILE: {json_dataset}")
                datasets_with_problems.append(json_dataset)
                # return None

    # Concatenate all datasets
    print("Merging Datasets")
    if datasets:
        datasets = [dataset.cast(expected_features) for dataset in datasets]
        print("Concatenating datasets...")
        merged_dataset = concatenate_datasets(datasets)
        if len(datasets_with_problems) > 1:
            print("Some datasets could not be added")
            print(datasets_with_problems)
        return merged_dataset
    else:
        raise ValueError("No datasets found to merge.")


# Example usage
if __name__ == "__main__":
    merged_dataset = merge_datasets(DATA_ROOT)
    merged_dataset.save_to_disk("dataset/full.hf")
