from huggingface_hub import hf_hub_download, list_repo_files
import json
import zipfile
from pathlib import Path
import os

DATA_ROOT = Path("dataset/")
hf_list = [
    "https://huggingface.co/datasets/ckodser/Iranian_olympiad_of_informatics_multimodal_questions",
    "https://huggingface.co/datasets/ckodser/driving_licence_with_image",
    "https://huggingface.co/datasets/amayuelas/aya-mm-exams-spanish-medical",
    "https://huggingface.co/datasets/azminetoushikwasi/bd-bcs-multimodal",
    "https://huggingface.co/datasets/srajwal1/dsssb_hindi",
    "https://huggingface.co/datasets/Jekaterina/multimodal-LT-exam",
    "https://huggingface.co/datasets/srajwal1/olympiad-hindi",
    "https://huggingface.co/datasets/srajwal1/driving-license-penn/tree/main",
    "https://huggingface.co/datasets/srajwal1/driving-license-hindi/tree/main",
    "https://huggingface.co/datasets/azminetoushikwasi/bd-bcs-multimodal",
    "https://huggingface.co/datasets/1024m/chemistry-multimodal-exams",
    "https://huggingface.co/datasets/dipikakhullar/gaokao",
    "https://huggingface.co/datasets/jjzha/flemish_multimodal_exams_physician",
    "https://huggingface.co/datasets/shayekh/BRTA-Driving-License-bn-BD",
    "https://huggingface.co/datasets/1024m/chemistry-multimodal-exams-telugu",
    "https://huggingface.co/datasets/jjzha/dutch-central-exam-mcq-multimodal-subset",
    "https://huggingface.co/datasets/danylo-boiko/zno-vision",
    "https://huggingface.co/datasets/rmahesh/NTSE_Punjabi_Multimodal",
    "https://huggingface.co/datasets/dokato/multimodal-SK-exams",
    "https://huggingface.co/datasets/dokato/multimodal-PL-exams",
    "https://huggingface.co/datasets/dokato/multimodal-CS-exams/",
    "https://huggingface.co/datasets/rmahesh/JEE_Main_Hindi_Multimodal",
    "https://huggingface.co/datasets/rmahesh/UP_CET_Hindi_Multimodal",
    "https://huggingface.co/datasets/amayuelas/aya-mm-exams-spanish-nursing",
    "https://huggingface.co/datasets/shayekh/ju_iq_multimodal",
]


def get_files(repo_id: str):
    dataset_name = repo_id.split("/")[-1]
    dataset_dir = DATA_ROOT / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    files = list_repo_files(repo_id, repo_type="dataset")
    for f in files:
        f = Path(f)
        if f.suffix == ".json":
            json_path = hf_hub_download(repo_id, filename=str(f), local_dir=dataset_dir)
            print(f"JSON file saved to: {json_path}")

        elif f.suffix == ".zip":
            zip_path = hf_hub_download(repo_id, filename=str(f), local_dir=dataset_dir)
            print(f"ZIP file saved to: {zip_path}")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                image_files = [
                    file
                    for file in zip_ref.namelist()
                    if file.lower().endswith((".png", ".jpg", ".jpeg"))
                ]
                for image_file in image_files:
                    zip_ref.extract(image_file, dataset_dir)
                    print(f"Extracted {image_file} to {dataset_dir}")
            os.remove(zip_path)
            print(f"Deleted ZIP file: {zip_path}")

        elif f.suffix in [".jpeg", ".jpg", ".png"]:
            image_path = hf_hub_download(
                repo_id,
                filename=str(f),
                repo_type="dataset",
                local_dir=dataset_dir,
            )
            print(f"Image file saved to: {image_path}")


if __name__ == "__main__":
    for repo_link in hf_list:
        repo_link = repo_link.replace("/tree/main", "")
        repo_id = repo_link.replace("https://huggingface.co/datasets/", "")
        try:
            print(f"Processing dataset: {repo_id}")
            get_files(repo_id)
        except Exception as e:
            print(f"Failed to process {repo_id}: {e}")
