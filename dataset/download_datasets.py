from huggingface_hub import hf_hub_download, list_repo_files
import json
import zipfile
from pathlib import Path
import os
from tqdm import tqdm

DATA_ROOT = Path("data/")

# hf_list = [
#     "https://huggingface.co/datasets/ckodser/Iranian_olympiad_of_informatics_multimodal_questions",
#     "https://huggingface.co/datasets/ckodser/driving_licence_with_image",
#     "https://huggingface.co/datasets/amayuelas/aya-mm-exams-spanish-medical",
#     "https://huggingface.co/datasets/azminetoushikwasi/bd-bcs-multimodal",
#     "https://huggingface.co/datasets/srajwal1/dsssb_hindi",
#     "https://huggingface.co/datasets/Jekaterina/multimodal-LT-exam",
#     "https://huggingface.co/datasets/srajwal1/olympiad-hindi",
#     "https://huggingface.co/datasets/srajwal1/driving-license-penn/tree/main",
#     "https://huggingface.co/datasets/srajwal1/driving-license-hindi/tree/main",
#     "https://huggingface.co/datasets/azminetoushikwasi/bd-bcs-multimodal",
#     "https://huggingface.co/datasets/1024m/chemistry-multimodal-exams",
#     "https://huggingface.co/datasets/dipikakhullar/gaokao",
#     "https://huggingface.co/datasets/jjzha/flemish_multimodal_exams_physician",
#     "https://huggingface.co/datasets/shayekh/BRTA-Driving-License-bn-BD",
#     "https://huggingface.co/datasets/1024m/chemistry-multimodal-exams-telugu",
#     "https://huggingface.co/datasets/jjzha/dutch-central-exam-mcq-multimodal-subset",
#     "https://huggingface.co/datasets/danylo-boiko/zno-vision",
#     "https://huggingface.co/datasets/rmahesh/NTSE_Punjabi_Multimodal",
#     "https://huggingface.co/datasets/dokato/multimodal-SK-exams",
#     "https://huggingface.co/datasets/dokato/multimodal-PL-exams",
#     "https://huggingface.co/datasets/dokato/multimodal-CS-exams/",
#     "https://huggingface.co/datasets/rmahesh/JEE_Main_Hindi_Multimodal",
#     "https://huggingface.co/datasets/rmahesh/UP_CET_Hindi_Multimodal",
#     "https://huggingface.co/datasets/amayuelas/aya-mm-exams-spanish-nursing",
#     "https://huggingface.co/datasets/shayekh/ju_iq_multimodal",
#     "https://huggingface.co/datasets/nicpopovic/amateur-radio-exam-germany",
#     "https://huggingface.co/datasets/danylo-boiko/sat",
#     "https://huggingface.co/datasets/shayekh/bn-vl-economics/",
#     "https://huggingface.co/datasets/shayekh/en-vl-biology",
#     "https://huggingface.co/datasets/josephimperial/vietnamese_geography_highschool_multimodal",
#     "https://huggingface.co/datasets/silviafernandez/mm-exams-biophysics",
#     "https://huggingface.co/datasets/sharad461/SSC-CGL-2023-hi-multimodal",
#     "https://huggingface.co/datasets/jebish7/PSC_Multimodal",
#     "https://huggingface.co/datasets/jebish7/PSC_Multimodal",
#     "https://huggingface.co/datasets/shayekh/bn-vl-geography",
#     "https://huggingface.co/datasets/azminetoushikwasi/hsc-bd-mm",
#     "https://huggingface.co/datasets/srajwal1/iitb_design/",
#     "https://huggingface.co/datasets/nazarkohut/driver-licence",
#     "https://huggingface.co/datasets/shayekh/bn-vl-biology",
#     "https://huggingface.co/datasets/josephimperial/vietnamese_mathematics_highschool_multimodal",
#     "https://huggingface.co/datasets/oltsy/rumk",
#     "https://huggingface.co/datasets/lubashi/UNESP_2014-2025_Multimodal_PTBR",
#     "https://huggingface.co/datasets/Maksim-KOS/fr-kan",
#     "https://huggingface.co/datasets/josephimperial/vietnamese_physics_highschool_multimodal",
#     "https://huggingface.co/datasets/jebish7/GATE_2022_Multimodal",
#     "https://huggingface.co/datasets/shayekh/en-vl-gate",
#     "https://huggingface.co/datasets/northern-64bit/Multimodal-EN-MCQ-Drivers-License-Exam",
#     "https://huggingface.co/datasets/bardia1383/Biology_university_entrance_preparation",
#     "https://huggingface.co/datasets/bardia1383/physics11_hard",
#     "https://huggingface.co/datasets/gabrielcmerlin/USP-University_Entrance_Exam-PTBR",
#     "https://huggingface.co/datasets/josephimperial/tagalog_professional_driving_multimodal",
#     "https://huggingface.co/datasets/Otavio12/Brazil-Medicine-Schools-Entrance-Exams-FAMERP-SANTA-CASA",
#     "https://huggingface.co/datasets/letmarchezi/unicamp_2019_2025",
#     "https://huggingface.co/datasets/AndreMitri/Unicamp_comvest_2011_to_2018",
#     "https://huggingface.co/datasets/maraljab/Geometry_Geology",
#     "https://huggingface.co/datasets/Mafarahani/Geology",
#     "https://huggingface.co/datasets/aruby/Arabic_Exams",
#     "https://huggingface.co/datasets/mponty/srmk-sr",
#     "https://huggingface.co/datasets/mponty/srmk-hr",
#     "https://huggingface.co/datasets/mponty/srmk-hu",
#     "https://huggingface.co/datasets/mponty/srmk-ro",
#     "https://huggingface.co/datasets/setayeshheydari1010/physics_geology",
#     "https://huggingface.co/datasets/DrishtiSharma/driving-license-test-marathi",
#     "https://huggingface.co/datasets/DrishtiSharma/driving-license-test-gujarati",
#     "https://huggingface.co/datasets/dipikakhullar/japanese_eju",
#     "https://huggingface.co/datasets/dipikakhullar/korean_csat",
#     "https://huggingface.co/datasets/mariagrandury/galician_culture_quiz",
#     "https://huggingface.co/datasets/mrshu/multimodal-sk-mcq-driving-license-exam",
#     "https://huggingface.co/datasets/AndreMitri/ENEM_multimodal_2021_to_2022",
#     "https://huggingface.co/datasets/letmarchezi/ENEM_Brazil_National_High_School_Exam_2022_to_2024",
#     "https://huggingface.co/datasets/mrshu/multimodal-cs-mcq-driving-license-exam",
#     "https://huggingface.co/datasets/johanobandoc/Natural_sciences/tree/main",
#     "https://huggingface.co/datasets/johanobandoc/Math/tree/main",
#     "https://huggingface.co/datasets/johanobandoc/Reading_comprehension/tree/main",
#     "https://huggingface.co/datasets/johanobandoc/Physics/tree/main",
#     "https://huggingface.co/datasets/johanobandoc/Driving_test/tree/main",
# ]


def get_files(repo_id: str):
    dataset_name = repo_id.split("/")[-1]
    dataset_dir = DATA_ROOT / dataset_name
    # Check if the dataset has already been processed
    if dataset_dir.exists():
        json_files = list(dataset_dir.glob("*.json"))
        if json_files:
            print(f"Dataset {dataset_name} already processed. Skipping.")
            return

    # Create the dataset directory
    files = list_repo_files(repo_id, repo_type="dataset")
    json_files = [f for f in files if f.endswith(".json")]
    images = [f for f in files if f.endswith(".png")]
    images_zip = [f for f in files if f.endswith(".zip")]

    if len(json_files) != 1:
        print(f"There should be only one json file.")
        return
    if len(images) < 1 and len(images_zip) != 1:
        print(f"No images found.")
        return
    dataset_dir.mkdir(parents=True, exist_ok=False)
    json_path = hf_hub_download(
        repo_id, filename=json_files[0], local_dir=dataset_dir, repo_type="dataset"
    )
    print(f"JSON file saved to: {json_path}")
    if len(images_zip) == 1:
        zip_path = hf_hub_download(
            repo_id, filename=images_zip[0], local_dir=dataset_dir, repo_type="dataset"
        )
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
    elif len(images) > 1:
        for image in tqdm(images):
            image_path = hf_hub_download(
                repo_id,
                filename=image,
                repo_type="dataset",
                local_dir=dataset_dir,
            )
        print(f"Images saved")


if __name__ == "__main__":
    with open("dataset/exams_list.txt", "r") as f:
        repo_links = f.readlines()

    #    repo_links = [line.strip() for line in repo_links]
    repo_links = [repo.strip().replace("/tree/main", "") for repo in repo_links]
    repo_links = set(
        [repo.replace("https://huggingface.co/datasets/", "") for repo in repo_links]
    )

    repo_processed = os.listdir(DATA_ROOT)
    import code

    code.interact(local=locals())
    missed_datasets = []
    for n, repo_id in enumerate(repo_links):
        if repo_id.split("/")[1] in repo_processed:
            print(f"Skiping repo: {repo_id}, already processed.")
            continue
        try:
            print(f"Processing dataset: {repo_id} ({n}/{len(repo_links)})")
            get_files(repo_id)
        except Exception as e:
            print(f"Failed to process {repo_id}: {e}")
            missed_datasets.append(repo_id)
    if len(missed_datasets) > 0:
        print("Some datasets could not be downloaded:")
        print(missed_datasets)
