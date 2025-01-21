from huggingface_hub import hf_hub_download, list_repo_files
import json
import zipfile

dataset_name ='dokato/multimodal-CS-exams'

# List all files in the repo
files = list_repo_files(dataset_name, repo_type="dataset")

# Download specific files
for file in files:
    if '.json' in file:
        json_data = hf_hub_download(dataset_name, filename=file, repo_type="dataset")
    if  '.zip' in file:
        images = hf_hub_download(dataset_name, filename=file, repo_type="dataset")
        with zipfile.ZipFile(images, 'r') as zip_ref:
            # List all PNG files
            png_files = [f for f in zip_ref.namelist() if f.lower().endswith('.png')]



with open(json_data, encoding='utf-8') as f:
    data = json.load(f)

    for question in data:
        if question['image_png']: 
            file_name = question['image_png']
            png_file = png_files[file_name]