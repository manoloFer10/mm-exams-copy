#!/bin/bash
#SBATCH -N 1
#SBATCH -A EUHPC_D12_071
#SBATCH -p boost_usr_prod
#SBATCH --job-name=evaluation
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=./slurm/eval_%j.log

export HF_HUB_OFFLINE=1
export PYTHONPATH=$(pwd)

# python main.py \
# --model pangea \
# --dataset /leonardo_work/EUHPC_D12_071/projects/mm-exams/stratified_dataset.hf/ \
# --model_path /leonardo_work/EUHPC_D12_071/projects/mm-exams/models/Pangea-7B-hf \
# --method zero-shot

# python main.py \
# --model molmo \
# --dataset /leonardo_work/EUHPC_D12_071/projects/mm-exams/dataset/dataset_0.hf/ \
# --model_path /leonardo_work/EUHPC_D12_071/projects/mm-exams/models/Molmo-7B-D-0924 \
# --method zero-shot \
# --output_name _0

python main.py \
--model qwen2.5-32b \
--dataset /leonardo_work/EUHPC_D12_071/projects/mm-exams/dataset/dataset_0.hf/ \
--model_path /leonardo_work/EUHPC_D12_071/projects/mm-exams/models/Qwen2.5-VL-7B-Instruct \
--method zero-shot \
--output_name _0
--subset multimdoal \
--output_name _no-image

# python main.py \
# --model aya-vision \
# --dataset dataset/dataset_3.hf \
# --method zero-shot \
# --api_key \
# --resume outputs/zero-shot/model_aya-vision/results_3.json \
# --output_name _3
