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
# --model_path /leonardo_work/EUHPC_D12_071/projects/mm-exams/models/Pangea-7B-hf

# python main.py \
# --model molmo \
# --dataset /leonardo_work/EUHPC_D12_071/projects/mm-exams/dataset-fewshot.hf/ \
# --model_path /leonardo_work/EUHPC_D12_071/projects/mm-exams/models/Molmo-7B-D-0924 \
# --method few-shot

python main.py \
--model deepseek \
--dataset /leonardo_work/EUHPC_D12_071/projects/mm-exams/stratified_dataset.hf/ \
--model_path /leonardo_work/EUHPC_D12_071/projects/mm-exams/models/deepseek-vl2-small \
--method zero-shot