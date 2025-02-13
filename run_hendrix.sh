#!/bin/bash
#SBATCH -N 1
#SBATCH -p gpu
#SBATCH --job-name=hendrix-eval
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --output=./slurm/eval_%j.log

export PYTHONPATH=$(pwd)

python main.py \
--model gemini-1.5-pro \
--num_samples 1 \
--dataset ./dataset/stratified_dataset.hf/
#--model_path /leonardo_work/EUHPC_D12_071/projects/mm-exams/models/qwen2-vl-7b-instruct/ \
#--selected_langs ['slovak'] \
#--api_key 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 \
