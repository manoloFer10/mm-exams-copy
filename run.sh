#!/bin/bash
#SBATCH -N 1
#SBATCH -A EUHPC_D12_071
#SBATCH -p boost_usr_prod
#SBATCH --job-name=evaluation
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=./slurm/eval_%j.log

export PYTHONPATH=$(pwd)

python main.py \
--model qwen2-7b \
--model_path /leonardo_work/EUHPC_D12_071/projects/mm-exams/models/qwen2-vl-7b-instruct/ \
--dataset /leonardo_work/EUHPC_D12_071/projects/mm-exams/merged_datasets.hf/
#--selected_langs ['slovak'] \
#--dataset dokato/multimodal-SK-exams \
#--api_key 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 \