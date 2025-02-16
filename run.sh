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

export PYTHONPATH=$(pwd)

python main.py \
--model qwen2.5-7b \
--dataset /leonardo_work/EUHPC_D12_071/projects/mm-exams/stratified_dataset.hf/ \
--model_path /leonardo_work/EUHPC_D12_071/projects/mm-exams/models/Qwen2.5-VL-7B-Instruct