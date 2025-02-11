#!/bin/bash
#SBATCH -N 1
#SBATCH -A EUHPC_D12_071
#SBATCH -p lrd_all_serial
#SBATCH --job-name=evaluation
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=./slurm/eval_%j.log

export PYTHONPATH=$(pwd)

python main.py \
--model gemini-1.5-pro \
--dataset /leonardo_work/EUHPC_D12_071/projects/mm-exams/stratified_dataset.hf/ \
--api_key \