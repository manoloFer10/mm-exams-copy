#!/bin/bash
#SBATCH -A EUHPC_D12_071
#SBATCH --nodes=1
#SBATCH -p boost_usr_prod
#SBATCH --ntasks-per-node=4
#SBATCH --job-name=CreateDataset
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20GB
#SBATCH --time=4:00:00

export PYTHONPATH=$(pwd)
python dataset/download.sh