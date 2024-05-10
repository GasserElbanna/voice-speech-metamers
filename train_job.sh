#!/bin/bash
#SBATCH --job-name=saganet_metamer
#SBATCH --output=/om2/user/annesyab/SLP_Project_2024/voice-speech-metamers/job_logs/job_%j.out
#SBATCH --error=/om2/user/annesyab/SLP_Project_2024/voice-speech-metamers/job_logs/job_%j.err
#SBATCH --mem=25Gb
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --partition=mcdermott

source /etc/profile.d/modules.sh
module load openmind/miniconda/3.9.1
module load openmind/git-lfs/2.13.2

source activate /om2/user/annesyab/summer2023/anaconda/envs/metamer_310

srun python train.py