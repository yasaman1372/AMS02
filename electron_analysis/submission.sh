#!/bin/bash
#SBATCH --job-name=echo
#SBATCH --output=/rwthfs/rz/cluster/home/op115134/Software/YasamanAnalysis/output/output_%5a.txt
#SBATCH --error=/rwthfs/rz/cluster/home/op115134/Software/YasamanAnalysis/output/output_%5a.txt
#SBATCH --time=6:00:00
#SBATCH --partition=amsclx
#SBATCH --array=0-0
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=3882M
#SBATCH --account=amsclx
#SBATCH --constraint=Rocky8


bash /rwthfs/rz/cluster/home/op115134/Software/YasamanAnalysis/jobs/slurmjob_$(printf "%05d" ${SLURM_ARRAY_TASK_ID}).sh
