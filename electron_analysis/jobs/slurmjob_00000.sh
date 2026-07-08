#! /bin/bash
#SBATCH --job-name=echo
#SBATCH --output=/rwthfs/rz/cluster/home/op115134/Software/YasamanAnalysis/output/output_00000.txt
#SBATCH --error=/rwthfs/rz/cluster/home/op115134/Software/YasamanAnalysis/output/output_00000.txt
#SBATCH --time=6:00:00
#SBATCH --partition=amsclx
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=3882M
#SBATCH --account=amsclx
#SBATCH --constraint=Rocky8

source /rwthfs/rz/cluster/home/op115134/Software/YasamanAnalysis/sandbox/environment.sh
export PSI_EXPORTS=OMP_NUM_THREADS
export PMI_BARRIER_ROUNDS=5

echo 
RET=${?}
echo
echo job 00000 exited with return code ${RET}

exit ${RET}
