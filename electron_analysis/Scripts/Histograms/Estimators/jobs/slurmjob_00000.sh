#! /bin/bash
#SBATCH --job-name=MyElectronAnalysis_011
#SBATCH --output=/rwthfs/rz/cluster/home/op115134/Software/YasamanAnalysis/Scripts/Histograms/Estimators/output/output_00000.txt
#SBATCH --error=/rwthfs/rz/cluster/home/op115134/Software/YasamanAnalysis/Scripts/Histograms/Estimators/output/output_00000.txt
#SBATCH --time=6:00:00
#SBATCH --partition=amsclx
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=3882M
#SBATCH --account=amsclx
#SBATCH --constraint=Rocky8

source /rwthfs/rz/cluster/home/op115134/Software/YasamanAnalysis/Scripts/Histograms/Estimators/sandbox/environment.sh
export PSI_EXPORTS=OMP_NUM_THREADS
export PMI_BARRIER_ROUNDS=5

2Dhistogram_Ecal_TRD_Estimators.py --data_type ISS /home/op115134/Software/YasamanAnalysis/test/LeptonAnalysis_Tree_ISS_00184_00026.root
RET=${?}
echo
echo job 00000 exited with return code ${RET}

exit ${RET}
