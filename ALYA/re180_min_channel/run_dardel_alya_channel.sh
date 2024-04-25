#!/bin/bash -l
#SBATCH -A naiss2023-3-13
#SBATCH -J channel_re180
#SBATCH -t 4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=2
#SBATCH -p main
#SBATCH --output "debug_det.out"
#SBATCH --error "debug_det.err"
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=polsm@kth.se

cd ./mesh
. doit.sh

cd ..

. ~/projects_dardel/env-tf-march24-2/activate_tf_march24.sh 

## check all slurm variables 
echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo ""
echo "Number of Nodes Allocated      = $SLURM_JOB_NUM_NODES"
echo "Number of Tasks Allocated      = $SLURM_NTASKS"
echo "Number of Cores/Task Allocated = $SLURM_CPUS_PER_TASK"
echo "Memory per node Allocated      = $SLURM_MEM_PER_NODE"
echo "Memory per cpu Allocated       = $SLURM_MEM_PER_CPU"

# launch training in 3D
echo "Script initiated at `date` on `hostname`"
srun -n 120 ./alya_3D_march24_cray.x channel
echo "Script initiated at `date` on `hostname`"
