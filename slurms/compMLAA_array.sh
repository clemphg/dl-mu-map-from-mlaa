#!/bin/bash -l
#SBATCH --partition=GPU24Go
#SBATCH  --cpus-per-task 6
#SBATCH --gres=gpu
#SBATCH --output slurm_out/comp_mlaa_tof_0.000001/output_comp_mlaa_%j.txt
#SBATCH --job-name MLAA
#SBATCH --array=0-9
#
echo array is: $SLURM_TASK_ARRAY_ID
cd ~
singularity exec --nv images/pytorch_parallelproj.sif python code/dl-mu-map-from-mlaa/scripts/compute_mlaa_slurmarray.py $SLURM_TASK_ARRAY_ID
exit 0
