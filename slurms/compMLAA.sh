#!/bin/bash -l
#SBATCH --partition=GPU24Go
#SBATCH  --cpus-per-task 6
#SBATCH --gres=gpu
#SBATCH --output slurm_out/comp_mlaa_tof_0.00001/output_comp_mlaa_%j.txt
#SBATCH --job-name MLAA
#
echo MLAA
cd ~
singularity exec --nv images/pytorch_parallelproj.sif python code/dl-mu-map-from-mlaa/scripts/compute_mlaa.py
exit 0
