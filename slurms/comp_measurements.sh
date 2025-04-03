#!/bin/bash -l
#SBATCH --partition=GPU24Go
#SBATCH  --cpus-per-task 6
#SBATCH --gres=gpu
#SBATCH --output slurm_out/comp_measurements/output_%j.txt
#SBATCH --job-name compMeasurements
#
echo compute measurements
cd ~
singularity exec --nv images/pytorch_parallelproj.sif python code/dl-mu-map-from-mlaa/scripts/compute_measurements.py
exit 0
