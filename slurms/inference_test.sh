#!/bin/bash -l
#SBATCH --partition=GPU48Go
#SBATCH  --cpus-per-task 6
#SBATCH --gres=gpu
#SBATCH --output slurm_out/inference_mu_unet/output_%j.txt
#SBATCH --job-name infmuUNet
#
echo inference muUNet
cd ~
singularity exec --nv images/pytorch_parallelproj.sif python code/dl-mu-map-from-mlaa/scripts/inference_test.py
exit 0
