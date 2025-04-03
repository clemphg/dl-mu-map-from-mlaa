#!/bin/bash -l
#SBATCH --partition=GPU48Go
#SBATCH  --cpus-per-task 12
#SBATCH --gres=gpu
#SBATCH --output slurm_out/train_mu_unet_k3_tau1e-6/output_train_unet_%j.txt
#SBATCH --job-name muUNetk3
#
echo train muUNet
cd ~
singularity exec --nv images/pytorch_train-dps.sif python code/dl-mu-map-from-mlaa/scripts/train2.py
exit 0
