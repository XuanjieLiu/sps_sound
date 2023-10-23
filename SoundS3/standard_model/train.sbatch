#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mem=6GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=s3plus
#SBATCH --output=ceiling_test.out
#SBATCH --error=ceiling_test.err
#SBATCH --partition=gpu
#SBATCH --qos=gpu-8

nvidia-smi -a
echo $CUDA_VISIBLE_DEVICES
hostname


source activate xuanjie
