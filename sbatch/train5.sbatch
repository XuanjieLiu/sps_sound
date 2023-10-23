#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mem=6GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=s3plus
#SBATCH --output=scale_singleInst_1dim_1.out
#SBATCH --error=scale_singleInst_1dim_1.err
#SBATCH --partition=gpu
#SBATCH --qos=gpu-8

nvidia-smi -a
echo $CUDA_VISIBLE_DEVICES
hostname


source activate xuanjie

cd /l/users/xuanjie.liu/sps_sound/SoundS3/standard_model
python main_train.py --name scale_singleInst_1dim_1_batevae  --seq_len 15  --data_folder cleanTrain_accordion   --n_runs 10  --beta_vae  --no_rnn