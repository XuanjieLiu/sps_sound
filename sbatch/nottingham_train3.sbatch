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
python main_train.py \
  --name nottingham_singleInst_1dim  \
  --seq_len 64  \
  --data_folder nottingham_eights_pool_accOnly_5000_easier   \
  --n_runs 10 \
  --additional_symm_steps 0 \
  --symm_start_step 0 \
  --rnn_num_layers 2 \
  --rnn_hidden_size 512 \
  --gru \
  --ae