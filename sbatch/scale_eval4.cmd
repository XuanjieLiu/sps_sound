cd ../SoundS3/standard_model
call python main_train.py --name scale_singleInst_1dim_ae_noSymm  --seq_len 15  --data_folder cleanTest   --n_runs 10  --eval_recons