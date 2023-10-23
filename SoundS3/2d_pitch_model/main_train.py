import os
import sys
import argparse
import torch

SCRIPT_DIR = '\\'.join(os.path.dirname(os.path.abspath(__file__)).split('\\')[:-1])
sys.path.append(os.path.dirname(SCRIPT_DIR))
SCRIPT_DIR = '\\'.join(os.path.dirname(os.path.abspath(__file__)).split('\\')[:-2])
sys.path.append(os.path.dirname(SCRIPT_DIR))

from train_config import CONFIG
from trainer_symmetry import BallTrainer, is_need_train

import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='unnamed')
    parser.add_argument('--seq_len', type=int, default=15)
    parser.add_argument('--data_folder', default='cleanTrain')
    parser.add_argument('--no_rnn', action='store_true')
    parser.add_argument('--no_symm', action='store_true')
    parser.add_argument('--no_rep', action='store_true')
    parser.add_argument('--symm_against_rnn', action='store_true')
    parser.add_argument('--additional_symm_steps', type=int, default=0) 
    parser.add_argument('--symm_start_step', type=int, default=0)

    # RNN params
    parser.add_argument('--rnn_num_layers', type=int, default=1)
    parser.add_argument('--rnn_hidden_size', type=int, default=256)

    args = parser.parse_args()

    CONFIG['name'] = args.name
    CONFIG['model_path'] =  f'{args.name}_Conv2dNOTGruConv2d_symmetry.pt'
    CONFIG['seq_len'] = args.seq_len
    CONFIG['train_data_path'] = f'../../data/{args.data_folder}'
    CONFIG['train_result_path'] = f'./dumpster/{args.name}TrainingResults/'
    CONFIG['train_record_path'] = f'./dumpster/{args.name}Train_record.txt'
    CONFIG['eval_record_path'] = f'./dumpster/{args.name}Eval_record.txt'
    CONFIG['no_rnn'] = args.no_rnn
    CONFIG['no_symm'] = args.no_symm
    CONFIG['no_repetition'] = args.no_rep
    CONFIG['additional_symm_steps'] = args.additional_symm_steps
    CONFIG['symm_start_step'] = args.symm_start_step    
    CONFIG['symm_against_rnn'] = args.symm_against_rnn
    CONFIG['rnn_num_layers'] = args.rnn_num_layers
    CONFIG['rnn_hidden_size'] = args.rnn_hidden_size
    CONFIG['learning_rate'] = args.lr

    # torch.manual_seed(21)

    trainer = BallTrainer(CONFIG)
    if is_need_train(CONFIG):
        trainer.train()
