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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='unnamed')
    parser.add_argument('--data_folder', default='cleanTrain')

    args = parser.parse_args()

    CONFIG['name'] = args.name
    CONFIG['model_path'] =  f'{args.name}_Conv2dNOTGruConv2d_symmetry.pt'
    CONFIG['train_data_path'] = f'../../data/{args.data_folder}'
    CONFIG['train_result_path'] = f'./dumpster/{args.name}TrainingResults/'
    CONFIG['train_record_path'] = f'./dumpster/{args.name}Train_record.txt'
    CONFIG['eval_record_path'] = f'./dumpster/{args.name}Eval_record.txt'
    CONFIG['seq_len'] = 15

    trainer = BallTrainer(CONFIG)
    if is_need_train(CONFIG):
        trainer.train()
