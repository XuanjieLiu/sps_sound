import os
import sys
import argparse
import torch
import numpy as np
import json

SCRIPT_DIR = '\\'.join(os.path.dirname(os.path.abspath(__file__)).split('\\')[:-1])
sys.path.append(os.path.dirname(SCRIPT_DIR))
SCRIPT_DIR = '\\'.join(os.path.dirname(os.path.abspath(__file__)).split('\\')[:-2])
sys.path.append(os.path.dirname(SCRIPT_DIR))
sys.path.append('{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/../'))
from importlib import reload
from trainer import Trainer, is_need_train

import torch

EXP_ROOT_PATH = '{}{}'.format(os.path.dirname(os.path.abspath(__file__)), '/exps')
sys.path.append(EXP_ROOT_PATH)

EXP_NAME_LIST = ['24.12.21_normal_train3']
PROJECT_NAME = 'intrinsic_rnn_2'

for exp_name in EXP_NAME_LIST:
    exp_path = os.path.join(EXP_ROOT_PATH, exp_name)
    os.chdir(exp_path)
    sys.path.append(exp_path)
    print(f'Exp path: {exp_path}')
    t_config = __import__('train_config')
    reload(t_config)
    sys.path.pop()
    print(t_config.CONFIG)
    t_config.CONFIG['name'] = exp_name
    t_config.CONFIG['project_name'] = PROJECT_NAME
    if is_need_train(t_config.CONFIG):
        trainer = Trainer(t_config.CONFIG)
        trainer.train()