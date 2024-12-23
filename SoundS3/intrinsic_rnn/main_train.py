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

from train_config import CONFIG
from trainer_symmetry import BallTrainer, is_need_train

import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # For the 1-dim setup with symm constraint, look at the lines along beta_vae
    # Just use beta_vae

    parser.add_argument('--name', default='unnamed')
    parser.add_argument('--seq_len', type=int, default=15)
    parser.add_argument('--data_folder', default='cleanTrain')
    parser.add_argument('--no_rnn', action='store_true')
    parser.add_argument('--no_symm', action='store_true')
    parser.add_argument('--no_rep', action='store_true')
    parser.add_argument('--symm_against_rnn', action='store_true')
    parser.add_argument('--additional_symm_steps', type=int, default=0) 
    parser.add_argument('--symm_start_step', type=int, default=0) # Set this to 15 to apply symm loss only on OOR steps
    parser.add_argument('--max_iter_num', type=int, default=15001)

    # RNN params
    parser.add_argument('--rnn_num_layers', type=int, default=1)
    parser.add_argument('--rnn_hidden_size', type=int, default=256)
    parser.add_argument('--gru', action='store_true')
    parser.add_argument('--beta_vae', action='store_true')
    parser.add_argument('--ae', action='store_true')

    # Hyper params
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--z_rnn_loss_scalar', type=float, default=2)

    parser.add_argument('--n_runs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=20)

    parser.add_argument('--kld_loss_scalar', type=float, default=0.01)
    parser.add_argument('--base_len', type=int, default=3)

    # Eval
    parser.add_argument('--eval_recons', action='store_true')

    args = parser.parse_args()

    CONFIG['seq_len'] = args.seq_len
    CONFIG['train_data_path'] = f'../../data/{args.data_folder}'
    CONFIG['no_rnn'] = args.no_rnn
    CONFIG['no_symm'] = args.no_symm
    CONFIG['no_repetition'] = args.no_rep
    CONFIG['additional_symm_steps'] = args.additional_symm_steps
    CONFIG['symm_start_step'] = args.symm_start_step    
    CONFIG['symm_against_rnn'] = args.symm_against_rnn
    CONFIG['rnn_num_layers'] = args.rnn_num_layers
    CONFIG['rnn_hidden_size'] = args.rnn_hidden_size
    CONFIG['GRU'] = args.gru
    CONFIG['learning_rate'] = args.lr
    CONFIG['z_rnn_loss_scalar'] = args.z_rnn_loss_scalar
    CONFIG['beta_vae'] = args.beta_vae
    CONFIG['ae'] = args.ae
    CONFIG['eval_recons'] = args.eval_recons
    CONFIG['batch_size'] = args.batch_size
    CONFIG['kld_loss_scalar'] = args.kld_loss_scalar
    CONFIG['max_iter_num'] = args.max_iter_num
    CONFIG['base_len']= args.base_len
    print(CONFIG['eval_recons'])
    if not args.eval_recons:
        with open(f'./new_dumpster/{args.name}_config.txt', 'w') as convert_file:
            for key, value in CONFIG.items():
                convert_file.write('%s:%s\n' % (key, value))

    # torch.manual_seed(21)


    l_recons = []
    l_preds = []
    l_priors = []
    # Loop for multiple runs
    for i in range(args.n_runs):
        # if i == 8:
        #     continue
        name = args.name
        if args.n_runs > 1:
            name = args.name + '_' + str(i)
        CONFIG['name'] = name
        CONFIG['model_path'] =  f'checkpoints/oct17/{name}_Conv2dNOTGruConv2d_symmetry.pt'
        CONFIG['train_result_path'] = f'./new_dumpster/{name}TrainingResults/'
        CONFIG['train_record_path'] = f'./new_dumpster/{name}Train_record.txt'
        CONFIG['eval_record_path'] = f'./new_dumpster/{name}Eval_record.txt'

        trainer = BallTrainer(CONFIG)

        if args.eval_recons or is_need_train(CONFIG):
            out = trainer.train()
            if args.eval_recons:
                self_recon, pred_recon, rnn_prior = out
                l_recons.append(np.mean(self_recon))
                l_preds.append(np.mean(pred_recon))
                l_priors.append(np.mean(rnn_prior))
                for idx, m in enumerate([self_recon, pred_recon, rnn_prior]):
                    m = torch.tensor(m)
                    print(idx, torch.mean(m))
    if args.eval_recons:
        rd = lambda x: round(x, 4)
        print('l_recons:')
        print(f'{rd(np.mean(l_recons))}$\pm${rd(np.std(l_recons))}')
        print('l_preds:')
        print(f'{rd(np.mean(l_preds))}$\pm${rd(np.std(l_preds))}')
        print('l_priors:')
        print(f'{rd(np.mean(l_priors))}$\pm${rd(np.std(l_priors))}')
