import sys
import os
from os import path
import shutil
sys.path.append(path.join(path.dirname(path.abspath(__file__)), 'standard_model'))

# DATASET_NAME = 'single_note_GU'
DATASET_NAME = sys.argv[1]
DATASET_PATH = '../data/' + DATASET_NAME

# EXP_GROUP_MODEL_PATH = './afterClean/vae_symm_4_repeat'
EXP_GROUP_MODEL_PATH = 'standard_model/checkpoints/'
CHECKPOINT_NAME = sys.argv[2]

# RESULT_NAME = 'test_set_vae_symm_4_repeat'
RESULT_NAME = sys.argv[3]
RESULT_PATH = './linearityEvalResults/encode_' + RESULT_NAME + '/'
try:
    shutil.rmtree(RESULT_PATH)
except FileNotFoundError:
    pass
os.mkdir(RESULT_PATH)

sys.path.append(path.abspath(EXP_GROUP_MODEL_PATH))
# from normal_rnn import Conv2dGruConv2d
# from train_config import CONFIG
# from trainer_symmetry import LOG_K

from standard_model.normal_rnn import Conv2dGruConv2d
from standard_model.train_config import CONFIG
from standard_model.trainer_symmetry import LOG_K

if not 'scale' in CHECKPOINT_NAME:
    CONFIG['seq_len'] = 15
    CONFIG['rnn_num_layers'] = 2
    CONFIG['rnn_hidden_size'] = 512
    CONFIG['GRU'] = True
else:
    CONFIG['seq_len'] = 15
    CONFIG['rnn_num_layers'] = 1
    CONFIG['rnn_hidden_size'] = 256
    CONFIG['GRU'] = False

if 'beta' in CHECKPOINT_NAME or '1dim' in CHECKPOINT_NAME:
    print('BETA')
    CONFIG['beta_vae'] = True

if '_ae_' in CHECKPOINT_NAME:
    print('ae')
    CONFIG['ae'] = True

single_inst = True

import torch
from tqdm import tqdm

from shared import DEVICE
from sound_dataset import Dataset, norm_log2

def main():
    model = Conv2dGruConv2d(CONFIG).to(DEVICE)
    model.load_state_dict(torch.load(
        path.join(
            EXP_GROUP_MODEL_PATH, CHECKPOINT_NAME, 
        ), map_location=DEVICE, 
    ))
    model.eval()

    dataset = Dataset(DATASET_PATH, CONFIG, cache_all=True)
    instruments = {}
    for instrument_name, pitch, datapoint in tqdm(
        dataset.data, desc='encode',
    ):
        print(f"pitch: {pitch}")
        # if pitch not in range(60, 84):
        if pitch < 60 or pitch > 84:
            continue
        norm_point = norm_log2(datapoint, k=LOG_K).to(DEVICE)
        _, mu, _ = model.batch_seq_encode_to_z(
            norm_point.unsqueeze(0), 
        )
        # mu: batch_i, t, z_i
        z = mu[0, 0, :]
        z_pitch = z[0]

        if instrument_name not in instruments:
            instruments[instrument_name] = ([], [])

        if single_inst and instrument_name != 'Accordion':
            continue
        pitches, z_pitches = instruments[instrument_name]
        pitches.append(pitch)
        z_pitches.append(z_pitch.detach())
    
    for instrument_name, (pitches, z_pitches) in tqdm(
        instruments.items(), desc='write disk', 
    ):
        with open(path.join(RESULT_PATH, instrument_name + '_pitch.txt'), 'w') as f:
            for pitch in pitches:
                print(pitch, file=f)
        with open(path.join(RESULT_PATH, instrument_name + '_z_pitch.txt'), 'w') as f:
            for z_pitch in z_pitches:
                print(z_pitch.item(), file=f)

if __name__ == '__main__':
    main()
