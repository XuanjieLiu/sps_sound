import sys
import os
from os import path
import shutil
from typing import Dict
import soundfile as sf

sys.path.append(path.join(path.dirname(path.abspath(__file__)), 'standard_model'))

N_STD = 2   # how many std's of z_pitch. controls eval range.
EVAL_SYNTH_AUDIO = True
RESOLUTION = 30 if EVAL_SYNTH_AUDIO else 100

# DATASET_NAME = 'single_note_GU'
DATASET_NAME = sys.argv[1]
DATASET_PATH = '../data/' + DATASET_NAME

# EXP_GROUP_MODEL_PATH = './afterClean/vae_symm_4_repeat'
EXP_GROUP_MODEL_PATH = 'standard_model/checkpoints/'
CHECKPOINT_NAME = sys.argv[2]

# RESULT_NAME = 'test_set_vae_symm_4_repeat'
RESULT_NAME = sys.argv[3]
RESULT_PATH = './linearityEvalResults/decode_' + RESULT_NAME + '/'
try:
    shutil.rmtree(RESULT_PATH)
except FileNotFoundError:
    pass
os.mkdir(RESULT_PATH)

sys.path.append(path.abspath(EXP_GROUP_MODEL_PATH))
from standard_model.normal_rnn import Conv2dGruConv2d
from standard_model.train_config import CONFIG
from standard_model.trainer_symmetry import LOG_K

# from example_model.normal_rnn import Conv2dGruConv2d
# from example_model.train_config import CONFIG
# from example_model.trainer_symmetry import LOG_K

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

import torch
from torchaudio.transforms import GriffinLim
import numpy as np
from tqdm import tqdm
try:
    from yin import yin
except ImportError as e:
    module_name = str(e).split('No module named ', 1)[1].strip().strip('"\'')
    if module_name in (
        'yin', 
    ):
        print(f'Missing module {module_name}. Please download at')
        print(f'https://github.com/Daniel-Chin/Python_Lib')
        input('Press Enter to quit...')
    raise e

from shared import DEVICE, CPU
from dataset_config import *
from sound_dataset import Dataset, norm_log2, norm_log2_reverse

class Instrument:
    def __init__(self):
        self.timbres = []
        self.z_pitch = None
        self.yin_pitch = None
    
    def meanTimbre(self):
        return torch.stack(self.timbres).mean(dim=0)

def main():
    model = Conv2dGruConv2d(CONFIG).to(DEVICE)
    model.load_state_dict(torch.load(
        path.join(
            EXP_GROUP_MODEL_PATH, CHECKPOINT_NAME, 
        ), map_location=DEVICE, 
    ))
    model.eval()

    grifLim = GriffinLim(
        n_fft=WIN_LEN, win_length=WIN_LEN, hop_length=HOP_LEN, 
    ).to(DEVICE)

    dataset = Dataset(DATASET_PATH, cache_all=True)
    instruments: Dict[str, Instrument] = {}
    all_z_pitches = []
    for instrument_name, pitch, datapoint in tqdm(
        dataset.data, desc='encode', 
    ):
        norm_point = norm_log2(datapoint.to(DEVICE), k=LOG_K)
        _, mu, _ = model.batch_seq_encode_to_z(
            norm_point.unsqueeze(0), 
        )
        # mu: batch_i, t, z_i
        z = mu[0, 0, :].detach()
        z_timbre = z[1:]
        if CONFIG['beta_vae']:
            z_timbre = torch.zeros_like(z_timbre)
        z_pitch = z[0]
        all_z_pitches.append(z_pitch)

        if instrument_name not in instruments:
            instruments[instrument_name] = Instrument()
        instrument = instruments[instrument_name]
        # instrument.pitches.append(pitch)
        instrument.timbres.append(z_timbre)
    
    all_z_pitches = torch.Tensor(all_z_pitches)
    z_pitch_mean = all_z_pitches.mean()
    z_pitch_std  = all_z_pitches.std()
    print(f'z_pitch_mean={z_pitch_mean}, z_pitch_std={z_pitch_std}')

    for instrument_name, instrument in tqdm(
        instruments.items(), desc='decode & yin', 
    ):
        mean_timbre = instrument.meanTimbre()
        f0s = []
        instrument.z_pitch = torch.linspace(
            z_pitch_mean - z_pitch_std * N_STD, 
            z_pitch_mean + z_pitch_std * N_STD, 
            RESOLUTION, 
        )
        audio_eval_out = []
        for z_pitch in instrument.z_pitch.to(DEVICE):
            z: torch.Tensor = torch.cat((
                z_pitch.unsqueeze(0), 
                mean_timbre, 
            ))
            # assert z.shape == (3, )
            spec = model.batch_decode_from_z(z.unsqueeze(0))[0, :, :]
            spec = norm_log2_reverse(spec, k=LOG_K).detach()
            wav: torch.Tensor = grifLim(spec)
            assert wav.shape == (1, N_SAMPLES_PER_NOTE)
            wav = wav.view(N_SAMPLES_PER_NOTE).to(CPU).numpy()

            clipped: np.ndarray = wav[:N_SAMPLES_PER_NOTE]
            f0 = yin(clipped, SR, N_SAMPLES_PER_NOTE)
            f0s.append(f0)
            audio_eval_out.append(clipped)
        instrument.yin_pitch = freq2pitch(torch.Tensor(f0s))
        if EVAL_SYNTH_AUDIO:
            sf.write(path.join(
                RESULT_PATH, instrument_name + '_synth.wav',
            ), np.concatenate(audio_eval_out), SR)

    for instrument_name, instrument in tqdm(
        instruments.items(), desc='write disk', 
    ):
        with open(path.join(RESULT_PATH, instrument_name + '_z_pitch.txt'), 'w') as f:
            for z_pitch in instrument.z_pitch:
                print(z_pitch.item(), file=f)
        with open(path.join(RESULT_PATH, instrument_name + '_yin_pitch.txt'), 'w') as f:
            for yin_pitch in instrument.yin_pitch:
                print(yin_pitch.item(), file=f)

def freq2pitch(f):
    return np.log(f) * 17.312340490667562 - 36.37631656229591

if __name__ == '__main__':
    main()
