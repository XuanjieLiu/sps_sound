import sys
from os import path
sys.path.append(path.join(path.dirname(path.abspath(__file__)), '../../'))

from winsound import PlaySound, SND_MEMORY, SND_FILENAME

import matplotlib.pyplot as plt
from normal_rnn import Conv2dGruConv2d, LAST_H, LAST_W, IMG_CHANNEL, CHANNELS
from train_config import CONFIG
from PIL import Image, ImageTk

import os
import torch
from trainer_symmetry import save_spectrogram, tensor2spec, norm_log2, norm_log2_reverse, LOG_K
import torchaudio.transforms as T
import torchaudio
from SoundS3.sound_dataset import Dataset, PersistentLoader
import matplotlib
from SoundS3.symmetry import rotation_x_mat, rotation_y_mat, rotation_z_mat, do_seq_symmetry, symm_rotate
import numpy as np
import argparse

from SoundS3.shared import DEVICE

matplotlib.use('AGG')

WAV_PATH = '../../data/cleanTrain_accordion/'

parser = argparse.ArgumentParser()

parser.add_argument('--name', default='scale_singleInst_1dim_betaVae_1_checkpoint_10000')
parser.add_argument('--wav_name', type=str, default='Accordion-69.97.wav')
parser.add_argument('--n_trans', type=int, default=4)
parser.add_argument('--trans_interval', type=float, default=0.5)


parser.add_argument('--seq_len', type=int, default=15)
parser.add_argument('--base_len', type=int, default=3)

# RNN params
parser.add_argument('--rnn_num_layers', type=int, default=1)
parser.add_argument('--rnn_hidden_size', type=int, default=256)
parser.add_argument('--gru', action='store_true')



args = parser.parse_args()

NAME = args.name
CONFIG['seq_len'] = args.seq_len
CONFIG['train_data_path'] = WAV_PATH
CONFIG['rnn_num_layers'] = args.rnn_num_layers
CONFIG['rnn_hidden_size'] = args.rnn_hidden_size
CONFIG['GRU'] = args.gru
CONFIG['base_len'] = args.base_len

MODEL_PATH = f'./checkpoints/{args.name}.pt'
SPEC_PATH_ORIGIN = 'origin.png'
n_fft = 1024
win_length = 1024
hop_length = 512
sample_rate = 16000
RANGE = 18.

def init_vae(model_path=MODEL_PATH):
    model = Conv2dGruConv2d(CONFIG).to(DEVICE)
    model.eval()
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"Model is loaded")
        print(model_path)
    else:
        assert False
    return model


def save_audio(spec, name, sample_rate=16000):
    griffin_lim = T.GriffinLim(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
    )
    recon_waveform = griffin_lim(spec.cpu())
    torchaudio.save(name, recon_waveform, sample_rate)


def decoded_tensor2spec(tensor):
    reverse_tensor = norm_log2_reverse(tensor, k=LOG_K)
    spec = tensor2spec(reverse_tensor[0])
    return spec


def generate_translated_audios():
    save_dir = f'{args.name}__{args.wav_name}'
    os.makedirs(save_dir, exist_ok=True)
    write_args_in_lines(args, os.path.join(save_dir, 'params.txt'))

    interval = args.trans_interval
    n_trans = args.n_trans
    vae = init_vae()
    dataset = Dataset(
        CONFIG['train_data_path'], CONFIG, cache_all=False
    )
    selected_wav_spec_tensor = dataset.get(args.wav_name)
    selected_spec_frame = tensor2spec(selected_wav_spec_tensor)
    save_spectrogram(selected_spec_frame[0], os.path.join(save_dir, SPEC_PATH_ORIGIN), need_norm_reverse=False)
    tensor = selected_wav_spec_tensor.unsqueeze(0).to(DEVICE)
    normed_tensor = norm_log2(tensor, k=LOG_K)
    z_gt, mu, logvar = vae.batch_seq_encode_to_z(normed_tensor)
    base_latent_code = mu
    start_trans = -n_trans * interval
    start_latent_code = base_latent_code + start_trans

    for i in range(n_trans*2+1):
        trans_latent_code = start_latent_code + i * interval
        name_prefix = f'{i-n_trans}'
        save_spectrogram_and_audio(trans_latent_code, vae, save_dir, name_prefix)

def write_args_in_lines(args, file_path):
    with open(file_path, 'w') as f:
        for k, v in vars(args).items():
            f.write(f'{k}: {v}\n')

def save_spectrogram_and_audio(latent_code, model, save_dir, name_prefix):
    recon_tensor = model.batch_seq_decode_from_z(latent_code)
    spec = decoded_tensor2spec(recon_tensor)
    spec_path = os.path.join(save_dir, f'{name_prefix}.png')
    audio_path = os.path.join(save_dir, f'{name_prefix}.wav')
    save_spectrogram(spec[0], spec_path, need_norm_reverse=False)
    save_audio(spec, audio_path)


if __name__ == "__main__":
    generate_translated_audios()

