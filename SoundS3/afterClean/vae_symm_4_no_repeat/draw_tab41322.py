from tkinter.tix import TCL_TIMER_EVENTS
from winsound import PlaySound, SND_MEMORY, SND_FILENAME

import matplotlib.pyplot as plt

import os
import sys
SCRIPT_DIR = '\\'.join(os.path.dirname(os.path.abspath(__file__)).split('\\')[:-2])
sys.path.append(os.path.dirname(SCRIPT_DIR))
from normal_rnn import Conv2dGruConv2d, LAST_H, LAST_W, IMG_CHANNEL, CHANNELS
from train_config import CONFIG
from tkinter import *
from PIL import Image, ImageTk
from torchvision.utils import save_image
import torch
from trainer_symmetry import save_spectrogram, tensor2spec, norm_log2, norm_log2_reverse, LOG_K
import torchaudio.transforms as T
import torchaudio

from SoundS3.sound_dataset import Dataset
import matplotlib
from SoundS3.symmetry import rotation_x_mat, rotation_y_mat, rotation_z_mat, do_seq_symmetry, symm_rotate
import numpy as np
import seaborn as sns

matplotlib.use('AGG')

TICK_INTERVAL = 0.1
CODE_LEN = CONFIG['latent_code_num']
IMG_ROOT = 'vae3DBallEval_ImgBuffer'
SPEC_PATH_ORIGIN = IMG_ROOT + "/origin.png"
SPEC_PATH_SELF_RECON = IMG_ROOT + "/self_recon.png"
SPEC_PATH_PRED_RECON = IMG_ROOT + "/pred_recon.png"
Z_GRAPH_PATH_SELF_RECON = IMG_ROOT + "/z_graph_self_recon.png"
SPEC_PATH_TRANSFORMED_SELF_RECON = IMG_ROOT + "/transformed_self_recon.png"
SPEC_PATH_TRANSFORMED_PRED_RECON = IMG_ROOT + "/transformed_pred_recon.png"
Z_GRAPH_PATH_TRANSFORMED_SELF_RECON = IMG_ROOT + "/transformed_z_graph_self_recon.png"
WAV_PATH_SELF_RECON = IMG_ROOT + "/self_recon.wav"
WAV_PATH_PRED_RECON = IMG_ROOT + "/pred_recon.wav"
WAV_PATH_TRANSFORMED_SELF_RECON = IMG_ROOT + "/transformed_self_recon.wav"
WAV_PATH_TRANSFORMED_PRED_RECON = IMG_ROOT + "/transformed_pred_recon.wav"
DIY_WAVE_NAME = IMG_ROOT + "/diy_wave.wav"
WAV_PATH = '../../../data/test'

n_fft = 2046
win_length = None
hop_length = 512
sample_rate = 16000
RANGE = 6.


def decoded_tensor2spec(tensor):
    reverse_tensor = norm_log2_reverse(tensor, k=LOG_K)
    spec = tensor2spec(reverse_tensor[0])
    return spec

def get_z_at_pitch(model, dataset, file_path):
    selected_wav_spec_tensor = dataset.get(f'{file_path}')
    tensor = selected_wav_spec_tensor.unsqueeze(0)

    normed_tensor = norm_log2(tensor, k=LOG_K)
    z_gt, mu, logvar = model.batch_seq_encode_to_z(normed_tensor)
    selected_wav_latent_code = mu

    z_seq = mu[0].cpu().detach()[0, :]
    return z_seq

def do_the_thing(pit_range=[48, 73], norm=2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model
    model = Conv2dGruConv2d(CONFIG).to(device)
    model.eval()
    model_path = 'checkpoint.pt'
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Model loaded: {model_path}")

    # Get a list of instruments
    inst_pits = {}
    insts = []
    for file in os.listdir(WAV_PATH):
        inst = file.split('-')[0]
        pit = int(file.split('-')[1][: -4])

        # Identical wavs
        if inst in ['Lute', 'Handbells']:
            continue

        if inst not in inst_pits.keys():
            inst_pits[inst] = []
        inst_pits[inst].append(pit)
    for inst, pits in inst_pits.items():
        has_all_pits = True
        for pit in range(pit_range[0], pit_range[1]):
            if pit not in pits:
                has_all_pits = False
        if has_all_pits:
            insts.append(inst)

    print(insts)

    # Get the stdev
    dataset = Dataset(WAV_PATH)
    std_mat = None
    for chosen_pit in range(pit_range[0], pit_range[1]):
        std_mat_line = torch.zeros(len(insts), 3)
        for inst_idx, inst in enumerate(insts):
            file_name = f'{inst}-{chosen_pit}.wav'
            z = get_z_at_pitch(model, dataset, file_name)
            std_mat_line[inst_idx] = z
        if std_mat == None:
            std_mat = std_mat_line
        else:
            std_mat = torch.cat((std_mat, std_mat_line), dim=0)
    mean_mat = torch.mean(std_mat, dim=0)
    std_mat = torch.std(std_mat, dim=0)
    print(mean_mat)
    print(std_mat)

    # generate z at the first pitch in each wav file
    delta = None
    tim_correct = torch.tensor(0)
    tim_ctr = 0
    
    for chosen_pit in range(pit_range[0], pit_range[1]):
        z_mat = torch.zeros(len(insts), 3)
        for inst_idx, inst in enumerate(insts):
            file_name = f'{inst}-{chosen_pit}.wav'
            z = get_z_at_pitch(model, dataset, file_name)
            z = (z - mean_mat) / std_mat
            z_mat[inst_idx] = z
        
        # 41323
        stdev = torch.std(z_mat, dim=0)
        _, indices = torch.sort(stdev, dim=0)
        tim_correct += torch.sum(indices[1:] > 0)
        tim_ctr += 2

        # Permutate all combinations
        for i in range(1, len(insts)):
            z_shifted = torch.cat((z_mat[i:], z_mat[:i]), dim=0)
            if delta == None:
                delta = z_mat-z_shifted
            else:
                delta = torch.cat((delta, z_mat-z_shifted), dim=0)

    delta_p = torch.norm(delta[:, :1], p=norm, dim=1)
    delta_tim = torch.norm(delta[:, 1:], p=norm, dim=1)

    print(f'deltaP / deltaTim: {torch.mean(delta_p / delta_tim)}')
    print(f'prec_tim: {tim_correct / tim_ctr}')

    # same timbre, different pitches 
    delta = None
    p_correct = torch.tensor(0)
    p_ctr = torch.tensor(0)
    
    for inst_idx, inst in enumerate(insts):
        z_mat = torch.zeros(pit_range[1] - pit_range[0], 3)
        for pit_idx, chosen_pit in enumerate(range(pit_range[0], pit_range[1])):
            file_name = f'{inst}-{chosen_pit}.wav'
            z = get_z_at_pitch(model, dataset, file_name)
            z = (z - mean_mat) / std_mat
            z_mat[pit_idx] = z

        # 41323
        stdev = torch.std(z_mat, dim=0)
        _, indices = torch.sort(stdev, dim=0)
        p_correct += torch.sum(indices[2:] < 1)
        p_ctr += 1

        # Permutate all combinations
        for i in range(1, pit_range[1] - pit_range[0]):
            z_shifted = torch.cat((z_mat[i:], z_mat[:i]), dim=0)
            if delta == None:
                delta = z_mat-z_shifted
            else:
                delta = torch.cat((delta, z_mat-z_shifted), dim=0)


    delta_p = torch.norm(delta[:, :1], p=norm, dim=1)
    delta_tim = torch.norm(delta[:, 1:], p=norm, dim=1)
    
    print(f'deltaTim / deltaP: {torch.mean(delta_tim / delta_p)}')
    print(f'prec_p: {p_correct / p_ctr}')

    return

if __name__ == '__main__':
    do_the_thing()