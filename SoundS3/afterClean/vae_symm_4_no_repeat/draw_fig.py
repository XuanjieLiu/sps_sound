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
    print(f'{file_path}')
    selected_wav_spec_tensor = dataset.get(f'{file_path}')
    tensor = selected_wav_spec_tensor.unsqueeze(0)

    normed_tensor = norm_log2(tensor, k=LOG_K)
    z_gt, mu, logvar = model.batch_seq_encode_to_z(normed_tensor)
    selected_wav_latent_code = mu

    z_seq = mu[0].cpu().detach()[0, :]
    return z_seq

def do_the_thing():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model
    model = Conv2dGruConv2d(CONFIG).to(device)
    model.eval()
    model_path = 'checkpoint.pt'
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Model loaded: {model_path}")

    # generate z at the first pitch in each wav file
    dataset = Dataset(WAV_PATH)
    test_insts = ['Accordion', 'Clarinet', 'Electric Piano', 'Flute', 'Guitar', 'Piano', 'Saxophone', 'Trumpet', 'Violin']
    # test_insts = ['Piano', 'Accordion', 'Acoustic Bass', 'Banjo', 'Bassoon', 'Celesta', 'Church Bells', 'Clarinet', 'Clavichord', 'Dulcimer', 'Electric Bass', 'Electric Guitar', 'Electric Organ', 'Electric Piano', 'English Horn', 'Flute', 'Fretless Bass', 'Glockenspiel', 'Guitar', 'Harmonica', 'Harp', 'Harpsichord', 'Horn', 'Kalimba', 'Koto', 'Mandolin', 'Marimba', 'Oboe', 'Ocarina', 'Organ', 'Pan Flute', 'Piccolo', 'Recorder', 'Reed Organ', 'Sampler', 'Saxophone', 'Shakuhachi', 'Shamisen', 'Shehnai', 'Sitar', 'Soprano Saxophone', 'Steel Drum', 'Timpani', 'Trombone', 'Trumpet', 'Vibraphone', 'Viola', 'Violin', 'Violoncello', 'Whistle', 'Xylophone']
    out = []
    z_all = []
    z_insts = {}
    z_pits = {}
    inst_out = []

    chosen_pit = 50
    norm = 2
    std_mat = torch.tensor([1.1206, 0.6677, 0.4246])

    for file in os.listdir(WAV_PATH):
        inst = file.split('-')[0]
        pit = int(file.split('-')[1][: -4])

        if pit != chosen_pit:
            continue

        if inst not in test_insts:
            continue

        if inst not in inst_out:
            inst_out.append(inst)
        
        z = get_z_at_pitch(model, dataset, file) / std_mat
        z_all.append(z)
        
        if inst not in z_insts.keys():
            z_insts[inst] = z
        
    delta_p = []
    delta_tim = []
    
    p_anchor = z_insts[test_insts[0]][0:1]
    tim_anchor = z_insts[test_insts[0]][1:]

    for inst in test_insts[1: ]:
        p = z_insts[inst][0: 1]
        tim = z_insts[inst][1: ]
        delta_p.append(torch.norm(p - p_anchor, p=norm).item())
        delta_tim.append(torch.norm(tim - tim_anchor, p=norm).item())
    
    test_insts[test_insts.index('Electric Piano')] = 'E. Piano'
    labels = test_insts[1:]

    plt.rcParams.update({
        'text.usetex': True, 
        'font.family': 'serif', 
        'font.serif': ['Computer Modern'], 
        'font.size': 7, 
    })

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig = plt.figure(figsize=(5.5, 2))

    (ax1, ax2, ax3) = fig.subplots(1, 3, sharey=True)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    for ax in [ax1, ax2, ax3]:
        rects1 = ax.bar(
            x - width/2, delta_p, width, 
            facecolor='w', edgecolor='k', 
        )
        rects2 = ax.bar(
            x + width/2, delta_tim, width, 
            facecolor='k', edgecolor='k', 
        ) 
        ax.set_xticks(x, labels, rotation=-90)

        ax.bar_label(rects1, padding=2, fmt='')
        ax.bar_label(rects2, padding=2, fmt='')
    
    fig.legend(
        [rects1, rects2], 
        [r'$||\Delta z_{\mathrm{pitch}}||_2$', r'$||\Delta z_{\mathrm{timbre}}||_2$'], 
        ncols=2, loc='lower center', bbox_to_anchor=(.5, .9), 
    )

    fig.tight_layout()
    plt.subplots_adjust(top=.9)

    plt.savefig(f'deltaZ_L{norm}.pdf')
    plt.savefig(f'deltaZ_L{norm}.png')

    print(torch.mean(torch.tensor(delta_p) / torch.tensor(delta_tim)))

    print(delta_p)
    print(delta_tim)

    return

if __name__ == '__main__':
    do_the_thing()