import math
import random
import time
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
import os
from normal_rnn import Conv2dGruConv2d, repeat_one_dim
from SoundS3.shared import DEVICE
from SoundS3.sound_dataset import Dataset, PersistentLoader, norm_log2, norm_log2_reverse
from SoundS3.symmetry import make_translation_batch, make_random_rotation_batch, do_seq_symmetry, symm_trans, \
    symm_rotate, make_rand_zoom_batch, symm_zoom
from SoundS3.loss_counter import LossCounter
from SoundS3.common_utils import create_path_if_not_exist
import matplotlib.pyplot as plt
import librosa
import torchaudio.transforms as T
import torchaudio

import matplotlib

matplotlib.use('AGG')
LOG_K = 12.5

n_fft = 2046
win_length = None
hop_length = 512
sample_rate = 16000
time_frame_len = 4

DT_BATCH_MULTIPLE = 4
# LAST_SOUND = 16 # len(RNN) change me 


def tensor_arrange(num, start, end, is_have_end):
    if is_have_end:
        return torch.arange(num) * abs(end - start) / (num - 1) + start
    else:
        return torch.arange(num) * abs(end - start) / num + start


def save_spectrogram(tensor, file_path, title=None, ylabel="freq_bin", aspect="auto", xmax=None,
                     need_norm_reverse=True):
    spec = norm_log2_reverse(tensor, k=LOG_K) if need_norm_reverse else tensor
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    im = axs.imshow(librosa.power_to_db(spec), origin="lower", aspect=aspect, interpolation='nearest')
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    plt.savefig(file_path)
    plt.clf()
    plt.close('all')

def tensor2spec(tensor):
    return tensor.permute(1, 2, 0, 3).reshape(tensor.size(1), tensor.size(2),
                                              tensor.size(0) * tensor.size(3)).detach().cpu()


def is_need_train(train_config):
    loss_counter = LossCounter([])
    iter_num = loss_counter.load_iter_num(train_config['train_record_path'])
    if train_config['max_iter_num'] > iter_num:
        print("Continue training")
        return True
    else:
        print("No more training is needed")
        return False


def vector_z_score_norm(vector, mean=None, std=None):
    if mean is None:
        mean = torch.mean(vector, [k for k in range(vector.ndim - 1)])
    if std is None:
        std = torch.std(vector, [j for j in range(vector.ndim - 1)])
    return (vector - mean) / std, mean, std


def symm_rotate_first3dim(z, rotator):
    z_R = torch.matmul(z[..., 0:3].unsqueeze(1), rotator)
    return torch.cat((z_R.squeeze(1), z[..., 3:]), -1)


class BallTrainer:
    def __init__(self, config, is_train=True):
        self.model = Conv2dGruConv2d(config).to(DEVICE)
        self.batch_size = config['batch_size']
        self.dataset = Dataset(config['train_data_path'], config, cache_all=config['seq_len']==15)
        # self.train_data_loader = PersistentLoader(
        #     self.dataset, self.batch_size, set_break = config['eval_recons']
        # )
        self.train_data_loader = DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True
        )
        print(len(self.dataset))
        self.mse_loss = nn.MSELoss(reduction='sum').to(DEVICE)
        self.model.to(DEVICE)
        self.model_path = config['model_path']
        self.kld_loss_scalar = config['kld_loss_scalar']
        self.z_rnn_loss_scalar = config['z_rnn_loss_scalar']
        self.z_symm_loss_scalar = config['z_symm_loss_scalar']
        self.enable_sample = config['enable_sample']
        self.checkpoint_interval = config['checkpoint_interval']
        self.t_batch_multiple = config['t_batch_multiple']
        self.r_batch_multiple = config['r_batch_multiple']
        self.z_batch_multiple = config['z_batch_multiple']
        self.t_range = config['t_range']
        self.r_range = config['r_range']
        self.z_range = config['z_range']
        self.learning_rate = config['learning_rate']
        self.scheduler_base_num = config['scheduler_base_num']
        self.max_iter_num = config['max_iter_num']
        self.base_len = config['base_len']
        self.train_result_path = config['train_result_path']
        self.train_record_path = config['train_record_path']
        self.eval_record_path = config['eval_record_path']
        self.log_interval = config['log_interval']
        self.eval_interval = config['eval_interval']
        self.sample_prob_param_alpha = config['sample_prob_param_alpha']
        self.sample_prob_param_beta = config['sample_prob_param_beta']
        self.enable_SRS = config['enable_SRS']
        self.is_save_img = config['is_save_img']
        self.griffin_lim = T.GriffinLim(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
        )
        self.config = config
        
        # Eval recons
        self.eval_recons_self = []
        self.eval_recons_pred = []
        self.eval_recons_prior = []

    def save_result_imgs(self, img_list, name, seq_len):
        result = torch.cat([img[0] for img in img_list], dim=0)
        save_image(result, self.train_result_path + str(name) + '.png', seq_len)

    def get_sample_prob(self, step):
        alpha = self.sample_prob_param_alpha
        beta = self.sample_prob_param_beta
        return alpha / (alpha + np.exp((step + beta) / alpha))

    def gen_sample_points(self, base_len, total_len, step, enable_sample):
        if not enable_sample:
            return []
        sample_rate = self.get_sample_prob(step)
        sample_list = []
        for i in range(base_len, total_len):
            r = np.random.rand()
            if r > sample_rate:
                sample_list.append(i)
        return sample_list

    def resume(self):
        if os.path.exists(self.model_path):
            self.model.load_state_dict(self.model.load_tensor(self.model_path))
            print(f"Model is loaded")
        else:
            print("New model is initialized")

    def scheduler_func(self, curr_iter):
        return self.scheduler_base_num ** curr_iter

    def train(self):
        if not self.config['eval_recons']:
            create_path_if_not_exist(self.train_result_path)
            self.model.train()
            self.resume()
            train_loss_counter = LossCounter(['loss_ED', 'loss_ERnnD', 'loss_Rnn',
                                            'loss_TRnnTrD_x1', 'loss_RRnnRrD_x1', 'loss_ZRnnZrD_x1',
                                            'loss_TRnnTr_z1', 'loss_RRnnRr_z1', 'loss_ZRnnZr_z1',
                                            'KLD'])
            iter_num = train_loss_counter.load_iter_num(self.train_record_path)
            curr_iter = iter_num
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: self.scheduler_func(curr_iter))
        else:
            self.model.eval()
            self.model.load_state_dict(self.model.load_tensor(self.model_path))
            train_loss_counter = LossCounter(['loss_ED', 'loss_ERnnD', 'loss_Rnn',
                                            'loss_TRnnTrD_x1', 'loss_RRnnRrD_x1', 'loss_ZRnnZrD_x1',
                                            'loss_TRnnTr_z1', 'loss_RRnnRr_z1', 'loss_ZRnnZr_z1',
                                            'KLD'])
            iter_num = 0
            curr_iter = 0
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        for i in range(iter_num, self.max_iter_num):
            print(i)
            curr_iter = iter_num
            # data = next(self.train_data_loader)
            data = None
            for batch_ndx, sample in enumerate(self.train_data_loader):
                data = sample
                break
            data = data.to(DEVICE) # change me 
            data = norm_log2(data, k=LOG_K)
            is_log = (i % self.log_interval == 0 and i != 0)
            # save_spectrogram(tensor2spec(data[0])[0], f'{self.train_result_path}{i}-data.png')
            # for j in range(0, len(data[0])):
            #     save_spectrogram(tensor2spec(data[0][j].unsqueeze(0))[0], f'{self.train_result_path}{i}-{j}-data.png')
            optimizer.zero_grad()
            I_sample_points = self.gen_sample_points(self.base_len, data.size(1), i, self.enable_sample) # sampling for to transit from teacher forcing to student forcing
            T_sample_points = self.gen_sample_points(self.base_len, data.size(1), i, self.enable_sample)

            z_gt, mu, logvar = self.model.batch_seq_encode_to_z(data)

            if self.config['ae'] or self.config['eval_recons']:
                z_gt = mu
            z_gt_p = z_gt[..., 0:1]
            z_gt_c = z_gt[..., 1:]
            z_gt_cr = repeat_one_dim(z_gt_c, sample_range=10)
            z_gt_cr_extended = repeat_one_dim(z_gt_c, sample_range=10, repeat_times=z_gt_c.shape[1] + self.config['additional_symm_steps'])
            z_gt_cr = z_gt_cr_extended[:z_gt_cr.shape[0], :z_gt_cr.shape[1], :z_gt_cr.shape[2]]
            if self.config['no_repetition'] or self.config['beta_vae']:
                z_combine = torch.cat((z_gt_p, z_gt_c), -1)
            else:
                z_combine = torch.cat((z_gt_p, z_gt_cr), -1)
            T, Tr = make_translation_batch(batch_size=self.batch_size * DT_BATCH_MULTIPLE, dim=np.array([1]),
                                           t_range=self.t_range)
            z0_rnn_extended = self.model.predict_with_symmetry(z_gt_p, I_sample_points, lambda z: z, self.config['additional_symm_steps']) # Except there is no symm function. and this is just pred w/ RNN?
            # extended with out-of-range predictions for long imagination
            z0_rnn = z0_rnn_extended[:, :z_gt.shape[1]-1, :]
            
            # Recons, KLD
            vae_loss = self.calc_vae_loss(data, z_combine, mu, logvar, is_log * i) 
            if self.config['ae'] and not self.config['eval_recons']:
                vae_loss = vae_loss[0], torch.zeros_like(vae_loss[1])

            # Pred_recon, rnn_prior
            rnn_loss = self.calc_rnn_loss(data[:, 1:, :, :, :], z_gt_p, z0_rnn, is_log * i, z_gt_cr[:, :-1, :])
            
            # Ablate the RNN loss
            if self.config['no_rnn']:
                rnn_loss = torch.zeros_like(rnn_loss[0]), torch.zeros_like(rnn_loss[1])

            R_loss = (torch.zeros(2))
            Z_loss = (torch.zeros(2))
            # The actual symm step
            T_loss = self.batch_symm_loss(
    
                data[:, 1:, :, :, :], z_gt_p, z0_rnn_extended, T_sample_points, DT_BATCH_MULTIPLE,
                lambda z: symm_trans(z, T), lambda z: symm_trans(z, Tr), z_gt_cr_extended[:, :-1, :]
            )
            if self.config['no_symm'] or self.config['beta_vae']:
                T_loss = (torch.zeros(2))

            loss = self.loss_func(vae_loss, rnn_loss, T_loss, R_loss, Z_loss, train_loss_counter)
            if not self.config['eval_recons']:
                loss.backward()
                optimizer.step()
                scheduler.step()
            else:
                self.eval_recons_self.append(vae_loss[0].detach().item())
                self.eval_recons_pred.append(rnn_loss[0].detach().item())
                self.eval_recons_prior.append(rnn_loss[1].detach().item())

            if self.config['eval_recons'] and (i+1) * self.batch_size >= len(self.dataset):
                print(f'Eval ends. total iters: {i+1}')
                return self.eval_recons_self, self.eval_recons_pred, self.eval_recons_prior

            if is_log and not self.config['eval_recons']:
                self.model.save_tensor(self.model.state_dict(), self.model_path)
                print(train_loss_counter.make_record(i))
                train_loss_counter.record_and_clear(self.train_record_path, i)
            if i % self.checkpoint_interval == 0 and i != 0 and not self.config['eval_recons']:
                self.model.save_tensor(self.model.state_dict(), f'./checkpoints/{self.config["name"]}_checkpoint_{i}.pt')

    def save_audio(self, spec, name, sample_rate=16000):
        recon_waveform = self.griffin_lim(spec.cpu())
        torchaudio.save(name, recon_waveform, sample_rate)

    def batch_symm_loss(self, x1, z_gt, z0_rnn, sample_points, symm_batch_multiple, symm_func, symm_reverse_func, z_gt_cr):
        z_gt_repeat = z_gt.repeat(symm_batch_multiple, 1, 1)
        z0_S_rnn = self.model.predict_with_symmetry(z_gt_repeat, sample_points, symm_func, self.config['additional_symm_steps']) # prediction with symm
        z0_rnn_repeat = z0_rnn.repeat(symm_batch_multiple, 1, 1)
        z_gt_cr_repeat = z_gt_cr.repeat(symm_batch_multiple, 1, 1)[:, 0:self.config['seq_len']-1+self.config['additional_symm_steps'], :]
        x1_repeat = x1.repeat(symm_batch_multiple, 1, 1, 1, 1)[:, 0:self.config['seq_len']-1, :, :, :]
        zloss_S_rnn_Sr_D__x1, zloss_S_rnn_Sr__z1 = \
            self.calc_symm_loss(x1_repeat, z_gt_repeat, z0_rnn_repeat, z0_S_rnn, symm_reverse_func, z_gt_cr_repeat)
        return zloss_S_rnn_Sr_D__x1 / symm_batch_multiple, zloss_S_rnn_Sr__z1 / symm_batch_multiple

    def batch_normalize(self, tensor: torch.Tensor):
        mean = torch.mean(tensor)
        std = torch.std(tensor)
        return (tensor - mean) / std


    def calc_rnn_loss(self, x1, z_gt, z0_rnn, log_num=0, z_gt_cr=None):
        if z_gt_cr is None:
            z_next = z0_rnn
        else:
            z_next = torch.cat((z0_rnn, z_gt_cr), -1)
        recon_next = self.model.batch_seq_decode_from_z(z_next)
        if self.config['eval_recons']:
            xloss_ERnnD = nn.BCELoss(reduction='mean')(recon_next, x1)
            zloss_Rnn = nn.MSELoss()(self.batch_normalize(z0_rnn), self.batch_normalize(z_gt[:, 1:, :]))
            if zloss_Rnn.item() > 1:
                print(f"Outlier !!! ")
                print(z0_rnn)
                print(z_gt[:, 1:, :])
        else:
            xloss_ERnnD = nn.BCELoss(reduction='sum')(recon_next, x1)
            zloss_Rnn = self.z_rnn_loss_scalar * self.mse_loss(z0_rnn, z_gt[:, 1:, :])
        if log_num != 0 and not self.config['eval_recons']:
            save_spectrogram(tensor2spec(recon_next[0])[0], f'{self.train_result_path}{log_num}-recon_pred.png')
            # self.save_audio(tensor2spec(recon_next[0]), f'{self.train_result_path}{log_num}-recon_pred.wav')

        return xloss_ERnnD, zloss_Rnn

    def calc_vae_loss(self, data, z_gt, mu, logvar, log_num=0):
        recon = self.model.batch_seq_decode_from_z(z_gt)
        if self.config['eval_recons']:
            recon_loss = nn.BCELoss(reduction='mean')(recon, data)
        else:
            recon_loss = nn.BCELoss(reduction='sum')(recon, data)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - torch.exp(logvar)) * self.kld_loss_scalar
        if log_num != 0 and not self.config['eval_recons']:
            save_spectrogram(tensor2spec(data[0])[0], f'{self.train_result_path}{log_num}-gt.png')
            # self.save_audio(tensor2spec(data[0]), f'{self.train_result_path}{log_num}-gt.wav')
            save_spectrogram(tensor2spec(recon[0])[0], f'{self.train_result_path}{log_num}-recon.png')
            # self.save_audio(tensor2spec(recon[0]), f'{self.train_result_path}{log_num}-recon.wav')
        return recon_loss, KLD

    def calc_symm_loss(self, x1, z_gt, z0_rnn, z0_S_rnn, symm_reverse_func, z_gt_cr):
        z0_S_rnn_Sr = do_seq_symmetry(z0_S_rnn, symm_reverse_func)[:, 0:self.config['seq_len']-1+self.config['additional_symm_steps'], :]
        # Symm loss against RNN predictions
        if self.config['additional_symm_steps'] > 0 or self.config['symm_against_rnn']:
            zloss_S_rnn_Sr__z1 = self.z_symm_loss_scalar * self.mse_loss(z0_S_rnn_Sr[:, self.config['symm_start_step']:, :], z0_rnn[:, self.config['symm_start_step']:, :])
            # batch decode predicted z (after reverse symmetry)
            z0_S_rnn_Sr_D = self.model.batch_seq_decode_from_z(torch.cat((z0_S_rnn_Sr[:, self.config['symm_start_step']:, :], z_gt_cr[:, self.config['symm_start_step']:, :]), -1))
            z0_rnn_D = self.model.batch_seq_decode_from_z(torch.cat((z0_rnn[:, self.config['symm_start_step']:, :], z_gt_cr[:, self.config['symm_start_step']:, :]), -1))
            zloss_S_rnn_Sr_D__x1 = nn.BCELoss(reduction='sum')(z0_S_rnn_Sr_D, z0_rnn_D)
            zloss_S_rnn_Sr_D__x1 = 0 # Interesting... If I let zloss_S_rnn_Sr_D__x1 = 0, the gradient does not explode?

            # Rescale 
            zloss_S_rnn_Sr__z1 *= (zloss_S_rnn_Sr__z1-1) / (zloss_S_rnn_Sr__z1-1+self.config['additional_symm_steps'])
            zloss_S_rnn_Sr_D__x1 *= (zloss_S_rnn_Sr__z1-1) / (zloss_S_rnn_Sr__z1-1+self.config['additional_symm_steps'])
        # Symm loss against GT
        else:
            zloss_S_rnn_Sr__z1 = self.z_symm_loss_scalar * self.mse_loss(z0_S_rnn_Sr, z_gt[:, 1:self.config['seq_len'], :])
            z0_S_rnn_Sr_D = self.model.batch_seq_decode_from_z(torch.cat((z0_S_rnn_Sr, z_gt_cr), -1))
            zloss_S_rnn_Sr_D__x1 = nn.BCELoss(reduction='sum')(z0_S_rnn_Sr_D, x1)
        return zloss_S_rnn_Sr_D__x1, zloss_S_rnn_Sr__z1

    def loss_func(self, vae_loss, rnn_loss, T_loss, R_loss, Z_loss, loss_counter):
        xloss_ED, KLD = vae_loss
        xloss_ERnnD, zloss_Rnn = rnn_loss
        zloss_T_rnn_Sr_T__x1, zloss_T_rnn_Tr__z1 = T_loss
        zloss_R_rnn_Sr_R__x1, zloss_R_rnn_Rr__z1 = R_loss
        zloss_Z_rnn_Sr_Z__x1, zloss_Z_rnn_Rr__z1 = Z_loss

        loss = 0
        loss += xloss_ED + KLD + xloss_ERnnD
        loss += zloss_Rnn
        loss += zloss_T_rnn_Tr__z1 + zloss_R_rnn_Rr__z1 + zloss_Z_rnn_Rr__z1
        loss += zloss_T_rnn_Sr_T__x1 + zloss_R_rnn_Sr_R__x1 + zloss_Z_rnn_Sr_Z__x1

        loss_counter.add_values([xloss_ED.item(), xloss_ERnnD.item(), zloss_Rnn.item(),
                                 zloss_T_rnn_Sr_T__x1.item(), zloss_R_rnn_Sr_R__x1.item(), zloss_Z_rnn_Sr_Z__x1.item(),
                                 zloss_T_rnn_Tr__z1.item(), zloss_R_rnn_Rr__z1.item(), zloss_Z_rnn_Rr__z1.item(),
                                 KLD.item()
                                 ])
        return loss
