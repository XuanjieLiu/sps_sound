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
from eval_linearity import eval_linearity
import wandb

WANDB_API_KEY="532007cd7a07c1aa0d1194049c3231dadd1d418e"
wandb.login(key=WANDB_API_KEY)

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


def sample_lengths(total=15, prompt_range=None, auto_g_range=None):
    if prompt_range is None:
        prompt_range = list(range(3, 6))
    if auto_g_range is None:
        auto_g_range = list(range(4, 9))
    prompt_1, prompt_2 = random.sample(prompt_range, 2)
    auto_g_1 = random.sample(auto_g_range, 1)[0]
    auto_g_2 = total - max([prompt_1, prompt_2]) - auto_g_1
    return prompt_1, prompt_2, auto_g_1, auto_g_2


class Trainer:
    def __init__(self, config, is_train=True):
        self.model = Conv2dGruConv2d(config).to(DEVICE)
        self.batch_size = config['batch_size']
        self.dataset = Dataset(config['train_data_path'], config, cache_all=config['seq_len']==15)
        self.eval_dataset = Dataset(config['eval_data_path'], config, cache_all=config['seq_len']==15)
        self.eval_data_loader = DataLoader(
            self.eval_dataset, batch_size=self.batch_size, shuffle=True
        )
        self.train_data_loader = DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True
        )
        print(len(self.dataset))
        self.mse_loss = nn.MSELoss(reduction='sum').to(DEVICE)
        self.model.to(DEVICE)
        self.model_path = config['model_path']
        self.kld_loss_scalar = config['kld_loss_scalar']
        self.z_rnn_loss_scalar = config['z_rnn_loss_scalar']
        self.checkpoint_interval = config['checkpoint_interval']
        self.learning_rate = config['learning_rate']
        self.max_iter_num = config['max_iter_num']
        self.base_len = config['base_len']
        self.train_result_path = config['train_result_path']
        self.train_record_path = config['train_record_path']
        self.eval_record_path = config['eval_record_path']
        self.log_interval = config['log_interval']
        self.eval_interval = config['eval_interval']
        self.griffin_lim = T.GriffinLim(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
        )
        self.config = config
        self.is_save_img = config['is_save_img']
        self.is_intrinsic_train = config['is_intrinsic_train']
        self.is_normal_train = config['is_normal_train']
        self.is_stupid_train = config['is_stupid_train']
        self.linear_schedule = config['linear_schedule']
        self.is_stupid_add_len = config['is_stupid_add_len']

        self.linear_schedule_rate = self.linear_scheduler_func(0)
        self.curr_train_iter = 0


    def save_result_imgs(self, img_list, name, seq_len):
        result = torch.cat([img[0] for img in img_list], dim=0)
        save_image(result, self.train_result_path + str(name) + '.png', seq_len)


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


    def linear_scheduler_func(self, curr_iter):
        if self.linear_schedule <= 0:
            return 0
        else:
            sample_rate = 1 - 1 / (self.linear_schedule * self.max_iter_num) * curr_iter
        return max(0, sample_rate)


    def intrinsic_train(self, x, z_gt):
        prompt_1, prompt_2, auto_g_1, auto_g_2 = sample_lengths()
        x_loss, z_loss = self.intrinsic_one_step(z_gt, x, prompt_1, prompt_2, auto_g_1, auto_g_2)
        return x_loss, z_loss
    

    def intrinsic_one_step(self, z_gt, x, prompt_1, prompt_2, auto_g_1, auto_g_2):
        total_len_1 = prompt_1 + auto_g_1
        z_prompt_1 = z_gt
        z_gt_1 = z_gt[:, :total_len_1, :]
        sample_1 = list(range(prompt_1))
        if self.linear_schedule > 0:
            add_sample_1 = random.sample(range(prompt_1, total_len_1), int(auto_g_1 * self.linear_schedule_rate))
            sample_1.extend(add_sample_1)
            print(f"Sample 1 points: {sample_1}")
        z1_gen = self.model.predict_with_symmetry(z_prompt_1, sample_1, lambda z: z, total_len_1)
        x_1 = x[:, 1:total_len_1, :, :, :]
        x_loss_1, z_loss_1 = self.calc_rnn_loss(x_1, z_gt_1, z1_gen)

        total_len_2 = prompt_2 + auto_g_2
        z_prompt_2 = z1_gen[:, z1_gen.size(1) - prompt_2:, :]
        z_prompt_2 = torch.cat((z_prompt_2, z_gt[:, total_len_1:, :]), 1)
        z_gt_2 = z_gt[:, total_len_1 - prompt_2:total_len_1 + auto_g_2, :]
        sample_2 = list(range(prompt_2))
        if self.linear_schedule > 0:
            add_sample_2 = random.sample(range(prompt_2, total_len_2), int(auto_g_2 * self.linear_schedule_rate))
            sample_2.extend(add_sample_2)
            print(f"Sample 2 points: {sample_2}")
        z2_gen = self.model.predict_with_symmetry(z_prompt_2, sample_2, lambda z: z, total_len_2)
        x_2 = x[:, total_len_1 - prompt_2 + 1:total_len_1 + auto_g_2, :, :, :]
        x_loss_2, z_loss_2 = self.calc_rnn_loss(x_2, z_gt_2, z2_gen)
        return x_loss_1 + x_loss_2, z_loss_1 + z_loss_2
    

    def eval(self, iter_num):
        self.config['eval_recons'] = True
        self.model.eval()
        with torch.no_grad():
            data = None
            for batch_ndx, sample in enumerate(self.eval_data_loader):
                data = sample
                break
            data = data.to(DEVICE)
            data = norm_log2(data, k=LOG_K)
            z_gt, mu, logvar = self.model.batch_seq_encode_to_z(data)
            pred_x_loss, pred_z_loss = self.normal_train(data, z_gt)
            wandb.log({
                'eval_pred_x_loss': pred_x_loss.item(),
                'eval_z_loss': pred_z_loss.item(),
                'iter': iter_num,
            })
        self.config['eval_recons'] = False
        self.model.train()
    

    def stupid_train_2(self, z_gt):
        base_len_1, base_len_2 = random.sample(range(3, 7), 2)
        z_gen_1 = self.normal_predict(z_gt, base_len_1, is_schedule=False)
        z_gen_2 = self.normal_predict(z_gt, base_len_2, is_schedule=False)
        max_base_len = max(base_len_1, base_len_2)
        z_loss = self.mse_loss(z_gen_1[:, max_base_len:, :], z_gen_2[:, max_base_len:, :])
        return z_loss * self.z_rnn_loss_scalar
        

    def stupid_train(self, x, z_gt):
        shift = random.randint(1, 10)
        if self.is_stupid_add_len:
            add_len = shift
        else:
            add_len = 0
        total_len = z_gt.size(1) + add_len
        z_gen = self.normal_predict(z_gt, base_len=None, is_schedule=False, total_len=total_len)
        x1 = x[:, 1:, :, :, :]
        x1_gen = self.model.batch_seq_decode_from_z(z_gen)
        x1_gen_shift = x1_gen[:, shift:, :, :, :]
        
        z_gen_shift = z_gen[:, shift:, :]
        z_gt_shift = z_gt[:, shift:, :]
        shift_base_len = min(z_gt_shift.size(1), self.base_len)
        z_shift_gen = self.normal_predict(z_gt_shift, base_len=shift_base_len, is_schedule=False, total_len=total_len - shift)
        x1_shift_gen = self.model.batch_seq_decode_from_z(z_shift_gen)

        # Basic RNN loss
        xloss= nn.BCELoss(reduction='sum')(x1_gen[:, :x1_gen.size(1)-add_len, ...], x1)
        zloss = self.mse_loss(z_gen[:, :z_gen.size(1)-add_len, ...], z_gt[:, 1:, :])

        # Shift gen to gt loss
        xloss_shift = nn.BCELoss(reduction='sum')(
            x1_shift_gen[:, :x1_shift_gen.size(1)-add_len, ...], 
            x1[:, shift:, ...])
        zloss_shift = self.mse_loss(z_shift_gen[:, :z_shift_gen.size(1)-add_len, ...], z_gt[:, 1+shift:, :])

        # Shift gen to gen shift loss
        xloss_gen_shift = nn.BCELoss(reduction='sum')(x1_gen_shift, x1_shift_gen.detach()) + nn.BCELoss(reduction='sum')(x1_shift_gen, x1_gen_shift.detach())
        zloss_gen_shift = self.mse_loss(z_gen_shift, z_shift_gen)

        if self.curr_train_iter % 1000 == 0 and not self.config['eval_recons'] and self.is_save_img:
            save_spectrogram(tensor2spec(x1_gen[0])[0], f'{self.train_result_path}{self.curr_train_iter}-stupid_pred.png')

        return (xloss + xloss_shift + xloss_gen_shift), self.z_rnn_loss_scalar * (zloss + zloss_shift + zloss_gen_shift)



    def normal_predict(self, z_gt, base_len=None, is_schedule=False, total_len=None):
        if total_len is None:
            total_len = z_gt.size(1)
        if base_len is None:
            base_len = self.base_len
        sample = list(range(base_len))
        if self.linear_schedule > 0 and is_schedule and not self.config['eval_recons']:
            add_sample = random.sample(range(base_len, total_len), int((total_len - base_len) * self.linear_schedule_rate))
            sample.extend(add_sample)
            print(f"Sample points: {sample}")
        z_gen = self.model.predict_with_symmetry(z_gt, sample, lambda z: z, total_len)
        return z_gen
    
    
    def normal_train(self, x, z_gt, base_len=None):
        z_gen = self.normal_predict(z_gt, base_len, is_schedule=True)
        x1 = x[:, 1:, :, :, :]
        x_loss, z_loss = self.calc_rnn_loss(x1, z_gt, z_gen)
        return x_loss, z_loss


    def encode_first_frame_to_z(self, data):
        data = data.to(DEVICE)
        data = norm_log2(data, k=LOG_K)
        data = data[:, 0:1, :, :, :]
        with torch.no_grad():
            z_gt, mu, logvar = self.model.batch_seq_encode_to_z(data)
            return mu[:, 0, :].cpu()
        

    def init_wandb(self):
        wandb.init(
            project=self.config['project_name'], 
            name=self.config['name'],
            config=self.config,
            group= 'intrinsic' if self.is_intrinsic_train else 'normal',
        )


    def train(self):
        self.init_wandb()
        if not self.config['eval_recons']:
            create_path_if_not_exist(self.train_result_path)
            self.model.train()
            self.resume()
            train_loss_counter = LossCounter(['loss_ED', 'loss_ERnnD', 'loss_z', 'KLD'])
            iter_num = train_loss_counter.load_iter_num(self.train_record_path)
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        else:
            self.model.eval()
            self.model.load_state_dict(self.model.load_tensor(self.model_path))
            train_loss_counter = LossCounter(['loss_ED', 'loss_ERnnD', 'loss_z', 'KLD'])
            iter_num = 0
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        for i in range(iter_num, self.max_iter_num):
            print(i)
            self.curr_train_iter = i
            self.linear_schedule_rate = self.linear_scheduler_func(i)
            # print(f"Linear schedule rate: {self.linear_schedule_rate}")
            # data = next(self.train_data_loader)
            data = None
            for batch_ndx, sample in enumerate(self.train_data_loader):
                data = sample
                break
            data = data.to(DEVICE) # change me
            data = norm_log2(data, k=LOG_K)
            is_log = (i % self.log_interval == 0 and i != 0)
            optimizer.zero_grad()
            z_gt, mu, logvar = self.model.batch_seq_encode_to_z(data)
            if self.config['ae'] or self.config['eval_recons']:
                z_gt = mu
        
            # Recons, KLD
            vae_loss = self.calc_vae_loss(data, z_gt, mu, logvar) 
            if self.config['ae'] and not self.config['eval_recons']:
                vae_loss = vae_loss[0], torch.zeros_like(vae_loss[1])

            total_pred_x_loss = torch.zeros(1, device=DEVICE)
            total_pred_z_loss = torch.zeros(1, device=DEVICE)
            # Pred_recon, rnn_prior
            if self.is_intrinsic_train:
                pred_x_loss, pred_z_loss = self.intrinsic_train(data, z_gt)
                total_pred_x_loss += pred_x_loss
                total_pred_z_loss += pred_z_loss
            if self.is_normal_train:
                pred_x_loss, pred_z_loss = self.normal_train(data, z_gt)
                total_pred_x_loss += pred_x_loss
                total_pred_z_loss += pred_z_loss
            if self.is_stupid_train:
                pred_x_loss, pred_z_loss = self.stupid_train(data, z_gt)
                total_pred_x_loss += pred_x_loss
                total_pred_z_loss += pred_z_loss

            # compute loss
            loss = self.loss_func(i, vae_loss, total_pred_x_loss, total_pred_z_loss, train_loss_counter)
            loss.backward()
            optimizer.step()

            if is_log and not self.config['eval_recons']:
                self.eval(i)
                r2_train = eval_linearity(self.dataset, self.encode_first_frame_to_z)
                r2_eval = eval_linearity(self.eval_dataset, self.encode_first_frame_to_z)
                wandb.log({'r2_train': r2_train, 'r2_eval': r2_eval,'iter': i})
                self.model.save_tensor(self.model.state_dict(), self.model_path)
                print(train_loss_counter.make_record(i))
                train_loss_counter.record_and_clear(self.train_record_path, i)
            if i % self.checkpoint_interval == 0 and i != 0 and not self.config['eval_recons']:
                self.model.save_tensor(self.model.state_dict(), f'{self.config["name"]}_checkpoint_{i}.pt')
        wandb.finish()
        self.curr_train_iter = 0
                

    def save_audio(self, spec, name, sample_rate=16000):
        recon_waveform = self.griffin_lim(spec.cpu())
        torchaudio.save(name, recon_waveform, sample_rate)

    def batch_normalize(self, tensor: torch.Tensor):
        mean = torch.mean(tensor)
        std = torch.std(tensor)
        return (tensor - mean) / std


    def calc_rnn_loss(self, x1, z_gt, z0_rnn, z_gt_cr=None, is_save_img=True):
        if z_gt_cr is None:
            z_next = z0_rnn
        else:
            z_next = torch.cat((z0_rnn, z_gt_cr), -1)
        recon_next = self.model.batch_seq_decode_from_z(z_next)
        if self.config['eval_recons']:
            xloss_ERnnD = nn.BCELoss(reduction='mean')(recon_next, x1)
            zloss_Rnn = nn.MSELoss(reduction='mean')(self.batch_normalize(z0_rnn), self.batch_normalize(z_gt[:, 1:, :]))
            if zloss_Rnn.item() > 1:
                print(f"Outlier !!! ")
                print(z0_rnn)
                print(z_gt[:, 1:, :])
        else:
            xloss_ERnnD = nn.BCELoss(reduction='sum')(recon_next, x1)
            zloss_Rnn = self.z_rnn_loss_scalar * self.mse_loss(z0_rnn, z_gt[:, 1:, :])
        if self.curr_train_iter % 1000 == 0 and not self.config['eval_recons'] and self.is_save_img and is_save_img:
            save_spectrogram(tensor2spec(recon_next[0])[0], f'{self.train_result_path}{self.curr_train_iter}-recon_pred.png')
            # self.save_audio(tensor2spec(recon_next[0]), f'{self.train_result_path}{log_num}-recon_pred.wav')

        return xloss_ERnnD, zloss_Rnn

    def calc_vae_loss(self, data, z_gt, mu, logvar):
        recon = self.model.batch_seq_decode_from_z(z_gt)
        if self.config['eval_recons']:
            recon_loss = nn.BCELoss(reduction='mean')(recon, data)
        else:
            recon_loss = nn.BCELoss(reduction='sum')(recon, data)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - torch.exp(logvar)) * self.kld_loss_scalar
        if self.curr_train_iter % 1000 == 0 and not self.config['eval_recons']:
            save_spectrogram(tensor2spec(data[0])[0], f'{self.train_result_path}{self.curr_train_iter}-gt.png')
            # self.save_audio(tensor2spec(data[0]), f'{self.train_result_path}{log_num}-gt.wav')
            save_spectrogram(tensor2spec(recon[0])[0], f'{self.train_result_path}{self.curr_train_iter}-recon.png')
            # self.save_audio(tensor2spec(recon[0]), f'{self.train_result_path}{log_num}-recon.wav')
        return recon_loss, KLD


    def loss_func(self, iter, vae_loss, pred_x_loss, pred_z_loss, loss_counter):
        xloss_ED, KLD = vae_loss
        loss = torch.zeros(1, device=DEVICE)
        loss += xloss_ED + KLD + pred_x_loss + pred_z_loss
        loss_counter.add_values([xloss_ED.item(), 
                                 pred_x_loss.item(), 
                                 pred_z_loss.item(), 
                                 KLD.item()
                                 ])
        wandb.log({
            'xloss_ED': xloss_ED.item(),
            'pred_x_loss': pred_x_loss.item(),
            'pred_z_loss': pred_z_loss.item(),
            'KLD': KLD.item(),
            'iter': iter,
        })
        return loss


if __name__ == '__main__':
    print(sample_lengths())
