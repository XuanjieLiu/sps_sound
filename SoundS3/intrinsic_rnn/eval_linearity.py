from SoundS3.sound_dataset import norm_log2
from SoundS3.sound_dataset import Dataset, get_all_data
import torch
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def eval_linearity(dataset: Dataset, data_enc_func: callable):
    all_data, all_pitch, all_instr = get_all_data(dataset)
    x = data_enc_func(all_data)
    model = LinearRegression()
    model.fit(x, all_pitch)
    r2 = r2_score(all_pitch, model.predict(x))
    return r2
