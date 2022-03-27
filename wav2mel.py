from utils.stft import TacotronSTFT
from utils.util import read_wav_np
from dataset.audio_processing import pitch
import os
import glob
import tqdm
import torch
import argparse
from utils.stft import TacotronSTFT
from utils.util import read_wav_np
from dataset.audio_processing import pitch
from utils.hparams import HParam
import torch.nn.functional as F
from utils.util import str_to_int_list
import numpy as np

def preprocess(data_path, hp, file):
    stft = TacotronSTFT(
        filter_length=hp.audio.n_fft,
        hop_length=hp.audio.hop_length,
        win_length=hp.audio.win_length,
        n_mel_channels=hp.audio.n_mels,
        sampling_rate=hp.audio.sample_rate,
        mel_fmin=hp.audio.fmin,
        mel_fmax=hp.audio.fmax,
    )
    sr, wav = read_wav_np('./ref_wav/Eleanor.wav', hp.audio.sample_rate)
    p = pitch(wav, hp)  # [T, ] T = Number of frames
    wav = torch.from_numpy(wav).unsqueeze(0)
    mel, mag = stft.mel_spectrogram(wav)  # mel [1, 80, T]  mag [1, num_mag, T]
    mel = mel.squeeze(0)  # [num_mel, T]
    mag = mag.squeeze(0)  # [num_mag, T]
    e = torch.norm(mag, dim=0)  # [T, ]
    p = p[: mel.shape[1]]
    np.save("./ref_wav/Eleanor.npy", mel.numpy(), allow_pickle=True)

def main(args, hp):
    preprocess(args.config, hp, hp.data.train_filelist)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="yaml file for configuration"
    )
    args = parser.parse_args()

    hp = HParam(args.config)

    main(args, hp)