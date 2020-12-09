import glob
import os
import random

import torch
from torch.utils.data import DataLoader, Dataset


class MelWavDataset(Dataset):
    def __init__(self, data_dir, data_length=32, hop_length=256):

        self.data_dir = data_dir
        self.data_length = data_length
        self.hop_length = hop_length

        self.file_names = [i[:-7]
                           for i in glob.glob(os.path.join(self.data_dir, "*.mel.pt"))]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):

        wav_dir = self.file_names[index] +  ".pt"
        mel_dir = self.file_names[index] + ".mel.pt"

        wav = torch.load(wav_dir)
        mel = torch.load(mel_dir)
        
        if self.data_length is not "MAX":
            start_point = random.randint(0, len(mel[0]) - self.data_length - 1)
            mel = mel[:, start_point : start_point + self.data_length]

            start_point *= self.hop_length
            wav = wav[start_point: start_point +
                    (self.data_length * self.hop_length)].unsqueeze(0)

        return mel, wav
