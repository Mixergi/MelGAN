import argparse
import glob
import math
import multiprocessing
import os
import random

import numpy as np
import librosa
import torch
from tqdm import tqdm


def main(args):
    data_dir = glob.glob(os.path.join(args.data_dir, "*.wav"))
    random.shuffle(data_dir)

    train_data = data_dir[:-(args.valid_num+args.test_num+args.sample_num)]
    valid_data = data_dir[-(args.valid_num+args.test_num+args.sample_num):-(args.test_num+args.sample_num)]
    test_data = data_dir[-(args.test_num+args.sample_num):-args.sample_num]
    sample_data = data_dir[-args.sample_num:]

    pool = multiprocessing.Pool(args.num_workers)

    print("Train Data Processing Starting...")
    for i in tqdm(range(math.ceil(len(train_data)/args.num_workers))):
        start_index = i * args.num_workers
        pool.starmap(process_audio, ((data_path, args.train_dir, args.hop_length)
                                     for data_path in train_data[start_index:start_index+args.num_workers]))

    print("Valid Data Process Starting...")
    for i in tqdm(range(math.ceil(len(valid_data)/args.num_workers))):
        start_index = i * args.num_workers
        pool.starmap(process_audio, ((data_path, args.valid_dir, args.hop_length)
                                     for data_path in valid_data[start_index:start_index+args.num_workers]))
                                     
    print("Test Data Processing Starting...")
    for i in tqdm(range(math.ceil(len(test_data)/args.num_workers))):
        start_index = i * args.num_workers
        pool.starmap(process_audio, ((data_path, args.test_dir, args.hop_length)
                                     for data_path in test_data[start_index:start_index+args.num_workers]))

    print("Sample Data Processing Starting...")
    for i in tqdm(range(math.ceil(len(sample_data)/args.num_workers))):
        start_index = i * args.num_workers
        pool.starmap(process_audio, ((data_path, args.sample_dir, args.hop_length)
                                     for data_path in sample_data[start_index:start_index+args.num_workers]))

    pool.close()

def process_audio(data_path, save_dir, hop_length=256):

    file_name = os.path.split(data_path)[-1][:-4]

    wav, sr = librosa.load(data_path)

    mel_basis = librosa.filters.mel(sr, 1024, 80)
    spectrogram = librosa.stft(wav, 1024, hop_length, 1024)
    mel_spectrogram = np.dot(mel_basis, np.abs(spectrogram).astype(np.float32))

    wav = torch.from_numpy(wav)
    mel_spectrogram = torch.from_numpy(mel_spectrogram)

    torch.save(wav, os.path.join(save_dir, file_name + ".pt"))
    torch.save(mel_spectrogram, os.path.join(save_dir, file_name + ".mel.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()   

    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--train_dir", default="./train")
    parser.add_argument("--valid_dir", default="./valid")
    parser.add_argument("--sample_dir", default="./sample")
    parser.add_argument("--test_dir", default="./test")
    parser.add_argument("--valid_num", default=5, type=int)
    parser.add_argument("--test_num", default=5, type=int)
    parser.add_argument("--sample_num", default=1, type=int)
    parser.add_argument("--hop_length", default=256, type=int)
    parser.add_argument(
        "--num_workers", default=multiprocessing.cpu_count(), type=int)

    args = parser.parse_args()

    os.makedirs(args.train_dir, exist_ok=True)
    os.makedirs(args.valid_dir, exist_ok=True)
    os.makedirs(args.test_dir, exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)

    main(args)
