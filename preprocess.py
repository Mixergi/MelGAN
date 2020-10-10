import argparse
import gc
import math
import multiprocessing
import os
import random

import librosa
import numpy as np
from tqdm import tqdm

from hparams import audio_params


def wav_to_log_mel(wav_data, sample_rate, n_fft, hop_length, win_length, n_mels):
    mel_spectrogram = librosa.feature.melspectrogram(wav_data,
                                                     sample_rate,
                                                     n_fft=n_fft,
                                                     hop_length=hop_length,
                                                     win_length=win_length,
                                                     n_mels=n_mels)

    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    log_mel_spectrogram = log_mel_spectrogram.T[::-1]
    return log_mel_spectrogram


def preprocess(data_dir,
               save_dir,
               spectrogram_length,
               random_start,
               sample_rate,
               n_fft,
               hop_length,
               win_length,
               n_mels):

    wav_data, _ = librosa.load(data_dir, sr=sample_rate)

    mel_size = hop_length * spectrogram_length

    if len(wav_data) < mel_size:
        wav_data = np.concatenate((wav_data, np.zeros(
            hop_length * spectrogram_length - len(wav_data))))

    else:
        if random_start:
            start_index = random.randrange(0, len(wav_data) - mel_size + 1)
            wav_data = wav_data[start_index:start_index+mel_size]

        else:
            wav_data = wav_data[:mel_size]

    log_mel_spectrogram = wav_to_log_mel(
        wav_data[:-1], sample_rate, n_fft, hop_length, win_length, n_mels)

    save_array = np.array([log_mel_spectrogram, wav_data])
    np.save(save_dir, save_array)


def preprocess_loop(data_dir,
                    save_dir,
                    num_workers,
                    spectrogram_length,
                    random_start,
                    sample_rate,
                    n_fft,
                    hop_length,
                    win_length,
                    n_mels):

    pool = multiprocessing.Pool(num_workers)

    for i in tqdm(range(math.ceil(len(data_dir)/num_workers))):
        file_path = data_dir[i*num_workers:(i+1)*num_workers]
        save_path = [os.path.join(save_dir, file_name.split(
            '\\')[-1][:-4]) for file_name in file_path]

        pool.starmap(preprocess, [(file_path, save, spectrogram_length, random_start,
                                   sample_rate, n_fft, hop_length, win_length, n_mels) for file_path, save in zip(file_path, save_path)])
    
    pool.close()


def main(args):
    os.makedirs(args.train_dir, exist_ok=True)
    os.makedirs(args.valid_dir, exist_ok=True)
    os.makedirs(args.test_dir, exist_ok=True)

    file_list = os.listdir(args.data_dir)
    file_dir = [os.path.join(args.data_dir, file_name)
                for file_name in file_list]
    random.shuffle(file_dir)

    train_file = file_dir[args.test_num + args.valid_num:]
    valid_file = file_dir[args.test_num:args.test_num + args.valid_num]
    test_file = file_dir[:args.test_num]

    print("Preprocessing Train Data")
    preprocess_loop(train_file,
                    args.train_dir,
                    args.num_workers,
                    args.spectrogram_length,
                    args.random_start,
                    args.sample_rate,
                    args.n_fft,
                    args.hop_length,
                    args.win_length,
                    args.n_mels)

    print("Preprocessing Valid Data")
    preprocess_loop(valid_file,
                    args.valid_dir,
                    args.num_workers,
                    args.spectrogram_length,
                    args.random_start,
                    args.sample_rate,
                    args.n_fft,
                    args.hop_length,
                    args.win_length,
                    args.n_mels)

    print("Preprocessing Test Data")
    preprocess_loop(test_file,
                    args.test_dir,
                    args.num_workers,
                    args.spectrogram_length,
                    args.random_start,
                    args.sample_rate,
                    args.n_fft,
                    args.hop_length,
                    args.win_length,
                    args.n_mels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--train_dir", default="./train")
    parser.add_argument("--valid_dir", default="./valid")
    parser.add_argument("--valid_num", default="5", type=int)
    parser.add_argument("--test_dir", default="./test")
    parser.add_argument("--test_num", default="5", type=int)
    parser.add_argument("--spectrogram_length", default=200, type=int)
    parser.add_argument("--random_start", default=True, type=int)
    parser.add_argument(
        "--sample_rate", default=audio_params["sample_rate"], type=int)
    parser.add_argument("--n_fft", default=audio_params["n_fft"], type=int)
    parser.add_argument(
        "--hop_length", default=audio_params["hop_length"], type=int)
    parser.add_argument(
        "--win_length", default=audio_params["win_length"], type=int)
    parser.add_argument(
        "--n_mels", default=audio_params["n_mels"], type=int)
    parser.add_argument(
        "--num_workers", default=multiprocessing.cpu_count(), type=int)

    args = parser.parse_args()

    main(args)
