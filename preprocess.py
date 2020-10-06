import argparse
import gc
import math
import multiprocessing
import os
import random

import librosa
import numpy as np
import tqdm

from hparams import audio_params


def wav_to_log_mel(wav_data, sample_rate, n_fft, hop_size, win_length):
    mel_spectrogram = librosa.feature.melspectrogram(wav_data,
                                                     sample_rate,
                                                     n_fft=n_fft,
                                                     win_length=win_length)

    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    log_mel_spectrogram = np.swapaxes(log_mel_spectrogram, 1, 0)
    return log_mel_spectrogram


def preprocess_save(file_dir, save_dir, sample_rate, n_fft, hop_size, win_length):
    wav_data, _ = librosa.load(file_dir, sample_rate)
    log_mel_spectrogram = wav_to_log_mel(wav_data[:-(len(wav_data) % hop_size + 1)],
                                         sample_rate,
                                         n_fft,
                                         hop_size,
                                         win_length)

    np.save(save_dir, np.array([log_mel_spectrogram, wav_data[:-(len(wav_data) % hop_size)]]))


def preprocess_loop(file_dir, save_dir, num_workers, sample_rate, n_fft, hop_size, win_length):
    
    p = multiprocessing.Pool(num_workers)

    for i in tqdm.tqdm(range(math.ceil(len(file_dir)/num_workers))):
        file_path = file_dir[i*num_workers:(i+1)*num_workers]
        save_path = [os.path.join(save_dir, file_name.split('\\')[-1][:-4]) for file_name in file_path]

        p.starmap(preprocess_save, [(file_name, save_name, sample_rate, n_fft, hop_size, win_length)
                                     for file_name, save_name in zip(file_path, save_path)])

        gc.collect()

    p.close()


def main(args):
    os.makedirs(args.train_dir, exist_ok=True)
    os.makedirs(args.valid_dir, exist_ok=True)
    os.makedirs(args.test_dir, exist_ok=True)

    file_dir = os.listdir(args.data_dir)
    file_dir = [os.path.join(args.data_dir, file_name) for file_name in file_dir]
    random.shuffle(file_dir)

    train_file = file_dir[args.test_num + args.valid_num:]
    valid_file = file_dir[args.test_num:args.test_num + args.valid_num]
    test_file = file_dir[:args.test_num]

    print("Preprocessing Train Data")
    preprocess_loop(train_file,
                    args.train_dir,
                    args.num_workers,
                    audio_params["sample_rate"],
                    audio_params["n_fft"],
                    audio_params["hop_size"],
                    audio_params["win_length"])

    print("Preprocessing Valid Data")
    preprocess_loop(valid_file,
                    args.valid_dir,
                    args.num_workers,
                    audio_params["sample_rate"],
                    audio_params["n_fft"],
                    audio_params["hop_size"],
                    audio_params["win_length"])

    print("Preprocessing Test Data")
    preprocess_loop(test_file,
                    args.test_dir,
                    args.num_workers,
                    audio_params["sample_rate"],
                    audio_params["n_fft"],
                    audio_params["hop_size"],
                    audio_params["win_length"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="./data") 
    parser.add_argument("--train_dir", default="./train")
    parser.add_argument("--valid_dir", default="./valid")
    parser.add_argument("--valid_num", default="5", type=int)
    parser.add_argument("--test_dir", default="./test")
    parser.add_argument("--test_num", default="5", type=int)
    parser.add_argument("--num_workers", default=multiprocessing.cpu_count(), type=int)

    args = parser.parse_args()

    main(args)
