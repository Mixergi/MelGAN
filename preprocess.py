import argparse
import multiprocessing
import os

import librosa
import tqdm


def main(args):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--train_dir", default="./train")
    parser.add_argument("--valid_dir", default="./valid")
    parser.add_argument("--valid_num", default="2")
    parser.add_argument("--test_dir", default="./test")
    parser.add_argument("--test_num", default="2")
    parser.add_argument("--num_workers", default=multiprocessing.cpu_count(), type=int)
    args = parser.parse_args()

    main(args)

