import os
import random

import numpy as np
import tensorflow as tf

class MelGAN_Dataset:
    def __init__(self, data_dir, shuffle=True):
        self.data_dir = data_dir
        self.shuffle = shuffle

    def _generator(self):

        file_list = os.listdir(self.data_dir)

        if self.shuffle:
            random.shuffle(file_list)

        for file_dir in file_list:
            file_dir = os.path.join(self.data_dir, file_dir)
            mels, wav = np.load(file_dir, allow_pickle=True)

            wav = np.expand_dims(wav, -1)

            return mels, wav

    def create_dataset(self):
        return tf.data.Dataset.from_generator(
            self._generator, 
            output_types=(tf.float32, tf.float32)
            ).prefetch(tf.data.experimental.AUTOTUNE).batch(1)

    def get_lengh(self):
        return len(os.listdir(self.data_dir))
