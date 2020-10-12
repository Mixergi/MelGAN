import math
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

            yield mels, wav

    def create(self, batch_size=1):

        self.batch_size=batch_size

        return tf.data.Dataset.from_generator(
            self._generator, 
            output_types=(tf.float32, tf.float32)
            ).prefetch(tf.data.experimental.AUTOTUNE).batch(batch_size)

    def get_length(self):
        return math.ceil(len(os.listdir(self.data_dir)) / batch_size)
