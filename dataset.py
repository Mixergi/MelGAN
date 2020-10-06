import os

import numpy as np
import tensorflow as tf


def data_generator(data_dir):
    for file_dir in os.listdir(data_dir):
        file_dir = os.path.join(data_dir, file_dir)
        mels, wav = np.load(file_dir, allow_pickle=True)

        wav = np.expand_dims(wav, -1)

        return mels, wav


def create_dataset(data_dir):
    return tf.data.Dataset.from_generator(
        data_generator,
        output_types=(tf.float32, tf.float32),
        args=data_dir
    ).prefetch(tf.data.experimental.AUTOTUNE).batch(1)
