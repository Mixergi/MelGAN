import math
import os
import time

import numpy as np
import tensorflow as tf
import tqdm


class MelGAN_Trainer():
    def __init__(self, discriminator, generator, **kwargs):
        
        self.discriminator = discriminator
        self.generator = generator

        self.metrics_name = [
            'adversarial_loss',
            'feature_matching_loss',
            'gen_loss',
            'real_loss',
            'fake_loss',
            'dis_loss'
        ]

        self.train_metrics = {}
        self.valid_metrics = {}

    def init_metrics(self):
        for name in self.metrics_name:
            self.train_metrics[name] = tf.keras.metrics.Mean()
            self.valid_metrics[name] = tf.keras.metrics.Mean()

    def compile(self, D_opt, G_opt, d_learning_rate, g_learning_rate):
        self.D_opt = getattr(tf.keras.optimizers, D_opt)(d_learning_rate)
        self.G_opt = getattr(tf.keras.optimizers, G_opt)(g_learning_rate)

    def train(self, train_dataset, epochs=1, valid_dataset=None):
        pass

    @tf.function()
    def _train_step(self, batch, training=True):
        mels, y = batch
        y_hat = self._generator_step(mels, y, training)
        self._discriminator_step(y, y_hat, training)
    
    @tf.function()
    def _discriminator_step(self, y, y_hat, training=True):
        with tf.GradientTape() as tape:
            pass

        if training:
            pass
        
        else:
            pass
    
    @tf.function()
    def _generator_step(self, mels, y, training=True):
        with tf.GradientTape() as tape:
            pass

        if training:
            pass

        else:
            pass