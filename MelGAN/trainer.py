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

        self.mse = tf.keras.losses.MeanSquaredError()
        self.mae = tf.keras.losses.MeanAbsoluteError()

        self.metrics_name = [
            'adversarial_loss',
            'feature_matching_loss',
            'generator_loss',
            'real_loss',
            'fake_loss',
            'discriminator_loss'
        ]

        self.train_metrics = {}
        self.valid_metrics = {}

    def init_metrics(self):
        for name in self.metrics_name:
            self.train_metrics[name] = tf.keras.metrics.Mean()
            self.valid_metrics[name] = tf.keras.metrics.Mean()

    def compile(self, D_opt, G_opt):
        self.D_opt = D_opt
        self.G_opt = G_opt

    def train(self, train_dataset, epochs=1, valid_dataset=None):

        for epoch in tqdm.tqdm(range(epochs)):
            train_data = train_dataset.create_dataset()
            train_data_length = train_data.get_length()

            for batch in tqdm.tqdm(train_data, total=train_data_length):
                self._train_step(batch, training=True)

            if valid_dataset:
                valid_data = valid_dataset.create_dataset()
                valid_data_length = valid_data.get_length()

                for batch in tqdm.tqdm(valid_data, total=valid_data_length):
                    self._train_step(batch, training=False)

    @tf.function()
    def _train_step(self, batch, training=True):
        mels, y = batch
        y_hat = self._generator_step(mels, y, training)
        self._discriminator_step(y, y_hat, training)

    @tf.function()
    def _discriminator_step(self, y, y_hat, training=True):
        with tf.GradientTape() as tape:
            p = self.discriminator(y)

            p_hat = self.discriminator(y_hat)

            real_loss = 0
            fake_loss = 0

            for i in range(len(p)):
                real_loss += self.mse(p[i][-1],
                                      tf.ones_like(p[i][-1], tf.float32))
                fake_loss += self.mse(p_hat[i][-1],
                                      tf.ones_like(p_hat[i][-1], tf.float32))

            real_loss /= len(p)
            fake_loss /= len(p_hat)

            discriminator_loss = real_loss + fake_loss

        if training:
            gradient = tape.gradient(
                discriminator_loss, self.discriminator.trainable_variables)
            self.d_opt.apply_gradients(
                zip(gradient, self.discriminator.trainable_variables))

            self.train_metrics['real_loss'].update_state(real_loss)
            self.train_metrics['fake_loss'].update_state(fake_loss)
            self.train_metrics['discriminator_loss'].update_state(
                discriminator_loss)

        else:
           self.valid_metrics['real_loss'].update_state(real_loss)
           self.valid_metrics['fake_loss'].update_state(fake_loss)
           self.valid_metrics['discriminator_loss'].update_state(
               discriminator_loss)

    @tf.function()
    def _generator_step(self, mels, y, training=True):
        with tf.GradientTape() as tape:
            y_hat = self.generator(mels)
            p_hat = self.discriminator(y_hat)

            adversarial_loss = 0

            for i in range(len(p_hat)):
                adversarial_loss += self.mse(p_hat[i][-1],
                                             tf.ones_like(p_hat[i][-1], dtype=tf.float32))

            adversarial_loss /= len(p_hat)

            p = self.discriminator(y)

            feature_matching_loss = 0

            for i in range(len(p_hat)):
                for j in range(len(p_hat[i]) - 1):
                    feature_matching_loss += self. mae(p_hat[i][j], [i][j])

            feature_matching_loss /= len(p_hat) * len(p_hat[0])
            generator_loss = adversarial_loss + feature_matching_loss * 10

        if training:
            gradient = tape.gradient(
                generator_loss, self.generator.trainable_variables)
            self.G_opt.apply_gradients(
                zip(gradient, self.generator.trainable_variables))

            self.train_metrics['adversarial_loss'].update_state(
                adversarial_loss)
            self.train_metrics['feature_matching_loss'].update_state(
                feature_matching_loss)
            self.train_metrics['generator_loss'].update_state(generator_loss)

        else:
            self.valid_metrics['adversarial_loss'].update_state(
                adversarial_loss)
            self.valid_metrics['feature_matching_loss'].update_state(
                feature_matching_loss)
            self.valid_metrics['generator_loss'].update_state(generator_loss)

        y_hat = self.generator(mels)

        return y_hat
