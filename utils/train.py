import datetime
import math
import os

import numpy as np
import tensorflow as tf
import tqdm


class MelGAN_Trainer():
    def __init__(self, discriminator, generator, **kwargs):

        self.discriminator = discriminator
        self.generator = generator

        self.mse = tf.keras.losses.MeanSquaredError()
        self.mae = tf.keras.losses.MeanAbsoluteError()

        self.use_valid = False

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

        self.init_metrics()

    def init_metrics(self):
        for metrics in self.metrics_name:
            self.train_metrics[metrics] = tf.keras.metrics.Mean()
            self.valid_metrics[metrics] = tf.keras.metrics.Mean()

    def compile(self, D_opt, G_opt):
        self.D_opt = D_opt
        self.G_opt = G_opt

    def train(self, train_dataset, epochs=1, valid_dataset=None, use_tensorboard=False):

        if valid_dataset:
            self.use_valid = True

        current_time =  datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
        self.train_summary_writter = tf.summary.create_file_writer(train_log_dir)

        if self.use_valid:
            valid_log_dir = 'logs/gradient_tape/' + current_time + '/test'
            self.valid_summary_writter = tf.summary.create_file_writer(valid_log_dir)

        for epoch in tqdm.tqdm(range(epochs)):
            train_data = train_dataset.create_dataset()
            train_data_length = train_dataset.get_length()
            
            print(f'\n{epoch} epochs training_loop')
            for batch in tqdm.tqdm(train_data, total=train_data_length):
                self._train_step(batch, training=True)

            if self.use_valid:
                valid_data = valid_dataset.create_dataset()
                valid_data_length = valid_dataset.get_length()
                print(f'{epoch} epochs valid_loop')
                for batch in tqdm.tqdm(valid_data, total=valid_data_length):
                    self._train_step(batch, training=False)
            
            print()

            for metrics in self.metrics_name:
                print(f'train_{metrics}: {self.train_metrics[metrics].result()}', endl='     ')

                self.train_metrics[metrics].reset_states()

            print()

            if self.use_valid:
                for metrics in self.metrics_name:
                    print(f'valid_{metrics}: {self.valid_metrics[metrics].result()}', endl='     ')
                    self.valid_metrics[metrics].reset_states()
            
            if use_tensorboard:
                self._write_on_tensorboard(epoch)

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

            for output in p:
                real_loss += tf.reduce_mean(tf.maximum(0.0, 1 - output[-1]))

            fake_loss = 0

            for output in p_hat:
                fake_loss += tf.reduce_mean(tf.maximum(0.0, 1 + output[-1])) 

            discriminator_loss = real_loss + fake_loss

        if training:
            gradient = tape.gradient(
                discriminator_loss, self.discriminator.trainable_variables)
            self.D_opt.apply_gradients(
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

            for i in p_hat:
                adversarial_loss += -tf.reduce_mean(i[-1])


            adversarial_loss /= len(p_hat)

            p = self.discriminator(y)

            feature_matching_loss = 0

            for i in range(len(p_hat)):
                for j in range(len(p_hat[i]) - 1):
                    feature_matching_loss += self.mae(p_hat[i][j], p[i][j])

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

    def _write_on_tensorboard(self, epoch):
        with self.train_summary_writter.as_default():
            for key, value in self.train_metrics.items():
                tf.summary.scalar(key, value.result(), epoch)

        if self.use_valid:
            with self.valid_summary_writter.as_default():
                for key, value in self.valid_metrics.items():
                    tf.summary.scalar(key, value.result(), epoch)
