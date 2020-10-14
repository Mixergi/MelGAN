import datetime
import math
import os

import numpy as np
import tensorflow as tf
from tqdm import tqdm

class MelGAN_Trainer():
    def __init__(self, generator, discriminator, **kwargs):

        self.generator = generator
        self.discriminator = discriminator

        self.mse = tf.keras.losses.MeanSquaredError()
        self.mae = tf.keras.losses.MeanAbsoluteError()

        self.metrics_name = [
            "adversarial_loss",
            "feature_matching_loss",
            "generator_loss",
            "real_loss",
            "fake_loss",
            "discriminator_loss"
        ]

        self.train_metrics = {}
        self.valid_metrics = {}
        
        self.__init_metrics()

        self.valid_use = False
    
    def __init_metrics(self):

        for name in self.metrics_name:
            self.train_metrics[name] = tf.keras.metrics.Mean()
            self.valid_metrics[name] = tf.keras.metrics.Mean()
        
    def compile(self, generator_opt, discriminator_opt):
        
        self.generator_opt = generator_opt
        self.discriminator_opt = discriminator_opt

    def train(self, train_dataset, epochs=1, batch_size=1, valid_dataset=None, use_tensorboard=False):
        
        if valid_dataset:
            self.valid_use = True
        else:
            self.valid_use = False

        if use_tensorboard:
            current_time =  datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
            self.train_summary_writter = tf.summary.create_file_writer(train_log_dir)
            
            if self.valid_use:
                valid_log_dir = 'logs/gradient_tape/' + current_time + '/test'
                self.valid_summary_writter = tf.summary.create_file_writer(valid_log_dir)

        for epoch in tqdm(range(epochs)):
            print(f'{epoch + 1} epochs...')

            for batch in tqdm(train_dataset.create(batch_size), total=train_dataset.get_length()):
                self._train_step(batch, training=True)

            if self.valid_use:
                for batch in valid_dataset.create(batch_size):
                    self._train_step(batch, training=False)

            for key, value in self.train_metrics.items():
                print(f"{key}: {value.result()}", end='\t')

            print()

            if self.valid_use:
                for key, value in self.valid_metrics.items():
                    print(f"{key}: {value.result()}", end='\t')

            print()

            if use_tensorboard:
                self._write_on_tensorboard(epoch)

    @tf.function
    def _train_step(self, batch, training=True):
        
        x, y = batch

        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            y_hat = self.generator(x)
            
            p_hat = self.discriminator(y_hat)
            p = self.discriminator(y)

            #Generator loss
            adversarial_loss = 0

            for output in p_hat:
                adversarial_loss += tf.reduce_mean(output[-1])
            
            adversarial_loss /= len(p_hat)

            feature_matching_loss = 0

            for output_hat, output in zip(p_hat, p):
                for feature_hat, feature in zip(output_hat, output):
                    feature_matching_loss += self.mse(feature_hat, feature)

            feature_matching_loss /= len(p_hat) * len(output_hat)

            generator_loss = adversarial_loss + 10 * feature_matching_loss

            #Discriminator loss
            real_loss = 0
            fake_loss = 0

            for ouput in p:
                real_loss += tf.reduce_mean(tf.maximum(0.0, 1 - ouput[-1]))

            for ouput in p_hat:
                fake_loss += tf.reduce_mean(tf.maximum(0.0, 1 + ouput[-1]))

            discriminator_loss = real_loss + fake_loss

        if training:
            gradient = g_tape.gradient(
                generator_loss, self.generator.trainable_variables
            )
            self.generator_opt.apply_gradients(
                zip(gradient, self.generator.trainable_variables)
            )

            gradient = d_tape.gradient(
                discriminator_loss, self.discriminator.trainable_variables
            )
            self.discriminator_opt.apply_gradients(
                zip(gradient, self.discriminator.trainable_variables)
            )

            self.train_metrics['adversarial_loss'].update_state(
                adversarial_loss)
            self.train_metrics['feature_matching_loss'].update_state(
                feature_matching_loss)
            self.train_metrics['generator_loss'].update_state(generator_loss)
            self.train_metrics['real_loss'].update_state(real_loss)
            self.train_metrics['fake_loss'].update_state(fake_loss)
            self.train_metrics['discriminator_loss'].update_state(
                discriminator_loss
            )
    
        else:
            self.valid_metrics['adversarial_loss'].update_state(
                adversarial_loss)
            self.valid_metrics['feature_matching_loss'].update_state(
                feature_matching_loss)
            self.valid_metrics['generator_loss'].update_state(generator_loss)
            self.valid_metrics['real_loss'].update_state(real_loss)
            self.valid_metrics['fake_loss'].update_state(fake_loss)
            self.valid_metrics['discriminator_loss'].update_state(
                discriminator_loss
            )
    
    def _write_on_tensorboard(self, epoch):
        with self.train_summary_writter.as_default():
            for key, value in self.train_metrics.items():
                tf.summary.scalar(key, value.result(), epoch)

        if self.use_valid:
            with self.valid_summary_writter.as_default():
                for key, value in self.valid_metrics.items():
                    tf.summary.scalar(key, value.result(), epoch)
