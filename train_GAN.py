import argparse
import os

import tensorflow as tf

from dataset import MelGAN_Dataset
from hparams import hparams
from MelGAN.Discriminator import Discriminator
from MelGAN.Generator import Generator
from utils.train import MelGAN_Trainer


def main(args):
    train_dataset = MelGAN_Dataset(args.train_dir, True)
    if args.valid_dir:
        valid_dataset = MelGAN_Dataset(args.valid_dir, True)

    generator = Generator()
    discriminator = Discriminator()

    trainer = MelGAN_Trainer(discriminator, generator)

    D_opt = getattr(tf.keras.optimizers, args.discriminator_opt)(args.discriminator_learning_rate)
    G_opt = getattr(tf.keras.optimizers, args.generator_opt)(args.generator_learning_rate)

    trainer.compile(D_opt, G_opt)
    
    trainer.train(train_dataset, args.epochs, valid_dataset, args.use_tensorboard)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_dir", default="./train")
    parser.add_argument("--valid_dir")
    parser.add_argument("--test_dir")
    parser.add_argument("--epochs", default=hparams["epochs"], type=int)
    parser.add_argument("--discriminator_opt", default=hparams["discriminator_opt"])
    parser.add_argument("--discriminator_learning_rate", default=hparams["discriminator_learning_rate"])
    parser.add_argument("--generator_opt", default=hparams["generator_opt"])
    parser.add_argument("--generator_learning_rate", default=hparams["discriminator_learning_rate"])
    parser.add_argument("--use_tensorboard", default=False)

    args = parser.parse_args()

    main(args)
