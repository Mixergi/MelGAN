import argparse
import glob
from multiprocessing import cpu_count
import os

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import soundfile as sf

from melgan.generator import Generator
from melgan.discriminator import Discriminator
from dataset import MelWavDataset


def main(args):

    device = torch.device(args.device)

    train_dataset = MelWavDataset(
        args.train_dir, args.data_length, args.hop_length)
    valid_dataset = MelWavDataset(
        args.valid_dir, args.data_length, args.hop_length)
    test_dataset = MelWavDataset(
        args.test_dir, args.data_length, args.hop_length)
    sample_dataset = MelWavDataset(
        args.sample_dir, "MAX", args.hop_length)

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.batch_num_workers)
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.batch_num_workers)
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.batch_num_workers)

    start_epoch = 0

    generator_save_dir = glob.glob(os.path.join(args.save_dir, "*_generator.pt"))
    discriminator_save_dir = glob.glob(os.path.join(args.save_dir, "*_discriminator.pt"))

    if generator_save_dir:
        generator = torch.load(generator_save_dir[-1], map_location="cpu").to(device)
        generator_name = os.path.split(generator_save_dir[-1])[-1]
        start_epoch = int(generator_name[:generator_name.find('_')]) - 1
    else:
        generator = Generator().to(device)

    if discriminator_save_dir:
        discriminator = torch.load(discriminator_save_dir[-1], map_location="cpu").to(device)
    else:
        discriminator = Discriminator().to(device)

    g_opt = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.9))
    d_opt = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.9))

    if args.use_tensorboard:
        writer = SummaryWriter(args.tensorboard_save_dir, filename_suffix='MelGAN_Train')

    for epoch in range(start_epoch, args.epochs):

        print(f"({epoch+1}/{args.epochs}) epochs")

        train_generator_loss = 0
        train_discriminator_loss = 0

        valid_generator_loss = 0
        valid_discriminator_loss = 0

        for mel, wav in tqdm(train_dataloader):
            mel = mel.to(device)
            wav = wav.to(device)

            y_hat = generator(mel)
            p_hat = discriminator(y_hat)
            p = discriminator(wav)

            generator_loss = get_generator_loss(p, p_hat)

            generator.zero_grad()
            generator_loss.backward()
            g_opt.step()

            y_hat = generator(mel)
            p_hat = discriminator(y_hat)
            p = discriminator(wav)

            discriminator_loss = get_discriminator_loss(p, p_hat)

            discriminator.zero_grad()
            discriminator_loss.backward()
            d_opt.step()

            train_generator_loss += generator_loss
            train_discriminator_loss += discriminator_loss

        train_generator_loss /= len(train_dataloader)
        train_discriminator_loss /= len(train_dataloader)

        for mel, wav in valid_dataloader:
            with torch.no_grad():
                mel = mel.to(device)
                wav = wav.to(device)

                y_hat = generator(mel)
                p_hat = discriminator(y_hat)
                p = discriminator(wav)

                generator_loss = get_generator_loss(p, p_hat)

                discriminator_loss = get_discriminator_loss(p, p_hat)

                valid_generator_loss += generator_loss
                valid_discriminator_loss += discriminator_loss

        valid_generator_loss /= len(valid_dataloader)
        valid_discriminator_loss /= len(valid_dataloader)

        if args.use_tensorboard:
            writer.add_scalar("Generator/Train",
                              train_generator_loss, epoch + 1)
            writer.add_scalar("Discriminator/Train",
                              train_discriminator_loss, epoch + 1)
            writer.add_scalar("Generator/Valid",
                              valid_generator_loss, epoch + 1)
            writer.add_scalar("Discriminator/Valid",
                              valid_discriminator_loss, epoch + 1)

        if (epoch + 1) % args.save_interval == 0:
            save_models(epoch, args.save_dir, generator, discriminator)

        if (epoch + 1) % args.sample_save_interval == 0:
            for i in range(len(sample_dataset)):
                with torch.no_grad():

                    file_name = os.path.split(sample_dataset.file_names[i])[-1]

                    mel = sample_dataset[i][0].to(device).unsqueeze(0)

                    wav = generator(mel).squeeze().cpu()

                    sf.write(os.path.join(
                        args.sample_save_dir, f"{epoch+1}_{file_name}.wav"), wav.detach().numpy(), 22050)

        print(f"\n training_generator_loss: {train_generator_loss}\t\
                training_discriminator_loss: {train_discriminator_loss}\t\
                validtion_generator_loss: {valid_generator_loss}\t\
                validation_discriminator_loss: {valid_discriminator_loss}")

    for mel, wav in test_dataloader:
        with torch.no_grad():
            mel = mel.to(device)
            wav = wav.to(device)

            y_hat = generator(mel)
            p_hat = discriminator(y_hat)
            p = discriminator(wav)

            generator_loss = get_generator_loss(p, p_hat)

            discriminator_loss = get_discriminator_loss(p, p_hat)

    print(f"Test Generator Loss: {generator_loss}\t Test Discriminator Loss: {discriminator_loss}")

    save_models(args.epochs, args.save_dir, generator, discriminator)


def get_generator_loss(p, p_hat):

    adversarial_loss = 0
    feature_matching_loss = 0

    for output_hat in p_hat:
        adversarial_loss += -output_hat[-1].mean()

    for output, output_hat in zip(p_hat, p):
        for feature, feature_hat in zip(output, output_hat):
            feature_matching_loss += F.l1_loss(feature_hat, feature).mean()

    generator_loss = adversarial_loss + 10 * feature_matching_loss

    return generator_loss


def get_discriminator_loss(p, p_hat):

    real_loss = 0
    fake_loss = 0

    for real_output, fake_ouptut in zip(p, p_hat):
        real_loss += F.relu(1 - real_output[-1]).mean()
        fake_loss += F.relu(1 + fake_ouptut[-1]).mean()

    discriminator_loss = real_loss + fake_loss

    return discriminator_loss

def save_models(epoch, save_dir, generator, discriminator):

    for save_path in glob.glob(os.path.join(save_dir, "*")):
        os.remove(save_path)

    torch.save(generator, os.path.join(save_dir, f"{epoch+1}_generator.pt"))
    torch.save(discriminator, os.path.join(save_dir, f"{epoch+1}_discriminator.pt"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--save_dir", default="./models")
    parser.add_argument("--sample_save_dir", default="./sample_wav")
    parser.add_argument("--train_dir", default="./train")
    parser.add_argument("--valid_dir", default="./valid")
    parser.add_argument("--test_dir", default="./test")
    parser.add_argument("--sample_dir", default="./sample")
    parser.add_argument("--data_length", default=32, type=int)
    parser.add_argument("--hop_length", default=256, type=int)
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--batch_num_workers", default=cpu_count(), type=int)
    parser.add_argument("--save_interval", default=5, type=int)
    parser.add_argument("--sample_save_interval", default=5, type=int)
    parser.add_argument("--use_tensorboard", default=False, type=bool)
    parser.add_argument("--tensorboard_save_dir", default="./runs")
    parser.add_argument("--device", default="cpu")

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.sample_save_dir, exist_ok=True)

    if args.use_tensorboard:
        os.makedirs(args.tensorboard_save_dir, exist_ok=True)

    main(args)
