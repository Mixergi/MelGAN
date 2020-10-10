audio_params = {
    "sample_rate": 22050,
    "hop_length": 256,
    "n_fft": 1024,
    "win_length": 1024,
    "n_mels": 80
}

hparams = {
    "discriminator_opt": "Adam",
    "discriminator_learning_rate": 0.0001,
    "generator_opt": "Adam",
    "generator_learning_rate": 0.0001,
    "epochs": 2000
}