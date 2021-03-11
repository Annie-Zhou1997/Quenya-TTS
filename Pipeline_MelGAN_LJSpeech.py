"""
Train non-autoregressive spectrogram inversion model on the german single speaker dataset by Hokuspokus
"""

import os
import random
import warnings

import torch

from MelGAN.MelGANDataset import MelGANDataset
from MelGAN.MelGANGenerator import MelGANGenerator
from MelGAN.MelGANMultiScaleDiscriminator import MelGANMultiScaleDiscriminator
from MelGAN.melgan_train_loop import train_loop

warnings.filterwarnings("ignore")

torch.manual_seed(13)
random.seed(13)


def get_file_list():
    file_list = list()
    for wav_file in os.listdir("/mount/resources/speech/corpora/LJSpeech/16kHz/wav"):
        if ".wav" in wav_file:
            file_list.append("/mount/resources/speech/corpora/LJSpeech/16kHz/wav/" + wav_file)
    return file_list


if __name__ == '__main__':
    print("Preparing")
    fl = get_file_list()
    model_save_dir = "Models/MelGAN/MultiSpeaker/LJSpeech"
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    cache_dir = "Corpora/LJSpeech"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    train_dataset = MelGANDataset(list_of_paths=fl[:-100], cache_dir=os.path.join(cache_dir, "melgan_train_cache.json"))
    valid_dataset = MelGANDataset(list_of_paths=fl[-100:], cache_dir=os.path.join(cache_dir, "melgan_valid_cache.json"))
    generator = MelGANGenerator()
    generator.reset_parameters()
    multi_scale_discriminator = MelGANMultiScaleDiscriminator()

    print("Training model")
    train_loop(batchsize=64,
               epochs=600000,  # just kill the process at some point
               generator=generator,
               discriminator=multi_scale_discriminator,
               train_dataset=train_dataset,
               valid_dataset=valid_dataset,
               device=torch.device("cuda:2"),
               generator_warmup_steps=200000,
               model_save_dir=model_save_dir)