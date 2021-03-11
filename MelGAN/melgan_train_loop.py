import json
import os
import time

import torch
import torch.multiprocessing
from adabound import AdaBound
from torch.utils.data.dataloader import DataLoader

from MelGAN.MultiResolutionSTFTLoss import MultiResolutionSTFTLoss


def train_loop(batchsize=64,
               epochs=10,
               generator=None,
               discriminator=None,
               train_dataset=None,
               valid_dataset=None,
               device=None,
               model_save_dir=None,
               generator_warmup_steps=200000,
               epochs_per_save=10):
    torch.backends.cudnn.benchmark = True
    # we have fixed input sizes, so we can enable benchmark mode

    train_losses = dict()
    train_losses["adversarial"] = list()
    train_losses["multi_res_spectral_convergence"] = list()
    train_losses["multi_res_log_stft_mag"] = list()
    train_losses["generator_total"] = list()
    train_losses["discriminator_mse"] = list()

    valid_losses = dict()
    valid_losses["adversarial"] = list()
    valid_losses["multi_res_spectral_convergence"] = list()
    valid_losses["multi_res_log_stft_mag"] = list()
    valid_losses["generator_total"] = list()
    valid_losses["discriminator_mse"] = list()

    step_counter = 0

    criterion = MultiResolutionSTFTLoss().to(device)
    discriminator_criterion = torch.nn.MSELoss().to(device)

    g = generator.to(device)
    d = discriminator.to(device)
    g.train()
    d.train()
    optimizer_g = AdaBound(g.parameters())
    optimizer_d = AdaBound(d.parameters())

    torch.multiprocessing.set_sharing_strategy('file_system')
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batchsize,
                              shuffle=True,
                              num_workers=16,
                              pin_memory=True,
                              drop_last=True,
                              prefetch_factor=8,
                              persistent_workers=True)
    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=50,
                              shuffle=False,
                              num_workers=10,
                              pin_memory=True,
                              drop_last=False,
                              prefetch_factor=5,
                              persistent_workers=True)

    start_time = time.time()

    for epoch in range(epochs):

        train_losses_this_epoch = dict()
        train_losses_this_epoch["adversarial"] = list()
        train_losses_this_epoch["multi_res_spectral_convergence"] = list()
        train_losses_this_epoch["multi_res_log_stft_mag"] = list()
        train_losses_this_epoch["generator_total"] = list()
        train_losses_this_epoch["discriminator_mse"] = list()

        valid_losses_this_epoch = dict()
        valid_losses_this_epoch["adversarial"] = list()
        valid_losses_this_epoch["multi_res_spectral_convergence"] = list()
        valid_losses_this_epoch["multi_res_log_stft_mag"] = list()
        valid_losses_this_epoch["generator_total"] = list()
        valid_losses_this_epoch["discriminator_mse"] = list()

        optimizer_g.zero_grad()
        optimizer_d.zero_grad()
        for datapoint in train_loader:
            step_counter += 1
            ############################
            #         Generator        #
            ############################

            gold_wave = datapoint[0].to(device)
            melspec = datapoint[1].to(device)
            pred_wave = g(melspec)
            spectral_loss, magnitude_loss = criterion(pred_wave.squeeze(1), gold_wave)
            train_losses_this_epoch["multi_res_spectral_convergence"].append(float(spectral_loss))
            train_losses_this_epoch["multi_res_log_stft_mag"].append(float(magnitude_loss))
            if step_counter > generator_warmup_steps:  # generator needs warmup
                d_outs = d(pred_wave)
                adversarial_loss = 0.0
                for i in range(len(d_outs)):
                    adversarial_loss += discriminator_criterion(d_outs[i][-1],
                                                                d_outs[i][-1].new_ones(d_outs[i][-1].size()))
                adversarial_loss /= (len(d_outs))
                train_losses_this_epoch["adversarial"].append(float(adversarial_loss))
                generator_total_loss = spectral_loss + magnitude_loss + adversarial_loss
            else:
                train_losses_this_epoch["adversarial"].append(0.0)
                generator_total_loss = spectral_loss + magnitude_loss
            train_losses_this_epoch["generator_total"].append(float(generator_total_loss))
            # generator step time
            optimizer_g.zero_grad()
            generator_total_loss.backward()
            optimizer_g.step()
            optimizer_g.zero_grad()

            ############################
            #       Discriminator      #
            ############################

            if step_counter > generator_warmup_steps:  # generator needs warmup
                new_pred = g(melspec).detach()
                fake_loss = 0.0
                d_outs = d(new_pred)
                for i in range(len(d_outs)):
                    fake_loss += discriminator_criterion(d_outs[i][-1], d_outs[i][-1].new_zeros(d_outs[i][-1].size()))
                fake_loss /= len(d_outs)
                real_loss = 0.0
                d_outs = d(gold_wave.unsqueeze(1))
                for i in range(len(d_outs)):
                    real_loss += discriminator_criterion(d_outs[i][-1], d_outs[i][-1].new_ones(d_outs[i][-1].size()))
                real_loss /= len(d_outs)
                discriminator_mse_loss = fake_loss + real_loss
                train_losses_this_epoch["discriminator_mse"].append(float(discriminator_mse_loss))
                # discriminator step time
                optimizer_d.zero_grad()
                discriminator_mse_loss.backward()
                optimizer_d.step()
                optimizer_d.zero_grad()
            else:
                train_losses_this_epoch["discriminator_mse"].append(0.0)
            # print("Step {}".format(step_counter))

        ############################
        #         Evaluate         #
        ############################

        with torch.no_grad():
            g.eval()
            d.eval()
            for datapoint in valid_loader:
                gold_wave = datapoint[0].to(device)
                melspec = datapoint[1].to(device)
                pred_wave = g(melspec)
                spectral_loss, magnitude_loss = criterion(pred_wave.squeeze(1), gold_wave)
                valid_losses_this_epoch["multi_res_spectral_convergence"].append(float(spectral_loss))
                valid_losses_this_epoch["multi_res_log_stft_mag"].append(float(magnitude_loss))
                if step_counter > generator_warmup_steps:  # generator needs warmup
                    d_outs = d(pred_wave)
                    adversarial_loss = 0.0
                    for i in range(len(d_outs)):
                        adversarial_loss += discriminator_criterion(d_outs[i][-1],
                                                                    d_outs[i][-1].new_ones(d_outs[i][-1].size()))
                    adversarial_loss /= (len(d_outs))
                    valid_losses_this_epoch["adversarial"].append(float(adversarial_loss))
                    generator_total_loss = spectral_loss + magnitude_loss + adversarial_loss
                else:
                    valid_losses_this_epoch["adversarial"].append(0.0)
                    generator_total_loss = spectral_loss + magnitude_loss
                valid_losses_this_epoch["generator_total"].append(float(generator_total_loss))
                if step_counter > generator_warmup_steps:  # generator needs warmup
                    fake_loss = 0.0
                    d_outs = d(pred_wave)
                    for i in range(len(d_outs)):
                        fake_loss += discriminator_criterion(d_outs[i][-1],
                                                             d_outs[i][-1].new_zeros(d_outs[i][-1].size()))
                    fake_loss /= len(d_outs)
                    real_loss = 0.0
                    d_outs = d(gold_wave.unsqueeze(1))
                    for i in range(len(d_outs)):
                        real_loss += discriminator_criterion(d_outs[i][-1],
                                                             d_outs[i][-1].new_ones(d_outs[i][-1].size()))
                    real_loss /= len(d_outs)
                    discriminator_mse_loss = fake_loss + real_loss
                    valid_losses_this_epoch["discriminator_mse"].append(float(discriminator_mse_loss))
                else:
                    valid_losses_this_epoch["discriminator_mse"].append(0.0)
            valid_gen_mean_epoch_loss = sum(valid_losses_this_epoch["generator_total"][-len(valid_dataset):]) / len(
                valid_dataset)
            if step_counter > generator_warmup_steps and epoch % epochs_per_save == 0:
                # only then it gets interesting
                torch.save({"generator": g.state_dict(),
                            "discriminator": d.state_dict(),
                            "generator_optimizer": optimizer_g.state_dict(),
                            "discriminator_optimizer": optimizer_d.state_dict()},
                           os.path.join(model_save_dir, "checkpoint_{}.pt".format(step_counter)))
            print("Epoch:                  {}".format(epoch + 1))
            print("Valid Generator Loss:   {}".format(valid_gen_mean_epoch_loss))
            print("Time elapsed:           {} Minutes".format(round((time.time() - start_time) / 60), 2))
            print("Steps:                  {}".format(step_counter))
            g.train()
            d.train()

            # average the losses of the epoch for a smooth plotting experience
            train_losses["adversarial"].append(
                sum(train_losses_this_epoch["adversarial"]) / len(train_losses_this_epoch["adversarial"]))
            train_losses["multi_res_spectral_convergence"].append(
                sum(train_losses_this_epoch["multi_res_spectral_convergence"]) / len(
                    train_losses_this_epoch["multi_res_spectral_convergence"]))
            train_losses["multi_res_log_stft_mag"].append(sum(train_losses_this_epoch["multi_res_log_stft_mag"]) / len(
                train_losses_this_epoch["multi_res_log_stft_mag"]))
            train_losses["generator_total"].append(
                sum(train_losses_this_epoch["generator_total"]) / len(train_losses_this_epoch["generator_total"]))
            train_losses["discriminator_mse"].append(
                sum(train_losses_this_epoch["discriminator_mse"]) / len(train_losses_this_epoch["discriminator_mse"]))
            valid_losses["adversarial"].append(
                sum(valid_losses_this_epoch["adversarial"]) / len(valid_losses_this_epoch["adversarial"]))
            valid_losses["multi_res_spectral_convergence"].append(
                sum(valid_losses_this_epoch["multi_res_spectral_convergence"]) / len(
                    valid_losses_this_epoch["multi_res_spectral_convergence"]))
            valid_losses["multi_res_log_stft_mag"].append(sum(valid_losses_this_epoch["multi_res_log_stft_mag"]) / len(
                valid_losses_this_epoch["multi_res_log_stft_mag"]))
            valid_losses["generator_total"].append(
                sum(valid_losses_this_epoch["generator_total"]) / len(valid_losses_this_epoch["generator_total"]))
            valid_losses["discriminator_mse"].append(
                sum(valid_losses_this_epoch["discriminator_mse"]) / len(valid_losses_this_epoch["discriminator_mse"]))
            with open(os.path.join(model_save_dir, "train_loss.json"), 'w') as plotting_data_file:
                json.dump(train_losses, plotting_data_file)
            with open(os.path.join(model_save_dir, "valid_loss.json"), 'w') as plotting_data_file:
                json.dump(valid_losses, plotting_data_file)