import os
import time
import torch
import numpy as np
import utils
from collections import defaultdict

from options import *
from model.hidden import Hidden
from utils.metrics import AverageMeter
from utils.utils import message_to_image_square
from utils import logger
from utils import dataset


class Trainer:
    def __init__(self, model, hidden_config, train_options, out_folder, device):
        """
        Initialize the trainer

        Parameter:
            model: HiDDeN model
            hidden_config (NetConfiguration): The configuration of the model
            train_options (TrainingOptions): The training options
            out_folder (str): The folder to save the results
            device (torch.device): The device to run the training on
        """
        self.model = model
        self.hidden_config = hidden_config
        self.train_options = train_options
        self.out_folder = out_folder
        self.device = device

        self.train_data, self.val_data = dataset.get_data_loaders(hidden_config, train_options)

        file_count = len(self.train_data.dataset)
        if file_count % train_options.batch_size == 0:
            self.steps_in_epoch = file_count // train_options.batch_size
        else:
            self.steps_in_epoch = file_count // train_options.batch_size + 1

    def train(self):
        print_each = 100
        images_to_save = 8
        saved_images_size = (self.hidden_config.H, self.hidden_config.W)
        loss_best = 1000

        for epoch in range(self.train_options.start_epoch, self.train_options.number_of_epochs + 1):
            logger.info('\nStarting epoch {}/{}'.format(epoch, self.train_options.number_of_epochs))
            logger.info('Batch size = {}\nSteps in epoch = {}'.format(self.train_options.batch_size,
                                                                      self.steps_in_epoch))
            training_losses = defaultdict(AverageMeter)
            loss_train = 0
            loss_val = 0
            epoch_start = time.time()
            step = 1
            # step_noise = cal_step_noise(epoch)
            step_noise = min(epoch, 50)
            logger.log(f"Noise Step is {step_noise}")

            for image, _ in self.train_data:
                image = image.to(self.device)
                message = torch.Tensor(
                    np.random.choice([0, 1], (image.shape[0], self.hidden_config.message_length))).to(self.device)
                secret_image = self._set_sec_img(message)

                losses, _ = self.model.train([image, secret_image], step_noise)

                for name, loss in losses.items():
                    training_losses[name].update(loss)
                    if name[:4] == 'loss':
                        loss_train += loss
                if step % print_each == 0 or step == self.steps_in_epoch:
                    logger.log(
                        'Epoch: {}/{} Step: {}/{}'.format(epoch, self.train_options.number_of_epochs, step,
                                                          self.steps_in_epoch))
                    logger.log(training_losses)
                    logger.log('-' * 20)

                step += 1

            loss_train /= len(self.train_data)
            train_duration = time.time() - epoch_start
            logger.log('Epoch {} training duration {:.2f} sec'.format(epoch, train_duration))
            logger.log('loss {}'.format(loss_train))
            logger.log('-' * 40)
            utils.utils.write_losses(os.path.join(self.out_folder, 'train.csv'), training_losses, epoch, train_duration)

            # Validation
            first_iteration = True
            validation_losses = defaultdict(AverageMeter)
            logger.log('Running validation for epoch {}/{}'.format(epoch, self.train_options.number_of_epochs))
            for image, _ in self.val_data:
                image = image.to(self.device)
                message = torch.Tensor(
                    np.random.choice([0, 1], (image.shape[0], self.hidden_config.message_length))).to(self.device)
                secret_image = self._set_sec_img(message)

                losses, (encoded_images, noised_images, decoded_images, decoded_cover) = \
                    self.model.val([image, secret_image], step_noise)

                residual_image = encoded_images - image

                for name, loss in losses.items():
                    validation_losses[name].update(loss)
                    if name[:4] == 'loss':
                        loss_val += loss
                if first_iteration:
                    if self.hidden_config.enable_fp16:
                        image = image.float()
                        encoded_images = encoded_images.float()
                    utils.utils.save_images(image.cpu()[:images_to_save, :, :, :],
                                            encoded_images[:images_to_save, :, :, :].cpu(),
                                            epoch,
                                            os.path.join(self.out_folder, 'images'), resize_to=saved_images_size)
                    utils.utils.save_noised_images(encoded_images.cpu()[:images_to_save, :, :, :],
                                                   noised_images[:images_to_save, :, :, :].cpu(),
                                                   epoch,
                                                   os.path.join(self.out_folder, 'images'), resize_to=saved_images_size)
                    utils.utils.save_secret_images(secret_image.cpu()[:images_to_save, :, :, :],
                                                   decoded_images[:images_to_save, :, :, :].cpu(),
                                                   epoch,
                                                   os.path.join(self.out_folder, 'images'), resize_to=saved_images_size)
                    utils.utils.save_cover_images(image.cpu()[:images_to_save, :, :, :],
                                                  decoded_cover[:images_to_save, :, :, :].cpu(),
                                                  epoch,
                                                  os.path.join(self.out_folder, 'images'), resize_to=saved_images_size)
                    utils.utils.save_residual_images(image.cpu()[:images_to_save, :, :, :],
                                                     residual_image[:images_to_save, :, :, :].cpu(),
                                                     epoch,
                                                     os.path.join(self.out_folder, 'images'),
                                                     resize_to=saved_images_size)
                    first_iteration = False

            logger.log(validation_losses)
            logger.log('-' * 40)
            loss_val /= len(self.val_data)
            val_duration = time.time() - epoch_start
            logger.log('Epoch {} training duration {:.2f} sec'.format(epoch, val_duration))
            logger.log('loss {}'.format(loss_val))
            logger.log('-' * 40)
            if loss_val < loss_best:
                utils.utils.save_checkpoint(self.model, self.train_options.experiment_name, epoch,
                                            os.path.join(self.out_folder, 'checkpoints'))
                loss_best = loss_val
            elif epoch % 20 == 0:
                utils.utils.save_checkpoint(self.model, self.train_options.experiment_name, epoch,
                                            os.path.join(self.out_folder, 'checkpoints'))
            utils.utils.write_losses(os.path.join(self.out_folder, 'validation.csv'), validation_losses, epoch,
                                     time.time() - epoch_start)

    def _train(self):
        pass

    def _val(self):
        pass

    def _set_sec_img(self, mess):
        mess_sqrt = torch.sqrt(torch.tensor(self.hidden_config.message_length, dtype=torch.float32))
        repeat_row = int(self.hidden_config.H / mess_sqrt)
        repeat_col = int(self.hidden_config.W / mess_sqrt)
        # assert self.hidden_config.H % mess_sqrt.item() == 0
        sec_img = (message_to_image_square(mess, row_repeat=repeat_row, rows=self.hidden_config.H,
                                           column_repeat=repeat_col, columns=self.hidden_config.W)).to(self.device)
        return sec_img


def cal_step_noise(epoch):
    return min(epoch, 50)
