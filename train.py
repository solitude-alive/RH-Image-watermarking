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


def train(model: Hidden,
          device: torch.device,
          hidden_config: NetConfiguration,
          train_options: TrainingOptions,
          this_run_folder: str,
          tb_logger):
    """
    Trains the HiDDeN model
    :param model: The model
    :param device: torch.device object, usually this is GPU (if available), otherwise CPU.
    :param hidden_config: The network configuration
    :param train_options: The training settings
    :param this_run_folder: The parent folder for the current training run to store training artifacts/results/logs.
    :param tb_logger: TensorBoardLogger object which is a thin wrapper for TensorboardX logger.
                Pass None to disable TensorboardX logging
    :return:
    """

    train_data, val_data = utils.utils.get_data_loaders(hidden_config, train_options)
    file_count = len(train_data.dataset)
    if file_count % train_options.batch_size == 0:
        steps_in_epoch = file_count // train_options.batch_size
    else:
        steps_in_epoch = file_count // train_options.batch_size + 1

    print_each = 100
    images_to_save = 8
    saved_images_size = (hidden_config.H, hidden_config.W)

    loss_best = 1000

    for epoch in range(train_options.start_epoch, train_options.number_of_epochs + 1):
        logger.log('\nStarting epoch {}/{}'.format(epoch, train_options.number_of_epochs))
        logger.log('Batch size = {}\nSteps in epoch = {}'.format(train_options.batch_size, steps_in_epoch))
        training_losses = defaultdict(AverageMeter)
        loss_train = 0
        loss_val = 0
        epoch_start = time.time()
        step = 1
        for image, _ in train_data:
            image = image.to(device)
            message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], hidden_config.message_length))).to(device)

            mess_sqrt = torch.sqrt(torch.tensor(hidden_config.message_length, dtype=torch.float32))
            repeat_row = int(hidden_config.H / mess_sqrt)
            repeat_col = int(hidden_config.W / mess_sqrt)
            assert hidden_config.H % mess_sqrt.item() == 0
            secret_images = (message_to_image_square(message, row_repeat=repeat_row, rows=hidden_config.H,
                                                     column_repeat=repeat_col, columns=hidden_config.W)).to(device)

            losses, _ = model.train_on_batch([image, secret_images], epoch)

            for name, loss in losses.items():
                training_losses[name].update(loss)
                if name[:4] == 'loss':
                    loss_train += loss
            if step % print_each == 0 or step == steps_in_epoch:
                logger.log(
                    'Epoch: {}/{} Step: {}/{}'.format(epoch, train_options.number_of_epochs, step, steps_in_epoch))
                utils.utils.log_progress(training_losses)
                logger.log('-' * 20)

            step += 1

        loss_train /= len(train_data)
        train_duration = time.time() - epoch_start
        logger.log('Epoch {} training duration {:.2f} sec'.format(epoch, train_duration))
        logger.log('loss {}'.format(loss_train))
        logger.log('-' * 40)
        utils.utils.write_losses(os.path.join(this_run_folder, 'train.csv'), training_losses, epoch, train_duration)
        if tb_logger is not None:
            tb_logger.save_losses(training_losses, epoch)
            tb_logger.save_grads(epoch)
            tb_logger.save_tensors(epoch)

        first_iteration = True
        validation_losses = defaultdict(AverageMeter)
        logger.log('Running validation for epoch {}/{}'.format(epoch, train_options.number_of_epochs))
        for image, _ in val_data:
            image = image.to(device)
            message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], hidden_config.message_length))).to(device)

            mess_sqrt = torch.sqrt(torch.tensor(hidden_config.message_length, dtype=torch.float32))
            repeat_row = int(hidden_config.H / mess_sqrt)
            repeat_col = int(hidden_config.W / mess_sqrt)
            assert hidden_config.H % mess_sqrt.item() == 0

            secret_images = (message_to_image_square(message, row_repeat=repeat_row, rows=hidden_config.H,
                                                     column_repeat=repeat_col, columns=hidden_config.W)).to(device)

            losses, (encoded_images, noised_images, decoded_images, decoded_cover) = \
                model.validate_on_batch([image, secret_images], epoch)

            residual_image = encoded_images - image

            for name, loss in losses.items():
                validation_losses[name].update(loss)
                if name[:4] == 'loss':
                    loss_val += loss
            if first_iteration:
                if hidden_config.enable_fp16:
                    image = image.float()
                    encoded_images = encoded_images.float()
                utils.utils.save_images(image.cpu()[:images_to_save, :, :, :],
                                        encoded_images[:images_to_save, :, :, :].cpu(),
                                        epoch,
                                        os.path.join(this_run_folder, 'images'), resize_to=saved_images_size)
                utils.utils.save_noised_images(encoded_images.cpu()[:images_to_save, :, :, :],
                                               noised_images[:images_to_save, :, :, :].cpu(),
                                               epoch,
                                               os.path.join(this_run_folder, 'images'), resize_to=saved_images_size)
                utils.utils.save_secret_images(secret_images.cpu()[:images_to_save, :, :, :],
                                               decoded_images[:images_to_save, :, :, :].cpu(),
                                               epoch,
                                               os.path.join(this_run_folder, 'images'), resize_to=saved_images_size)
                utils.utils.save_cover_images(image.cpu()[:images_to_save, :, :, :],
                                              decoded_cover[:images_to_save, :, :, :].cpu(),
                                              epoch,
                                              os.path.join(this_run_folder, 'images'), resize_to=saved_images_size)
                utils.utils.save_residual_images(image.cpu()[:images_to_save, :, :, :],
                                                 residual_image[:images_to_save, :, :, :].cpu(),
                                                 epoch,
                                                 os.path.join(this_run_folder, 'images'), resize_to=saved_images_size)
                first_iteration = False

        utils.utils.log_progress(validation_losses)
        logger.log('-' * 40)
        loss_val /= len(val_data)
        val_duration = time.time() - epoch_start
        logger.log('Epoch {} training duration {:.2f} sec'.format(epoch, val_duration))
        logger.log('loss {}'.format(loss_val))
        logger.log('-' * 40)
        if loss_val < loss_best:
            utils.utils.save_checkpoint(model, train_options.experiment_name, epoch,
                                        os.path.join(this_run_folder, 'checkpoints'))
            loss_best = loss_val
        elif epoch % 20 == 0:
            utils.utils.save_checkpoint(model, train_options.experiment_name, epoch,
                                        os.path.join(this_run_folder, 'checkpoints'))
        utils.utils.write_losses(os.path.join(this_run_folder, 'validation.csv'), validation_losses, epoch,
                                 time.time() - epoch_start)
