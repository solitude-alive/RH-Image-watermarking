import os
import time
import torch
import numpy as np
import utils
import logging
import torchvision
import torch.nn.functional as F

from options import *
from utils.utils import message_to_image_square
from utils import utils
from model.hidden import Hidden

from torchvision import datasets, transforms
from PIL import Image

import pandas as pd


def get_test_image(hidden_config: NetConfiguration, image):
    """ Get torch data loaders for training and validation. The data loaders take a crop of the image,
    transform it into tensor, and normalize it."""
    data_transforms = transforms.Compose([
        transforms.Resize((hidden_config.H, hidden_config.W)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    image_tensor = data_transforms(image)
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor


def save_single_image(image, folder, name):
    # scale values to range [0, 1] from original range of [-1, 1]
    image = (image + 1) / 2

    filename = os.path.join(folder, name.split('.')[0] + '.png')
    torchvision.utils.save_image(image, filename)


def encode(model: Hidden,
           device: torch.device,
           hidden_config: NetConfiguration,
           img,
           name: str,
           encoded_path: str,
           original_path: str,
           secret_path: str,
           ):
    """
    Evaluates the HiDDeN model
    :param model: The model
    :param device: torch.device object, usually this is GPU (if avaliable), otherwise CPU.
    :param hidden_config: The network configuration
    :return:
    """

    test_tensor = get_test_image(hidden_config, img)

    image = test_tensor.to(device)
    message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], hidden_config.message_length))).to(device)

    mess_sqrt = torch.sqrt(torch.tensor(hidden_config.message_length, dtype=torch.float32))
    repeat_row = int(hidden_config.H / mess_sqrt)
    repeat_col = int(hidden_config.W / mess_sqrt)

    secret_images = (message_to_image_square(message, row_repeat=repeat_row, rows=hidden_config.H,
                                             column_repeat=repeat_col, columns=hidden_config.W)).to(device)

    encoded_images = model.encoder_decoder.encoder(image, secret_images)

    save_single_image(image, original_path, name)
    save_single_image(encoded_images, encoded_path, name)
    save_single_image(secret_images, secret_path, name)

    return message


def decode(model: Hidden,
           device: torch.device,
           hidden_config: NetConfiguration,
           img,
           name: str,
           secret_path: str,
           ):
    """
    Evaluates the HiDDeN model
    :param model: The model
    :param device: torch.device object, usually this is GPU (if avaliable), otherwise CPU.
    :param hidden_config: The network configuration
    :return:
    """

    test_tensor = get_test_image(hidden_config, img)
    image = test_tensor.to(device)

    decoded_images = model.encoder_decoder.decoder(image)
    save_single_image(decoded_images, secret_path, name)


def load_model(device, run_folder, model_name=None):
    if model_name is None:
        model_name = "noise--epoch-100.pyt"
    options_file = os.path.join(run_folder, "options-and-config.pickle")
    train_options, hidden_config, noise_config = utils.load_options(options_file)
    loaded_checkpoint_file_name = os.path.join(run_folder, "checkpoints", model_name)
    checkpoint = torch.load(loaded_checkpoint_file_name)

    model = Hidden(hidden_config, device)

    # if we are continuing, we have to load the model params
    assert checkpoint is not None
    logging.info(f'Loading checkpoint from file {loaded_checkpoint_file_name}')
    utils.model_from_checkpoint(model, checkpoint)

    print("Load checkpoint successful!")

    return model, hidden_config


# csv function
def csv_save(dic, csv_folder, file_name, columns_name):
    df = pd.DataFrame.from_dict(dic, orient="index")
    df.columns = columns_name
    df.to_csv(os.path.join(csv_folder, file_name))
    print("save csv to {}/{}".format(str(csv_folder), file_name))


def secret_save(secret_dict, csv_folder, file_name):
    df = pd.DataFrame.from_dict(secret_dict, orient="index")
    df.to_csv(os.path.join(csv_folder, file_name), index_label='key')


# make dir
def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


# encode image
def encode_image(device, model, config, path_dict, dataset_folder):
    model.encoder_decoder.eval()

    with torch.no_grad():
        start_time = time.time()

        print("------------------ Encode Start ------------------")

        secret_dic = {}

        for i in os.listdir(dataset_folder):
            img = Image.open(os.path.join(dataset_folder, i)).convert('RGB')
            mess = encode(model, device, config, img, i,
                          path_dict['encoded_ori_path'], path_dict['original_path'], path_dict['secret_path'])

            secret_dic[i.split('.')[0]] = mess.cpu().numpy().reshape(-1).astype(np.int64)

        secret_save(secret_dic, path_dict['csv_path'], "secret.csv")
        print("secret.csv save to {}/{}!".format(str(path_dict['csv_path']), "secret.csv"))

        print('Encode complete! Duration {:.2f} sec'.format(time.time() - start_time))


# decode image
def decode_image(device, model, config, output_folder, data_folder):
    model.encoder_decoder.eval()

    with torch.no_grad():
        start_time = time.time()

        # print("------------------ Decode Start ------------------")

        make_dir(output_folder)

        for i in os.listdir(data_folder):
            img = Image.open(os.path.join(data_folder, i))
            decode(model, device, config, img, i, output_folder)

        print('Decode complete! Duration {:.2f} sec'.format(time.time() - start_time))


def make_dirs(output_folder):
    encoded_path = os.path.join(output_folder, "encoded")
    encoded_ori_path = os.path.join(encoded_path, "original")
    original_path = os.path.join(output_folder, "original")
    secret_path = os.path.join(output_folder, "secret")

    csv_path = os.path.join(output_folder, "csv")

    decode_path = os.path.join(output_folder, "decoded")

    make_dir(output_folder)
    make_dir(encoded_path)
    make_dir(encoded_ori_path)
    make_dir(original_path)
    make_dir(secret_path)
    make_dir(csv_path)
    make_dir(decode_path)

    dic = {
        'encoded_path': encoded_path,
        'encoded_ori_path': encoded_ori_path,
        'original_path': original_path,
        'secret_path': secret_path,
        'csv_path': csv_path,
        'decoded_path': decode_path
    }

    return dic

# os.environ['CUDA_VISIBLE_DEVICES'] = "5"  # 指定第1块GPU
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# this_run_folder = "/lab/zgyang/watermarking/Tencent/watermarking-2/runs/noise 2023.02.28--21-11-42"
# result_folder = "result_2-02.28-new"
# img_folder = "/lab/zgyang/dataset/test1K/ImageNet1K/test"
# path_dic = make_dirs(result_folder)

# model, config = load_model(device, this_run_folder)
# encode_image(device, model, config, path_dic, img_folder)
# decode_image(device, model, config, path_dic['decode_path'], img_folder)
