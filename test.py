import os
import torch
import cv2
import numpy as np

from model.test_model import TestModel
from utils import logger, utils


def img_to_tensor(img):
    img = torch.from_numpy(img).float()
    img = img.permute(2, 0, 1)
    img = img.unsqueeze(0)
    img = (img / 255) * 2 - 1
    return img


def sec_to_tensor(mess):
    mess = torch.tensor(mess)
    mess_sqrt = torch.sqrt(torch.tensor(mess.shape[1], dtype=torch.float32))
    repeat_row = int(256 / mess_sqrt)
    repeat_col = int(256 / mess_sqrt)
    sec_img = (utils.message_to_image_square(mess, row_repeat=repeat_row, rows=256,
                                             column_repeat=repeat_col, columns=256))
    return sec_img


def save_single_image(img, path, name):
    img = img.squeeze(0)
    img = img.detach().cpu().numpy()
    img = (img + 1) / 2 * 255
    img = img.astype(np.uint8)
    img = img.transpose(1, 2, 0)
    cv2.imwrite(os.path.join(path, name), img)


if __name__ == "__main__":
    # set random seed
    torch.manual_seed(0)
    np.random.seed(0)

    # Set logger
    logger.configure("./runs/test", format_strs=["stdout", "log"])

    # Set path
    # prj_dir = "runs/noise 2023.04.25--07-20-39"   # target file name is "1"
    # model_name = "noise--epoch-300.pyt"
    # prj_dir = "runs/noise 2024.01.25--12-46-07"   # target file name is "2"
    prj_dir = "runs/noise 2024.01.26--14-19-14"     # target file name is "3"
    model_name = "noise--epoch-100.pyt"
    config_file = os.path.join(prj_dir, "options-and-config.pickle")
    # Load config file
    train_options, hidden_config, noise_config = utils.load_options(config_file)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.log(device)

    # Load model
    model = TestModel(hidden_config, device)
    # Load model parameters
    model.load_model(prj_dir, model_name)

    # load image
    image_path = "data/000000000408.jpg"
    img_ori_cv = cv2.resize(cv2.imread(image_path), (256, 256))
    img_ori = img_to_tensor(img_ori_cv).to(device)
    # set secret image
    sec = np.random.choice([0, 1], size=(1, 64))
    img_sec = sec_to_tensor(sec).to(device)
    # Encode
    img_enc = model.encode(img_ori, img_sec)
    # Decode
    img_dec = model.decode(img_enc)

    # Save image
    save_single_image(img_ori, "./data", "3_ori.jpg")
    save_single_image(img_sec, "./data", "3_sec.jpg")
    save_single_image(img_enc, "./data", "3_enc.jpg")
    save_single_image(img_dec, "./data", "3_dec.jpg")
