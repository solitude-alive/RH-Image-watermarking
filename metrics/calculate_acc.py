import os
import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import re
import time
import json
import glob
import torchgeometry
import random


class ConvToBit(nn.Module):
    """
    input is an image with ndarray, size = (height, width)
    output is a secret bits with ndarray, size = (len,)
    """
    def __init__(self, kernel_size=8, stride=8):

        super(ConvToBit, self).__init__()
        self.conv = nn.Conv2d(1, 1, (kernel_size, kernel_size), stride=stride, padding=0, bias=False)
        self.conv.weight.data = torch.ones((1, 1, kernel_size, kernel_size))

    def forward(self, image, kernel_size=8, flag=False):
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        image = self.conv(image)

        if flag:
            threshold = 0   # input image value max = 1
        else:
            threshold = 255 * kernel_size * kernel_size / 2 # input image value max = 255

        bit_np = image.squeeze().detach().numpy()
        bit_np = np.where(bit_np > threshold, 1, 0)
        bit_np.resize((bit_np.size,))

        return bit_np


# Reference: https://docs.opencv.org/4.x/d2/d96/tutorial_py_table_of_contents_imgproc.html

def image_threshold(image, method="global"):
    if method == "global":
        ret, res = cv.threshold(image, 127, 255, cv.THRESH_BINARY)
    elif method == "adaptive mean":
        res = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, blockSize=255, C=0)
    elif method == "adaptive gaussian":
        res = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, blockSize=255, C=2)
    elif method == "otsu":
        ret, res = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    else:
        print("method error!")
    return res


def image_save(image, path):
    cv.imwrite(image, path)


def csv_save(dic, csv_folder, file_name, columns_name):
    df = pd.DataFrame.from_dict(dic, orient="index")
    df.columns = columns_name
    df.to_csv(os.path.join(csv_folder, file_name))
    print("save csv to {}/{}".format(str(csv_folder), file_name))


def avg_csv_save(dic, csv_folder, file_name, row_name):
    df = pd.DataFrame.from_dict(dic, orient="columns")
    df.index = row_name
    df.to_csv(os.path.join(csv_folder, file_name))
    print("save csv to {}/{}".format(str(csv_folder), file_name))


def read_secret_csv(csv_folder, file_name):
    secret_img = pd.read_csv(os.path.join(csv_folder, file_name), index_col=0, dtype={'key': str}).to_dict(orient="index")
    dic = {}
    for key in secret_img:
        dic[key] = np.array(list(secret_img[key].values()))
    return dic


def get_numbers_from_filename(filename):
    return re.search(r'\d+', filename).group(0)


def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

#
# file_path = "images"
#
# cover_path = os.path.join(file_path, "cover")
# secret_path = os.path.join(file_path, "secret")
# encoded_path = os.path.join(file_path, "encoded")
# decoded_path = os.path.join(file_path, "decoded")
# csv_file = os.path.join(file_path, "csv")
#
# secret_origin = os.path.join(secret_path)
# secret_test = os.path.join(secret_path, "test")
#
# make_dir(cover_path)
# make_dir(decoded_path)
#
# kernel = int(256 / 8)
# image_to_bit = ConvToBit(kernel_size=kernel, stride=kernel)
#
# secret_dict = read_secret_csv("secret.csv")
# test_folder = "images/decoded"


def cal_acc(decoded_folder, csv_path, kernel=32):
    csv_index_name = []
    result_dict = {}
    acc_list = []

    image_to_bit = ConvToBit(kernel_size=kernel, stride=kernel)
    secret_dict = read_secret_csv(csv_path, "secret.csv")

    # print(secret_dict)

    for folder in os.listdir(decoded_folder):
        print("folder name: {}".format(folder))
        start_time = time.time()
        img_folder = os.path.join(decoded_folder, folder)

        csv_index_name.append(folder)
        acc = 0

        for i in os.listdir(img_folder):
            img_de_path = os.path.join(img_folder, i)
            img = cv.imread(img_de_path)  # img: ndarray

            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # gray.size = (height, width)
            gray_binary = image_threshold(gray, "adaptive mean")

            bit = image_to_bit(gray_binary, kernel_size=kernel)

            secret_name = i.split('.')[0]
            secret_gt = secret_dict[secret_name]


            img_acc = sum((bit == secret_gt).astype(int)) / len(secret_gt)

            if secret_name in result_dict.keys():
                result_dict[secret_name].append(img_acc)
            else:
                result_dict[secret_name] = [img_acc]

            acc += img_acc

        acc_list.append(acc / 1000)

        print("----- Duration {:.2f} sec -----".format(time.time() - start_time))

    result_dict["average"] = acc_list

    avg_dict = {"average": result_dict["average"]}

    avg_csv_save(avg_dict, csv_path, "average.csv", csv_index_name)

    csv_save(result_dict, csv_path, "result.csv", csv_index_name)








