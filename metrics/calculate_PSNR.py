import os
import cv2 as cv
import numpy as np
import pandas as pd


def calculate_psnr(original_image, target_image):
    """
    input is ndarray, size = (height, width, 3)
    """
    mse = np.mean((original_image - target_image) ** 2)
    image_psnr = 10 * np.log10(255 ** 2 / mse)
    return image_psnr


def calculate_psnr_ch(original_image, target_image):
    """
    input is ndarray, size = (height, width, 3)
    """
    image_psnr =[]
    for c in range(3):
        mse = np.mean((original_image[:, :, c] - target_image[:, :, c]) ** 2)
        image_psnr_c = 10 * np.log10(255 ** 2 / mse)
        image_psnr.append(image_psnr_c)
    return image_psnr


def csv_save(dic, csv_folder, file_name, columns_name):
    df = pd.DataFrame.from_dict(dic, orient="index")
    df.columns = columns_name
    df.to_csv(os.path.join(csv_folder, file_name))
    print("save csv to {}/{}".format(str(csv_folder), file_name))


def cal_psnr(original_encoded_path, encoded_path, csv_path):
    original_dict = {}
    psnr_dict = {}
    sum_psnr = [0, 0, 0]

    for i in os.listdir(original_encoded_path):
        sct_path = os.path.join(original_encoded_path, i)
        sct = cv.imread(sct_path)  # sct: ndarray

        original_dict[i[:-4]] = sct

    for i in os.listdir(encoded_path):
        img_path = os.path.join(encoded_path, i)
        img = cv.imread(img_path)

        gt_name = i[:-4]

        img_psnr = cv.PSNR(original_dict[gt_name], img)
        img_psnr_2 = calculate_psnr(original_dict[gt_name], img)
        img_psnr_list = calculate_psnr_ch(original_dict[gt_name], img)

        sum_psnr[0] += img_psnr
        sum_psnr[1] += img_psnr_2
        sum_psnr[2] += sum(img_psnr_list) / 3

        psnr_dict[i] = [img_psnr, img_psnr_2, img_psnr_list]

    psnr_dict["average"] = np.array(sum_psnr) / 1000

    columns_name = ["opencv", "calculate", "calculate channels"]
    csv_save(psnr_dict, csv_path, "psnr.csv", columns_name)


