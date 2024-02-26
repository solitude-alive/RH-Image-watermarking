import os
import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import re
import time
import random


# Reference: https://docs.opencv.org/4.x/d2/d96/tutorial_py_table_of_contents_imgproc.html

def image_perspective(image, input_fac, output_fac, image_size):
    height, width = image.shape[:2]

    # pts1 = np.float32([[0, 0], [256, 0], [0, 256], [256, 256]])  # Pixels on the target image
    # pts2 = np.float32([[0, 0], [226, 20], [30, 226], [256, 256]])  # Pixels on the output image
    pts1 = input_fac
    pts2 = output_fac
    m = cv.getPerspectiveTransform(pts1, pts2)
    res = cv.warpPerspective(image, m, (image_size, image_size))
    return res


def image_gaussian(image, kernel_size=5):
    res = cv.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return res


def image_brightness_contrast(image, alpha=2.5, beta=50):
    new = alpha * image.astype(float) + beta
    res = np.clip(new, 0, 255)
    return res


def get_crop_rectangle(size, ratio):
    """
    Returns a random rectangle inside the image, where the size is random and is controlled by height_ratio_range and width_ratio_range.
    This is analogous to a random crop. For example, if height_ratio_range is (0.7, 0.9), then a random number in that range will be chosen
    (say it is 0.75 for illustration), and the image will be cropped such that the remaining height equals 0.75. In fact,
    a random 'starting' position rs will be chosen from (0, 0.25), and the crop will start at rs and end at rs + 0.75. This ensures
    that we crop from top/bottom with equal probability.
    The same logic applies to the width of the image, where width_ratio_range controls the width crop range.
    :param size: The original image size
    :param ratio: The range of remaining height ratio
    :return: "Cropped" rectange with width and height drawn randomly factor
    """
    image_height = size[0]
    image_width = size[1]

    remaining_height = int(np.rint(ratio * image_height))
    remaining_width = int(np.rint(ratio * image_width))

    if remaining_height == image_height:
        height_start = 0
    else:
        height_start = np.random.randint(0, image_height - remaining_height)

    if remaining_width == image_width:
        width_start = 0
    else:
        width_start = np.random.randint(0, image_width - remaining_width)

    return height_start, height_start + remaining_height, width_start, width_start + remaining_width


def image_cropout_black(image, factor=0.8):
    image_size = (image.shape[0], image.shape[1])
    h_start, h_end, w_start, w_end = get_crop_rectangle(size=image_size, ratio=factor)

    cropout_mask = np.zeros_like(image)
    cropout_mask[h_start: h_end, w_start: w_end, :] = 1

    image_crop = image * cropout_mask
    return image_crop


def image_cropout_original(image, original_image, factor=0.8):
    image_size = (image.shape[0], image.shape[1])
    h_start, h_end, w_start, w_end = get_crop_rectangle(size=image_size, ratio=factor)

    cropout_mask = np.zeros_like(image)
    cropout_mask[h_start: h_end, w_start: w_end, :] = 1

    original_mask = 1 - cropout_mask

    image_crop = image * cropout_mask + original_image * original_mask
    return image_crop


def image_save(image, path):
    cv.imwrite(image, path)


def get_numbers_from_filename(filename):
    return re.search(r'\d+', filename).group(0)


def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def make_attack_folder(attack_path):
    perspective_path = os.path.join(attack_path, "perspective")
    gaussian_path = os.path.join(attack_path, "gaussian")
    jpeg_path = os.path.join(attack_path, "jpeg")
    brightness_contrast_path = os.path.join(attack_path, "c_b")
    cropout_black_path = os.path.join(attack_path, "cropout_black")
    cropout_original_path = os.path.join(attack_path, "cropout_original")

    dic = {
        'perspective_path': perspective_path,
        'gaussian_path': gaussian_path,
        'jpeg_path': jpeg_path,
        'brightness_contrast_path': brightness_contrast_path,
        'cropout_black_path': cropout_black_path,
        'cropout_original_path': cropout_original_path
    }

    return dic


# Attack for test
def cropout_black(folder_name, img, name):
    # cropout with black padding
    for f in [0.8, 0.9, 0.95]:
        cropout_black_f_path = folder_name + '_' + str(f)
        make_dir(cropout_black_f_path)
        img_cropout_black = image_cropout_black(img, f)
        cropout_black_file = os.path.join(cropout_black_f_path, name)
        image_save(cropout_black_file, img_cropout_black)


def cropout_original(folder_name, img, name, ori_img):
    # cropout with origianl padding
    ori_name = name
    ori_path = os.path.join(ori_img, ori_name)
    img_ori = cv.imread(ori_path)
    for f in [0.8, 0.9, 0.95]:
        cropout_original_f_path = folder_name + '_' + str(f)
        make_dir(cropout_original_f_path)
        img_cropout_original = image_cropout_original(img, img_ori, f)
        cropout_original_file = os.path.join(cropout_original_f_path, name)
        image_save(cropout_original_file, img_cropout_original)


def gaussian_noise(folder_name, img, name):
    # GaussianBlur
    for k in [5, 9]:
        gaussian_k_path = folder_name + str(k)
        make_dir(gaussian_k_path)
        img_gaussian = image_gaussian(img, k)
        gaussian_file = os.path.join(gaussian_k_path, name)
        image_save(gaussian_file, img_gaussian)


def jpeg_compression(folder_name, img, name):
    # Jpeg Compression
    for quality in range(55, 101, 10):
        jpeg_q_path = folder_name + str(quality)
        make_dir(jpeg_q_path)
        if name[-4:] == '.png':
            jpeg_name = name[:-4] + ".jpg"
        else:
            jpeg_name = name
        jpeg_file = os.path.join(jpeg_q_path, jpeg_name)
        cv.imwrite(jpeg_file, img, [int(cv.IMWRITE_JPEG_QUALITY), quality])


def perspective(folder_name, img, name):
    # Perspective Transformation
    img_size = img.shape[0]
    for d in [2, 7, 12, 20, 30]:
        tl_x = random.uniform(-d, d)  # Top left corner, top
        tl_y = random.uniform(-d, d)  # Top left corner, left
        bl_x = random.uniform(-d, d)  # Bot left corner, bot
        bl_y = random.uniform(-d, d)  # Bot left corner, left
        tr_x = random.uniform(-d, d)  # Top right corner, top
        tr_y = random.uniform(-d, d)  # Top right corner, right
        br_x = random.uniform(-d, d)  # Bot right corner, bot
        br_y = random.uniform(-d, d)  # Bot right corner, right
        rect = np.array([
            [tl_x, tl_y],
            [tr_x + img_size, tr_y],
            [br_x + img_size, br_y + img_size],
            [bl_x, bl_y + img_size]], dtype="float32")

        dst = np.array([
            [0, 0],
            [img_size, 0],
            [img_size, img_size],
            [0, img_size]], dtype="float32")

        perspective_d_path = folder_name + str(d)
        make_dir(perspective_d_path)
        img_perspective = image_perspective(img, rect, dst, img_size)
        perspective_file = os.path.join(perspective_d_path, name)
        image_save(perspective_file, img_perspective)


def con_bri(folder_name, img, name):
    # Contrast and Brightness Control
    for a in [0.5, 0.8, 1.2, 1.5]:
        for b in [-50, 50]:
            c_b_path = folder_name + "_" + str(a) + "_" + str(b)
            make_dir(c_b_path)
            img_con_bri = image_brightness_contrast(img, alpha=a, beta=b)
            con_bri_file = os.path.join(c_b_path, name)
            image_save(con_bri_file, img_con_bri)


def attack(ori_en_image_folder, attack_image_folder, ori_image_folder):
    start_time = time.time()
    print("------------------ Attack Start ------------------")

    attack_path = make_attack_folder(attack_image_folder)

    begin_time = time.time()
    for i in os.listdir(ori_en_image_folder):
        img_path = os.path.join(ori_en_image_folder, i)
        img = cv.imread(img_path)  # img: ndarray
        cropout_black(attack_path['cropout_black_path'], img, i)
    print("cropout_black complete! Duration {:.2f} sec".format(time.time() - begin_time))

    begin_time = time.time()
    for i in os.listdir(ori_en_image_folder):
        img_path = os.path.join(ori_en_image_folder, i)
        img = cv.imread(img_path)  # img: ndarray
        cropout_original(attack_path['cropout_original_path'], img, i, ori_image_folder)
    print("cropout_original complete! Duration {:.2f} sec".format(time.time() - begin_time))

    begin_time = time.time()
    for i in os.listdir(ori_en_image_folder):
        img_path = os.path.join(ori_en_image_folder, i)
        img = cv.imread(img_path)  # img: ndarray
        gaussian_noise(attack_path['gaussian_path'], img, i)
    print("gaussian_noise complete! Duration {:.2f} sec".format(time.time() - begin_time))

    begin_time = time.time()
    for i in os.listdir(ori_en_image_folder):
        img_path = os.path.join(ori_en_image_folder, i)
        img = cv.imread(img_path)  # img: ndarray
        jpeg_compression(attack_path['jpeg_path'], img, i)
    print("jpeg_compression complete! Duration {:.2f} sec".format(time.time() - begin_time))

    begin_time = time.time()
    for i in os.listdir(ori_en_image_folder):
        img_path = os.path.join(ori_en_image_folder, i)
        img = cv.imread(img_path)  # img: ndarray
        perspective(attack_path['perspective_path'], img, i)
    print("perspective complete! Duration {:.2f} sec".format(time.time() - begin_time))

    begin_time = time.time()
    for i in os.listdir(ori_en_image_folder):
        img_path = os.path.join(ori_en_image_folder, i)
        img = cv.imread(img_path)  # img: ndarray
        con_bri(attack_path['brightness_contrast_path'], img, i)
    print("brightness_contrast complete! Duration {:.2f} sec".format(time.time() - begin_time))

    print("--------------- Attack complete! Duration {:.2f} sec ---------------".format(time.time() - start_time))
