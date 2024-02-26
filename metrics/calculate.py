import os
import torch
import time
from metrics.eval import make_dirs, encode_image, load_model, decode_image
from metrics.attack import attack
from metrics.calculate_PSNR import cal_psnr
from metrics.calculate_acc import cal_acc

os.environ['CUDA_VISIBLE_DEVICES'] = "5"  # 指定第1块GPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
this_run_folder = "/lab/zgyang/watermarking/Tencent/watermarking-2/runs/noise 2023.02.28--21-11-42"
out_folder = "result_2-02.28-new"

start_time = time.time()

for i in ["ImageNet1K", "COCO1K"]:
    print("test folder: {} begin:".format(i))
    img_folder = os.path.join("/lab/zgyang/dataset/test1K/", i, "test")

    result_folder = os.path.join(out_folder, i)
    path_dic = make_dirs(result_folder)

    start_time_1 = time.time()

    # load model
    model, config = load_model(device, this_run_folder)

    # encode image and save
    encode_image(device, model, config, path_dic, img_folder)

    # attack image and save
    attack(path_dic['encoded_ori_path'], path_dic['encoded_path'], path_dic['original_path'])

    # decode image and save
    begin_time = time.time()

    print("------------------ Decode Start ------------------")

    for method in os.listdir(path_dic['encoded_path']):
        decoding_folder = os.path.join(path_dic['encoded_path'], method)
        decoded_path = os.path.join(path_dic['decoded_path'], method)

        if not os.path.exists(decoded_path):
            os.mkdir(decoded_path)
        print("folder name:{}".format(method))
        decode_image(device, model, config, output_folder=decoded_path, data_folder=decoding_folder)

    print("------------------ Decode complete! Duration {:.2f} sec ------------------".format(time.time() - begin_time))

    # calculate metrics
    print("------------------ Calculate Metrics Start ------------------")

    print("Calculate PSNR")
    begin_time = time.time()

    cal_psnr(path_dic['original_path'], path_dic['encoded_ori_path'], path_dic['csv_path'])

    print("Duration {:.2f} sec".format(time.time() - begin_time))

    print("Calculate Acc")
    begin_time = time.time()

    cal_acc(path_dic['decoded_path'], path_dic['csv_path'], kernel=int(256 / 8))

    print("Duration {:.2f} sec".format(time.time() - begin_time))

    print('{} Complete!!! Duration {:.2f} sec'.format(i, (time.time() - start_time_1)))

print('ALL Complete!!! Duration {:.2f} sec'.format(time.time() - start_time))


