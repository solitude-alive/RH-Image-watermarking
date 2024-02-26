import torch
import torch.nn as nn
from model.conv import ConvBNTanh
from model.networks import UnetGenerator, ResnetGenerator, ColorAttentionMechanism


class Encoder(nn.Module):
    """
    Inserts a watermark into an image.
    """

    def __init__(self, container_channels: int, encoded_channels: int, generator_name: str,
                 use_c_att: bool, use_up: bool):
        super(Encoder, self).__init__()

        if generator_name == "unet":
            self.model = UnetGenerator(input_nc=container_channels, output_nc=container_channels,
                                       num_downs=8, use_dropout=True, use_up=use_up)
        elif generator_name == "resnet":
            self.model = ResnetGenerator(input_nc=container_channels, output_nc=container_channels,
                                         n_blocks=8, use_dropout=True)
        else:
            raise NotImplementedError('generator_name [%s] is not implemented' % generator_name)

        self.final_tanh_layer = ConvBNTanh(channels_in=container_channels, channels_out=encoded_channels)

    def forward(self, cover_image, secret_image):
        container_img = torch.cat([cover_image, secret_image], dim=1)
        img = self.model(container_img)
        encoded_img = self.final_tanh_layer(img)
        return encoded_img


# import numpy as np
#
# def get_sec_img(img_tensor):
#     message = torch.Tensor(np.random.choice([0, 1], (img_tensor.shape[0], 64)))
#     mess_sqrt = torch.sqrt(torch.tensor(64, dtype=torch.float32))
#     repeat_row = int(256 / mess_sqrt)
#     repeat_col = int(256 / mess_sqrt)
#     assert 256 % mess_sqrt.item() == 0
#     sec_img = (message_to_image_square(message, row_repeat=repeat_row, rows=256,
#                                        column_repeat=repeat_col, columns=256)).to("cuda")
#     return sec_img
#
# def message_to_image_square(messages, row_repeat=2, rows=6, column_repeat=2, columns=6):
#     im = []
#     for batch_me in messages:
#         batch_im = []
#         batch_me_list = batch_me.tolist()
#         for row in range(int(rows / row_repeat)):
#             batch_im_row = []
#             for col in range(int(columns / column_repeat)):
#                 me = batch_me_list.pop(0)
#                 if me == 0:
#                     me = me - 1
#                 new = [me] * column_repeat * row_repeat
#                 new = np.array(new)
#                 new = new.reshape((row_repeat, column_repeat))
#                 batch_im_row.append(new)
#             batch_im_row = np.concatenate(batch_im_row, axis=1)
#             batch_im.append(batch_im_row)
#         batch_im = np.concatenate(batch_im, axis=0)
#         im.append(batch_im)
#
#     im = torch.tensor(np.array(im), dtype=torch.float32)
#     im = torch.unsqueeze(im, dim=1)
#     return im
