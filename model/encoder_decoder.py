import torch.nn as nn
from model.encoder import Encoder
from model.decoder import Decoder
import torch.nn.functional as F
import functools
from options import NetConfiguration
from noise_layer.transform_net import transform_net


class EncoderDecoder(nn.Module):
    """
    Combines Encoder->Noiser->Decoder into single pipeline.
    The input is the cover image and the watermark message. The module inserts the watermark into the image
    (obtaining encoded_image), then applies Noise layers (obtaining noised_image), then passes the noised_image
    to the Decoder which tries to recover the watermark (called decoded_message). The module outputs
    a three-tuple: (encoded_image, noised_image, decoded_message)
    """
    def __init__(self, config: NetConfiguration):

        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(config.container_channels, config.encoded_channels, config.generator_name,
                               config.c_att, config.up)
        self.decoder = Decoder(config.encoded_channels, config.secret_channels, config.generator_name)

    def forward(self, cover_images, secret_images, epoch):
        encoded_image = self.encoder(cover_images, secret_images)

        noised_image = transform_net(encoded_image, epoch)
        noised_image = noised_image.to('cuda')

        decoded_image = self.decoder(noised_image)
        return encoded_image, noised_image, decoded_image
