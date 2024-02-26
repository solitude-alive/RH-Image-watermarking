import torch.nn as nn
from options import NetConfiguration
from model.conv import ConvBNTanh
from model.networks import UnetGenerator, ResnetGenerator


class Cover_decoder(nn.Module):
    """
    decode the cover image from encoded image.
    """
    def __init__(self, container_channels: int, encoded_channels: int, generator_name: str):
        super(Cover_decoder, self).__init__()
        self.conv1 = ConvBNTanh(channels_in=encoded_channels, channels_out=container_channels)

        if generator_name == "unet":
            self.model = UnetGenerator(input_nc=container_channels, output_nc=container_channels,
                                       num_downs=8, use_dropout=True)
        elif generator_name == "resnet":
            self.model = ResnetGenerator(input_nc=container_channels, output_nc=container_channels,
                                         n_blocks=8, use_dropout=True)
        else:
            raise NotImplementedError('generator_name [%s] is not implemented' % generator_name)

        self.final_layer = ConvBNTanh(channels_in=container_channels, channels_out=encoded_channels)

    def forward(self, encoded_image):
        img = self.conv1(encoded_image)
        img = self.model(img)
        decoded_cover_image = self.final_layer(img)
        return decoded_cover_image


class CNN_F(nn.Module):
    """
    Combines Encoder->Noiser->Decoder into single pipeline.
    The input is the cover image and the watermark message. The module inserts the watermark into the image
    (obtaining encoded_image), then applies Noise layers (obtaining noised_image), then passes the noised_image
    to the Decoder which tries to recover the watermark (called decoded_message). The module outputs
    a three-tuple: (encoded_image, noised_image, decoded_message)
    """
    def __init__(self, config: NetConfiguration):

        super(CNN_F, self).__init__()
        self.decode_cover_image = Cover_decoder(config.container_channels, config.encoded_channels,
                                                config.generator_name)

    def forward(self, encoded_image):
        decoded_cover_image = self.decode_cover_image(encoded_image)
        return decoded_cover_image

