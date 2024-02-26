import torch.nn as nn
from model.conv import ConvBNTanh
from model.networks import UnetGenerator, ResnetGenerator


class Decoder(nn.Module):
    """
    Decoder module. Receives a watermarked image and extracts the watermark.
    The input image may have various kinds of noise applied to it,
    such as Crop, JpegCompression, and so on. See Noise layers for more.
    """
    def __init__(self, encoded_channels: int, secret_channels: int, generator_name: str):
        super(Decoder, self).__init__()

        if generator_name == "unet":
            self.model = UnetGenerator(input_nc=encoded_channels, output_nc=encoded_channels,
                                       num_downs=5, use_dropout=True)
        elif generator_name == "resnet":
            self.model = ResnetGenerator(input_nc=encoded_channels, output_nc=encoded_channels, use_dropout=True)
        else:
            raise NotImplementedError('generator_name [%s] is not implemented' % generator_name)

        self.final_tanh_layer = ConvBNTanh(channels_in=encoded_channels, channels_out=secret_channels)

    def forward(self, encoded_image):
        img = self.model(encoded_image)
        decoded_img = self.final_tanh_layer(img)
        return decoded_img
