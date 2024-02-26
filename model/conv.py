import torch.nn as nn


class ConvBNRelu(nn.Module):
    """
    Building block used in network. Is a sequence of Convolution, Batch Normalization, and ReLU activation
    """

    def __init__(self, channels_in, channels_out, stride=1):
        super(ConvBNRelu, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)


class ConvBNTanh(nn.Module):
    """
    Building block used in network. Is a sequence of Convolution, Batch Normalization, and ReLU activation
    """

    def __init__(self, channels_in, channels_out, stride=1):
        super(ConvBNTanh, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(channels_out),
            nn.Tanh()
        )

    def forward(self, x):
        return self.layers(x)
