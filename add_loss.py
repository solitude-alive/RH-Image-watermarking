import torch
import torch.nn as nn
import torchvision
from lpips_pytorch import LPIPS, lpips


class Res50(nn.Module):
    """

    """

    def __init__(self):
        super(Res50, self).__init__()
        res = torchvision.models.resnet50(pretrained=True)

    def forward(self, img):
        return 1


class VGGLoss(nn.Module):
    """
    Part of pre-trained VGG16. This is used in case we want perceptual loss instead of Mean Square Error loss.
    See for instance https://arxiv.org/abs/1603.08155
    """

    def __init__(self, block_no: int, layer_within_block: int, use_batch_norm_vgg: bool):
        super(VGGLoss, self).__init__()
        if use_batch_norm_vgg:
            vgg16 = torchvision.models.vgg16_bn(pretrained=True)
        else:
            vgg16 = torchvision.models.vgg16(pretrained=True)
        curr_block = 1
        curr_layer = 1
        layers = []
        for layer in vgg16.features.children():
            layers.append(layer)
            if curr_block == block_no and curr_layer == layer_within_block:
                break
            if isinstance(layer, nn.MaxPool2d):
                curr_block += 1
                curr_layer = 1
            else:
                curr_layer += 1

        self.vgg_loss = nn.Sequential(*layers)

    def forward(self, img):
        return self.vgg_loss(img)


class LPIPLoss(nn.Module):
    """
    LPIPS loss
    """

    def __init__(self):
        super(LPIPLoss, self).__init__()
        self.criterion = LPIPS(
            net_type='alex',  # choose a network type from ['alex', 'squeeze', 'vgg']
            version='0.1'  # Currently, v0.1 is supported
        )

    def forward(self, x, y):
        return self.criterion(x, y)

