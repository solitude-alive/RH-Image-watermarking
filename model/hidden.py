import numpy as np
import torch
import torch.nn as nn
import os

from options import NetConfiguration
from model.discriminator import Discriminator
from model.encoder_decoder import EncoderDecoder
from model.CNN_F import CNN_F
from model.filter import FilterLow, FilterHigh
from add_loss import LPIPLoss
from utils import utils, logger


class Hidden:
    def __init__(self, configuration: NetConfiguration, device: torch.device):
        """
        Initializes the Hidden network

        Parameters:
            configuration (NetConfiguration): Configuration for the net, Hidden_configuration
            device (torch.device): CPU or GPU
        """
        super(Hidden, self).__init__()

        self.encoder_decoder = EncoderDecoder(configuration).to(device)
        self.discriminator = Discriminator(configuration).to(device)
        self.cnn_f = CNN_F(configuration).to(device)

        # Using more discriminators
        if configuration.more_dis:
            self.discriminator_low = Discriminator(configuration).to(device)
            self.discriminator_high = Discriminator(configuration).to(device)

            self.optimizer_discrim_low = torch.optim.Adam(self.discriminator_low.parameters())
            self.optimizer_discrim_high = torch.optim.Adam(self.discriminator_high.parameters())

            # define the filter
            self.filter_low = FilterLow().to(device)
            self.filter_high = FilterHigh().to(device)
        else:
            self.filter_low = None
            self.filter_high = None

        # Adam for discriminate
        self.optimizer_discrim = torch.optim.Adam(self.discriminator.parameters())

        self.optimizer_enc_dec_cnn = torch.optim.Adam(list(self.encoder_decoder.parameters()) +
                                                      list(self.cnn_f.parameters()))

        self.config = configuration
        self.device = device

        self.bce_with_logits_loss = nn.BCEWithLogitsLoss().to(device)
        self.mse_loss = nn.MSELoss().to(device)
        self.l1_loss = nn.L1Loss().to(device)

        # Defined the labels used for training the discriminator/adversarial loss
        self.cover_label = 1
        self.encoded_label = 0

        # using LPIPS
        self.lpip_loss = LPIPLoss()
        self.lpip_loss.to(device)

        # define the attributes
        # model input
        self.images = None
        self.secret_images = None
        self.batch_size = None
        # model output
        self.encoded_images = None
        self.noised_images = None
        self.decoded_images = None
        self.cnn_f_images = None
        # loss
        self.g_loss_adv = None
        self.g_loss_dis_low = None
        self.g_loss_dis_high = None
        self.g_loss_enc_l1 = None
        self.g_loss_enc_lpip = None
        self.g_loss_enc = None
        self.g_loss_dec = None
        self.c_loss = None
        self.g_loss = None
        self.d_loss = None
        self.d_loss_on_low = None
        self.d_loss_on_high = None
        self.losses = None
        self.bitwise_avg_err = None

    def train(self, batch, step_noise):
        """
        Train the model on a single batch of data

        Parameters:
            batch (list): batch of training data, in the form [images, messages]
            step_noise (int): step in the noise layer, max is 50
        Returns:
            losses (dict): dictionary of error metrics from Encoder, Decoder, and Discriminator on the current batch
            encoded_images (tensor): encoded images
            noised_images (tensor): noised images
            decoded_images (tensor): decoded images
            cnn_f_images (tensor): cnn_f images
        """
        self._set_input(batch)
        self._optimize_parameters(step_noise)
        self._loss_stack()
        return self.losses, (self.encoded_images, self.noised_images, self.decoded_images, self.cnn_f_images)

    def val(self, batch, step_noise):
        self._set_input(batch)
        self._val(step_noise)
        self._loss_stack()
        return self.losses, (self.encoded_images, self.noised_images, self.decoded_images, self.cnn_f_images)

    def _loss_stack(self):
        decoded_rounded = self.decoded_images.detach().cpu().numpy().round().clip(-1, 1)
        self.bitwise_avg_err = np.sum(np.abs(decoded_rounded - self.secret_images.detach().cpu().numpy())) / (
                self.batch_size * self.secret_images.shape[1])
        self.losses = {
            'loss            ': self.g_loss.item(),
            'g_loss          ': self.g_loss_enc.item(),
            'enc_l1          ': self.g_loss_enc_l1.item(),
            'enc_lpip        ': self.g_loss_enc_lpip.item(),
            'dec_mse         ': self.g_loss_dec.item(),
            'adversarial_bce ': self.g_loss_adv.item(),
            'c_loss_l1       ': self.c_loss.item(),
            'g_loss_dis_low  ': self.g_loss_dis_high.item(),
            'g_loss_dis_high ': self.g_loss_dis_low.item(),
            'bitwise-error   ': self.bitwise_avg_err,
            'dis_bce         ': self.d_loss.item(),
            'dis_low_bce     ': self.d_loss_on_low.item(),
            'dis_high_bce    ': self.d_loss_on_high.item(),
        }

    def _set_input(self, batch: list):
        """
        Sets the input for the network

        Parameters:
            batch (list): batch of training data, in the form [images, messages]
        """
        self.images, self.secret_images = batch
        self.batch_size = self.images.shape[0]

    def _optimize_parameters(self, step_noise):
        with torch.enable_grad():
            self._forward(step_noise)
            # update encoder, decoder and cnn_f
            self.optimizer_enc_dec_cnn.zero_grad()
            self._backward_enc_dec_cnn()
            self.optimizer_enc_dec_cnn.step()
            # update discriminator
            self.optimizer_discrim.zero_grad()
            self._backward_dis()
            self.optimizer_discrim.step()
            # update discriminator_low and discriminator_high
            self.optimizer_discrim_low.zero_grad()
            self.optimizer_discrim_high.zero_grad()
            self._backward_more_dis()
            self.optimizer_discrim_low.step()
            self.optimizer_discrim_high.step()

    def _val(self, step_noise):
        with torch.no_grad():
            self._forward(step_noise)
            self._cal_loss_enc_dec_cnn()
            self._cal_loss_dis()
            self._cal_loss_more_dis()

    def _forward(self, step_noise: int):
        """
        Runs a forward pass on the network

        Parameters:
            step_noise (int): step in the noise layer, max is 50
        """
        self.encoded_images, self.noised_images, self.decoded_images = \
            (self.encoder_decoder(self.images, self.secret_images, step_noise))
        self.cnn_f_images = self.cnn_f(self.encoded_images)

    def _cal_loss_enc_dec_cnn(self):
        g_target_label_encoded = torch.full((self.batch_size, 1), self.cover_label, dtype=torch.float32, device=self.device)
        # discriminator
        self.g_loss_adv = self.bce_with_logits_loss(self.discriminator(self.encoded_images), g_target_label_encoded)
        # discriminator more
        self.g_loss_dis_low = self.bce_with_logits_loss(self.discriminator_low(self.encoded_images),
                                                        g_target_label_encoded)
        self.g_loss_dis_high = self.bce_with_logits_loss(self.discriminator_high(self.encoded_images),
                                                         g_target_label_encoded)
        self.g_loss_enc_l1 = self.l1_loss(self.encoded_images, self.images)
        self.g_loss_enc_lpip = self.lpip_loss(self.encoded_images, self.images)
        self.g_loss_enc = self.g_loss_enc_l1 + self.g_loss_enc_lpip
        self.g_loss_dec = self.mse_loss(self.decoded_images, self.secret_images)
        self.c_loss = self.l1_loss(self.cnn_f(self.encoded_images), self.images)
        self.g_loss = (self.config.adversarial_loss * self.g_loss_adv +
                       self.config.adversarial_loss * self.g_loss_dis_low +
                       self.config.adversarial_loss * self.g_loss_dis_high +
                       self.config.encoder_loss * self.g_loss_enc +
                       self.config.decoder_loss * self.g_loss_dec +
                       self.config.cnn_f_loss * self.c_loss)

    def _backward_enc_dec_cnn(self):
        self._cal_loss_enc_dec_cnn()
        self.g_loss.backward()

    def _cal_loss_dis_basic(self, net_d, real, fake):
        """
        Calculate GAN loss for the discriminator

        Parameters:
            net_d (network)      -- the discriminator D
            real  (tensor array) -- real images
            fake  (tensor array) -- images generated by a generator (fake.detach())
        """
        d_target_label_cover = torch.full((self.batch_size, 1), self.cover_label, device=self.device)
        d_target_label_encoded = torch.full((self.batch_size, 1), self.encoded_label, device=self.device)

        # train on cover
        d_on_cover = net_d(real)
        d_loss_on_cover = self.bce_with_logits_loss(d_on_cover, d_target_label_cover.float())
        # train on fake
        d_on_encoded = net_d(fake)
        d_loss_on_encoded = self.bce_with_logits_loss(d_on_encoded, d_target_label_encoded.float())
        d_loss = d_loss_on_cover + d_loss_on_encoded
        return d_loss

    def _cal_loss_dis(self):
        self.d_loss = self._cal_loss_dis_basic(self.discriminator, self.images, self.encoded_images.detach())

    def _backward_dis(self):
        self._cal_loss_dis()
        self.d_loss.backward()

    def _cal_loss_more_dis(self):
        # discriminator_low
        cover_low = self.filter_low(self.images)
        encoded_low = self.filter_low(self.encoded_images)
        self.d_loss_on_low = self._cal_loss_dis_basic(self.discriminator_low, real=cover_low, fake=encoded_low.detach())

        # discriminator_high
        cover_high = self.filter_high(self.images)
        encoded_high = self.filter_high(self.encoded_images)
        self.d_loss_on_high = self._cal_loss_dis_basic(self.discriminator_high, cover_high, encoded_high.detach())

    def _backward_more_dis(self):
        self._cal_loss_more_dis()
        self.d_loss_on_low.backward()
        self.d_loss_on_high.backward()

    def load_model(self, project_pah, model_name, load_dis=False):
        """
        Load the model from the project path
        """
        ckp_path = os.path.join(project_pah, "checkpoints", model_name)
        logger.log(f"loading model from {ckp_path}")
        # load from the checkpoint
        checkpoint = torch.load(ckp_path)
        self.encoder_decoder.load_state_dict(checkpoint["enc-dec-model"])
        self.cnn_f.load_state_dict(checkpoint["cnn-f-model  "])
        if load_dis:
            self.discriminator.load_state_dict(checkpoint["discrim-model"])
        epoch = checkpoint["epoch"]

    def to_stirng(self):
        return '{}\n{}'.format(str(self.encoder_decoder), str(self.discriminator))
