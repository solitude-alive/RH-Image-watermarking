import torch
import os

from options import NetConfiguration
from model.encoder_decoder import EncoderDecoder
from utils import logger


class TestModel:
    def __init__(self, configuration: NetConfiguration, device: torch.device):
        """
        Initializes the Hidden network for testing

        Parameters:
            configuration (NetConfiguration): Configuration for the net, Hidden_configuration
            device (torch.device): CPU or GPU
        """
        self.net_config = configuration
        self.device = device
        self.encoder_decoder = EncoderDecoder(self.net_config).to(device)

        self.encoder = None
        self.decoder = None

    def load_model(self, project_pah, model_name):
        """
        Load the model from the project path
        """
        ckp_path = os.path.join(project_pah, "checkpoints", model_name)
        logger.log(f"Loading model from {ckp_path}")
        # load from the checkpoint
        checkpoint = torch.load(ckp_path)
        self.encoder_decoder.load_state_dict(checkpoint["enc-dec-model"])
        logger.log("Loaded model successfully")

        self.encoder = self.encoder_decoder.encoder
        self.decoder = self.encoder_decoder.decoder

    def encode(self, img_cover, img_secret):
        """
        Encode the image
        """
        self.encoder.eval()
        with torch.no_grad():
            img_enc = self.encoder(img_cover, img_secret)
        return img_enc

    def decode(self, img_enc):
        """
        Decode the image
        """
        self.decoder.eval()
        with torch.no_grad():
            img_dec = self.decoder(img_enc)
        return img_dec
