import argparse
import socket
import torch
import os
import pickle
import pprint

from options import *
from utils import utils, logger
from model.hidden import Hidden


def load_args():
    parent_parser = argparse.ArgumentParser(description='Training of nets')
    parent_parser.add_argument('--hostname', default=socket.gethostname(),
                               help='the  host name of the running server')
    subparsers = parent_parser.add_subparsers(dest='command', help='Sub-parser for commands')

    new_run_parser = subparsers.add_parser('new', help='starts a new run')
    new_run_parser.add_argument('--data-dir', '-d', required=True, type=str,
                                help='The directory where the data is stored.')
    new_run_parser.add_argument('--batch-size', '-b', required=True, type=int, help='The batch size.')
    new_run_parser.add_argument('--epochs', '-e', default=300, type=int,
                                help='Number of epochs to run the simulation.')
    new_run_parser.add_argument('--name', required=True, type=str, help='The name of the experiment.')

    new_run_parser.add_argument('--size', '-s', default=128, type=int,
                                help='The size of the images (images are square so this is height and width).')
    new_run_parser.add_argument('--message', '-m', default=30, type=int,
                                help='The length in bits of the watermark.')
    new_run_parser.add_argument('--continue-from-folder', '-c', default='', type=str,
                                help='The folder from where to continue a previous run. Leave blank if you are '
                                     'starting a new experiment.')
    new_run_parser.add_argument('--tensorboard', action='store_true',
                                help='Use to switch on Tensorboard logging.')
    new_run_parser.add_argument('--enable-fp16', dest='enable_fp16', action='store_true',
                                help='Enable mixed-precision training.')

    new_run_parser.set_defaults(tensorboard=False)
    new_run_parser.set_defaults(enable_fp16=False)

    continue_parser = subparsers.add_parser('continue', help='Continue a previous run')
    continue_parser.add_argument('--folder', '-f', required=True, type=str,
                                 help='Continue from the last checkpoint in this folder.')
    continue_parser.add_argument('--data-dir', '-d', required=False, type=str,
                                 help='The directory where the data is stored. Specify a value only if you want to '
                                      'override the previous value.')
    continue_parser.add_argument('--epochs', '-e', required=False, type=int,
                                 help='Number of epochs to run the simulation. Specify a value only if you want to '
                                      'override the previous value.')

    args = parent_parser.parse_args()
    return args


class Setting:

    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.hostname = args.hostname

        self.train_options = None
        self.noise_config = None
        self.hidden_config = None
        self.model = None

        self.set_config()
        self.log_config()
        self.write_pickle()

        self.set_model()

    def set_config(self):
        args = self.args
        args.command = 'new'
        args.batch_size = 8 * 2 * 2
        args.epochs = 300   # default is 500
        args.data_dir = "/zgyang/dataset/alask10k"
        args.name = "noise"
        args.noise = None
        args.size = 256
        args.message = 64
        args.continue_from_folder = ''
        args.tensorboard = False
        args.enable_fp16 = False
        args.generator_name = "unet"  # | "unet" | "resnet" |

        assert args.command == 'new'
        start_epoch = 1
        self.train_options = TrainingOptions(
            batch_size=args.batch_size,
            number_of_epochs=args.epochs,
            train_folder=os.path.join(args.data_dir, "train"),
            validation_folder=os.path.join(args.data_dir, "val"),
            runs_folder=os.path.join('.', "runs"),
            start_epoch=start_epoch,
            experiment_name=args.name
        )

        self.noise_config = args.noise if args.noise is not None else []
        self.hidden_config = NetConfiguration(
            h=args.size, w=args.size, message_length=args.message,
            container_channels=4, encoded_channels=3,
            secret_channels=1,
            use_discriminator=True,
            discriminator_blocks=3, discriminator_channels=64,
            decoder_loss=1,
            encoder_loss=1,
            adversarial_loss=1e-3,
            cnn_f_loss=0.5,
            enable_fp16=False,
            generator_name=args.generator_name,
            use_up=True,
            use_more_dis=True,
            use_c_att=False,
            use_s_att=False,
            use_w_gan=False,
        )

    def set_model(self):
        self.model = Hidden(self.hidden_config, self.device)

        logger.log("HiDDeN model: {}\n".format(self.model.to_stirng()))
        logger.log("Model Configuration:\n")
        logger.log(pprint.pformat(vars(self.hidden_config)))
        logger.log("\nTraining train_options:\n")
        logger.log(pprint.pformat(vars(self.train_options)))

    def log_config(self):
        this_run_folder = utils.create_folder_for_run(self.train_options.runs_folder, self.args.name)

        logger.configure(this_run_folder, format_strs=["stdout", "log"])

    def write_pickle(self):
        this_run_folder = logger.get_dir()
        with open(os.path.join(this_run_folder, 'options-and-config.pickle'), 'wb+') as f:
            pickle.dump(self.train_options, f)
            pickle.dump(self.noise_config, f)
            pickle.dump(self.hidden_config, f)

    def print_args(self):
        logger.log("--------------")
        logger.log(self.args)
        logger.log(self.hostname)
        logger.log(self.device)
