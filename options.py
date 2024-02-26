class TrainingOptions:
    """
    Configuration options for the training
    """

    def __init__(self,
                 batch_size: int,
                 number_of_epochs: int,
                 train_folder: str, validation_folder: str, runs_folder: str,
                 start_epoch: int, experiment_name: str):
        self.batch_size = batch_size
        self.number_of_epochs = number_of_epochs
        self.train_folder = train_folder
        self.validation_folder = validation_folder
        self.runs_folder = runs_folder
        self.start_epoch = start_epoch
        self.experiment_name = experiment_name


class NetConfiguration:
    """
    The network configuration.
    """

    def __init__(self, h: int, w: int,
                 message_length: int,
                 container_channels: int,
                 encoded_channels: int,
                 secret_channels: int,
                 use_discriminator: bool,
                 discriminator_blocks: int,
                 discriminator_channels: int,
                 decoder_loss: float,
                 encoder_loss: float,
                 adversarial_loss: float,
                 cnn_f_loss: float,
                 enable_fp16: bool = False,
                 generator_name: str = "unet",
                 use_up: bool = True,
                 use_more_dis: bool = False,
                 use_s_att: bool = False,
                 use_c_att: bool = False,
                 use_w_gan: bool = False,
                 ):
        self.H = h
        self.W = w
        self.message_length = message_length
        self.container_channels = container_channels
        self.encoded_channels = encoded_channels
        self.secret_channels = secret_channels
        self.use_discriminator = use_discriminator
        self.discriminator_blocks = discriminator_blocks
        self.discriminator_channels = discriminator_channels
        self.decoder_loss = decoder_loss
        self.encoder_loss = encoder_loss
        self.adversarial_loss = adversarial_loss
        self.cnn_f_loss = cnn_f_loss
        self.enable_fp16 = enable_fp16
        self.generator_name = generator_name
        self.up = use_up
        self.more_dis = use_more_dis
        self.c_att = use_c_att
        self.s_att = use_s_att
        self.w_gan = use_w_gan
