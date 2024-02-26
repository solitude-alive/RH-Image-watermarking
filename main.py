import os

from default_setting import load_args, Setting
from utils import logger

from trainer import Trainer

os.environ['CUDA_VISIBLE_DEVICES'] = "0"  # Set the GPU id here


def main():
    args = load_args()

    settings = Setting(args)

    trainer = Trainer(model=settings.model, hidden_config=settings.hidden_config,
                      train_options=settings.train_options, out_folder=logger.get_dir(), device=settings.device)

    trainer.train()

    from calculate import cal_metrics

    cal_metrics(settings.device, settings.model, settings.hidden_config, out_folder="./runs/test_results")


if __name__ == '__main__':
    main()
