from argparse import ArgumentParser, Namespace
from datetime import datetime
import logging
from pathlib import Path

import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from multihead_pl_module import MultiPartClassifier
from utils_base import check_is_valid_torchvision_architecture
from evaluate_testsets import raw_geoaccuracy_to_df


def parse_args():
    args = ArgumentParser()
    args.add_argument("-c", "--config", type=Path, required=True)
    args.add_argument("--out_dir", type=Path, default="../data/training")
    return args.parse_args()


def main():

    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    # load all model and training parameters from file
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        hparams_model = config["model_params"]
        hparams_trainer = config["trainer_params"]
    assert check_is_valid_torchvision_architecture(hparams_model["arch"])

    # initialize model
    model = MultiPartClassifier(hparams=Namespace(**hparams_model))

    # training configuration
    out_dir = args.out_dir / args.config.stem / datetime.now().strftime("%y%m%d-%H%M%S")

    logging.info(f"Output directory: {out_dir}")

    checkpoint_dir = out_dir
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=checkpoint_dir,
        save_top_k=1,
        filename="{epoch}-{val_loss:.3f}",
    )
    trainer = Trainer(
        **hparams_trainer,
        logger=TensorBoardLogger(save_dir=str(out_dir), name=""),
        callbacks=[
            LearningRateMonitor(logging_interval="epoch"),
            checkpoint_callback,
        ],
        val_check_interval=hparams_model["val_check_interval"],
    )

    # training pipeline
    trainer.fit(model)

    # evaluate best checkpoint
    outputs = trainer.test(model, ckpt_path="best", verbose=False)
    df = raw_geoaccuracy_to_df(outputs[0])
    df.to_csv(out_dir / "results_testsets_best.csv")
    df.style.to_latex(out_dir / "results_testsets_best.tex")


if __name__ == "__main__":
    main()
