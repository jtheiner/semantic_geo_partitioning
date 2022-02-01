from argparse import ArgumentParser, Namespace
import logging
from pathlib import Path

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import pandas as pd

from multihead_pl_module import MultiPartClassifier


def raw_geoaccuracy_to_df(output):
    records = []
    # one record: "dataset/partitioning_level/km_threshold": accuracy
    for k_full, acc in output.items():
        testset_name, partitioning_level, t_km = k_full.split("/")
        records.append(
            {
                "dataset": testset_name,
                "partitioning_level": partitioning_level,
                "km": t_km.split("_")[-1],
                "geo_acc": acc,
            }
        )

    df = pd.DataFrame.from_records(records)
    df = df.set_index(["partitioning_level", "dataset", "km"]).unstack(["km"])
    return df[
        [
            ("geo_acc", "1"),
            ("geo_acc", "25"),
            ("geo_acc", "200"),
            ("geo_acc", "750"),
            ("geo_acc", "2500"),
        ]
    ]


def parse_args():
    args = ArgumentParser()
    args.add_argument(
        "-c",
        "--config",
        type=Path,
        help="Path to hparams.yaml",
    )
    args.add_argument(
        "-ckpt",
        "--checkpoint",
        type=str,
        help="Path for checkpoint file (.ckpt)",
    )
    args.add_argument(
        "--output", type=Path, help="If not provided, output is only printed to console"
    )
    args.add_argument("--batch_size", type=int, default=128)
    args.add_argument("--cpuonly", action="store_true")
    args.add_argument("--precision", choices=[16, 32, 64], type=int, default=16)
    return args.parse_args()


def main():

    # for interactive
    args = Namespace()
    args.config = Path("")
    args.checkpoint = ""
    args.output = None

    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    model = MultiPartClassifier.load_from_checkpoint(
        args.checkpoint, hparams_file=str(args.config)
    )

    testset_logger = TensorBoardLogger(
        save_dir=str(args.config.parent.parent), name="tb_logs_testsets"
    )

    model.hparams["batch_size"] = args.batch_size

    trainer = Trainer(
        gpus=None if args.cpuonly else 1,
        precision=args.precision,
        logger=testset_logger,
    )

    outputs = trainer.test(model, verbose=False)

    df = raw_geoaccuracy_to_df(outputs[0])

    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=True)
        ckpt_fname = Path(args.checkpoint).stem
        df.to_csv(args.output / f"results_testsets_{ckpt_fname}.csv")
        df.to_latex(args.output / f"results_testsets_{ckpt_fname}.tex")

    df = df * 100
    df = df.round(decimals=1)

    print(df.iloc[df.index.get_level_values("partitioning_level") == "fine"])
    print()
    print(df.iloc[df.index.get_level_values("partitioning_level") == "hierarchy"])


if __name__ == "__main__":
    main()
