import logging
import json
from math import ceil
from argparse import Namespace
from pathlib import Path
from typing import Tuple, Union

import torch
from pytorch_lightning import LightningModule


import utils_base
import utils_eval
from datasets.image_preprocessing import tfms
from datasets.image_dataset import ImageFiveCropsTestDataset
from datasets.msgpack_dataset import MsgPackIterableDatasetMultiTargetWithDynLabels
from partitioning.utils import BasePartitioning
from partitioning.s2p import S2Hierarchy
from partitioning.semp import SemPHierarchy


class MultiPartClassifier(LightningModule):
    def __init__(self, hparams: Namespace):
        super().__init__()
        self.save_hyperparameters(hparams, logger=True)

        self.partitionings, self.hierarchy = self.__init_partitionings()
        self.model, self.classifier = self.__build_model(
            self.hparams.arch, self.hparams.image_size, self.hparams.weights
        )

    def __init_partitionings(self):
        """Build (multi-)partitioning from config file
        hparams example s2:
            partitionings:
                ptype: s2
                shortnames: [coarse, middle, fine]
                files:
                - <...>/cells_50_5000.csv
                - <...>/cells_50_2000.csv
                - <...>/cells_50_1000.csv
                base_part_kwargs:
                    skiprows: 2
                    index_col: class_label
                    col_class_label: hex_id
                    col_latitude: latitude_mean
                    col_longitude: longitude_mean

        hparams example semp:
            partitionings:
                ptype: semp
                shortnames: [coarse, middle, fine]
                h_dict: <...>/h_dict.json
                files:
                - <...>/m_5212.csv
                - <...>/m_11252.csv
                - <...>/m_14371.csv
                base_part_kwargs:
                  skiprows: 0
                  index_col: class_index
                  col_class_label: class_label
                  col_latitute: latitude_mean
                  col_longitude: longitude_mean


        Raises:
            KeyError: if `ptype` not in {s2, semp}

        Returns:
            Tuple[List[BasePartitioning], Union[None, Union[S2Hierarchy, SemPHierarchy]]]: List of partitionings and optional Hierachy when using a multiple partitionings
        """

        partitionings = []
        for shortname, path in zip(
            self.hparams.partitionings["shortnames"],
            self.hparams.partitionings["files"],
        ):
            path = Path(path)
            if self.hparams.partitionings["ptype"] in ["s2", "semp"]:
                partitionings.append(
                    BasePartitioning(
                        csv_file=path,
                        shortname=shortname,
                        **self.hparams.partitionings["base_part_kwargs"],
                    )
                )
            else:
                raise KeyError(
                    f'Only "s2" or "semp" implemented '
                    f'available but received "{self.hparams.partitionings["ptype"]}"'
                )

        if len(self.hparams.partitionings["files"]) == 1:  # single partitioning
            return partitionings, None  # no need to build the hierarchy mapping

        if self.hparams.partitionings["ptype"] == "s2":
            return partitionings, S2Hierarchy(partitionings)
        elif self.hparams.partitionings["ptype"] == "semp":
            return (
                partitionings,
                SemPHierarchy(
                    partitionings, Path(self.hparams.partitionings["h_dict"])
                ),
            )
        return None, None

    def __build_model(
        self,
        arch: str,
        image_size: Union[int, None] = None,
        weights: Union[None, str, Path] = None,
    ) -> Tuple[torch.nn.Module, torch.nn.Module]:
        """Initialize backbone and classifier from Namespace.

        Args:
            arch (str): CNN backbone from torchvision model namespace (https://pytorch.org/vision/stable/models.html) or "efficientnet-b{0-7}"
            image_size (Union[int, None], optional): Model input image size (image_size, image_size). Defaults to None.
            weights (Union[None, str, Path], optional): Filepath to pre-trained model; load matching weights. Defaults to None.

        Returns:
            Tuple[torch.nn.Module, torch.nn.Module]: returns model and classifier module individual
        """
        logging.info("Build model")
        model, nfeatures = utils_base.build_base_model(arch, image_size)

        classifier = torch.nn.ModuleList(
            [
                torch.nn.Linear(nfeatures, len(self.partitionings[i]))
                for i in range(len(self.partitionings))
            ]
        )

        if weights:
            logging.info("Load weights from pre-trained model")
            model, classifier = utils_base.load_weights_if_available(
                model, classifier, self.hparams.weights
            )

        return model, classifier

    def forward(self, x):
        # feature extraction
        x = self.model(x)

        # classification for each individual partitioning
        yhats = []
        for i in range(len(self.partitionings)):
            yhats.append(self.classifier[i](x))
        return yhats

    def training_step(self, batch, batch_idx):

        images, target = batch

        # single partitioning
        if not isinstance(target, list) and len(target.shape) == 1:
            target = [target]

        output = self(images)
        losses = [
            torch.nn.functional.cross_entropy(output[i], target[i])
            for i in range(len(output))
        ]
        loss = sum(losses)

        self.log("train/loss", loss, on_step=True, on_epoch=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=None):

        images, target, true_lats, true_lngs = batch

        # single partitioning
        if not isinstance(target, list) and len(target.shape) == 1:
            target = [target]

        output = self(images)
        losses = [
            torch.nn.functional.cross_entropy(output[i], target[i])
            for i in range(len(output))
        ]

        loss = sum(losses)

        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=False, logger=True
        )

        # we also log the geolocational accuracy

        # hierarchical prediction, i.e. multiplication of conected classes between individual partitionings
        if self.hierarchy is not None:
            hierarchy_logits = [
                yhat[:, self.hierarchy.M[:, i]] for i, yhat in enumerate(output)
            ]
            hierarchy_logits = torch.stack(hierarchy_logits, dim=-1)
            hierarchy_preds = torch.prod(hierarchy_logits, dim=-1)

        # for each partitioning (+ hierarchy) log GCD error@km threshold
        pnames = [p.shortname for p in self.partitionings]
        if self.hierarchy is not None:
            pnames.append("hierarchy")

        distances_dict = {}
        for i, pname in enumerate(pnames):
            # get predicted coordinates
            if i == len(self.partitionings):
                i = i - 1
                pred_class_indexes = torch.argmax(hierarchy_preds, dim=1)
            else:
                pred_class_indexes = torch.argmax(output[i], dim=1)
            pred_latlngs = [
                self.partitionings[i].get_lat_lng(idx)
                for idx in pred_class_indexes.tolist()
            ]
            pred_lats, pred_lngs = map(list, zip(*pred_latlngs))
            pred_lats = torch.tensor(pred_lats, dtype=torch.float)
            pred_lngs = torch.tensor(pred_lngs, dtype=torch.float)
            # calculate error
            distances = utils_eval.vectorized_gc_distance(
                pred_lats,
                pred_lngs,
                true_lats.type_as(pred_lats),
                true_lngs.type_as(pred_lats),
            )
            distances_dict[f"val/{pname}/gcd"] = distances

        return {"val_loss": loss, **distances_dict}

    def validation_epoch_end(self, outputs) -> None:
        # aggregate distances, i.e. global computation of the GCD error @threshold [km]
        gcd_dict = utils_eval.summarize_gcd_stats(outputs)
        # val/<partitioning_level>/gcd_<km> = acc
        self.log_dict(gcd_dict, logger=True, on_epoch=True)

    def configure_optimizers(self):

        # optimize all weights with a global strategy
        optim_full_model = torch.optim.SGD(
            self.parameters(), **self.hparams.optim["params"]
        )

        return {
            "optimizer": optim_full_model,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.MultiStepLR(
                    optim_full_model, **self.hparams.scheduler["params"]
                ),
                "interval": "epoch",
                "name": "lr",
            },
        }

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        images, meta_batch = batch

        cur_batch_size = images.shape[0]
        ncrops = images.shape[1]

        # reshape input from [bs, crops, **] to [bs x crops, **]
        images = torch.reshape(images, (cur_batch_size * ncrops, *images.shape[2:]))

        # forward pass
        yhats = self(images)

        # softmax for each crop of each partitioning
        yhats = [torch.nn.functional.softmax(yhat, dim=1) for yhat in yhats]

        # respape output back to access individual crops
        yhats = [
            torch.reshape(yhat, (cur_batch_size, ncrops, *list(yhat.shape[1:])))
            for yhat in yhats
        ]

        # calculate max over crops
        yhats = [torch.max(yhat, dim=1)[0] for yhat in yhats]

        # make hierarchical prediction
        # multiplication of connected cells
        if self.hierarchy is not None:
            hierarchy_logits = torch.stack(
                [yhat[:, self.hierarchy.M[:, i]] for i, yhat in enumerate(yhats)],
                dim=-1,
            )
            hierarchy_preds = torch.prod(hierarchy_logits, dim=-1)

        # calculate great circle distances
        distances_dict = {}
        if self.hierarchy is not None:
            nparts = len(self.partitionings) + 1
        else:
            nparts = len(self.partitionings)

        for i in range(nparts):
            # get pred class indices
            if self.hierarchy is not None and i == len(self.partitionings):
                pname = "hierarchy"
                pred_classes = torch.argmax(hierarchy_preds, dim=1)
                i = i - 1
            else:
                pname = self.partitionings[i].shortname
                pred_classes = torch.argmax(yhats[i], dim=1)

            # calculate GCD
            pred_lats, pred_lngs = map(
                list,
                zip(
                    *[
                        self.partitionings[i].get_lat_lng(c)
                        for c in pred_classes.tolist()
                    ]
                ),
            )
            pred_lats = torch.tensor(pred_lats, dtype=torch.float)
            pred_lngs = torch.tensor(pred_lngs, dtype=torch.float)
            distances = utils_eval.vectorized_gc_distance(
                pred_lats,
                pred_lngs,
                meta_batch["latitude"].type_as(pred_lats),
                meta_batch["longitude"].type_as(pred_lngs),
            )
            distances_dict[f"test/{pname}/gcd"] = distances
        return distances_dict

    def test_epoch_end(self, outputs):

        # TODO: single dataloader output -> not type(outputs, List)?

        testset_names = [Path(s["meta_info"]).stem for s in self.hparams.test]

        for dataset_idx, dataloader_outputs in enumerate(outputs):
            result = utils_eval.summarize_gcd_stats(
                dataloader_outputs
            )  # per testset results

            # replace `test` in test/<partitioning_level>/gcd_<km> with testset_name
            log_dict = {
                key.replace("test", testset_names[dataset_idx]): acc
                for key, acc in result.items()
            }
            # <testset_name>/<partitioning_level>/gcd_<km> = acc
            self.log_dict(log_dict)

    def train_dataloader(self):

        with open(self.hparams.train_label_mapping, "r") as f:
            self._target_mapping = json.load(f)

        logging.info(f"Training set size (samples): {len(self._target_mapping)}")
        logging.info(
            f"Estimated number of training batches: {ceil(len(self._target_mapping) / self.hparams.batch_size)}"
        )

        dataset = MsgPackIterableDatasetMultiTargetWithDynLabels(
            path=self.hparams.msgpack_train_dir,
            target_mapping=self._target_mapping,
            key_img_id=self.hparams.key_img_id,
            key_img_encoded=self.hparams.key_img_encoded,
            shuffle=True,
            transformation=tfms("train", self.hparams.image_size),
        )
        return torch.utils.data.DataLoader(
            dataset,
            num_workers=self.hparams.num_workers_per_loader,
            batch_size=self.hparams.batch_size,
            shuffle=False,  # sample-wise pseudo-shuffling in dataset
            pin_memory=True,
        )

    def val_dataloader(self):
        # map image ids to class labels
        with open(self.hparams.val_label_mapping, "r") as f:
            target_mapping = json.load(f)

        dataset = MsgPackIterableDatasetMultiTargetWithDynLabels(
            path=self.hparams.msgpack_val_dir,
            target_mapping=target_mapping,
            key_img_id=self.hparams.key_img_id,
            key_img_encoded=self.hparams.key_img_encoded,
            shuffle=False,
            transformation=tfms("valid", self.hparams.image_size),
            meta_path=self.hparams.val_meta_path,
            cache_size=1024,
        )

        return torch.utils.data.DataLoader(
            dataset,
            num_workers=self.hparams.num_workers_per_loader,
            batch_size=self.hparams.batch_size,
            pin_memory=True,
        )

    def test_dataloader(self):

        batch_size_eff = ceil(self.hparams.batch_size / 5)  # due to five crop inference

        datasets = []
        # prepare one dataloader for each test set
        for testset in self.hparams.test:
            img_path = Path(testset["meta_info"]).parent / "img"
            logging.info(
                f"Init testset (five crop): img_base_path: {img_path}, {testset['meta_info']}"
            )
            dataset = ImageFiveCropsTestDataset(
                testset["meta_info"], img_path, image_size=self.hparams.image_size
            )
            datasets.append(dataset)
        return [
            torch.utils.data.DataLoader(
                d,
                num_workers=self.hparams.num_workers_per_loader,
                batch_size=batch_size_eff,
            )
            for d in datasets
        ]
