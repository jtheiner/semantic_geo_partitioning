from pathlib import Path
from typing import Union
import torch
from PIL import Image
import pandas as pd
import torchvision

from datasets.image_preprocessing import IMAGENET_MEAN, IMAGENET_STD


class ImageFiveCropsTestDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        meta_csv: Union[str, Path],
        image_dir: Path,
        img_id_col: Union[str, int] = "img_id",
        image_size=224,
    ):
        self.image_dir = image_dir
        self.img_id_col = img_id_col
        self.image_size = image_size
        self.meta_info = pd.read_csv(meta_csv)
        self.meta_info["img_path"] = self.meta_info[img_id_col].apply(
            lambda img_id: str(self.image_dir / img_id)
        )

    def __len__(self):
        return len(self.meta_info.index)

    def __getitem__(self, idx):
        meta = self.meta_info.iloc[idx]
        meta = meta.to_dict()
        meta["img_id"] = meta[self.img_id_col]

        image = Image.open(meta["img_path"]).convert("RGB")
        image = torchvision.transforms.Resize(320)(image)
        if self.image_size < 250:
            image = torchvision.transforms.Resize(256)(image)
        crops = torchvision.transforms.FiveCrop(self.image_size)(image)

        tfms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

        crops_transformed = []
        for crop in crops:
            crops_transformed.append(tfms(crop))

        return torch.stack(crops_transformed, dim=0), meta


class CenterCropTestDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        meta_csv: Union[str, Path],
        image_dir: Union[str, Path],
        img_id_col: Union[str, int] = "img_id",
        dim_expand_multi_crop=False,
        image_size=224,
    ):
        self.dim_expand_multi_crop = dim_expand_multi_crop
        self.image_dir = Path(image_dir)
        self.meta_info = pd.read_csv(meta_csv)
        self.meta_info["img_path"] = self.meta_info[img_id_col].apply(
            lambda img_id: str(self.image_dir / img_id)
        )
        self.img_id_col = img_id_col
        self.image_size = image_size

    def __len__(self):
        return len(self.meta_info.index)

    def __getitem__(self, idx):
        meta = self.meta_info.iloc[idx]
        meta = meta.to_dict()
        meta["img_id"] = meta[self.img_id_col]

        image = Image.open(meta["img_path"]).convert("RGB")
        if self.image_size < 250:
            image = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize(256),
                    torchvision.transforms.CenterCrop(self.image_size),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                ]
            )(image)
        else:
            image = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize(320),
                    torchvision.transforms.CenterCrop(self.image_size),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                ]
            )(image)

        if self.dim_expand_multi_crop:
            # pseudo dimension for multi-crop inference
            image = image.unsqueeze(0)
        return image, meta
