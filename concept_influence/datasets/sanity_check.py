from typing import Union

import torch

from utils import convert_semantic_map_to_masks


class DummyConceptInfluenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dilation: Union[None, int],
        image_size=(16, 16),
        dataset_size=4,
        num_classes=10,
    ) -> None:
        super().__init__()

        self.dilation = dilation
        self.image_size = image_size
        self.dataset_size = dataset_size
        self.num_classes = num_classes

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, _):

        semantic_image = torch.randint(0, self.num_classes, size=self.image_size)
        segmentation_masks = convert_semantic_map_to_masks(
            semantic_image, self.dilation
        )

        attribution_map = torch.rand((3, *self.image_size))

        return {
            "attribution_map": attribution_map,  # shape (C, H, W)
            "segmentation_masks": segmentation_masks,  # Dict[Any, torch.tensor(H, W)]
            "metadata": {"image_id": "NotImplemented"},
        }
