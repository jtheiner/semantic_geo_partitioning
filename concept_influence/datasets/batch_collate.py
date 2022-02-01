import torch


def collate_fn_ci(batch):

    attributions_batch = torch.stack(
        [x["attribution_map"] for x in batch], dim=0
    )  # (BS, C, H, W)
    attributions_batch, _ = attributions_batch.max(dim=1)  # max over channel dimension

    segmentation_maps = [x["segmentation_masks"] for x in batch]

    metadata = {}
    for k in batch[0]["metadata"].keys():
        if isinstance(batch[0]["metadata"][k], str):
            metadata[k] = [x["metadata"][k] for x in batch]
        else:
            metadata[k] = torch.stack([x["metadata"][k] for x in batch])

    return {
        "attribution_maps": attributions_batch,
        "segmentation_maps": segmentation_maps,
        "metadata": metadata,
    }
