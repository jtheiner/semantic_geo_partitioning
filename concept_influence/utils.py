import torch
import numpy as np


def top_k_intersection(a1: torch.Tensor, a2: torch.Tensor, k=1000) -> float:
    """
    Args:
        a1: binary mask as float tensor
        a2: attribution map, i.e. feature importance for each pixel - needs same shape as a1
        k: intersect top-k pixel from attribution map

    Returns: portion of relevant pixels within the mask

    """
    values_masked, r1 = a1.view(-1).sort(descending=True)
    r1 = r1[values_masked == 1.0]
    r2 = a2.view(-1).argsort(descending=True)[:k]
    intersection = np.intersect1d(r1.detach().cpu(), r2.detach().cpu())
    return intersection.shape[0] / k


def concept_influence(attribution_map: torch.tensor, segmentation_maps):
    #  segmentation_maps: Dict[Any : torch.tensor]
    class_labels = []
    intersection = []
    concept_size_px = []
    for label, seg_mask in segmentation_maps.items():
        class_labels.append(label)
        intersection.append(top_k_intersection(seg_mask, attribution_map))
        concept_size_px.append(seg_mask.sum().item())

    return class_labels, intersection, concept_size_px


class PadObject:
    def __init__(self, beta):
        """Increase mask size by a number of pixels beta, a.k.a. dilation:

        https://en.wikipedia.org/wiki/Dilation_(morphology)

        mask: binary mask encoded as {0, 1}
        border: number of pixel to wrap
        """
        self.beta = beta
        self.nearest = lambda x, y: set(
            [
                (x - 1, y),
                (x - 1, y),
                (x + 1, y),
                (x, y - 1),
                (x, y + 1),
                (x - 1, y - 1),
                (x + 1, y + 1),
            ]
        )

    def __call__(self, mask):
        mask = mask.squeeze()
        for _ in range(self.beta):
            where = torch.where((mask == 1.0))
            # filter by image edges
            wrap_indexes = torch.bitwise_and(
                torch.bitwise_and(where[0] > 0, where[0] < mask.shape[1] - 1),
                torch.bitwise_and(where[1] > 0, where[1] < mask.shape[0] - 1),
            )
            where = torch.stack(where, dim=1)[wrap_indexes].tolist()
            # get all indices to update
            if len(where) == 0:
                mask = mask.unsqueeze(0)
                return mask
            update = list(set.union(*map(set, [self.nearest(x, y) for x, y in where])))
            xx, yy = list(zip(*update))
            xx = list(xx)
            yy = list(yy)
            mask[xx, yy] = 1.0

        mask = mask.unsqueeze(0)
        return mask


def convert_semantic_map_to_masks(semantic_image: torch.tensor, dilation=False):
    # semantic image with integer encoded classes -> dict of binary maps
    binary_maps = {}
    for seg_class_index in semantic_image.unique().tolist():
        mask = (semantic_image == seg_class_index).type(torch.FloatTensor)
        if dilation:
            mask = PadObject(mask)
        binary_maps[seg_class_index] = mask

    return binary_maps
