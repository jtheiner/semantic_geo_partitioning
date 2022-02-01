from ctypes import Union
from typing import Dict, List
import torch


def vectorized_gc_distance(latitudes, longitudes, latitudes_gt, longitudes_gt):
    R = 6371
    factor_rad = 0.01745329252
    longitudes = factor_rad * longitudes
    longitudes_gt = factor_rad * longitudes_gt
    latitudes = factor_rad * latitudes
    latitudes_gt = factor_rad * latitudes_gt
    delta_long = longitudes_gt - longitudes
    delta_lat = latitudes_gt - latitudes
    subterm0 = torch.sin(delta_lat / 2) ** 2
    subterm1 = torch.cos(latitudes) * torch.cos(latitudes_gt)
    subterm2 = torch.sin(delta_long / 2) ** 2
    subterm1 = subterm1 * subterm2
    a = subterm0 + subterm1
    c = 2 * torch.asin(torch.sqrt(a))
    gcd = R * c
    return gcd


def summarize_gcd_stats(
    outputs: List[Dict[str, torch.Tensor]], thresholds_km=[1, 25, 200, 750, 2500]
):
    """Computes the geolocational accuracy at multiple error levels for a bunch of per sample distances.

    Keys without containing `gcd` are ignored

    Args:
        outputs (List[Dict[str, torch.Tensor]]): Each batch contains keys of format `<prefix>/<partitioning_level>/gcd` and a tensor of per-sample distances
        thresholds_km (List[int]): Thresholds to compute the geolocational accuracy at given spatial levels.

    Returns:
        Dict[str, Any]: Global metric values in format `<prefix>/<partitioning_level>/gcd_{threshold [km]}` = accuracy
    """

    def _gcd_threshold_eval(gc_dists):
        results = {}
        for thres in thresholds_km:
            results[thres] = torch.true_divide(
                torch.sum(gc_dists <= thres), len(gc_dists)
            ).item()
        return results

    # aggregate only all GCD metrics
    metric_names = [s for s in list(outputs[0].keys()) if "gcd" in s]

    gcd_dict = {}
    for metric_name in metric_names:
        prefix, p_name, _ = metric_name.split("/")
        # distances for each sample, i.e. removed batch dimension
        distances_flat = torch.cat([output[metric_name] for output in outputs], dim=0)

        gcd_results = _gcd_threshold_eval(distances_flat)
        for gcd_thres, acc in gcd_results.items():
            gcd_dict[f"{prefix}/{p_name}/gcd_{gcd_thres}"] = acc

    return gcd_dict
