import logging
from functools import partial
from multiprocessing import Pool
from collections import defaultdict

import pandas as pd
import numpy as np


def __filter_by_list(df, keep_nodes: set, h_key="h"):
    def _filter(locs):
        filtered = []
        for l in locs:
            if l in keep_nodes:
                filtered.append(l)
        if len(filtered) == 0:
            return None
        return filtered

    df[h_key] = df[h_key].apply(lambda locs: _filter(locs))
    return df


def parallel_filter_by_list(df, keep_nodes: set, num_worker=8, h_key="h"):

    splits = np.array_split(df, num_worker)
    _fnc_partial = partial(__filter_by_list, keep_nodes=keep_nodes, h_key=h_key)
    with Pool(num_worker) as p:
        df = pd.concat(p.map(_fnc_partial, splits))

    num_samples = len(df.index)
    df = df[df[h_key].notnull()]
    logging.info(
        f"Removed samples from dataset: {(num_samples - len(df.index)) / num_samples * 100:.3f}%"
    )

    return df


def re_assign_locations(h: pd.Series, parents_dict: dict) -> pd.Series:
    # take finest original location and replace parents with locations from parents_dict
    def _re_assign(locs):
        if locs is None or len(locs) == 0:
            return None

        finest = locs[0]

        for l in locs:
            if l in parents_dict:
                parents = parents_dict[l]
                parents.insert(0, l)
                return parents
            else:
                logging.error(
                    f"finest: {finest} not in parents dict, try next parent {l}"
                )
        return None

    return h.apply(lambda x: _re_assign(x))


def __remove_cycle(s: pd.Series):
    def _remove(locs):
        filtered = []
        if locs is None:
            return None
        last = None
        for i in range(len(locs)):
            if last == locs[i]:
                continue
            last = locs[i]
            filtered.append(last)
        return filtered

    return s.apply(lambda x: _remove(x))


def remove_cycle_parallel(s: pd.Series, num_worker=8):
    splits = np.array_split(s, num_worker)
    with Pool(num_worker) as p:
        filtered = pd.concat(p.map(__remove_cycle, splits))
    return filtered


def re_count_locations(h: pd.Series) -> pd.DataFrame:
    counts = defaultdict(int)
    for locs in h:
        if locs is None:
            counts[None] += 1
        else:
            for l in locs:
                counts[l] += 1
    counts_df = pd.DataFrame.from_dict(
        dict(counts), orient="index", columns=["counts_replaced"]
    )
    return counts_df


def assign(t, valid, parents_dict: dict):
    if t is None:
        return None
    if t in valid:
        return t
    for l in parents_dict[t]:
        if l in valid:
            return l
    return None
