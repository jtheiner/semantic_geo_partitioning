import json
from functools import partial
from multiprocessing import Pool
import pandas as pd


def _parse_string2json(line, keep_keys=None):
    if keep_keys is not None:
        row = json.loads(line)
        row = {k: row[k] for k in row if k in keep_keys}
        return row
    return json.loads(line)


def df2jsonl(df, path, **kwargs):
    df.to_json(path, orient="records", lines=True, **kwargs)


def jsonl2df(path, index_col, keep_keys=None, nthreads=8, nrows=None):

    rows = []
    with open(path) as f:
        for i, line in enumerate(f.readlines()):
            if nrows is not None and i == nrows:
                break
            rows.append(line)

    _fnc_parse = partial(_parse_string2json, keep_keys=keep_keys)
    with Pool(nthreads) as p:
        rows = p.map(_fnc_parse, rows)
    df = pd.DataFrame(rows)
    df.set_index(index_col, inplace=True)
    return df


def dump(filename, data):
    with open(filename, "w") as fw:
        json.dump(data, fw)


def load(filename):
    with open(filename) as fr:
        return json.load(fr)
