import pickle
from argparse import ArgumentParser, Namespace
import logging
from pathlib import Path
import warnings

import pandas as pd

warnings.filterwarnings("ignore", message=".*initial implementation of Parquet.*")
log = logging.getLogger("GeoCoder")
logging.basicConfig(level=logging.INFO)

from utils.json_io import df2jsonl
from utils.geocoding import (
    NominatimCoder,
    fix_countries,
    get_locations,
    init_input,
    keep_location_ids_only,
    initial_reverse_geocode,
    query_address_vector,
    add_to_df,
    format_locations,
)


def _save_to_cache(cache_file: Path, data):
    log.info(f"Dump pickable object to file: {cache_file}")
    with open(cache_file, "wb") as f:
        pickle.dump(data, f)


def _load_from_cache(cache_file: Path):

    with open(cache_file, "rb") as f:
        log.info(f"Read rows from file: {cache_file}")
        data = pickle.load(f)
    return data


args = ArgumentParser()
args.add_argument(
    "--input_csv",
    type=Path,
    required=True,
    help="CSV with columns 'image_id', 'latitude' and 'longitude'",
)
args.add_argument(
    "--min_num_loc_total",
    type=int,
    help="When provided, keep only locations wuith at least n images",
)
args.add_argument(
    "--file_locations_filter",
    type=Path,
    help="When provided, filter dataset by given list of locations",
)
args.add_argument(
    "--output_directory",
    type=Path,
    default=Path("../data/semantic_partitioning/reverse_geocoding"),
)

args.add_argument("--host", type=str, default="localhost")
args.add_argument("--port", type=str, default="8081")
args.add_argument("--worker", default=8)
args.add_argument(
    "--simplify_geometry_tolerance",
    type=float,
    default=0.001,
)
args.add_argument("--disable_tqdm", action="store_true", default=False)
args.add_argument(
    "--nrows",
    type=int,
)
args = args.parse_args()


# DEBUG Ipython
# args = Namespace()
# args.input_csv = Path("../data/images/mp16/meta_mp16_train.csv")
# args.output_directory = Path("../data/semantic_partitioning/reverse_geocoding")
# args.nrows = None
# args.host = "localhost"
# args.port = "8081"
# args.worker = 16
# args.simplify_geometry_tolerance = 0.001
# args.disable_tqdm = False
# args.min_num_loc_total = 50  # None to use args.file_locations_filter
# args.file_locations_filter = None  # Path or None

dataset_out_prefix = args.input_csv.stem
if args.min_num_loc_total:
    dataset_out_prefix += f"-min_{args.min_num_loc_total}"
if args.nrows:
    dataset_out_prefix += f"-nrows_{args.nrows}"
file_dataset_out = args.output_directory / f"{dataset_out_prefix}.jsonl.bz2"
file_locations_out = args.output_directory / f"locations_{dataset_out_prefix}.feather"

# read dataset of latitudes and longitudes and eventually apply geo-filtering
df = init_input(
    args.input_csv,
    "image_id",
    "latitude",
    "longitude",
    filter_by_region=None,
    filter_key=None,
    filter_value=None,
    nrows=args.nrows,
)

geocoder = NominatimCoder(args.host, args.port, args.worker)


cache_dir = args.output_directory / "cache"
cache_dir.mkdir(exist_ok=True, parents=True)

file_cache = cache_dir / f"{dataset_out_prefix}_raw_reverse_initial_reverse_geocode.pkl"
if file_cache.exists():
    raw_data = _load_from_cache(file_cache)
else:
    raw_data = initial_reverse_geocode(df, geocoder, disable_tqdm=args.disable_tqdm)
    _save_to_cache(file_cache, raw_data)


file_cache = cache_dir / f"{dataset_out_prefix}_query_address_vector.pkl"
if file_cache.exists():
    raw_data = _load_from_cache(file_cache)
else:
    raw_data = query_address_vector(raw_data, geocoder, disable_tqdm=args.disable_tqdm)
    _save_to_cache(file_cache, raw_data)


df = add_to_df(raw_data, df)
# add missing information for countries
df = fix_countries(df, geocoder)


#%%
df_out = df.copy(deep=True)


df_out["h_raw"] = df_out["h_raw"].apply(
    lambda x: x["address"] if (isinstance(x, dict) and "address" in x) else None
)
# remove samples with unsucessfull geocoding
print(df_out[["h_raw"]].isna().value_counts())
df_out.dropna(subset=["h_raw"], inplace=True)
print(df_out.agg({"h_raw": len}).value_counts())


def keep_location_ids_only(
    df,
    id_keys=["osm_type", "osm_id"],
):
    def _minimize(raw_response):

        r = []
        for h in raw_response:
            r.append("".join([str(h[k]) for k in id_keys]))
        return r

    log.info("Minimize hierarchy")
    df["h_raw"] = df["h_raw"].apply(lambda x: _minimize(x))

    return df


df_out = keep_location_ids_only(df_out)
df_out["h_raw"] = df_out["h_raw"].apply(
    lambda h_raw: [x for x in h_raw if "None" not in x]
)


#%%

keep_locs = None
if args.min_num_loc_total:
    log.info(f"Remove locations that occur sporadically: {args.min_num_loc_total}")
    locations_count = df_out.explode("h_raw")["h_raw"].value_counts()
    log.info(f"Number of unique locations: {len(locations_count)}")
    keep_locs = locations_count.loc[locations_count >= args.min_num_loc_total].index
    log.info(
        f"Number of unique locations after applying filtering by distribution: {len(keep_locs)}"
    )


elif args.file_locations_filter:
    log.info(
        f"Remove locations that does not exist in: {args.file_locations_filter.name}"
    )
    # from file, e.g. from training dataset
    # load only osmids, e.g. N20975578
    if "feather" not in args.file_locations_filter.name:
        raise NotImplementedError  # just need a set of location ids
    keep_locs = set(
        pd.read_feather(args.file_locations_filter, columns=["id"])["id"].values
    )

if keep_locs is not None:
    df_out["h"] = df_out["h_raw"].apply(
        lambda h: [idx for idx in h if idx in keep_locs] if h is not None else None
    )
    print(df_out.agg({"h": len}).value_counts())
    if args.min_num_loc_total:
        assert df_out.explode("h")["h"].value_counts().min() >= args.min_num_loc_total

size_initial = len(df_out.index)
log.info(f"Initial dataset size: {size_initial}")
df_out = df_out[(df_out["h"].notnull() & (df_out["h"].map(len) > 0))]
log.info(f"Size after removing Nones: {len(df_out.index)}")


log.info(f"Write dataset with address vector to file: {file_dataset_out}")
try:
    file_dataset_out.parent.mkdir(parents=True)
except FileExistsError as e:
    log.warn(f"Output directory already exists: {file_dataset_out.parent}")
log.info("Dump to file:")
df2jsonl(df_out, file_dataset_out, compression={"method": "bz2"})

# DEBUG
# file_dataset_out = args.output_directory / f"{dataset_out_prefix}.jsonl"
# df2jsonl(df_out, file_dataset_out)

# DEBUG
# df_out = pd.read_json(file_dataset_out, orient="records", lines=True)

if args.min_num_loc_total:
    # get additional information for each location
    nodes = get_locations(df_out, geocoder)

    gdf_locations = format_locations(
        nodes, args.worker, args.simplify_geometry_tolerance
    )
    log.info("Dump to file:")
    log.info(gdf_locations.head(2))
    log.info(f"Write location metadata with geometry to file: {file_locations_out}")
    gdf_locations.to_feather(file_locations_out, compression="lz4")


geocoder.session.close()
