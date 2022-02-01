
# Contents
- [Reverse Geocoding](#reverse-geocoding)
- [Extended MP-16 Dataset (EMP-16)](#extended-mp-16-dataset--emp-16-)
  * [Data Format](#data-format)
  * [Download](#download)
  * [Visualization](#visualization)
  * [Dataset Usage](#dataset-usage)
- [Semantic Partitioning Construction](#semantic-partitioning-construction)
  * [Create Base Hierarchy](#create-base-hierarchy)
  * [Create Single and Multi Partitioning](#create-single-and-multi-partitioning)

***

# Reverse Geocoding
Given a GPS coordinate anywhere on earth, we aim to find the nearest address by utilizing [Nominatim](https://nominatim.org/).
After a full Nominatim [https://nominatim.org/release-docs/latest/admin/Installation/](installation), we can the following script:

```sh
python reverse_geocoding.py --input_csv ../data/images/mp16/meta_mp16_train.csv --min_num_loc_total 50 --host <HOST> --port <PORT>
python reverse_geocoding.py --input_csv ../data/images/yfcc25600/meta_yfcc25600.csv --file_locations_filter ../data/semantic_partitioning/reverse_geocoding/locations_meta_mp16_train-min_50.feather --host <HOST> --port <PORT>
```
# Extended MP-16 Dataset (EMP-16)

For our work, we use Nominatim V3.4.2 as geocoder and a planet-scale OpenStreetMap (OSM) dump from Mai, 2020.
Source for reverse geocoding procedure is the MP-16 dataset comprising photos and respective GPS coordinates.
Additionally, we provide the output for validation set (YFCC-Val26k).

The dataset is avalable at: https://data.uni-hannover.de/dataset/extended-mp-16-dataset
The metadata (output of the reverse geocoding) of both datasets is licensed under the [Open Data Commons Open Database License (ODbL)](https://opendatacommons.org/licenses/odbl/) by the OpenStreetMap Foundation (OSMF).

## Data Format

The postproccessed output of the reverse geocoding for each sample (image):
```json
{"id":"54\/ff\/2588992263.jpg","latitude":-16.885703,"longitude":-179.998283,"h_raw":["N262907788","R4632581","R571747"],"h":["R4632581","R571747"]}
```
where `h` holds the filtered address vector `h_raw` as we only keep locations that occurs at least *min_num_loc_total* times. 
For The MP-16 training dataset it was set to 50. The locations for validation are filtered according to the available locations of the training set. 

Each unique location (joined *osm_type* and *osm_id*) contains additional metadata such as:
```json
{
    "id": "R4632581",
    "osm_type": "R",
    "osm_id": 4632581,
    "geometry": "<shapely.geometry.multipolygon.MultiPolygon>",
    "latitude_center": -16.3515846,
    "longitude_center": 179.048837131356,
    "category": "boundary",
    "type": "administrative",
    "admin_level": 4,
    "rank_address": 8,
    "isarea": true,
    "name_en": "Northern",
    "localname": "Northern",
    "country_code": "fj",
    "indexed_date": "2020-05-22T22:11:57+00:00",
    "importance": 0.29157890191013,
    "calculated_importance": 0.29157890191013,
    "extratags": {"<...>"},
}
```

## Download

```sh
mkdir -p ../data/semantic_partitioning/reverse_geocoding
cd ../data/semantic_partitioning/reverse_geocoding
wget https://data.uni-hannover.de/dataset/extended-mp-16-dataset/resource/1f7436c2-f3d2-4090-9c7d-9cf711edd7e3/download/meta_mp16_train-min_50.jsonl.bz2 -O meta_mp16_train-min_50.jsonl.bz2
wget https://data.uni-hannover.de/dataset/extended-mp-16-dataset/resource/02b4fcc5-62c4-429d-8c6f-014fb4e505ad/download/locations_meta_mp16_train-min_50.feather -O locations_meta_mp16_train-min_50.feather
wget https://data.uni-hannover.de/dataset/extended-mp-16-dataset/resource/689cc2fc-85b3-4b54-83aa-1b88d87890e4/download/meta_yfcc25600.jsonl.bz2 -O meta_yfcc25600.jsonl.bz2
```
Original photos can be directly downloaded from Flickr utilizing a script from [another repo](https://github.com/TIBHannover/GeoEstimation#Training-from-Scratch).


## Visualization
Raw dataset visualization on the world map: [inspect_dataset_folium.py](inspect_dataset_folium.py)


## Dataset Usage

```python
import pandas as pd
import geopandas as gpd
# one sample per line
df_dataset = pd.read_json("../data/semantic_partitioning/reverse_geocoding/meta_mp16_train-min_50.jsonl.bz2", orient="records", lines=True)
# metadata for each location including the geometry if available and its centroid
gdf_locations = gpd.read_feather("../data/semantic_partitioning/reverse_geocoding/locations_meta_mp16_train-min_50.feather)
```

# Semantic Partitioning Construction

## Create Base Hierarchy

```sh
# TODO
```

## Create Single and Multi Partitioning

```sh
# TODO
```
