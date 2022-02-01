import logging
from pathlib import Path
from typing import Union
from multiprocessing import Pool
from functools import partial
from urllib3 import Retry
from requests.adapters import HTTPAdapter

import requests
import pandas as pd
import geopandas as gpd
import shapely.geometry
import tqdm


log = logging.getLogger("GeoCoder")


def init_input(
    input_csv: Path,
    col_id: str = "image_id",
    col_lat: str = "latitude",
    col_lng: str = "longitude",
    filter_by_region: Union[None, bool] = None,
    nrows: Union[None, int] = None,
    filter_key=None,
    filter_value=None,
) -> pd.DataFrame:

    log.info(f"Read: {input_csv}")
    df = pd.read_csv(input_csv, usecols=[col_id, col_lat, col_lng], nrows=nrows)
    df = df.rename(columns={col_id: "id", col_lat: "latitude", col_lng: "longitude"})

    log.info(f"Prepare DataFrame to filter by region")
    """
    Read coordinates from CSV and filter for one region
    """

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.longitude, df.latitude),
        crs={"init": "epsg:4326"},
    )
    log.info(f"DataFrame size: {len(gdf.index)}")

    if filter_by_region is not None:
        log.info(f"Filtering dataset by region name: {filter_key}:{filter_value}")
        raise NotImplementedError

    gdf = gdf[["id", "latitude", "longitude", "geometry"]]
    log.info("Sort by coordinates to speedup reverse geocoding queries...")
    gdf = gdf.sort_values(by=["longitude", "latitude"])

    return pd.DataFrame(gdf.drop(columns="geometry"))


class NominatimCoder:
    def __init__(
        self,
        domain="localhost",
        port=8081,
        worker=4,
    ):

        self.domain = domain
        self.port = port
        self.worker = worker

        self.session = requests.Session()
        adapter = HTTPAdapter(max_retries=Retry(connect=3, backoff_factor=0.5))
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def query_details(self, place_id: int) -> Union[dict, None]:
        query = f"place_id={place_id}&format=json&polygon_geojson=1"

        # r = requests.get(f"http://{self.domain}:{self.port}/details.php?{query}")
        r = self.session.get(f"http://{self.domain}:{self.port}/details.php?{query}")

        if r.status_code == 200:
            r = r.json()
            if "error" in r:
                log.warning(r)
                return None
            return r
        else:
            log.error(f"{r.status_code} for {place_id} - {query}")
            return None

    def query_details_osmid(self, osm_id: str):
        # https://nominatim.org/release-docs/3.4.2/api/Details/
        query = f"osmtype={osm_id[0]}&osmid={osm_id[1:]}&format=json&polygon_geojson=1"
        url = f"http://{self.domain}:{self.port}/details.php?{query}"
        # r = requests.get(url)
        r = self.session.get(url)

        if r.status_code == 200:
            r = r.json()
            if "error" in r:
                log.warning(r)
                return None
            return r
        else:
            log.error(f"{r.status_code} for {osm_id}, {url}")
            return None

    def query_details_hierarchy(self, location: dict) -> Union[dict, None]:
        """Query details, i.e. the extended hierarchy of a given location

        See: https://nominatim.org/release-docs/develop/api/Details/
        See https://nominatim.org/release-docs/develop/api/Output for output format
        """
        if location is None:
            return None
        if "place_id" not in location:
            if location is None:
                log.warning(f"place_id not in {location}: ignore")
            else:
                log.warning(f"invalid location: {location}")
            return None

        # query details by osm_type and osm_id (place_id only valid for local installation and osm_id is not unique)
        osm_type = location["osm_type"][0].upper()
        osm_id = location["osm_id"]
        query_string = f"osmtype={osm_type}&osmid={osm_id}&addressdetails=1&format=json"

        # r = requests.get(f"http://{self.domain}:{self.port}/details.php?{query_string}")
        r = self.session.get(
            f"http://{self.domain}:{self.port}/details.php?{query_string}"
        )

        if r.status_code == 200:
            r = r.json()
            if "error" in r:
                log.error(f"{r} - {query_string}")
                return None
            return r
        else:
            log.warning(f"{r.status_code} for {location}")
            return None

    def query_reverse(self, p, zoom=18):

        query = (
            f"lat={p[0]:.8f}&lon={p[1]:.8f}&zoom={zoom}&addressdetails=0&format=json"
        )

        r = self.session.get(f"http://{self.domain}:{self.port}/reverse.php?{query}")
        # r = requests.get(f"http://{self.domain}:{self.port}/reverse.php?{query}")
        if r.status_code == 200:
            r = r.json()
            if "error" in r:
                log.error(f"{p}: {r} - {query}")
                return None
            return r
        log.error(f"{r.status_code} for {query}")
        return None

    def query_forward(self, localname, key_help="country"):
        query = f"{key_help}={localname}&limit=1&format=json"

        # r = requests.get(f"http://{self.domain}:{self.port}/search.php?{query}")
        r = self.session.get(f"http://{self.domain}:{self.port}/search.php?{query}")
        if r.status_code == 200:
            r = r.json()
            if "error" in r:
                log.error(f"{localname} {r} - {query}")
                return None
            return r
        log.error(f"{r.status_code}")
        return None


def initial_reverse_geocode(df, geocoder: NominatimCoder, zoom=18, disable_tqdm=False):

    # raw reverse geo coding i.e. reverse.php?lat=lat&lon=lon&zoom=zoom
    latlngs = list(zip(df.latitude, df.longitude))
    log.info("Starting initial reverse geo coding")
    with Pool(geocoder.worker) as p:
        query_reverse_func = partial(geocoder.query_reverse, zoom=zoom)
        data = list(
            tqdm.tqdm(
                p.imap(query_reverse_func, latlngs),
                total=len(latlngs),
                miniters=1000,
                disable=disable_tqdm,
            )
        )
    return data


def query_address_vector(raw_data, geocoder, disable_tqdm=False):
    # get details for hierarchy i.e. details.php/osmtype=osmtype&osmid=osmid&addressdetails=1
    log.info("Querying details to access address vector")
    with Pool(geocoder.worker) as p:
        query_details_fnc = partial(geocoder.query_details_hierarchy)
        data_processed = list(
            tqdm.tqdm(
                p.imap(query_details_fnc, raw_data),
                total=len(raw_data),
                miniters=1000,
                disable=disable_tqdm,
            )
        )
    return data_processed


def add_to_df(data_processed, df):
    # fix Nominatim issue: first element (queried place_id) misses values for osm_id and osm_type and admin_level
    log.info("Fixing issue for missing osm_id for queried location...")
    for i, element in enumerate(data_processed):
        if element is None:
            continue
        for j, h in enumerate(element["address"]):
            if h["place_id"] == element["place_id"]:
                if (
                    h["osm_id"] is None
                    or h["osm_type"] is None
                    or h["admin_level"] is None
                ):
                    h["osm_id"] = element["osm_id"]
                    h["osm_type"] = element["osm_type"]
                    h["admin_level"] = element["admin_level"]
                    data_processed[i]["address"][j] = h

    df["h_raw"] = data_processed
    return df


def fix_countries(df, geocoder):

    log.info(
        "Starting geo coding for country localname to fill missing osm_id and osm_type"
    )
    cache = {}
    # get unique country names
    unique_countries = {}
    for c in df["h_raw"]:
        if c is None:
            continue
        for h in c["address"]:
            if h["type"] == "country":
                if h["localname"] not in unique_countries:
                    unique_countries[h["localname"]] = h

    # query needed for osm_id
    for localname, h in tqdm.tqdm(unique_countries.items()):
        if localname in cache:
            log.info("Load country from cache")
            country_details = cache[localname]
            if country_details is None:
                continue
        else:
            country_details = geocoder.query_forward(localname, key_help="country")
            if country_details is None:
                cache[localname] = None
                continue
            if len(country_details) == 0:
                log.error(f"cannot geocode at country level: {localname}")
                cache[localname] = None
                continue
            country_details = country_details[0]  # limited query to one result

        unique_countries[localname]["osm_id"] = country_details["osm_id"]
        unique_countries[localname]["osm_type"] = country_details["osm_type"][0].upper()
        unique_countries[localname]["place_id"] = country_details["place_id"]

        cache[localname] = unique_countries[localname]

        # query needed for admin_level
        if "admin_level" not in country_details:
            admin_level_details = geocoder.query_details(country_details["place_id"])
            if admin_level_details is None:
                continue

            unique_countries[localname]["admin_level"] = admin_level_details[
                "admin_level"
            ]
            cache[localname]["admin_level"] = admin_level_details["admin_level"]

    # add missing information to dataset
    log.info("Add missing country information to dataset")
    for i, raw_results in enumerate(df["h_raw"]):
        if raw_results is None:
            continue
        for j, h in enumerate(raw_results["address"]):
            if h["type"] == "country":
                if h["localname"] in unique_countries:
                    # store changes
                    df["h_raw"].iloc[i]["address"][j] = unique_countries[h["localname"]]
    return df


def keep_location_ids_only(
    df,
    id_keys=["osm_type", "osm_id"],
):
    def _minimize(raw_response):
        if raw_response is None:
            return None
        r = []
        for h in raw_response["address"]:
            r.append("".join([str(h[k]) for k in id_keys]))
        return r

    log.info("Minimize hierarchy")
    df["h_raw"] = df["h_raw"].apply(lambda x: _minimize(x))
    return df


def _shape_convert(x: dict):
    if x is None:
        return None
    if not isinstance(x, dict):
        log.warning(x)
        return None
    return shapely.geometry.shape(x)


def _replace_empty_dict(x):
    if len(x) > 0:
        return x
    return {}


def format_locations(nodes, num_worker=4, simplify_geometry_tolerance=None):
    # convert to GeoDataFrame and postprocess
    df_locations = pd.DataFrame.from_dict(nodes, orient="index")
    with Pool(num_worker) as p:
        df_locations.geometry = p.map(_shape_convert, df_locations.geometry.tolist())
    with Pool(num_worker) as p:
        df_locations.extratags = p.map(
            _replace_empty_dict, df_locations.extratags.tolist()
        )
    gdf = gpd.GeoDataFrame(df_locations)

    with Pool(num_worker) as p:
        centers = p.map(shapely.geometry.shape, gdf["centroid"].tolist())
    gdf["latitude_center"] = [c.y for c in centers]
    gdf["longitude_center"] = [c.x for c in centers]
    gdf.drop(
        columns=[
            "place_id",
            "parent_place_id",
            "addresstags",
            "names",
            "icon",
            "linked_places",
            "housenumber",
            "calculated_postcode",
        ],
        inplace=True,
        errors="ignore",
    )
    
    if simplify_geometry_tolerance:
        log.info(f"Simplify geometry with tolerance: {simplify_geometry_tolerance}")
        gdf.geometry = gdf.simplify(tolerance=simplify_geometry_tolerance)
    gdf["id"] = gdf.index
    return gdf


def get_locations(df, geocoder):
    """Extract all unique locations independent of the hierarchy level.
    E.g. a valid id_key is 'place_id' or 'osm_id' with 'osm_type'
    See https://wiki.openstreetmap.org/wiki/Nominatim/Development_overview for details about keys and meaning

    """
    log.info("Count locations...")

    nodes = df.explode("h")["h"].unique().tolist()
    log.info(f"Number of locations: {len(nodes)}")

    # enrich information
    cache = {}
    log.info("Add additional information for each location...")
    for node in tqdm.tqdm(nodes):

        if node in cache:
            continue

        meta = geocoder.query_details_osmid(node)
        cache[node] = meta.copy()

        if meta is None:
            continue

        for k, attribute in meta.items():
            if k == "names":
                if "name:en" in attribute:
                    # only english name or localname is required (no need to store around 50 names)
                    cache[node]["name_en"] = meta["names"]["name:en"]
                else:
                    # fallback to localname
                    cache[node]["name_en"] = meta["localname"]
            else:
                cache[node][k] = attribute

    return cache
