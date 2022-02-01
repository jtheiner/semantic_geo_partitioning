import geopandas as gpd
import branca.colormap
import folium


GDF_COLUMN_SELECT_OSM_NOMINATIM = [
    "id",
    "name_en",
    "localname",
    "category",
    "type",
    "admin_level",
    "rank_address",
    "isarea",
]
KEYADDRESS_LEVEL_OSM = "rank_address"
COLORS_OSM_RANKING = {
    "Continent": ["#eeeeee", "#eeeeee"],  # 2-3
    "Country": ["#3182bd", "#6baed6", "#9ecae1", "#c6dbef"],  # 4-7 blues
    "State": ["#e6550d", "#fd8d3c"],  # 8-9 orange
    "Region": ["#31a354", "#74c476"],  # 10-11 green
    "County": ["#756bb1", "#9e9ac8", "#bcbddc", "#dadaeb"],  # 12-15 purple
    "City": ["#0082C8"],  # 16
    "Island, town, moor, waterways": ["#ffcb32"],  # 17
    "Village, hamlet, municipality, district, borough, airport, national park": [
        "#636363",
        "#969696",
    ],  # 18, 19
    "Suburb, croft, subdivision, farm, locality, islet": [
        "#bdbdbd",
        "#d9d9d9",
    ],  # 20, 21
    "Building": 8 * ["#555555"],  # 22-30
}


#%%
file_locations_meta = "../data/semantic_partitioning/reverse_geocoding/locations_meta_mp16_train-min_50.feather"
gdf_meta = gpd.read_feather(file_locations_meta)
gdf_vis = gdf_meta[GDF_COLUMN_SELECT_OSM_NOMINATIM + ["geometry"]]

# subset selection
# gdf_vis = gdf_vis.loc[gdf_vis["category"] == "highway"]
# gdf_vis = gdf_vis.loc[gdf_vis["isarea"]]
# gdf_meta = gdf_meta.sample(frac=0.1)

#%%

colors = [item for sublist in COLORS_OSM_RANKING.values() for item in sublist]
colormap = branca.colormap.StepColormap(
    colors, vmin=0, vmax=30, index=list(range(2, 30))
)
colormap.caption = KEYADDRESS_LEVEL_OSM


m = folium.Map(height=1080, zoom_start=4, tiles="OpenStreetMap")
m.add_child(colormap)

feature_groups = []
for level in sorted(gdf_vis[KEYADDRESS_LEVEL_OSM].unique()):
    gdf_vis_at_level = gdf_vis.loc[gdf_vis[KEYADDRESS_LEVEL_OSM] == level]

    feature_group = folium.FeatureGroup(name=str(level), show=False)

    marker = folium.CircleMarker()
    choropleth_layer = folium.GeoJson(
        gdf_vis_at_level.to_json(),
        style_function=lambda feature: {
            "fillColor": colormap(feature["properties"][KEYADDRESS_LEVEL_OSM]),
            "color": "#555",
            "weight": 2,
            "fillOpacity": 0.45,
            "lineOpacity": 0.1,
        },
        tooltip=folium.features.GeoJsonTooltip(
            fields=GDF_COLUMN_SELECT_OSM_NOMINATIM, labels=True, sticky=False
        ),
        highlight_function=lambda _: {"weight": 5, "color": "black"},
        marker=marker,
    ).add_to(feature_group)
    feature_groups.append(feature_group)


for feature_group in feature_groups:
    feature_group.add_to(m)
folium.LayerControl(position="topright", collapsed=False).add_to(m)

#%%
# m.save(file_locations_meta.parent / f"{file_locations_meta.stem}.html")
m
