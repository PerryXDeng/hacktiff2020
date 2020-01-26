import os
from PIL import Image
import json
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from sys import platform

def get_image_size(img_path):
    return Image.open(img_path).size


def image_size_compliant(img_path, size_cutoff):
    w, h = get_image_size(img_path)
    return w < size_cutoff and h < size_cutoff


def filter_imagepaths(imagepaths, size_cutoff): 
    if size_cutoff is None:
        return imagepaths
    imagepaths = [path for path in imagepaths if image_size_compliant(path, size_cutoff)]
    return imagepaths


def getjsons(path):
    im_type = ""
    root = []
    if platform == "win32":
        symbol = "\\"
    else:
        symbol = "/"
    if "nadir" in path:
        root = path.split(symbol + "nadirs")
        #print(path.split("/nadir"))
        im_type = "nadir"
    elif "oblique" in path:
        root = path.split(symbol + "obliques")
        #print(path.split("/obliques"))
        im_type = "obliques"
<<<<<<< HEAD
    parts = root[0].split('/')
    packageid = parts[len(parts)-1]
    geojson = root[0]+"/"+packageid+".geojson"
=======
    parts = root[0].split(symbol);
    geojson = root[0]+symbol+parts[len(parts)-1]+".geojson"
>>>>>>> aa44cabbeefa8a1397e9d38ef47ae46d80096258
    json = path[:-3]+"json"
    return geojson, json, im_type, packageid


def get_obliqueness(projection_matrix):
    """Return a value (roughly between 0 and 1) of how oblique the given projection matrix's view is."""
    return np.degrees(np.arccos(-projection_matrix[2, 2])) / 60


### features from roof GeoJSON data ###

def _get_roof_polygons(geojson_path):
    with open(geojson_path) as f:
        geojson = json.load(f)
    roof_polygons = []
    for feature in geojson['features']:
        geometry = feature['geometry']
        assert geometry['type'] == 'Polygon', f"Unexpected geometry type: {geometry['type']!r}"
        [points] = geometry['coordinates']
        roof_polygons.append(np.array(points[::2], dtype=float))
    return roof_polygons

from pyproj import CRS, Transformer
from evtech.geodesy import utm_crs_from_latlon
_transformers = {}
def get_transformer_for_crs(crs):
    if crs not in _transformers:
        _transformers[crs] = Transformer.from_crs(CRS.from_user_input(4326), crs, always_xy=True)
    return _transformers[crs]
def get_transformer_for_lat_lon(lat, lon):
    return get_transformer_for_crs(utm_crs_from_latlon(lat, lon))

def _get_building_outlines(geojson_path):
    roof_polygons = _get_roof_polygons(geojson_path)
    
    lon, lat, _ = roof_polygons[0][0]
    transformer = get_transformer_for_lat_lon(lat, lon)
    def points_to_2D(points):
        return np.stack(transformer.transform(*points.T))[:2].T
    
    polygons = []
    for poly in roof_polygons:
        points = points_to_2D(poly)
        points *= 3.28084 # meters to feet
        
        try:
            polygons.append(Polygon(points).buffer(0))
        except ValueError as e:
            print(e)
    
    union = unary_union(polygons)
    if isinstance(union, MultiPolygon):
        return list(union.geoms)
    else:
        return [union]

def _get_building_outline(geojson_path):
    """Return the building outline with the largest area."""
    parts = _get_building_outlines(geojson_path)
    return max(parts, key=lambda polygon: polygon.area)

def get_building_features(geojson_path):
    outline = _get_building_outline(geojson_path)
    rect = outline.minimum_rotated_rectangle
    p1, p2, p3, p4, _ = rect.exterior.coords
    width = np.hypot(p2[0]-p1[0], p2[1]-p1[1])
    length = np.hypot(p2[0]-p3[0], p2[1]-p3[1])
    if length < width:
        width, length = length, width
    return {
        'outline_area': outline.area,
        'outline_perimeter': outline.length,
        'bounding_area': rect.area,
        'bounding_perimeter': rect.length,
        'length': length,
        'width': width,
    }
