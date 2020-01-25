import os
from PIL import Image

def get_image_size(img_path):
    return Image.open(img_path).size


def image_size_compliant(img_path, size_cutoff):
    w, h = get_image_size(img_path)
    return w < size_cutoff and h < size_cutoff


def getjsons(path):
    im_type = ""
    root = []
    if "/nadir" in path:
        root = path.split("/nadir")
        #print(path.split("/nadir"))
        im_type = "nadir"
    elif "/oblique" in path:
        root = path.split("/obliques")
        #print(path.split("/obliques"))
        im_type = "obliques"
    parts = root[0].split('/');
    geojson = root[0]+"/"+parts[len(parts)-1]+".geojson"
    json = path[:-3]+"json"
    return geojson, json, im_type

