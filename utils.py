import os
from PIL import Image

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


def get_obliqueness(projection_matrix):
    """Return a value (roughly between 0 and 1) of how oblique the given projection matrix's view is."""
    return np.degrees(np.arccos(-projection_matrix[2, 2])) / 60
