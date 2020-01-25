import evtech
from pathlib import Path
import json
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from typing import NamedTuple, List


class Package(NamedTuple):
    """Data for a single "package" (building)."""
    id: str
    nadirs: List[evtech.camera.Camera]
    obliques: List[evtech.camera.Camera]
    roof_polygons: List[np.ndarray]

data_dir = Path('./comp_data/')


def get_package_ids() -> List[str]:
    """Return a list of all the package IDs in the dataset."""
    return [package.name for package in data_dir.iterdir()]


def load_package(id: str) -> Package:
    """Load and return the Package with the given ID."""
    package_dir = data_dir / id
    
    # load the nadir/oblique cameras
    nadirs, obliques = evtech.load_dataset(package_dir)
    
    # load the GeoJSON polygons
    [geojson_path] = package_dir.glob('*.geojson')
    with geojson_path.open() as f:
        geojson = json.load(f)
    roof_polygons = []
    for feature in geojson['features']:
        geometry = feature['geometry']
        assert geometry['type'] == 'Polygon', f"Unexpected geometry type: {geometry['type']!r}"
        [points] = geometry['coordinates']
        roof_polygons.append(np.array(points[::2], dtype=float))
    
    # return a Package object
    return Package(id, nadirs, obliques, roof_polygons)



from pyproj import CRS, Transformer
def project_to_camera(camera, world_points):
    # Convert lat/lon/elev to camera CRS
    if not hasattr(camera, 'transformer'):
        camera.transformer = Transformer.from_crs(CRS.from_user_input(4326), camera.crs, always_xy=True)
    x,y,z = camera.transformer.transform(*world_points.T)
    pt = np.stack((x, y, z, np.ones(len(world_points))))

    # Do projection
    img_pt_h = camera.projection_matrix @ pt
    img_pt = img_pt_h[:2] / img_pt_h[2]

    # Offset pixel by bounds
    img_pt[0] -= camera.image_bounds[0]
    img_pt[1] -= camera.image_bounds[1]
    return img_pt.T

def draw_polygon(image, camera, world_points, color, thickness=2, offset=[0,0,3]):
    world_points = world_points.copy()
    world_points += offset
    world_points[:, 2] += camera.get_elevation()
    image_points = project_to_camera(camera, world_points)
    image_points = np.array([image_points])
    
    cv2.polylines(image, np.int32(image_points), isClosed=True, color=color, thickness=thickness)

def show_images(images, cols = 1, titles = None):
    """Display a list of images in a single figure with matplotlib.
    
    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.
    
    cols (Default = 1): Number of columns in figure (number of rows is 
                        set to np.ceil(n_images/float(cols))).
    
    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        # a.set_title(title)
        a.axis('off')
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99, wspace=0.01, hspace=0.01)
    plt.show()

def get_image_sizes(package_ids):
    print("Loading image sizes...")
    image_sizes = []
    for n, package_id in enumerate(package_ids):
        print(f"    Package {n+1}/{len(package_ids)}", end='\r')
        for img_path in (data_dir/package_id).glob('**/*.jpg'):
            size = Image.open(img_path).size
            image_sizes.append(size)
    print()
    return np.array(image_sizes)

def plot_image_sizes(package_ids):
    image_sizes = get_image_sizes(package_ids)
    
    print(f"Plotting the sizes of {len(image_sizes)} images...")
    plt.title('Image sizes')
    plt.scatter(*image_sizes.T, alpha=0.2)
    plt.xlabel('width')
    plt.ylabel('height')
    plt.gca().set_xlim(left=0)
    plt.gca().set_ylim(bottom=0)
    plt.show()

def visualize_package(package_id):
    print("Visualizing package ID:", package_id)
    
    package = load_package(package_id)
    cameras = package.obliques + package.nadirs
    colors = np.empty((len(package.roof_polygons), 3), dtype=np.uint8)
    colors[:, 0] = np.linspace(0, 180, len(colors)+1, dtype=np.uint8)[:-1]
    colors[:, 1] = 255
    colors[:, 2] = 200
    cv2.cvtColor(colors[None], cv2.COLOR_HSV2RGB, colors[None])
    colors = [tuple(int(v) for v in color) for color in colors]
    
    print("Drawing images...")
    annotated_images = []
    for camera in cameras[:4]:
        image = camera.load_image()
        """
        image_orig = image
        image = cv2.medianBlur(image, 5)
        meanShifted = cv2.pyrMeanShiftFiltering(image, 7, 10)
        image = meanShifted
        
        edges = cv2.Canny(image, 40, 100, apertureSize=3)
        plt.subplot(131)
        plt.imshow(image_orig)
        plt.axis('off')
        plt.subplot(132)
        plt.imshow(image)
        plt.axis('off')
        plt.subplot(133)
        plt.imshow(edges)
        plt.axis('off')
        plt.show()
        #"""
        
        for polygon, color in zip(package.roof_polygons, colors):
            draw_polygon(image, camera, polygon, color, thickness=2)#, offset=[0.000001,-0.000008,3])
        annotated_images.append(image)
    
    print("Showing images...")
    show_images(annotated_images, cols=2)

def process_roof(package_id):
    print("Processing package:", package_id)
    package = load_package(package_id)
    if not package.nadirs:
        return
    
    camera = package.nadirs[0]
    transformer = Transformer.from_crs(CRS.from_user_input(4326), camera.crs, always_xy=True)
    def points_to_2D(points):
        return np.stack(transformer.transform(*points.T))[:2].T
    
    all_points = points_to_2D(np.concatenate(package.roof_polygons))
    min_coords = all_points.min(axis=0)
    max_coord = (all_points - min_coords).max()
    
    if not any(max(Image.open(img_path).size) >= 697 for img_path in (data_dir/package_id).glob('**/*.jpg')):
        return
    
    plt.subplot(121)
    plt.imshow(camera.load_image()[:,:,::-1])
    
    plt.subplot(122)
    for poly in package.roof_polygons:
            # points = project_to_camera(camera, poly)
            points = points_to_2D(poly)
            points -= min_coords
            points *= 3.28084
            plt.plot(*np.concatenate((points, points[None,0])).T)
    plt.axis('scaled')
    
    plt.get_current_fig_manager().full_screen_toggle()
    plt.show()

def get_camera_vector(camera):
    cam_mat, rot_mat, trans_vec, rot_mat_x, rot_mat_y, rot_mat_z, euler_angles = cv2.decomposeProjectionMatrix(camera.projection_matrix)
    a, b, c = euler_angles.squeeze(axis=1)
    
    plt.title(f'{a:.1f} {b:.1f} {c:.1f}')
    plt.imshow(camera.load_image())
    plt.show()

def get_camera_vectors(package_id):
    package = load_package(package_id)
    for camera in package.obliques:
        get_camera_vector(camera)

def main(args):
    if args.package_ids:
        package_ids = args.package_ids
    else:
        package_ids = get_package_ids()
        package_ids.sort()
    
    if args.image_sizes:
        plot_image_sizes(package_ids)
    elif args.roof:
        for package_id in package_ids:
            process_roof(package_id)
    elif args.cam_vecs:
        for package_id in package_ids:
            get_camera_vectors(package_id)
    else:
        for package_id in package_ids:
            visualize_package(package_id)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--image-sizes', action='store_true')
    parser.add_argument('--roof', action='store_true')
    parser.add_argument('--cam-vecs', action='store_true')
    parser.add_argument('package_ids', nargs='*')
    main(parser.parse_args())
