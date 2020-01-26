import numpy as np
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import glob
from utils import getjsons, get_image_size, image_size_compliant, filter_imagepaths
from tqdm import tqdm

class SingleImageLabeledDataset(Dataset):
  def __init__(self, data_dir, packages_paths_filepath,
               transform=None, size_cutoff=None):
    self.size_cutoff = size_cutoff
    packages = np.loadtxt(packages_paths_filepath, dtype=int)
    package_paths = [os.path.join(data_dir, str(package)) for package in packages]
    print("filtering images")
    package_imagepaths = [filter_imagepaths(
                          glob.glob(os.path.join(package_path, "*/*.jpg")),
                          size_cutoff)
                          for package_path in tqdm(package_paths)]
    self.imagepaths = [imagepath for imagepaths in package_imagepaths for imagepath in imagepaths]
    self.imagepaths = [path for path in self.imagepaths if "oblique" not in path]
    print("done with filtering")
    self.transform = transform
    self.package_measurements = {}
    self.package_3d = {}

  def __len__(self):
    return len(self.imagepaths)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    img_path = self.imagepaths[idx]
    image = Image.open(img_path)
    
    geojson_path, imgjson_path, im_type = getjsons(img_path)
    metadata = torch.randn(1)
    measurements = torch.randn(1)
    
    if self.transform is not None:
      image = self.transform(image)

    observation = {'image': image, 'metadata': metadata,
                   'measurements': measurements, 'img_path': img_path}
    return observation

class MultiImageLabeledDataset(Dataset):
  def __init__(self, data_dir, packages_paths_filepath, transform=None):
    self.packages = np.loadtxt(packages_paths_filepath, dtype=int)
    self.transform = transform

  def __len__(self):
    return len(self.filepaths)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    image = io.imread(img_name)
    metadata = None
    measurements = None
    observation = {'image': image, 'metadata': metadata,
                   'measurements': measurements}
    if self.transform is not None:
      observation = self.transform(observation)
    return observation

