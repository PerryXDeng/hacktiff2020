import numpy as np
import os
import torch
from torch.utils.data import Dataset, Dataloader
from skimage import io
import glob
from .utils import getjsons, get_image_size, image_size_compliant

class SingleImageLabeledDataset(Dataset):
  def __init__(self, data_dir, packages_paths_filepath,
               transform=None, size_cutoff=None):
    packages = np.loadtxt(packages_paths_filepath, dtype=int)
    package_paths = [os.path.join(data_dir, package) for package in packages]
    package_imagepaths = [glob.glob(os.path.join(package_path, "*/*.jpg"))
                          for package_path in package_paths]
    self.imagepaths = []
    for package in package_imagepaths:
      for img_path in package:
        if size_cutoff is None:
          self.imagepaths.append(img_path)
        elif image_size_compliant(img_path, size_cutoff):
          self.imagepaths.append(img_path)
    self.transform = transform
    self.package_measurements = {}
    self.package_3d = {}

  def __len__(self):
    return len(self.filepaths)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    img_path = self.imagepaths[idx]
    image = io.imread(img_path)
    
    geojson_path, imgjson_path, im_type = getjsons(img_path)
    metadata = None
    measurements = None
    
    observation = {'image': image, 'metadata': metadata,
                   'measurements': measurements}
    if self.transform is not None:
      observation = self.transform(observation)
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

