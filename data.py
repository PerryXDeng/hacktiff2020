import numpy as np
import os
import torch
from torch.utils.data import Dataset, Dataloader
from skimage import io

class SingleImageLabeledDataset(Dataset):
  def __init__(self, data_dir, packages_paths_filepath, transform=None):
    packages = np.loadtxt(packages_paths_filepath, dtype=int)
    self.filepaths = [os.path.join(data_dir, house) for house in packages]
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

