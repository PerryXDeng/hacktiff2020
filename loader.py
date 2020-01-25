from __future__ import print_function, division
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
class HackLoader(Dataset):

    def __init__(self, split_txt, root_dir):
        self.root_dir = root_dir
        self.all_packs = np.loadtxt(split_txt)

    def __len__(self):
        return self.all_packs.size

    def __getitem__(self, idx):
        pack_id = str(int(self.all_packs[idx]))
        pack_path = os.path.join(self.root_dir, pack_id)
        nadir_path = os.path.join(pack_path, "nadir")
        obliques_path = os.path.join(pack_path, "obliques")
        ex_im = os.path.join(obliques_path, os.listdir(obliques_path)[0])
        image = Image.open(ex_im)
        return image

dset = HackLoader("example.txt", "Hackathon_Resources\data_packages")
print(dset[0])