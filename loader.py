from __future__ import print_function, division
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
import random

class HackLoader(Dataset):

    def __init__(self, split_txt, root_dir):
        self.root_dir = root_dir
        self.all_packs = np.loadtxt(split_txt)

    def __len__(self):
        return self.all_packs.size

    def __getitem__(self, idx):
        pack_id = str(int(self.all_packs[idx]))
        pack_path = os.path.join(self.root_dir, pack_id)
        nadir_path = os.path.join(pack_path, "nadirs")
        all_nadir_files = os.listdir(nadir_path)
        all_nadirs = [x for x in all_nadir_files if ".jpg" in x]
        rand_nadir = random.randint(0,len(all_nadirs)-1)
        ex_im = os.path.join(nadir_path, all_nadirs[rand_nadir])
        image = Image.open(ex_im)
        return image

#dset = HackLoader("eighty_list.txt", "C:\\Users\\Bowald\\Data\\2020_hackathon")
#dset[0].show()
#print(dset[0])