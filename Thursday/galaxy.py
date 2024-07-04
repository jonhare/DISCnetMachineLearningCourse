import os

import torch
from torchvision.transforms import ToTensor
from PIL import Image
from torch.utils.data import Dataset, random_split
import pandas as pd


class GalaxyDataset(Dataset):
    def __init__(self, basepath, image_dir="images_training_rev1", transform=None):
        self.basepath = basepath
        self.transform = transform
        self.image_dir = image_dir

        data = pd.read_csv(os.path.join(basepath, "training_solutions_rev1.csv"))

        self.image_ids = data['GalaxyID']
        self.target = torch.from_numpy(data.iloc[:, 1:].to_numpy()).float()
        self.target_names = list(data.columns)[1:]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        target = self.target[idx]

        image = Image.open(os.path.join(self.basepath, self.image_dir, f"{image_id}.jpg"))

        if self.transform is not None:
            image = self.transform(image)

        return image, target


ds = GalaxyDataset("/Users/jsh2/Work/galaxyzoo", transform=ToTensor())

generator = torch.Generator().manual_seed(42)
random_split(ds, [0.3, 0.3, 0.4], generator=generator)
