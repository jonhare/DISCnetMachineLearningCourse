import os

import torch
from torchvision.transforms import ToTensor
from torchvision.models import resnet18
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
train_ds, val_ds = random_split(ds, [0.7, 0.3], generator=generator)

device = "mps"
model = resnet18(num_classes=37).to(device)

loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True)
learning_rate = 0.01

opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(3):
    losses = 0
    count = 0
    for (x_, y_) in loader:
        opt.zero_grad()

        x_, y_ = x_.to(device), y_.to(device)

        prediction = model(x_)
        loss = torch.nn.functional.mse_loss(prediction, y_)
        loss.backward()
        opt.step()

        losses += loss.detach().cpu().item()
        count += x_.shape[0]
        print(losses / count)
