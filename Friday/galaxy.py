import torch
import torchvision
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from torchvision.models import resnet18
import numpy as np


BASE = "/Users/jsh2/Work/galaxyzoo/"
IMAGES = "images_training_rev1"
META = "training_solutions_rev1.csv"

BATCH_SIZE = 32

class GalaxyDataset(Dataset):
    def __init__(self, train=True, transforms=None):
        super().__init__()

        df = pd.read_csv(BASE + META)
        self.target_names = list(df.columns)[1:]
        self.ids = df.iloc[:, 0].to_numpy(dtype=int)
        self.meta_data = torch.from_numpy(df.iloc[:, 1:].to_numpy(dtype=np.float32))

        n = int(len(self) * 0.7)
        if train:
            self.ids = self.ids[:n]
            self.meta_data = self.meta_data[:n]
        else:
            self.ids = self.ids[n:]
            self.meta_data = self.meta_data[n:]

        self.transforms = transforms

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]
        image = Image.open(f"{BASE}/{IMAGES}/{idx}.jpg")

        if self.transforms is not None:
            image = self.transforms(image)

        target = self.meta_data[i]

        return image, target


if __name__ == '__main__':
    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
        torchvision.transforms.RandomCrop((400, 400)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomVerticalFlip(),
    ])
    train_ds = GalaxyDataset(train=True, transforms=train_transforms)

    val_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
        torchvision.transforms.CenterCrop((400, 400))
    ])
    val_ds = GalaxyDataset(train=False, transforms=val_transforms)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

    model = resnet18(num_classes=len(train_ds.target_names)).to("mps")

    loss = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)

    def val_loss():
        vloss = 0
        batches = 0
        model.eval()
        with torch.no_grad():
            for img, target in val_loader:
                prediction = model(img.to("mps"))
                vloss += loss(prediction, target.to("mps"))
                batches += 1
        model.train()
        print(f"Validation loss: {vloss/batches}")


    val_loss()
    for epoch in range(200):
        running_loss = 0
        for i, (img, target) in enumerate(train_loader):
            img = img.to("mps")
            target = target.to("mps")

            optimizer.zero_grad()

            prediction = model(img)
            l = loss(prediction, target)
            l.backward()
            optimizer.step()

            running_loss += l.detach().item()

            if i % 100 == 99:
                print(f"e: {epoch} iter: {i} Running loss: {running_loss / (i + 1)}")

        val_loss()
        scheduler.step()
