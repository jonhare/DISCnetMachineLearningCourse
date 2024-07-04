import os

import torch
from torchvision.transforms import ToTensor
from torchvision.models import resnet18
from PIL import Image
from torch.utils.data import Dataset, random_split
import pandas as pd
from tqdm import tqdm


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


ds = GalaxyDataset("/home/jsh2/galaxyzoo", transform=ToTensor())

generator = torch.Generator().manual_seed(42)
train_ds, val_ds = random_split(ds, [0.7, 0.3], generator=generator)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = resnet18(num_classes=37).to(device)

loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=128, shuffle=False)
learning_rate = 0.01

opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

def validate(val_loader, model, device):
    model.eval()

    with torch.no_grad():
        val_loss = 0
        count = 0
        for (x_, y_) in val_loader:
            x_, y_ = x_.to(device), y_.to(device)
            prediction = model(x_)
            loss = torch.nn.functional.mse_loss(prediction, y_)
            val_loss += loss.cpu().item()
            count += x_.shape[0]

        model.train()
        return val_loss / count


for epoch in range(100):
    losses = 0
    count = 0
    tloader = tqdm(loader)
    for (x_, y_) in tloader:
        opt.zero_grad()

        x_, y_ = x_.to(device), y_.to(device)

        prediction = model(x_)
        loss = torch.nn.functional.mse_loss(prediction, y_)
        loss.backward()
        opt.step()

        losses += loss.detach().cpu().item()
        count += x_.shape[0]
        tloader.set_postfix({"loss": losses / count})

    val_loss = validate(val_loader, model, device)
    print(epoch, val_loss)
    torch.save(model, f"./model_{epoch}.pt")
