import os.path as osp

import pandas as pd
from typing import Callable, Optional
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import torchvision


class VWDataset(Dataset):

    def __init__(self, root: str, train: bool = True, transform: Optional[Callable] = None):
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose(
                [transforms.Resize(224, antialias=False)])

        self.data = pd.read_csv(root + "/labels.csv")
        if train:
            mode = 'train'
            self.data = self.data.iloc[: int(self.data.shape[0] * 0.95)]
        else:
            mode = 'val'
            self.data = self.data.iloc[int(self.data.shape[0] * 0.95) + 1:]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        """Return a sample in the form: (uuid, image, label)"""
        action = torch.tensor(eval(self.data['cmd_vel'].iloc[idx])[
                              :2], dtype=torch.float)  # only the first two numbers are relevant
        img = torchvision.io.read_image(
            self.data['img_address'].iloc[idx]).float()
        img = self.transform(img)
        return img, action


if __name__ == "__main__":
    # import matplotlib.pyplot as plt
    # data = VWDataset("verti_wheelers/data")
    # uuid, img, action = next(iter(data))
    # plt.imshow(img.squeeze(), cmap='gray')
    # plt.show()
    pass
