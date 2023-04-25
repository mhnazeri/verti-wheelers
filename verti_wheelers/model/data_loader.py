import os.path as osp

import pickle
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

        self.data = None
        with open(f"{root}/labels.pickle", "rb") as f:
            self.data = pickle.load(f)

        if train:
            mode = 'train'
            self.data["cmd_vel"] = self.data["cmd_vel"][: int(len(self.data["cmd_vel"]) * 0.95)]
        else:
            mode = 'val'
            self.data["cmd_vel"] = self.data["cmd_vel"][int(len(self.data["cmd_vel"]) * 0.95) + 1:]

    def __len__(self):
        return len(self.data["cmd_vel"])

    def __getitem__(self, idx):
        """Return a sample in the form: (image, label)"""
        action = torch.tensor(self.data['cmd_vel'][idx], dtype=torch.float)
        img = torchvision.io.read_image(
            self.data['img_address'][idx]).float()
        img = self.transform(img)
        # rescale image between 0 and 1
        img = (img - img.min()) / (img.max() - img.min())
        # print(f'{img.min() = }')
        # print(f'{img.max() = }')
        return img, action


if __name__ == "__main__":
    # import matplotlib.pyplot as plt
    # data = VWDataset("verti_wheelers/data")
    # uuid, img, action = next(iter(data))
    # plt.imshow(img.squeeze(), cmap='gray')
    # plt.show()
    pass
