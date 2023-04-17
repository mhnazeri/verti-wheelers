"""Utility functions related to vision"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import torch
import torchvision.utils as vutils


def plot_images(batch: torch.Tensor, title: str):
    """Plot a batch of images

    Args:
        batch: (torch.Tensor) a batch of images with dimensions (batch, channels, height, width)
        title: (str) title of the plot and saved file
    """
    n_samples = batch.size(0)
    plt.figure(figsize=(n_samples // 2, n_samples // 2))
    plt.axis("off")
    plt.title(title)
    plt.imshow(
        np.transpose(vutils.make_grid(
            batch, padding=2, normalize=True), (1, 2, 0))
    )
    plt.savefig(f"{title}.png")


def draw_on_image(img, true_actions, action, gt=True):
    """Draw text on the image

    Args:
        img: (torch.Tensor) frame
        measurements: (dict) ground truth values
        action: (torch.Tensor) predicted actions
        gt: whether to draw true action or not
    """
    # if measurements:
    linear_gt = true_actions[0]
    angular_gt = true_actions[1]

    linear = action[0].item()
    angular = action[1].item()

    img = img.cpu().numpy()[-1]
    # print(f"{type(img) = }")
    # print(f"{img.shape = }")
    img_width = img.shape[1] // 2
    img = Image.fromarray(
        (((img - img.min()) / (-img.min() + img.max())) * 255).astype(np.uint8)
    )
    draw = ImageDraw.Draw(img)
    # load font
    fnt_path = Path("./verti_wheelers/").resolve() / 'misc_files/FUTURAM.ttf'
    print(f"{fnt_path = }")
    fnt = ImageFont.truetype(str(fnt_path), 8)
    draw.text((5, 30), f"Linear: {linear:.2f}", fill='red', font=fnt)
    draw.text((5, 50), f"Angular: {linear:.2f}", fill='red', font=fnt)

    if gt:
        draw.text((img_width, 30),
                  f"Linear_GT: {linear_gt:.2f}", fill='green', font=fnt)
        draw.text(
            (img_width, 50), f"Angular_GT: {angular_gt:.2f}", fill='green', font=fnt
        )

    return np.array(img)


def decode_mask(mask, nc: int = 23):
    """Decode segmentation map to an RGB image

    class labels are based on:
        https://carla.readthedocs.io/en/latest/ref_sensors/#semantic-segmentation-camera

    Args:
        mask: (numpy.ndarray) the segmentation image
        nc: (int) number of classes that segmentation have
    """
    if len(mask.shape) == 3:
        mask = np.argmax(mask, axis=0)

    label_colors = np.array(
        [
            (0, 0, 0),  # 0=Unlabeled
            # 1=Building, 2=Fence, 3=Other   , 4=Pedestrian, 5=Pole
            (70, 70, 70),
            (100, 40, 40),
            (55, 90, 80),
            (220, 20, 60),
            (153, 153, 153),
            # 6=RoadLine, 7=Road, 8=SideWalk, 9=Vegetation, 10=Vehicles
            (157, 234, 50),
            (128, 64, 128),
            (244, 35, 232),
            (107, 142, 35),
            (0, 0, 142),
            # 11=Wall, 12=TrafficSign, 13=Sky, 14=Ground, 15=Bridge
            (102, 102, 156),
            (220, 220, 0),
            (70, 130, 180),
            (81, 0, 81),
            (150, 100, 100),
            # 16=RailTrack, 17=GuardRail, 18=TrafficLight, 19=Static, 20=Dynamic
            (230, 150, 140),
            (180, 165, 180),
            (250, 170, 30),
            (110, 190, 160),
            (170, 120, 50),
            # 21=water, 22=terrain
            (45, 60, 150),
            (145, 170, 100),
        ]
    )

    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)

    for l in range(nc):
        idx = mask == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]

    rgb = np.stack([r, g, b], axis=0)
    rgb = rgb.astype(np.float)
    rgb = rgb / 255.0
    return rgb


if __name__ == "__main__":
    img = torch.randint(low=0, high=255, size=(3, 256, 256))
    img = draw_on_image(img, torch.tensor([2, 4]), torch.tensor([6, 8]))
    plt.imshow(img)
    plt.show()
