import os
import random
from matplotlib import pyplot as plt
import torchvision
import numpy as np
from PIL import Image
from attack.face_detection import GetFrame

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate

default_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def show(img, name):
    grid = torchvision.utils.make_grid(img, nrow=16, padding=100)
    npimg = grid.cpu().detach().numpy()
    # plt.imsave("/data/projects/CelebA-Spoof/results/plt.jpg", np.transpose(npimg, (1,2,0)), interpolation='nearest')
    npimg = np.transpose(npimg, (1, 2, 0))
    print(npimg.shape, npimg.max(), npimg.min())
    plt.imsave(f"/data/projects/CelebA-Spoof/results/{name}.jpg", npimg)


class CelebASpoofDataset(Dataset):
    def __init__(self, img_dir: str, transform=None):
        self.transform = transform
        self.img_paths = list(map(lambda x: os.path.join(img_dir, x), os.listdir(img_dir)))

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = Image.open(img_path)

        if self.transform:
            img = self.transform(img)
        else:
            img = default_transform(img)

        label = int(img_path.split('_')[-1][0])

        return {"filepath": img_path, "img": img, "label": label}

    def __len__(self):
        return len(self.img_paths)

class CroppedCelebASpoofDataset(Dataset):
    def __init__(self, img_dir: str, transform=None):
        self.transform = transform
        self.img_paths = list(map(lambda x: os.path.join(img_dir, x), os.listdir(img_dir)))
        self.mtcnn = GetFrame()

    def cropped_img(self, img):
        # Detect faces
        batch_boxes, batch_probs, batch_points = self.mtcnn.detect(img, landmarks=True)
        # Select faces
        if not self.mtcnn.keep_all:
            batch_boxes, batch_probs, batch_points = self.mtcnn.select_boxes(
                batch_boxes, batch_probs, batch_points, img, method=self.mtcnn.selection_method
            )
        if batch_boxes is None:
            return img.copy()
        raw_image_size = img.size
        batch_boxes = batch_boxes[0]
        box = [
            int(max(batch_boxes[0], 0)),
            int(max(batch_boxes[1], 0)),
            int(min(batch_boxes[2], raw_image_size[0])),
            int(min(batch_boxes[3], raw_image_size[1])),
        ]
        return img.crop(box).copy().resize(raw_image_size, Image.BILINEAR)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = Image.open(img_path)
        img = self.cropped_img(img)
        if self.transform:
            img = self.transform(img)
        else:
            img = default_transform(img)

        label = int(img_path.split('_')[-1][0])

        return {"filepath": img_path, "img": img, "label": label}

    def __len__(self):
        return len(self.img_paths)


def get_dataloader(batch_size=16, num_workers=2):
    dataset = CelebASpoofDataset(
        "/data/projects/CelebA-Spoof/attack/spoof_data/out_png_2000",
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    return dataloader


def get_cropped_dataloader(batch_size=16, num_workers=2):
    dataset = CroppedCelebASpoofDataset(
        "/data/projects/CelebA-Spoof/attack/spoof_data/out_png_2000",
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    return dataloader


def _test():
    dataloader = get_dataloader()
    # dataloader = get_cropped_dataloader()
    out_dir = "debug_load"
    os.makedirs(out_dir, exist_ok=True)
    for batch in dataloader:
        print(batch)
        img = batch["img"].numpy()
        label = batch["label"]
        for i in range(img.shape[0]):
            i_img = np.transpose(img[i] * 255, (1, 2, 0)).astype(np.uint8)
            im = Image.fromarray(i_img)
            i_label = label[i].numpy()
            img_path = os.path.join(out_dir, "{}_{}.jpg".format(i, i_label))
            im.save(img_path)
        break


if __name__ == "__main__":
    set_seed(0)
    _test()
