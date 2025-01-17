import json
import os
import pickle
import random
import logging
from matplotlib import pyplot as plt
import cv2
import torchvision
import nori2
import numpy as np
from PIL import Image

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
    npimg = grid.numpy()
    # plt.imsave("/data/projects/CelebA-Spoof/results/plt.jpg", np.transpose(npimg, (1,2,0)), interpolation='nearest')
    npimg = np.transpose(npimg, (1,2,0))
    print(npimg.shape, npimg.max(), npimg.min())
    plt.imsave(f"/data/projects/CelebA-Spoof/results/{name}.jpg", npimg)

def save_img(save_path, img, split_channel=False):
    img_ = np.array(img * 255).astype('uint8')
    if split_channel:
        for i in range(img_.shape[2]):
            ch_path = save_path + "@channel{}.jpg".format(i)
            ch = Image.fromarray(img_[:, :, i])
            ch.save(ch_path)
    else:
        Image.fromarray(img_).save(save_path)

class CelebASpoofDataset(Dataset):
    def __init__(self, input_json: str, transform=None, is_training: bool = True, cropped: bool = True):
        with open(input_json) as f:
            metas = json.load(f)
        self.metas = list(metas.items())
        self.nf = nori2.Fetcher()
        self.transform = transform
        self.is_training = is_training
        self.cropped = cropped

    def empty_data(self):
        if self.is_training:
            rand_idx = torch.randint(len(self.metas), (1,))[0]
            return self.__getitem__(rand_idx)
        else:
            return

    @classmethod
    def decode_image(cls, str_b: bytes) -> np.ndarray:
        return cv2.imdecode(np.fromstring(str_b, np.uint8), cv2.IMREAD_UNCHANGED)

    def __getitem__(self, index):
        filepath, meta = self.metas[index]
        nid, label = meta["data_id"], meta["label"]
        try:
            ndata = pickle.loads(self.nf.get(nid, retry=1))
        except Exception:
            import traceback
            traceback.print_exc()
            return self.empty_data()
        img = self.decode_image(ndata["img"])
        real_h, real_w = img.shape[:2]

        """bbox faceinfo:
            bbox = [61 45 61 112 0.9970805]
            bbox[0]: x value of the upper left corner
            bbox[1]: y value of the upper left corner
            bbox[2]: w value of bbox
            bbox[3]: h value of bbox
            bbox[4]: score of bbox
            - How to use bbox to crop face?
                1. Get the shape of image: real_h, real_w
                2. x1 = int(bbox[0]*(real_w / 224))
                y1 = int(bbox[1]*(real_h / 224))
                w1 = int(bbox[2]*(real_w / 224))
                h1 = int(bbox[3]*(real_h / 224))
                3. Then x1, y1, w1, h1 are the real bbox values of image
        """
        if self.cropped:
            try:
                x, y, w, h, score = ndata["bbox"].split()
                x, y, w, h = list(map(lambda x: int(float(x)), [x, y, w, h]))
                w = int(w*(real_w / 224))
                h = int(h*(real_h / 224))
                x = int(x*(real_w / 224))
                y = int(y*(real_h / 224))

                # Crop face based on its bounding box
                y1 = 0 if y < 0 else y
                x1 = 0 if x < 0 else x
                y2 = real_h if y1 + h > real_h else y + h
                x2 = real_w if x1 + w > real_w else x + w
                img = img[y1:y2, x1:x2, :]
                real_h, real_w = img.shape[:2]
            except:
                logging.info('Cropping Bounding Box of' + ' ' +
                             filepath + ' ' + 'goes wrong')

        if real_h * real_w < 100:
            return self.empty_data()

        # 这里的图片都是open-cv 读取的BGR图像，不是PIL读取的RGB图像，
        # 如果要使用torchvision的transform算子，需先通过下述方法转为BGR图像
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = Image.fromarray(img)
        # img = np.transpose(img, (2, 0, 1))
        if self.transform:
            img = self.transform(img)
        else:
            img = default_transform(img)
        
        # todo: sanity check
        # show(self.transform(img), "colerjitter")
        # show(default_transform(img), "default")
        # [0:40]: face attribute labels, [40]: spoof type label, [41]: illumination label, [42]: Environment label [43]: live/spoof label
        label = label[43]

        return {"filepath": filepath, "img": img, "label": label}

    def __len__(self):
        return len(self.metas)


def my_collate_fn(batch):
    # 过滤掉可能存在的 None
    batch = [item for item in batch if item is not None]
    return default_collate(batch) if batch else None


def get_train_dataloader(transform=None, batch_size=16, num_workers=2, cropped=False):
    train_dataset = CelebASpoofDataset(
        "/data/datasets/celeba_spoof/metas_resolved/intra_test_train_label.json",
        cropped=cropped,
        transform=transform,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    print(len(train_dataset), len(train_dataloader))
    return train_dataloader


def get_test_dataloader(batch_size=16, num_workers=2, cropped=True):
    test_dataset = CelebASpoofDataset(
        "/data/datasets/celeba_spoof/metas_resolved/intra_test_test_label.json",
        cropped=cropped,
        is_training=False,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=my_collate_fn,
    )
    print(len(test_dataset), len(test_dataloader))
    return test_dataloader


def _test(tnt=1000):
    dataloader = get_train_dataloader()
    out_dir = "out1"
    os.makedirs(out_dir, exist_ok=True)

    cnt_0, cnt_1, cnt = 0, 0, 1
    for batch in dataloader:
        if batch is None:
            continue
        img = batch["img"].numpy()
        label = batch["label"]
        for i in range(img.shape[0]):
            i_img = np.transpose(img[i] * 255, (1, 2, 0)).astype(np.uint8)
            i_label = label[i].numpy()
            if cnt_0 < tnt or cnt_1 < tnt:
                if cnt_0 < tnt and i_label == 0:
                    img_path = os.path.join(out_dir, "{}_{}.jpg".format(cnt, i_label))
                    Image.fromarray(i_img).save(img_path)
                    # cv2.imwrite(img_path, i_img)
                    cnt_0 += 1
                    cnt += 1
                elif cnt_1 < tnt and i_label == 1:
                    img_path = os.path.join(out_dir, "{}_{}.jpg".format(cnt, i_label))
                    # Image.fromarray(i_img).save(img_path)
                    # cv2.imwrite(img_path, i_img)
                    cnt_1 += 1
                    cnt += 1
            else:
                exit()

def _test_done(tnt=1000):
    dataloader = get_train_dataloader()
    out_dir = "out_png"
    os.makedirs(out_dir, exist_ok=True)

    cnt_0, cnt_1, cnt = 0, 0, 1
    for batch in dataloader:
        if batch is None:
            continue
        img = batch["img"].numpy()
        label = batch["label"]
        for i in range(img.shape[0]):
            i_img = np.transpose(img[i] * 255, (1, 2, 0)).astype(np.uint8)
            i_label = label[i].numpy()
            if cnt_0 < tnt or cnt_1 < tnt:
                if cnt_0 < tnt and i_label == 0:
                    if cnt in img_list: 
                        img_path = os.path.join(out_dir, "{}_{}.png".format(cnt, i_label))
                        Image.fromarray(i_img).save(img_path)
                    # cv2.imwrite(img_path, i_img)
                    cnt_0 += 1
                    cnt += 1
                elif cnt_1 < tnt and i_label == 1:
                    if cnt in img_list: 
                        img_path = os.path.join(out_dir, "{}_{}.png".format(cnt, i_label))
                        Image.fromarray(i_img).save(img_path)
                    # cv2.imwrite(img_path, i_img)
                    cnt_1 += 1
                    cnt += 1
            else:
                exit()

def _test_bigdata(tnt=2000):
    dataloader = get_train_dataloader()
    out_dir = "out_png_2000"
    os.makedirs(out_dir, exist_ok=True)

    cnt_0, cnt_1, cnt = 0, 0, 1
    for batch in dataloader:
        if batch is None:
            continue
        img = batch["img"].numpy()
        label = batch["label"]
        for i in range(img.shape[0]):
            i_img = np.transpose(img[i] * 255, (1, 2, 0)).astype(np.uint8)
            i_label = label[i].numpy()
            if cnt_0 < tnt or cnt_1 < tnt:
                if cnt_0 < tnt and i_label == 0:
                    img_path = os.path.join(out_dir, "{}_{}.png".format(cnt, i_label))
                    Image.fromarray(i_img).save(img_path)
                    cnt_0 += 1
                    cnt += 1
                elif cnt_1 < tnt and i_label == 1:
                    img_path = os.path.join(out_dir, "{}_{}.png".format(cnt, i_label))
                    Image.fromarray(i_img).save(img_path)
                    cnt_1 += 1
                    cnt += 1
            else:
                exit()


if __name__ == "__main__":
    set_seed(0)

    # img_list = list(map(lambda x: int(x.split('_')[0]), os.listdir('/data/projects/CelebA-Spoof/attack/spoof_data/out')))
    # print(img_list)
    # import pdb; pdb.set_trace()
    # _test_done()

    # _test()

    _test_bigdata()
