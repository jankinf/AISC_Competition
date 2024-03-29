import torch
import torch.nn as nn
import argparse
from torch.utils import data
import os
from PIL import Image
import torchvision.transforms as T
from util import get_fastmodel

parser = argparse.ArgumentParser(description='attacks in PyTorch')
parser.add_argument('--input_dir', default='/data/liyi/projects/aisc_facecomp/results/ctmim_eps40_steps30_alpha0.5_keypoints_dts_compare1_ord2_ens0178', type=str, help='directory of clean examples')
parser.add_argument('--model_idx', nargs='+', default=[2, 3, 4], type=int)
args = parser.parse_args()

class LFW(data.Dataset):
    IMG_DIR = "/data/projects/aisc_facecomp/data"
    def __init__(self, input_dir, transform) -> None:
        super().__init__()
        self.input_dir = input_dir
        self.transform = transform
        self.origin_image_files = [os.path.join(self.input_dir, "{:04d}.png".format(i + 1)) for i in range(3000)]
        self.compare_image_files = [os.path.join(LFW.IMG_DIR, "{:04d}_compare.png".format(i + 1)) for i in range(3000)]

        assert len(self.origin_image_files) == 3000
        assert len(self.compare_image_files) == 3000

    def __getitem__(self, index):
        origin_image_file = self.origin_image_files[index]
        compare_image_file = self.compare_image_files[index]
        origin_image = Image.open(origin_image_file).convert('RGB')
        compare_image = Image.open(compare_image_file).convert('RGB')
        if self.transform:
            origin_image = self.transform(origin_image)
            compare_image = self.transform(compare_image)

        return origin_image, origin_image_file, compare_image, compare_image_file

    def __len__(self):
        return len(self.origin_image_files)


preprocess = T.Compose([T.ToTensor()])

lfw = LFW(args.input_dir, preprocess)
lfw_loader = data.DataLoader(lfw, batch_size=250, shuffle=False, pin_memory=True, num_workers=8)

# model_idx = [2, 3, 4]
model_idx = args.model_idx
model_thresholds = {
    0: 0.33,
    1: 0.25,
    2: 0.23,
    3: 0.22,
    4: 0.22,
    5: 0.22,
    # 3: 0.34,
    # 4: 0.30,
    # 5: 0.28,
    6: 0.26,
    7: 0.23,
    8: 0.23,
    9: 0.21,
}


def test():
    score = 0
    criterion = nn.CosineSimilarity(dim=1, eps=1e-8)
    models = get_fastmodel(idx=model_idx)
    scores = []
    with torch.no_grad():
        for idx, model in zip(model_idx, models):
            threshold = model_thresholds[idx]
            score = 0
            for i, (origin, origin_file, compare_image, compare_file) in enumerate(lfw_loader):
                origin = origin.cuda()
                compare_image = compare_image.cuda()
                advs_logits = model(origin)
                compares_logits = model(compare_image)
                cos_test = criterion(advs_logits, compares_logits)

                score += (cos_test >= threshold).sum().item()
            print("model {}: {}".format(idx, score))
            scores.append(score / 3000)
    score = sum(scores) / 3
    print(f'total scores {scores}, scores {score}')
    with open("/data/projects/aisc_facecomp/offline_log.txt", "a") as fp:
        fp.write("{}:\nmodel{} -- scores:{}@tntscore:{}\n".format(os.path.basename(args.input_dir), sorted(model_idx), scores, score))


if __name__ == '__main__':
    test()
