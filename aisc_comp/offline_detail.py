import torch
import torch.nn as nn
import argparse
from torch.utils import data
import os
from PIL import Image
import torchvision.transforms as T
from util import get_fastmodel
import numpy as np
import pandas as pd
import time

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
        origin_image_files = [os.path.join(self.input_dir, "{:04d}.png".format(i + 1)) for i in range(3000)]
        compare_image_files = [os.path.join(LFW.IMG_DIR, "{:04d}_compare.png".format(i + 1)) for i in range(3000)]
        self.origin_image_files = []
        self.compare_image_files = []

        for ofile, cfile in zip(origin_image_files, compare_image_files):
            if not os.path.exists(ofile):
                continue
            self.origin_image_files.append(ofile)
            self.compare_image_files.append(cfile)


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
    # 3: 0.22,
    # 4: 0.22,
    # 5: 0.22,
    3: 0.34,
    4: 0.30,
    5: 0.28,
    6: 0.26,
    7: 0.23,
    8: 0.23,
    9: 0.21,
}


def test():
    score = 0
    criterion = nn.CosineSimilarity(dim=1, eps=1e-8)
    threshold_setting = '@' + '-'.join(["{}:{}".format(idx, model_thresholds[idx]) for idx in sorted(model_idx)])
    method_name = os.path.basename(args.input_dir)
    cos_stat_file = "cos_stat/{}/".format(time.strftime("%m-%d", time.localtime())) + method_name + '{}.csv'.format(threshold_setting)
    os.makedirs(os.path.dirname(cos_stat_file), exist_ok=True)
    if os.path.exists(cos_stat_file):
        exit()
    models = get_fastmodel(idx=model_idx)
    scores = []
    n_sample = len(lfw)
    allcos = np.array([]).reshape(n_sample, -1)
    with torch.no_grad():
        for idx, model in zip(model_idx, models):
            threshold = model_thresholds[idx]
            score = 0
            cos_model = np.array([]).reshape(-1, 1)
            for i, (origin, origin_file, compare_image, compare_file) in enumerate(lfw_loader):
                origin = origin.cuda()
                compare_image = compare_image.cuda()
                advs_logits = model(origin)
                compares_logits = model(compare_image)
                cos_test = criterion(advs_logits, compares_logits)
                cos_model = np.concatenate([cos_model, cos_test.detach().cpu().numpy().reshape(-1, 1)], axis=0)
                score += (cos_test >= threshold).sum().item()
            allcos = np.concatenate([allcos, cos_model.reshape(-1, 1)], axis=1)
            print("model {}: {}".format(idx, score))
            scores.append(score / n_sample)
        allcos = np.concatenate([allcos, allcos.mean(axis=1).reshape(-1, 1)], axis=1)
    score = sum(scores) / len(models)
    print(f'total scores {scores}, scores {score}')
    columns = list(map(str, model_idx)) + ["mean_cos"]

    df = pd.DataFrame(data=allcos, columns=columns)
    
    df.to_csv(cos_stat_file, index=False)
    with open("/data/projects/aisc_facecomp/offline_log.txt", "a") as fp:
        fp.write("{}:\nmodel{} -- scores:{}@tntscore:{}\n".format(method_name, threshold_setting, scores, score))


if __name__ == '__main__':
    test()
