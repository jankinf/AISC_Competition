# coding=utf-8
import argparse
import os.path
import torch
from torch.nn import functional as F
from src.tsn_predict import pretrain
from src.models import AENet
import torchvision
from PIL import Image
parser = argparse.ArgumentParser(description='eval')
parser.add_argument('--ckpt', type=str, default='/data/projects/CelebA-Spoof/src/ckpt_iter.pth.tar', help='path of ckpt to load')
args = parser.parse_args()


def main():
    print(args)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize([224, 224]),
        torchvision.transforms.ToTensor(),
    ])
    model = AENet(num_classes=2)
    checkpoint = torch.load(args.ckpt)
    pretrain(model, checkpoint['state_dict'])
    model = model.cuda()
    model.eval()

    # img_dir = "/data/projects/aisc_facecomp/finals/keypoints/face_aligned"
    # img_dir = "/data/projects/aisc_facecomp/finals/keypoints/aisc_self"
    img_dir = "/data/projects/aisc_facecomp/finals/keypoints/face_aligned224"
    paths = [
        "{}/ht_glasses.png".format(img_dir),
        "{}/ht.png".format(img_dir),
        "{}/liyi_glasses.png".format(img_dir),
        "{}/liyi.png".format(img_dir),
        "{}/qinyi.png".format(img_dir),
        "{}/qinyi02.png".format(img_dir),
        "{}/shibo.png".format(img_dir),
        "{}/shibo2.png".format(img_dir),
    ]
    for path in paths:
        im = Image.open(path)
        img = transform(im).cuda().unsqueeze(0)
        out = model(img)
        out = F.softmax(out, -1)
        pred = out.argmax(-1)
        score = out[:, pred]
        print("name: {}, pred: {}, score: {}".format(os.path.basename(path) ,pred.item(), score.item()))

if __name__ == '__main__':
    main()