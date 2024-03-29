from collections import defaultdict
import numpy as np
from PIL import Image
import os
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from utils import l2_sigmoid
from torch import nn
from torchvision import transforms as T
import argparse
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

parser = argparse.ArgumentParser()
parser.add_argument("--eval_dir", type=str, default="/data/projects/verification/face_recog/data/video_img")
parser.add_argument("--margin", type=int, default=0)
parser.add_argument("--img_size", type=int, default=160)
args = parser.parse_args()
mtcnn = MTCNN(image_size=args.img_size, margin=args.margin, device=torch.device('cuda'))

logfile = '/data/projects/verification/face_recog/results/adv/facenet_margin{}_img{}.log'.format(args.margin, args.img_size)
input_dir = args.eval_dir
img_list = os.listdir(input_dir)

os.makedirs(os.path.dirname(logfile), exist_ok=True)
results = defaultdict(float)
normalize = T.Normalize(mean=[127.5/255, 127.5/255, 127.5/255], std=[128.0/255, 128.0/255, 128.0/255])
with open(logfile, 'a') as fp:
    for pretrain in ['vggface2', 'casia-webface']:
        model = InceptionResnetV1(pretrained=pretrain).eval()
        model = nn.Sequential(normalize, model)
        model = model.cuda()
        with torch.no_grad():
            cos = nn.CosineSimilarity(dim=1, eps=1e-12)
            transform = T.ToTensor()
            reference_img = transform(Image.open("/data/projects/verification/face_recog/data/video_img/0.jpg")).unsqueeze(0).cuda()
            feat_img1 = model(reference_img).detach()
            for idx in range(len(img_list)):
                try:
                    target_img = Image.open(os.path.join(input_dir, img_list[idx]))
                    target_img = T.ToTensor()(target_img).cuda()
                    feat_img2 = model(target_img.unsqueeze(0))
                    cos_sim = cos(feat_img1, feat_img2)
                    # l2_sigmoid_sim = l2_sigmoid(feat_img1, feat_img2)
                    # logdata = "id:{}, file:{}, score_cos: {}, score_l2_sigmoid: {}\n".format(idx, img_list[idx], cos_sim.item(), l2_sigmoid_sim.item())
                    # logdata = "id:{}, file:{}, score_cos: {}\n".format(idx, img_list[idx], cos_sim.item())
                    # print(logdata, end='')
                    # fp.write(logdata)
                    # fp.flush()
                    results[pretrain] += cos_sim.item() / len(img_list)
                
                except Exception as e:
                    print(e)
                    continue

        fp.write("file:{} pretrain:{} cos:{}\n".format(input_dir, pretrain, results[pretrain]))