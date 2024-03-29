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

logfile = '/data/projects/verification/face_recog/results/baseline_clean/mtcnn_facenet_margin{}_img{}.log'.format(args.margin, args.img_size)
input_dir = args.eval_dir
img_list = os.listdir(input_dir)

os.makedirs(os.path.dirname(logfile), exist_ok=True)
with open(logfile, 'a') as fp:
    for pretrain in ['vggface2', 'casia-webface']:
        model = InceptionResnetV1(pretrained=pretrain).eval()
        model = model.cuda()
        fp.write("evaling {}\n".format(pretrain))
        with torch.no_grad():
            cos = nn.CosineSimilarity(dim=1, eps=1e-12)
            img = Image.open(os.path.join(input_dir, img_list[0]))
            img_cropped = mtcnn(img, save_path="/data/projects/verification/face_recog/data/debug/mtcnn_{}".format(img_list[0])).cuda()
            feat_img1 = model(img_cropped.unsqueeze(0)).detach()
             
            for idx in range(1, len(img_list)):
                try:
                    target_img = Image.open(os.path.join(input_dir, img_list[idx]))
                    tgt_img_cropped = mtcnn(target_img, save_path="/data/projects/verification/face_recog/data/debug/mtcnn_{}".format(img_list[idx])).cuda()
                    feat_img2 = model(tgt_img_cropped.unsqueeze(0))
                    cos_sim = cos(feat_img1, feat_img2)
                    l2_sigmoid_sim = l2_sigmoid(feat_img1, feat_img2)
                    logdata = "id:{}, file:{}, score_cos: {}, score_l2_sigmoid: {}\n".format(idx, img_list[idx], cos_sim.item(), l2_sigmoid_sim.item())
                    print(logdata, end='')
                    fp.write(logdata)
                    fp.flush()

                except Exception as e:
                    print(e)
                    continue
