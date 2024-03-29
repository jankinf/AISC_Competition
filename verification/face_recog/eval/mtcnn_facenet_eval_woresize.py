from collections import defaultdict
import numpy as np
from PIL import Image
import os
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from utils import l2_sigmoid
from torch import nn
from torch.nn import functional as F
from torchvision import transforms as T
import argparse
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

parser = argparse.ArgumentParser()
parser.add_argument("--eval_dir", type=str, default="/data/projects/verification/face_recog/data/video_img")
parser.add_argument("--margin", type=int, default=0)
parser.add_argument("--img_size", type=int, default=160)
args = parser.parse_args()

class Detector(MTCNN):
    def forward(self, ori_img):
        # Detect faces
        outs = []
        ori_img = T.ToTensor()(ori_img).unsqueeze(0)
        img = ori_img.permute(0, 2, 3, 1) * 255 # change to (NHWC)
        outs.append(img)
        batch_boxes, batch_probs, batch_points = self.detect(img, landmarks=True)
        # Select faces
        box, batch_probs, batch_points = self.select_boxes(
            batch_boxes, batch_probs, batch_points, img, method=self.selection_method
        )
        raw_image_size = img.shape[1:3]
        box = [
            int(max(box[0, 0, 0], 0)),
            int(max(box[0, 0, 1], 0)),
            int(min(box[0, 0, 2], raw_image_size[1])),
            int(min(box[0, 0, 3], raw_image_size[0])),
        ]
        out = img[:, box[1]:box[3], box[0]:box[2], :].permute(0, 3, 1, 2) / 255.
        
        return out

mtcnn = Detector(image_size=args.img_size, margin=args.margin, device=torch.device('cuda'))

logfile = '/data/projects/verification/face_recog/results/adv/mtcnn_facenet_margin{}_img{}_woresize.log'.format(args.margin, args.img_size)
input_dir = args.eval_dir
img_list = os.listdir(input_dir)

os.makedirs(os.path.dirname(logfile), exist_ok=True)
results = defaultdict(float)

with open(logfile, 'a') as fp:
    for pretrain in ['vggface2', 'casia-webface']:
        model = InceptionResnetV1(pretrained=pretrain).eval()
        model = model.cuda()
        with torch.no_grad():
            cos = nn.CosineSimilarity(dim=1, eps=1e-12)
            img = Image.open("/data/projects/verification/face_recog/data/video_img/0.jpg")
            # img_cropped = mtcnn(img, save_path="/data/projects/verification/face_recog/data/debug/mtcnn_{}".format(img_list[0])).cuda()
            img_cropped = mtcnn(img).cuda()
            # im = Image.fromarray((img_cropped.permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype(np.uint8)[0])
            # im.save("mtcnn_woresize.jpg")
            feat_img1 = model(img_cropped).detach()
             
            for idx in range(len(img_list)):
                try:
                    target_img = Image.open(os.path.join(input_dir, img_list[idx]))
                    # tgt_img_cropped = mtcnn(target_img, save_path="/data/projects/verification/face_recog/data/debug/mtcnn_{}".format(img_list[idx])).cuda()
                    tgt_img_cropped = mtcnn(target_img).cuda()
                    feat_img2 = model(tgt_img_cropped)
                    cos_sim = cos(feat_img1, feat_img2)
                    
                    results[pretrain] += cos_sim.item() / len(img_list)
                
                except Exception as e:
                    print(e)
                    continue

        fp.write("file:{} pretrain:{} cos:{}\n".format(input_dir, pretrain, results[pretrain]))