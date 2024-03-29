import numpy as np
from PIL import Image
import os
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from utils import l2_sigmoid
from torch import nn
from torchvision import transforms as T
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


logfile = '/data/projects/verification/face_recog/results/baseline_clean/facenet.log'
input_dir = "/data/projects/verification/face_recog/data/video_img"
img_list = os.listdir(input_dir)

os.makedirs(os.path.dirname(logfile), exist_ok=True)

normalize = T.Normalize(mean=[127.5/255, 127.5/255, 127.5/255], std=[128.0/255, 128.0/255, 128.0/255])
with open(logfile, 'a') as fp:
    for pretrain in ['vggface2', 'casia-webface']:
        model = InceptionResnetV1(pretrained=pretrain).eval()
        model = nn.Sequential(normalize, model)
        model = model.cuda()
        fp.write("evaling {}\n".format(pretrain))
        with torch.no_grad():
            cos = nn.CosineSimilarity(dim=1, eps=1e-12)
            transform = T.ToTensor()
            
            reference_img = transform(Image.open(os.path.join(input_dir, img_list[0]))).unsqueeze(0).cuda()
            feat_img1 = model(reference_img).detach()
            for idx in range(1, len(img_list)):
                try:
                    target_img = transform(Image.open(os.path.join(input_dir, img_list[idx]))).unsqueeze(0).cuda()
                    feat_img2 = model(target_img)
                    cos_sim = cos(feat_img1, feat_img2)
                    l2_sigmoid_sim = l2_sigmoid(feat_img1, feat_img2)
                    logdata = "id:{}, file:{}, score_cos: {}, score_l2_sigmoid: {}\n".format(idx, img_list[idx], cos_sim.item(), l2_sigmoid_sim.item())
                    print(logdata, end='')
                    fp.write(logdata)
                    fp.flush()

                except Exception as e:
                    print(e)
                    continue
