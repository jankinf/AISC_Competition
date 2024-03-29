import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as T
from torch.utils import data

from facenet_pytorch import InceptionResnetV1
from models.face_evoLVe.networks.MobileFace import MobileFace
from models.face_evoLVe.networks.Mobilenet import Mobilenet
from models.face_evoLVe.networks.ResNet import resnet
from models.face_evoLVe.networks.ShuffleNet import ShuffleNetV1
from models.face_evoLVe.networks.CosFace import CosFace
from models.face_evoLVe.networks.SphereFace import SphereFace
from models.face_evoLVe.networks.ArcFace import ArcFace
from models.face_evoLVe.networks.IR import IR
from models.face_evoLVe.networks.model_irse import IR_50
from models.face_evoLVe.networks.model_irse import IR_SE_50
from models.face_evoLVe.networks.model_irse import IR_152

import PIL
from PIL import Image

import numpy as np
import pandas as pd
import scipy.stats as st
import os
import math


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_models():
    ens_models = sorted([
        # 'MobileFace', 'ShuffleNet_V1_GDConv', 'Mobilenet', 'ResNet50', 'ArcFace', 'IR50'
        'IRSE50', 'IR50'
    ])
    models = []
    for model_name in ens_models:
        img_shape = (112, 112)       
        if 'IR' in model_name: 
            IR50_path = '/data/public/models/face_recognition/backbone_ir50_asia.pth'
            IR152_path = '/data/public/models/face_recognition/Backbone_IR_152_Epoch_112_Batch_2547328_Time_2019-07-13-02-59_checkpoint.pth'
            IRSE50_path = '/data/public/models/face_recognition/model_ir_se50.pth'
            print(f'Loading {model_name}')
            if model_name == 'IR50':
                model = IR_50(img_shape)
                model.load_state_dict(torch.load(IR50_path))

            elif model_name == 'IR152':
                model = IR_152(img_shape)
                model.load_state_dict(torch.load(IR152_path))

            elif model_name == 'IRSE50':
                model = IR_SE_50(img_shape)
                model.load_state_dict(torch.load(IRSE50_path))
            else:
                raise Exception
            model = model.to(device)
            model.eval()
            # models.append(nn.Sequential(T.Normalize(mean=[127.5/255, 127.5/255, 127.5/255], std=[128/255, 128/255, 128/255]), model))
            models.append(nn.Sequential(T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), model))

        elif model_name == 'MobileFace':
            model = MobileFace()
        elif model_name == 'Mobilenet':
            model = Mobilenet()
        elif model_name == 'ResNet50':
            model = resnet(depth=50)
        elif model_name == 'ShuffleNet_V1_GDConv':
            model = ShuffleNetV1(pooling='GDConv')
        elif model_name == 'CosFace':
            model = CosFace()
            img_shape = (112, 96)
        elif model_name == 'SphereFace':
            model = SphereFace()
            img_shape = (112, 96)
        elif model_name == 'ArcFace':
            model = ArcFace()
        else:
            raise Exception
        model = model.output['prelogits'][0].to(device)
        model.eval()
        # models.append(nn.Sequential(T.Normalize(mean=[127.5/255, 127.5/255, 127.5/255], std=[128/255, 128/255, 128/255]), model))
        models.append(nn.Sequential(T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), model))

    return models
