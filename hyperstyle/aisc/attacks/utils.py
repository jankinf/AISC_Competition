import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as T
from torch.utils import data
import cv2
from facenet_pytorch import InceptionResnetV1
import PIL
from PIL import Image

import numpy as np
import scipy.stats as st
import os
import math
from aisc.attacks.models.TC_Rank4.main import get_model as get_tcmodel
from aisc.attacks.models.TC_Rank4.model_irse import IR_50, IR_101, IR_152
from aisc.attacks.models.CurricularFace.main import load_models as cur_loads
from aisc.attacks.models.PFR.main import load_models as pfr_loads, load_model as pfr_load
from aisc.attacks.models.FaceX_Zoo.backbone.backbone_def import BackboneFactory

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MEAN = [127.5 / 255] * 3
STD = [128 / 255] * 3
CONFIG_FILE = "/data/projects/aisc_facecomp/third_party/FaceX_Zoo/training_mode/backbone_conf.yaml"
CKPT_DIR = "/data/projects/aisc_facecomp/third_party/FaceX_Zoo/ckpt/backbone"

class ModelLoader:
    """Load a model by network and weights file.

    Attributes: 
        model(object): the model definition file.
    """
    def __init__(self, backbone_factory):
        self.model = backbone_factory.get_backbone()

    def load_model_default(self, model_path):
        """The default method to load a model.
        
        Args:
            model_path(str): the path of the weight file.
        
        Returns:
            model(object): initialized model.
        """
        self.model.load_state_dict(torch.load(model_path)['state_dict'], strict=True) 
        model = self.model.cuda()
        return model

    def load_model(self, model_path):
        """The custom method to load a model.
        
        Args:
            model_path(str): the path of the weight file.
        
        Returns:
            model(object): initialized model.
        """
        model_dict = self.model.state_dict()
        pretrained_dict = torch.load(model_path)['state_dict']
        new_pretrained_dict = {}

        for k in model_dict:
            new_pretrained_dict[k] = pretrained_dict['backbone.'+k] # tradition training
            #new_pretrained_dict[k] = pretrained_dict['feat_net.'+k] # tradition training
            #new_pretrained_dict[k] = pretrained_dict['module.'+k]
            #new_pretrained_dict[k] = pretrained_dict['module.backbone.'+k]
            #new_pretrained_dict[k] = pretrained_dict[k] # co-mining
        
        model_dict.update(new_pretrained_dict)
        self.model.load_state_dict(model_dict)
        model = self.model.cuda()
        return model

def get_fastmodel(idx=-1):
    if idx == -1:
        idx = list(range(16))
    models = []
    if 0 in idx:
        model = InceptionResnetV1(pretrained='vggface2').to(device)
        # print(f"loading weight vggface2")
        model.eval()
        models.append(nn.Sequential(
            T.Resize(112),
            T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]),
            model
        ))
        print("loading model {}".format(0))
    if 1 in idx:
        model = InceptionResnetV1(pretrained='casia-webface').to(device)
        # print(f"loading weight casia-webface")
        model.eval()
        models.append(nn.Sequential(
            T.Resize(112),
            T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]),
            model
        ))
        print("loading model {}".format(1))
    if 2 in idx:
        models += cur_loads()
        print("loading model {}".format(2))
    if 3 in idx:
        models += pfr_load('ResNet50_IR')
        print("loading model {}".format(3))
    if 4 in idx:
        models += pfr_load('SEResNet50_IR')
        print("loading model {}".format(4))
    if 5 in idx:
        models += pfr_load('ResNet100_IR')
        print("loading model {}".format(5))
    if 6 in idx:
        models.append(get_tcmodel(IR_50, '/data/projects/aisc_facecomp/models/TC_Rank4/models/backbone_ir50_ms1m_epoch120.pth'))
        print("loading model {}".format(6))
    if 7 in idx:
        models.append(get_tcmodel(IR_50, '/data/projects/aisc_facecomp/models/TC_Rank4/models/Backbone_IR_50_LFW.pth'))
        print("loading model {}".format(7))
    if 8 in idx:
        models.append(get_tcmodel(IR_101, '/data/projects/aisc_facecomp/models/TC_Rank4/models/Backbone_IR_101_Batch_108320.pth'))
        print("loading model {}".format(8))
    if 9 in idx:
        models.append(get_tcmodel(IR_152, '/data/projects/aisc_facecomp/models/TC_Rank4/models/Backbone_IR_152_MS1M_Epoch_112.pth'))
        print("loading model {}".format(9))
    if 10 in idx:
        print("loading model {}".format(10))
        backbone_type = 'AttentionNet56'
        backbone_conf_file = CONFIG_FILE
        model_path = os.path.join(CKPT_DIR, 'attention56.pt')
        backbone_factory =  BackboneFactory(backbone_type, backbone_conf_file)
        model_loader = ModelLoader(backbone_factory)
        model = model_loader.load_model(model_path)
        model.eval()
        models.append(nn.Sequential(
            T.Resize(112),
            T.Normalize(mean=MEAN, std=STD),
            model
        ))
        models.append(model)

    if 11 in idx:
        print("loading model {}".format(11))
        backbone_type = 'AttentionNet92'
        backbone_conf_file = CONFIG_FILE
        model_path = os.path.join(CKPT_DIR, 'attention92.pt')
        backbone_factory =  BackboneFactory(backbone_type, backbone_conf_file)
        model_loader = ModelLoader(backbone_factory)
        model = model_loader.load_model(model_path)
        model.eval()
        models.append(nn.Sequential(
            T.Resize(112),
            T.Normalize(mean=MEAN, std=STD),
            model
        ))
    if 12 in idx:
        print("loading model {}".format(12))
        backbone_type = 'ResNet'
        backbone_conf_file = CONFIG_FILE
        model_path = os.path.join(CKPT_DIR, 'resnet152_irse.pt')
        backbone_factory =  BackboneFactory(backbone_type, backbone_conf_file)
        model_loader = ModelLoader(backbone_factory)
        model = model_loader.load_model(model_path)
        model.eval()
        models.append(nn.Sequential(
            T.Resize(112),
            T.Normalize(mean=MEAN, std=STD),
            model
        ))
    if 13 in idx:
        print("loading model {}".format(13))
        backbone_type = 'SwinTransformer'
        backbone_conf_file = CONFIG_FILE
        model_path = os.path.join(CKPT_DIR, 'swin_s.pt')
        backbone_factory =  BackboneFactory(backbone_type, backbone_conf_file)
        model_loader = ModelLoader(backbone_factory)
        model = model_loader.load_model(model_path)
        model.eval()
        models.append(nn.Sequential(
            T.Resize(224),
            T.Normalize(mean=MEAN, std=STD),
            model
        ))
    if 14 in idx:
        print("loading model {}".format(14))
        backbone_type = 'SwinTransformer'
        backbone_conf_file = CONFIG_FILE
        model_path = "/data/projects/aisc_facecomp/third_party/FaceX_Zoo/training_mode/conventional_training/out_dir/lr0.01/Epoch_49.pt"
        backbone_factory =  BackboneFactory(backbone_type, backbone_conf_file)
        model_loader = ModelLoader(backbone_factory)
        model = model_loader.load_model(model_path)
        model.eval()
        models.append(nn.Sequential(
            T.Resize(224),
            T.Normalize(mean=MEAN, std=STD),
            model
        ))
    if 15 in idx:
        print("loading model {}".format(15))
        backbone_type = 'SwinTransformer'
        backbone_conf_file = CONFIG_FILE
        model_path = "/data/projects/aisc_facecomp/third_party/FaceX_Zoo/training_mode/conventional_training/out_dir/adv_lr0.1_l2_eps0.06_alpha0.3_iter10/Epoch_20.pt"
        backbone_factory =  BackboneFactory(backbone_type, backbone_conf_file)
        model_loader = ModelLoader(backbone_factory)
        model = model_loader.load_model(model_path)
        model.eval()
        models.append(nn.Sequential(
            T.Resize(224),
            T.Normalize(mean=MEAN, std=STD),
            model
        ))

    return models


def save_img(save_path, img, split_channel=False):
    img_ = np.array(img * 255).astype('uint8')
    if split_channel:
        for i in range(img_.shape[2]):
            ch_path = save_path + "@channel{}.jpg".format(i)
            ch = Image.fromarray(img_[:, :, i])
            ch.save(ch_path)
    else:
        Image.fromarray(img_).save(save_path)


# DI
def input_diversity(x, resize_rate=1.15, diversity_prob=0.7):
    assert resize_rate >= 1.0
    assert diversity_prob >= 0.0 and diversity_prob <= 1.0
    if torch.rand(1) >= diversity_prob:
        return x
    img_size = max(*list(x.shape[-2:]))
    img_resize = int(img_size * resize_rate)
    rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
    rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
    h_rem = img_resize - rnd
    w_rem = img_resize - rnd
    pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
    pad_bottom = h_rem - pad_top
    pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
    pad_right = w_rem - pad_left
    padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)
    padded = F.interpolate(padded, size=[img_size, img_size], mode='bilinear', align_corners=False)
    return padded

