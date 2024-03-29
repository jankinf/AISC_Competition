import torch
import torch.nn as nn
import torchvision.transforms as transforms
from aisc.attacks.models.CurricularFace.backbone.model_resnet import ResNet_50, ResNet_101, ResNet_152
from aisc.attacks.models.CurricularFace.backbone.model_irse import IR_50, IR_101, IR_152, IR_SE_50, IR_SE_101, IR_SE_152
from aisc.attacks.models.CurricularFace.backbone.model_mobilefacenet import MobileFaceNet
'''
https://github.com/HuangYG123/CurricularFace
'''
BACKBONE_DICT = {
    'ResNet_50': ResNet_50, 
    'ResNet_101': ResNet_101, 
    'ResNet_152': ResNet_152,
    'IR_50': IR_50, 
    'IR_101': IR_101, 
    'IR_152': IR_152,
    'IR_SE_50': IR_SE_50, 
    'IR_SE_101': IR_SE_101, 
    'IR_SE_152': IR_SE_152,
    'MobileFaceNet': MobileFaceNet
}

def load_models():
    INPUT_SIZE = (112, 112)
    BACKBONE_RESUME_ROOT = "/data/projects/aisc_facecomp/models/CurricularFace/CurricularFace_Backbone.pth"
    model = BACKBONE_DICT['IR_101'](INPUT_SIZE)
    model.load_state_dict(torch.load(BACKBONE_RESUME_ROOT))
    model = nn.Sequential(
        transforms.Resize(112),
        transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]),
        model
    )
    model.eval()
    model = model.cuda()
    # print("loading model IR_101")
    return [model]
    
