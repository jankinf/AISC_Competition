import torch
from torch import nn
from torchvision import transforms
from models.PFR.Backbones.Backbone import MobileFacenet, CBAMResNet

backbones = {
    'ResNet50_IR': CBAMResNet(50, feature_dim=512, mode='ir'),
    'SEResNet50_IR': CBAMResNet(50, feature_dim=512, mode='ir_se'),
    'ResNet100_IR': CBAMResNet(100, feature_dim=512, mode='ir'),
    # 'MobileFaceNet': MobileFacenet(),
    # 'SEResNet100_IR': CBAMResNet(100, feature_dim=512, mode='ir_se')
}
ckpt_path = {
    'ResNet50_IR': "/data/projects/aisc_facecomp/models/PFR/ckpts/CASIA_WebFace_ResNet50_IR/Iter_64000_net.pth",
    'SEResNet50_IR': "/data/projects/aisc_facecomp/models/PFR/ckpts/CASIA_WebFace_SEResNet50_IR/Iter_64000_net.pth",
    'ResNet100_IR': "/data/projects/aisc_facecomp/models/PFR/ckpts/CASIA_WebFace_ResNet100_IR/Iter_64000_net.pth",
}
def load_model(backbone_net, device=torch.device('cuda')):
    if backbone_net in backbones:
        net = backbones[backbone_net]
    else:
        print(backbone_net + ' is not available!')

    net.load_state_dict(torch.load(ckpt_path[backbone_net]))
    net.eval()
    net = nn.Sequential(
        transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]),
        net
    )
    net = net.to(device)
    # print(f"loading model {backbone_net}")
    return [net]

def load_models(device=torch.device('cuda')):
    model_pool = []   
    for model_name in backbones.keys():
        print(f"loading model {model_name}")
        model_pool.append(load_model(model_name, device))
    return model_pool