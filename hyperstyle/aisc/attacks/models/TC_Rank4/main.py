import torch
from torch import nn
from torchvision import transforms
import torch.multiprocessing
from aisc.attacks.models.TC_Rank4.model_irse import IR_50, IR_101, IR_152
'''
https://github.com/BruceQFWang/TIANCHI_BlackboxAdversial
'''
def get_model(model, param, device=torch.device('cuda')):
    net = model([112,112]).to(device)
    net.load_state_dict(torch.load(param, map_location=device))
    net.eval()
    net = nn.Sequential(
        transforms.Resize(112),
        transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]),
        net
    )
    return net

def load_models(device=torch.device('cuda')):
    model_pool = []   
    model_pool.append(get_model(IR_50, '/data/projects/aisc_facecomp/models/TC_Rank4/models/backbone_ir50_ms1m_epoch120.pth', device))
    model_pool.append(get_model(IR_50, '/data/projects/aisc_facecomp/models/TC_Rank4/models/Backbone_IR_50_LFW.pth', device))
    model_pool.append(get_model(IR_101, '/data/projects/aisc_facecomp/models/TC_Rank4/models/Backbone_IR_101_Batch_108320.pth', device))
    model_pool.append(get_model(IR_152, '/data/projects/aisc_facecomp/models/TC_Rank4/models/Backbone_IR_152_MS1M_Epoch_112.pth', device))
    # print(f"loading tc_rank4 four models")

    return model_pool
