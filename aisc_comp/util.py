from turtle import shape
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
from models.TC_Rank4.main import get_model as get_tcmodel
from models.TC_Rank4.model_irse import IR_50, IR_101, IR_152
from models.CurricularFace.main import load_models as cur_loads
from models.PFR.main import load_models as pfr_loads, load_model as pfr_load
from models.TC_Rank4.main import load_models as tcr4_loads
from third_party.FaceX_Zoo.backbone.backbone_def import BackboneFactory
# from third_party.FaceX_Zoo.training_mode.conventional_training.train import FaceModel
# from third_party.FaceX_Zoo.addition_module.face_lightning.KDF.test_protocol.utils.model_loader import ModelLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MEAN = [127.5 / 255] * 3
STD = [128 / 255] * 3
CONFIG_FILE = "/data/projects/aisc_facecomp/third_party/FaceX_Zoo/training_mode/backbone_conf.yaml"
CKPT_DIR = "/data/projects/aisc_facecomp/third_party/FaceX_Zoo/ckpt/backbone"

def img_loader(path):
    try:
        img = cv2.imread(path)
        if len(img.shape) == 2:
            img = np.stack([img] * 3, 2)
        return img
    except IOError:
        print('Cannot load image ' + path)

class RAW_LFW(data.Dataset):
    def __init__(self, root, file_list, transform=None, loader=img_loader):
        self.root = root
        self.file_list = file_list
        self.transform = transform
        self.loader = loader
        self.nameLs = []
        self.nameRs = []
        self.folds = []
        self.flags = []

        with open(file_list) as f:
            pairs = f.read().splitlines()[1:]
        for i, p in enumerate(pairs):
            p = p.split('\t')
            if len(p) == 3:
                nameL = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
                nameR = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[2]))
                fold = i // 600
                flag = 1
            elif len(p) == 4:
                nameL = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
                nameR = p[2] + '/' + p[2] + '_' + '{:04}.jpg'.format(int(p[3]))
                fold = i // 600
                flag = -1
            self.nameLs.append(nameL)
            self.nameRs.append(nameR)
            self.folds.append(fold)
            self.flags.append(flag)

    def __getitem__(self, index):

        img_l = self.loader(os.path.join(self.root, self.nameLs[index]))
        img_r = self.loader(os.path.join(self.root, self.nameRs[index]))
        imglist = [img_l, cv2.flip(img_l, 1), img_r, cv2.flip(img_r, 1)]

        if self.transform is not None:
            for i in range(len(imglist)):
                imglist[i] = self.transform(imglist[i])

            imgs = imglist
            return imgs
        else:
            imgs = [torch.from_numpy(i) for i in imglist]
            return imgs

    def __len__(self):
        return len(self.nameLs)

class LFW(data.Dataset):
    IMG_DIR="/data/projects/aisc_facecomp/data"
    def __init__(self, input_dir, transform) -> None:
        super().__init__()
        self.input_dir = input_dir
        self.transform = transform
        self.origin_image_files = [os.path.join(self.IMG_DIR, "{:04d}.png".format(i + 1)) for i in range(3000)]
        self.compare_image_files = [os.path.join(self.IMG_DIR, "{:04d}_compare.png".format(i + 1)) for i in range(3000)]

        assert len(self.origin_image_files) == 3000
        assert len(self.compare_image_files) == 3000
        
    def __getitem__(self, index):
        origin_image_file = self.origin_image_files[index]
        compare_image_file = self.compare_image_files[index]
        
        origin_image = Image.open(origin_image_file).convert('RGB')
        compare_image = Image.open(compare_image_file).convert('RGB')
        
        if self.transform:
            origin_image = self.transform(origin_image)
            compare_image = self.transform(compare_image)
        
        return origin_image, origin_image_file, compare_image, compare_image_file
    
    def __len__(self):
        return len(self.origin_image_files)

class KP_LFW(data.Dataset):
    IMG_DIR="/data/projects/aisc_facecomp/data"
    MASK_DIR={
        "raw_keypoint": "/data/projects/aisc_facecomp/keypoints/show_masks",
        "comp_keypoint": "/data/projects/aisc_facecomp/keypoints/show_masks_comp",
        "comp_keypoint_v2": "/data/projects/aisc_facecomp/keypoints/show_masks_comp_v2",
        "comp_keypoint_2p_2eye": "/data/projects/aisc_facecomp/keypoints/show_masks_comp_2p_2eye",
        "comp_keypoint_3p_2eye_1mouse": "/data/projects/aisc_facecomp/keypoints/show_masks_comp_3p_2eye_1mouse",
        "comp_keypoint_3p_2eye_1nose": "/data/projects/aisc_facecomp/keypoints/show_masks_comp_3p_2eye_1nose",
        "hiltert_m5": "/data/projects/aisc_facecomp/masks/hilbert@m5",
        "hiltert_m7": "/data/projects/aisc_facecomp/masks/hilbert@m7",
        "comp_v2": "/data/projects/aisc_facecomp/keypoints/show_masks_seg_fill1254_v2",
        "origin_v2": "/data/projects/aisc_facecomp/keypoints/show_masks_seg_fill1254_origin_v2",
        "origin_tmode": "/data/projects/aisc_facecomp/keypoints/show_masks_seg_fill1254_origin_json_tmode",
        "origin_tmode_v2": "/data/projects/aisc_facecomp/keypoints/show_masks_seg_fill1254_origin_json_tmode_v2",
        "comp_tmode": "/data/projects/aisc_facecomp/keypoints/show_masks_seg_fill1254_comp_json_tmode",
        "comp_tmode_v2": "/data/projects/aisc_facecomp/keypoints/show_masks_seg_fill1254_comp_json_tmode_v2",
        "comp_tmode_v3": "/data/projects/aisc_facecomp/keypoints/fill1254_comp_tmode_v3",
        "fixed_mask_v1": "/data/projects/aisc_facecomp/keypoints/fixed_mask_v1",
        "fixed_mask_v2": "/data/projects/aisc_facecomp/keypoints/fixed_mask_v2",
    }
    # KEYPOINT="/data/projects/aisc_facecomp/keypoints/show_masks"
    def __init__(self, input_dir, transform, mtype="raw_keypoint", sample=3000) -> None:
        super().__init__()
        self.transform = transform
        self.mask_dir = self.MASK_DIR[mtype]
        self.origin_image_files = [os.path.join(input_dir, "{:04d}.png".format(i + 1)) for i in range(sample)]
        self.compare_image_files = [os.path.join(self.IMG_DIR, "{:04d}_compare.png".format(i + 1)) for i in range(sample)]
        
        if mtype == "comp_v2":
            self.keypoints_images_files = [os.path.join(self.mask_dir, "{:04d}_compare.png".format(i + 1)) for i in range(sample)]
        else:
            self.keypoints_images_files = [os.path.join(self.mask_dir, "{:04d}.png".format(i + 1)) for i in range(sample)]
        
        assert len(self.origin_image_files) == sample
        assert len(self.compare_image_files) == sample
        
    def __getitem__(self, index):
        origin_image_file = self.origin_image_files[index]
        compare_image_file = self.compare_image_files[index]
        keypoints_image_file = self.keypoints_images_files[index]
        
        origin_image = Image.open(origin_image_file).convert('RGB')
        compare_image = Image.open(compare_image_file).convert('RGB')
        keypoints_image = Image.open(keypoints_image_file).convert('RGB')
        
        if self.transform:
            origin_image = self.transform(origin_image)
            compare_image = self.transform(compare_image)
            keypoints_image = self.transform(keypoints_image)
        
        return origin_image, origin_image_file, compare_image, compare_image_file, keypoints_image, keypoints_image_file
    
    def __len__(self):
        return len(self.origin_image_files)

class KP_LFW_MULTGT(data.Dataset):
    IMG_DIR="/data/projects/aisc_facecomp/data"
    KEYPOINT="/data/projects/aisc_facecomp/keypoints/show_masks"
    MULTITGT="/data/projects/aisc_facecomp/raw_data/game3000"
    def __init__(self, input_dir, transform) -> None:
        super().__init__()
        self.transform = transform
        self.origin_image_files = [os.path.join(input_dir, "{:04d}.png".format(i + 1)) for i in range(3000)]
        self.compare_image_files = [os.path.join(input_dir, "{:04d}_compare.png".format(i + 1)) for i in range(3000)]
        self.keypoints_images_files = [os.path.join(self.KEYPOINT, "{:04d}.png".format(i + 1)) for i in range(3000)]
        self.tgt_dir = [os.path.join(self.MULTITGT, "{:04d}_compare".format(i + 1)) for i in range(3000)]
        
        assert len(self.origin_image_files) == 3000
        
    def __getitem__(self, index):
        origin_image_file = self.origin_image_files[index]
        keypoints_image_file = self.keypoints_images_files[index]
        tgt_image_dir = self.tgt_dir[index]
        
        origin_image = Image.open(origin_image_file).convert('RGB')
        keypoints_image = Image.open(keypoints_image_file).convert('RGB')
        compare_image_file = [os.path.join(tgt_image_dir, file) for file in os.listdir(tgt_image_dir)]
        compare_images = [Image.open(file).convert('RGB') for file in compare_image_file]
        
        if self.transform:
            origin_image = self.transform(origin_image)
            keypoints_image = self.transform(keypoints_image)
            compare_images_ts = []
            for img in compare_images:
                compare_images_ts.append(self.transform(img)[None])
            compare_images = torch.concat(compare_images_ts)
        
        return origin_image, origin_image_file, compare_images, compare_image_file, keypoints_image, keypoints_image_file
    
    def __len__(self):
        return len(self.origin_image_files)

def central_mask():
    mask = torch.zeros((1, 3, 112, 112))
    _, _, w, h = mask.shape
    pixel_nums = w * h * 0.1
    patch_nums = 1
    
    # 块的宽度
    l = int(np.sqrt(pixel_nums / patch_nums))
    
    start_x = start_y = (w - l) // 2
    mask[:, :, start_x:start_x+l, start_y:start_y+l] = 1.0
    
    return mask

def square_patch(loc="center"):
    # rate 0.1
    img = torch.zeros((1, 3, 112, 112))
    h, w = 36, 35
    tnt_pixel = math.floor(112 * 112 * 0.1)
    offset = tnt_pixel - h * w
    mask = torch.ones((h, w))
    mask[-1, offset:] = 0

    if loc == "center":
        start_x, start_y = 38, 38
    elif loc == "bottom":
        start_x, start_y = 70, 38
    elif loc == "top":
        start_x, start_y = 6, 38
    else:
        raise NotImplementedError
    
    img[:, :, start_x:start_x+h, start_y:start_y+w] = mask
    return img

def patch5_mask(): 
    mask = torch.zeros((1, 3, 112, 112))
    _, _, w, h = mask.shape
    pixel_nums = w * h * 0.1
    patch_nums = 5
    
    # 块的宽度
    l = int(np.sqrt(pixel_nums / patch_nums))
    inner_box_l = 3*l + 2
    
    start_x = start_y = (w - l) // 2
    mask[:, :, start_x:start_x+l, start_y:start_y+l] = 1.0
    mask[:, :, start_x-l-1:start_x-1, start_y-l-1:start_y-1] = 1.0
    mask[:, :, start_x-l-1:start_x-1, start_y+l+1:start_y+1+l+l] = 1.0
    mask[:, :, start_x+l+1:start_x+l+1+l, start_y-l-1:start_y-1] = 1.0
    mask[:, :, start_x+l+1:start_x+l+1+l, start_y+l+1:start_y+1+l+l] = 1.0
    
    return mask
    
# def cosine_similarity(tensor_a, tensor_b): # (x, 512) (y, 512) -> (x, y)
#     norm1 = torch.norm(tensor_a, dim=-1).reshape(-1, 1)
#     norm2 = torch.norm(tensor_b, dim=-1).reshape(1, -1)
#     end_norm = torch.matmul(norm1, norm2)
#     cosine_similarity = torch.matmul(tensor_a, tensor_b.t()) / end_norm
#     return cosine_similarity

def feature_layer(model):
    return model[1].conv2d_4a

def get_model(idx=-1):
    pretrained_weights = ['vggface2', 'casia-webface']
    models = []
    for weights in pretrained_weights:
        print(f"loading weight {weights}")
        model = InceptionResnetV1(pretrained=weights).to(device)
        model.eval()
        models.append(nn.Sequential(
            T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]),
            model
        ))
    models += cur_loads()
    models += pfr_loads()
    models += tcr4_loads()

    if idx != -1:
        pick_models = []
        for i in idx:
            pick_models.append(models[i])
            print("loading model {}".format(i))
        return pick_models
    return models

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
        return get_model(-1)
    models = []
    if 0 in idx:
        model = InceptionResnetV1(pretrained='vggface2').to(device)
        # print(f"loading weight vggface2")
        model.eval()
        models.append(nn.Sequential(
            T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]),
            model
        ))
        print("loading model {}".format(0))
    if 1 in idx:
        model = InceptionResnetV1(pretrained='casia-webface').to(device)
        # print(f"loading weight casia-webface")
        model.eval()
        models.append(nn.Sequential(
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
        print("loading model {}".format(14))
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

def save_image_file(img, file):
    img = PIL.Image.fromarray(np.uint8(np.round(img)))
    img.save(file, quality=100, subsampling=0)
    # img.save(file, quality=100)

## TI
def gkern(kernlen=15, nsig=3):
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel

## dem
def ensemble_input_diversity_v2(input_tensor, idx, bound=1.5, n=10):                             
    h, w = input_tensor.shape[2:] 
    tb = [(int(r * h), int(r * w)) for r in np.linspace(1, bound, n + 1)[1:]]
    rnd_h = torch.randint(h, tb[idx][0], ())
    rnd_w = torch.randint(w, tb[idx][1], ())
    rescaled = F.interpolate(input_tensor,
                             size=[rnd_h, rnd_w],
                             mode='bilinear',
                             align_corners=True)
    h_rem = tb[idx][0] - rnd_h
    w_rem = tb[idx][1] - rnd_w
    pad_top = torch.randint(0, h_rem, ())
    pad_bottom = h_rem - pad_top
    pad_left = torch.randint(0, w_rem, ())
    pad_right = w_rem - pad_left
    padded = nn.ConstantPad2d((pad_left, pad_right, pad_top, pad_bottom),
                              0.)(rescaled)
    padded = nn.functional.interpolate(padded, [h, w], mode='bilinear', align_corners=False)
    return padded

def ensemble_input_diversity(input_tensor,
                             idx,
                             image_shape=[112, 112],
                             ):
    width = max(*image_shape)
    tb = [int(r * width) for r in [1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40, 1.45, 1.50]]
    rnd = torch.randint(width, tb[idx], ())
    rescaled = F.interpolate(input_tensor,
                             size=[rnd, rnd],
                             mode='bilinear',
                             align_corners=True)
    h_rem = tb[idx] - rnd
    w_rem = tb[idx] - rnd
    pad_top = torch.randint(0, h_rem, ())
    pad_bottom = h_rem - pad_top
    pad_left = torch.randint(0, w_rem, ())
    pad_right = w_rem - pad_left
    padded = nn.ConstantPad2d((pad_left, pad_right, pad_top, pad_bottom),
                              0.)(rescaled)
    padded = nn.functional.interpolate(padded, image_shape, mode='bilinear', align_corners=False)
    return padded

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

# PI
def project_kern(kern_size):
    kern = np.ones((kern_size, kern_size), dtype=np.float32) / (kern_size ** 2 - 1)
    kern[kern_size // 2, kern_size // 2] = 0.0
    kern = kern.astype(np.float32)
    stack_kern = np.stack([kern, kern, kern])
    stack_kern = np.expand_dims(stack_kern, 1)
    stack_kern = torch.tensor(stack_kern)
    return stack_kern, kern_size // 2

def project_noise(x, stack_kern, padding_size):
    x = F.conv2d(x, stack_kern, padding = (padding_size, padding_size), groups=3)
    return x

# TI
class Translation_Kernel:
    def __init__(self, len_kernel=15, nsig=3, kernel_name='gaussian'):
        self.kernel_name = kernel_name
        self.len_kernel = len_kernel
        self.nsig = nsig

    def kernel_generation(self):
        if self.kernel_name == 'gaussian':
            kernel = self.gkern(self.len_kernel, self.nsig).astype(np.float32)
        elif self.kernel_name == 'linear':
            kernel = self.lkern(self.len_kernel).astype(np.float32)
        elif self.kernel_name == 'uniform':
            kernel = self.ukern(self.len_kernel).astype(np.float32)
        else:
            raise NotImplementedError

        stack_kernel = np.stack([kernel, kernel, kernel])
        stack_kernel = np.expand_dims(stack_kernel, 1)
        return stack_kernel

    def gkern(self, kernlen=15, nsig=3):
        """Returns a 2D Gaussian kernel array."""
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def ukern(self, kernlen=15):
        kernel = np.ones((kernlen,kernlen))* 1.0 /(kernlen*kernlen)
        return kernel

    def lkern(self, kernlen=15):
        kern1d = 1-np.abs(np.linspace((-kernlen+1)/2, (kernlen-1)/2, kernlen)/(kernlen+1)*2)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

# gaussian_kernel for filter high frequency information of images
def gaussian_kernel(device, kernel_size=15, sigma=2, channels=3):
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()  # kernel_size*kernel_size*2
    mean = (kernel_size - 1)/2.
    variance = sigma**2.
    gaussian_kernel = (1./(2.*math.pi*variance)) * torch.exp(-torch.sum((xy_grid - mean)**2., dim=-1) / (2*variance))
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)
    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels, padding=(kernel_size-1)//2, bias=False)
    gaussian_filter.weight.data = gaussian_kernel.to(device)
    gaussian_filter.weight.requires_grad = False
    return gaussian_filter

# SI
def scale_transform(input_tensor, m=5):
    outs = [(input_tensor) / (2**i) for i in range(m)]
    return outs
    

# ANDA
def get_theta(i, j):
    theta = torch.tensor([[[1, 0, i], [0, 1, j]]], dtype=torch.float)
    return theta

def get_thetas(n, min_r=-0.5, max_r=0.5):
    range_r = torch.linspace(min_r, max_r, n)
    thetas = []
    for i in range_r:
        for j in range_r:
            thetas.append(get_theta(i, j))
    thetas = torch.cat(thetas, dim=0)
    return thetas

def translation(thetas, imgs):
    grids = F.affine_grid(thetas, imgs.size()).to(imgs.device)
    output = F.grid_sample(imgs, grids)
    return output
