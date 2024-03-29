import os
import sys
from PIL import Image
import torch
import torchvision.transforms as transforms
from models.stylegan2.model import Generator
import pickle

sys.path.append(".")
sys.path.append("..")

from utils.common import tensor2im
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--image_path', default=None, type=str)
parser.add_argument('--outputs_path', default="./outputs", type=str)
args = parser.parse_args()
print(args)

EXPERIMENT_ARGS = {
    "model_path": "./pretrained_models/hyperstyle_ffhq.pt",
    "w_encoder_path": "./pretrained_models/faces_w_encoder.pt",
    "image_path": "./notebooks/images/face_image.jpg",
    "transform": transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
}

model_path = EXPERIMENT_ARGS['model_path']
def __get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
    return d_filt

image_path = args.image_path
input_image = Image.open(image_path)

img_transforms = EXPERIMENT_ARGS['transform']
transformed_image = img_transforms(input_image)
outputs_path = args.outputs_path
os.makedirs(outputs_path, exist_ok=True)
identity = os.path.splitext(os.path.basename(args.image_path))[0]
face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
with torch.no_grad():
    with open("/data/projects/hyperstyle/aisc/latent/{}.pkl".format(identity), "rb") as fp:
        latent_results = pickle.load(fp)

    inputs = transformed_image.unsqueeze(0).cuda()
    weights_deltas = latent_results["weights_deltas"]
    codes = latent_results["codes"]

    output_size = 1024
    decoder = Generator(output_size, 512, 8, channel_multiplier=2).cuda()
    ckpt = torch.load(model_path, map_location='cpu')
    decoder.load_state_dict(__get_keys(ckpt, 'decoder'), strict=True)
    images, _ = decoder([codes], weights_deltas=weights_deltas, input_is_latent=True, randomize_noise=False, return_latents=False)
    images = face_pool(images)
    # resize_amount = (1024, 1024)
    resize_amount = (256, 256)
    final_rec = tensor2im(images[0]).resize(resize_amount)
    final_rec.save(os.path.join(outputs_path, os.path.basename(image_path)))
