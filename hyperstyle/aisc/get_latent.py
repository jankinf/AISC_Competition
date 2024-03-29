# # HyperStyle Inference Notebook
# ## Prepare Environment and Download HyperStyle Code
#@title Clone HyperStyle Repo and Install Ninja { display-mode: "form" }

import os
import sys
import pprint
from PIL import Image
import torch
import torchvision.transforms as transforms
import pickle

sys.path.append(".")
sys.path.append("..")

from utils.common import tensor2im
from utils.inference_utils import run_inversion
from utils.model_utils import load_model
import argparse

EXPERIMENT_ARGS = {
    "model_path": "./pretrained_models/hyperstyle_ffhq.pt",
    "w_encoder_path": "./pretrained_models/faces_w_encoder.pt",
    "image_path": "./notebooks/images/face_image.jpg",
    "transform": transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
}

model_path = EXPERIMENT_ARGS['model_path']

net, opts = load_model(model_path, update_opts={"w_encoder_checkpoint_path": EXPERIMENT_ARGS['w_encoder_path']})
print('Model successfully loaded!')
pprint.pprint(vars(opts))


n_iters_per_batch = 5
opts.n_iters_per_batch = n_iters_per_batch
opts.resize_outputs = False  # generate outputs at full resolution
# opts.resize_outputs = True  # generate outputs at full resolution

img_transforms = EXPERIMENT_ARGS['transform']

dir_path = "/data/projects/hyperstyle/aisc/aligned1024"
identites = [
    "攻击者1",
    "攻击者2",
    "攻击者3",
    "受害者1",
    "受害者2",
    "受害者3",
]

with torch.no_grad():
    for identity in identites:
        image_path = os.path.join(dir_path, identity, "0.png")
        input_image = Image.open(image_path)
        transformed_image = img_transforms(input_image)

        inputs = transformed_image.unsqueeze(0).cuda()
        y_hat, latent, weights_deltas, codes = run_inversion(inputs, net, opts, return_intermediate_results=False)

        latent_results = {
            "weights_deltas": weights_deltas,
            "codes": codes,
        }

        with open("/data/projects/hyperstyle/aisc/latent/{}_0.pkl".format(identity), "wb") as fp:
            pickle.dump(latent_results, fp)
    