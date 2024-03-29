import torchvision.transforms as T
import argparse
import os
import pickle
from models.stylegan2.model import Generator
from models.encoders.psp import pSp
from aisc.attacks.method.encode_s_sproof import encode_s
from aisc.attacks.utils import *
from argparse import Namespace
import copy
from models.aenet import AENet

parser = argparse.ArgumentParser(description='attacks in PyTorch')

parser.add_argument('--steps', default=100, type=int, help='max iteration')
parser.add_argument('--alpha', default=0.01, type=float, help='step size for each iter')

parser.add_argument('--n_ens', default=1, type=int)
parser.add_argument('--resize_rate', default=1.15, type=float, help='DIM resize_rate')
parser.add_argument('--diversity_prob', default=0.7, type=float, help='DIM diversity_prob')

parser.add_argument('--model_idx', nargs='+', default=-1, type=int)

args = parser.parse_args()


def pretrain(model, state_dict):
    own_state = model.state_dict()
    for name, param in state_dict.items():
        realname = name.replace('module.', '')
        if realname in own_state:
            if isinstance(param, torch.nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            try:
                own_state[realname].copy_(param)
            except:
                print('While copying the parameter named {}, '
                      'whose dimensions in the model are {} and '
                      'whose dimensions in the checkpoint are {}.'
                      .format(realname, own_state[name].size(), param.size()))
                print("But don't worry about it. Continue pretraining.")


def __get_pretrained_w_encoder(opts):
    print("Loading pretrained W encoder...")
    opts_w_encoder = vars(copy.deepcopy(opts))
    opts_w_encoder['checkpoint_path'] = opts.w_encoder_checkpoint_path
    opts_w_encoder['encoder_type'] = opts.w_encoder_type
    opts_w_encoder['input_nc'] = 3
    opts_w_encoder = Namespace(**opts_w_encoder)
    w_net = pSp(opts_w_encoder)
    w_net = w_net.encoder
    w_net.eval()
    w_net.cuda()
    return w_net

def __get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
    return d_filt

EXPERIMENT_ARGS = {
    "model_path": "./pretrained_models/hyperstyle_ffhq.pt",
    "w_encoder_path": "./pretrained_models/faces_w_encoder.pt",
    "image_path": "./notebooks/images/face_image.jpg",
    "transform": T.Compose([T.Resize((256, 256)), T.ToTensor(), T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
}

model_path = EXPERIMENT_ARGS['model_path']
img_transforms = EXPERIMENT_ARGS['transform']
preprocess = T.Compose([
    T.ToTensor()
])

src = "攻击者3"
target = "受害者3"
image_path = "/data/projects/hyperstyle/aisc/aligned256/{}/0.png".format(src)
target_path = "/data/projects/hyperstyle/aisc/aligned256/{}/0.png".format(target)
target1024_path = "/data/projects/hyperstyle/aisc/aligned1024/{}/0.png".format(target)
mask_path = "/data/projects/hyperstyle/aisc/data/tmask/aligned256_m3/{}/0.png".format(src)
identity = os.path.splitext(os.path.basename(image_path))[0]
latent_path = "/data/projects/hyperstyle/aisc/latent/{}_0.pkl".format(target)
out_name = "{}_{}".format(src, target)

save_path = "/data/projects/hyperstyle/aisc/data/adv_encode_final/{}".format(out_name)
os.makedirs(save_path, exist_ok=True)
def main():
    with open(latent_path, "rb") as fp:
        latent_results = pickle.load(fp)

    compare_image = Image.open(target1024_path)
    compare_image1024 = img_transforms(compare_image).unsqueeze(0).cuda()
    
    origin_image = preprocess(Image.open(image_path)).unsqueeze(0).cuda()
    compare_image = preprocess(Image.open(target_path)).unsqueeze(0).cuda()
    keypoints_image = torch.from_numpy(np.array(Image.open(mask_path)))
    keypoints_image = torch.stack([keypoints_image] * 3, dim=0).unsqueeze(0).cuda()
    
    weights_deltas = latent_results["weights_deltas"]
    codes = latent_results["codes"]

    output_size = 1024
    decoder = Generator(output_size, 512, 8, channel_multiplier=2).cuda()
    ckpt = torch.load(model_path, map_location='cpu')
    decoder.load_state_dict(__get_keys(ckpt, 'decoder'), strict=True)
    opts = ckpt['opts']
    opts['checkpoint_path'] = model_path
    opts['load_w_encoder'] = True
    opts['w_encoder_checkpoint_path'] = EXPERIMENT_ARGS['w_encoder_path']
    opts = Namespace(**opts)
    encoder = __get_pretrained_w_encoder(opts)

    models = get_fastmodel(args.model_idx)
    black_models = get_fastmodel([2, 5, 6])

    as_model = AENet(num_classes=2)
    checkpoint = torch.load('/data/projects/CelebA-Spoof/src/ckpt_iter.pth.tar')
    pretrain(as_model, checkpoint['state_dict'])
    as_model = nn.Sequential(
        T.Resize(224),
        as_model
    ).cuda()
    as_model.eval()
    
    advs = encode_s(
        codes,
        weights_deltas,
        encoder,
        decoder,
        origin_image,
        compare_image,
        compare_image1024,
        keypoints_image,
        as_model,
        models,
        black_models,
        args.alpha,
        args.steps,
        resize_rate=args.resize_rate, 
        diversity_prob=args.diversity_prob,
        n_ens=args.n_ens,
        save_path=save_path,
    )
    
    save_img(os.path.join(save_path, "final.png"), advs[0].detach().permute(1, 2, 0).cpu())
    
if __name__ == '__main__':
    main()


        