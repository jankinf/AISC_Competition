# coding=utf-8
'''Implementation of I-FGSM attack in PyTorch'''
import argparse
import os.path
import torch
from torch.nn import functional as F
from attack.dataprocess import save_img
from attack.eval import verify_output
from src.tsn_predict import pretrain
from src.models import AENet
from src.tsn_predict import TSNPredictor as CelebASpoofDetector
from tqdm import tqdm
from PIL import Image
import torchvision
import numpy as np
from attack.utils import input_diversity
parser = argparse.ArgumentParser(description='I-FGSM attack in PyTorch')
parser.add_argument('--max_epsilon', default=16.0, type=float, help='max magnitude of adversarial perturbations')
parser.add_argument('--num_iter', default=10, type=int, help='max iteration')
parser.add_argument('--log_dir', default='/data/projects/aisc_facecomp/finals/anti_spoofing/logs', type=str, help='dir of logs')
parser.add_argument('--output_dir', default='/data/projects/aisc_facecomp/finals/anti_spoofing/output', type=str, help='directory of crafted adversarial examples')
parser.add_argument('--ckpt', type=str, help='path of ckpt to load')
parser.add_argument('--eval_ckpts', nargs='+', default=[], type=str, help='paths of ckpt')
args = parser.parse_args()

def i_fgsm(x, mask, model, num_iter, eps, alpha):
    x = x.cuda()
    y = torch.zeros(1).type_as(x).long()
    model = model.cuda()
    noise = torch.zeros_like(x, requires_grad=True).clone().detach()   
    mask = 1 - mask.float()
    print(mask[0, 0].sum() / mask[0, 0].numel())

    adv = x.clone()

    with torch.enable_grad():
        for _ in range(num_iter):
            noise.requires_grad = True
            advs_tmp = x + noise * mask
            aug_advs = [advs_tmp] + [input_diversity(advs_tmp, 1.3, 1.0) for _ in range(3)]
            batch_advs = torch.cat(aug_advs, dim=0)
            
            output = model(batch_advs)
            if output.ndim == 1:
                output = output.unsqueeze(0)
            ys = y.repeat(batch_advs.shape[0])

            loss = F.cross_entropy(output, ys, reduction="sum")
            loss.backward()
            
            cur_grad = noise.grad
            noise = noise - alpha * torch.sign(cur_grad)     
            
            noise = torch.clamp(noise, min=-eps, max=eps)
            noise = torch.clamp(noise + x, 0.0, 1.0) - x
            noise = noise.detach()
        adv = x + noise * mask
    
    return adv


def main():
    print(args)
    method = "i_fgsm"
    settings = f"iter{args.num_iter}_eps{args.max_epsilon}"
    output_dir = os.path.join(args.output_dir, method, settings)
    log_dir = os.path.join(args.log_dir, method, settings)
    log_file = os.path.join(log_dir, 'log.log')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    model = AENet(num_classes=2)
    checkpoint = torch.load(args.ckpt)
    pretrain(model, checkpoint['state_dict'])
    model.eval()
    model = torch.nn.Sequential(torchvision.transforms.Resize([224, 224]), model)

    kwargs = {
        "num_iter": args.num_iter,
        "eps": args.max_epsilon / 255,
        "alpha": args.max_epsilon / 255 / args.num_iter,
    }

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    detectors = {ckpt: CelebASpoofDetector(ckpt=ckpt) for ckpt in args.eval_ckpts}
    scores = {ckpt: [] for ckpt in args.eval_ckpts}
    
    mask_dir = "/data/projects/aisc_facecomp/finals/keypoints/tmasks224"
    img_dir = "/data/projects/aisc_facecomp/finals/keypoints/face_aligned224"
    paths = [
        "{}/ht_glasses.png".format(img_dir),
        "{}/ht.png".format(img_dir),
        "{}/liyi_glasses.png".format(img_dir),
        "{}/liyi.png".format(img_dir),
        "{}/qinyi.png".format(img_dir),
        "{}/qinyi02.png".format(img_dir),
        "{}/shibo.png".format(img_dir),
        "{}/shibo2.png".format(img_dir),
    ]
    for path in paths:
        name = os.path.basename(path)
        mask_path = os.path.join(mask_dir, name)
        out_path = os.path.join(output_dir, name)

        im = Image.open(path)
        mask = np.array(Image.open(mask_path)).astype(np.float32) / 255
        mask = np.stack([mask] * 3, axis=0).astype(np.bool8)
        mask = torch.from_numpy(mask).cuda().unsqueeze(0)

        img = transform(im).cuda().unsqueeze(0)
        
        adv_x = i_fgsm(img, mask, model, **kwargs)

        for ckpt in args.eval_ckpts:
            scores[ckpt].append(detectors[ckpt].predict(adv_x).tolist())
        
        save_img(out_path, adv_x[0].detach().permute(1, 2, 0).cpu())
        
    with open(log_file, 'a') as f:
        f.write('-' * 30 + '\n')
        f.write(f'params: {str(args)}\n')
        for ckpt in args.eval_ckpts:
            scores[ckpt] = np.array(scores[ckpt])
            f.write('ckpt: {}\n'.format(ckpt))
            f.write('scores: {}\n'.format(scores[ckpt]))
            f.write('\n')
    print('attack finished')

if __name__ == '__main__':
    main()