import torchvision.transforms as T
from torch.utils import data
import argparse
import os
from util import *
import functools

import torch
import torch.nn as nn
from util import *

parser = argparse.ArgumentParser(description='attacks in PyTorch')
parser.add_argument('--input_dir', default='/data/projects/aisc_facecomp/data', type=str, help='directory of clean examples')
parser.add_argument('--output_dir', default='./results/', type=str, help='directory of crafted adversarial examples')

parser.add_argument('--device', default='0', type=str, help='gpu device')
parser.add_argument('--use_gpu', default=True, type=bool, help='use gpu or not')
parser.add_argument('--workers', default=8, type=int, help='number of data loading workers (default: 8)')

parser.add_argument('--batch_size', type=int, default=1, help='mini-batch size (default: 1)')
parser.add_argument('--steps', default=5, type=int, help='max iteration')
parser.add_argument('--alpha', default=1.6, type=float, help='stepsize')

parser.add_argument('--speedup', action="store_true")
parser.add_argument('--sep', default=4, type=int, help='seperate data blocks to speed up')
parser.add_argument('--idx', default=0, type=int, help='index of current data block')
parser.add_argument('--model_idx', nargs='+', default=-1, type=int)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.device


def ensemble_grad(images, 
        compare_images,
        models, 
        steps, 
        alpha,
    ):

    comp_bs = compare_images.shape[0]
    assert comp_bs > 1
    images = images.clone().detach()
    noise = torch.zeros_like(images).clone().detach()   
    alpha = alpha / 255. * 2

    batch_size = images.shape[0]
    n_model = len(models)

    criterion = nn.CosineSimilarity(dim=3, eps=1e-8)
    
    advs = images.clone().detach()

    compare_feat = []
    with torch.no_grad():
        for model in models: 
            out = model(compare_images).detach() 
            compare_feat.append(out)
    compare_logits = torch.concat(compare_feat, dim=0).reshape(1, n_model, comp_bs, -1) # (1, n_model, comp_bs, 512)

    adv_list = []
    noise_list = []
    for t in range(steps):
        noise.requires_grad = True
        advs = noise + images
        
        # todo: sifgsm
        # si_advs = scale_transform(advs, m=4)
        # todo: difgsm
        # di_advs = [ensemble_grad_input_diversity(advs, i) for i in range(4)]    
        # todo: di+sifgsm
        disi_advs = [input_diversity(advs / 2**i) for i in range(4)]
        batch_advs = torch.cat(disi_advs, dim=0)
        aug_num = batch_advs.shape[0] // batch_size 

        origin_feat = []
        for model in models:   
            out = model(batch_advs) 
            origin_feat.append(out) 
        origin_logits = torch.concat(origin_feat, dim=0).reshape(len(models), aug_num, batch_size, -1)
        origin_logits = origin_logits.permute(1, 0, 2, 3)

        # loss backward
        loss = criterion(origin_logits, compare_logits)
        print("iter: {}, loss: {}, mean_loss: {}".format(t, loss.squeeze().detach().cpu().numpy(), loss.mean().item()))

        loss = loss.sum()
        loss.backward()
        
        noise = noise + alpha * torch.sign(noise.grad)
        
        advs =  images + noise
        advs = torch.clamp(advs, min=0, max=1).detach()
        noise = (advs - images).detach()

        adv_list.append(advs.cpu())
        noise_list.append(noise.cpu())

    return adv_list, noise_list

def main():
    save_path = args.output_dir
    os.makedirs(save_path, exist_ok=True)
    
    preprocess = T.Compose([
        T.ToTensor()
    ])
    
    assert args.batch_size == 1
    lfw = KP_LFW_MULTGT(args.input_dir, preprocess)
    
    if args.speedup:
        datalen = len(lfw) # 3000
        assert datalen % args.sep == 0
        bsize = datalen // args.sep
        start, end = bsize * args.idx, bsize * (args.idx + 1)
        index = list(range(start, end))
        lfw = torch.utils.data.Subset(lfw, index)
    lfw_loader = data.DataLoader(lfw, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    
    models = get_fastmodel(args.model_idx)
    os.makedirs(os.path.join(save_path, "adv"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "noise"), exist_ok=True)

    for i, (origin_image, origin_file, compare_image, compare_file, keypoints_image, keypoints_file) in enumerate(lfw_loader):
        # indice = []
        # for k in range(origin_image.shape[0]):
        #     if not os.path.exists(os.path.join(save_path, os.path.basename(origin_file[k]))):
        #         indice.append(k)

        # if not indice:
        #     continue
        # else:
        #     origin_image = origin_image[indice]
        #     compare_image = compare_image[indice]
        #     keypoints_image = keypoints_image[indice]
        #     origin_file = [origin_file[i] for i in indice]

        if args.use_gpu:
            origin_image = origin_image.cuda()
            compare_image = compare_image.cuda()
            keypoints_image = keypoints_image.cuda()

        adv_list, noise_list = ensemble_grad(
            origin_image,
            compare_image[0],
            models,
            args.steps,
            args.alpha,
        )

        for k in range(len(adv_list)):
            name = os.path.splitext(os.path.basename(origin_file[0]))[0]
            img = adv_list[k][0].permute(1, 2, 0)
            img = np.array(img * 255).astype('uint8')
            Image.fromarray(img).save(os.path.join(save_path, "adv", "{}@iter{}.png".format(name, k)))

            noise = noise_list[k][0].permute(1, 2, 0)
            noise = torch.abs(noise).sum(dim=2)
            # noise = torch.clamp(noise, 0).sum(dim=2)
            # noise = -torch.clamp(-noise, 0).sum(dim=2)
            # import pdb; pdb.set_trace()
            noise = (noise - noise.min()) / (noise.max() - noise.min())
            noise = np.array(noise * 255).astype('uint8')
            Image.fromarray(noise).save(os.path.join(save_path, "noise", "{}@iter{}.png".format(name, k)))
        # import pdb; pdb.set_trace()

    
if __name__ == '__main__':
    main()


        