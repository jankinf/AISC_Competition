import functools
import torchvision.transforms as T
from torch.utils import data
import argparse
import os
import numpy as np
from util import *

parser = argparse.ArgumentParser(description='attacks in PyTorch')
parser.add_argument('--input_dir', default='/data/projects/aisc_facecomp/data', type=str, help='directory of clean examples')
parser.add_argument('--device', default='0', type=str, help='gpu device')
parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 0)')
parser.add_argument('--steps', default=10, type=int, help='max iteration')
parser.add_argument('--alpha', default=1.6, type=float, help='stepsize')
parser.add_argument('--model_w_idx', nargs='+', default=[0], type=int, help='whitebox model idx')
parser.add_argument('--model_b_idx', nargs='+', default=[1], type=int, help='blackbox model idx')

parser.add_argument('--mtype', default="raw_keypoint", type=str, choices=["raw_keypoint", "comp_keypoint", "hiltert_m5", "hiltert_m7"])
parser.add_argument('--hard_ctl', action="store_true")
parser.add_argument('--hard_ctl_sample', default=1500, type=int)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.device

def attack(images, 
        compare_images,
        keypoints_images,
        models_w, 
        models_b, 
        steps, 
        alpha, 
        n,
        resize_rate=1.15, 
        diversity_prob=0.7,
        masktype='keypoints'
    ):

    images = images.clone().detach()
    noise = torch.zeros_like(images).clone().detach()   
    batch_size = images.shape[0]
    n_model = len(models_w)
    
    criterion = nn.CosineSimilarity(dim=3, eps=1e-8)
    
    if masktype == 'central':
        mask = central_mask().to(images.device)
    elif "square" in masktype:
        assert masktype in ['square_center', 'square_top', 'square_bottom']
        loc = masktype.split("_")[1]
        mask = square_patch(loc=loc).to(images.device)
    elif masktype == 'patch5':
        mask = patch5_mask().to(images.device)
    elif masktype == 'keypoints':
        mask = torch.zeros_like(images).to(images.device)
        mask[(images - keypoints_images) != 0] = 1.0
        mask = mask.detach()
    else:
        raise NotImplementedError
    
    images = (1 - mask) * images + mask * compare_images
    advs = images.clone().detach()
    
    compare_feat_w = []
    compare_feat_b = []
    with torch.no_grad():
        for model in models_w: 
            out = model(compare_images).detach() 
            compare_feat_w.append(out)
        for model in models_b: 
            out = model(compare_images).detach() 
            compare_feat_b.append(out)
    compare_logits_w = torch.concat(compare_feat_w, dim=0)
    compare_logits_b = torch.concat(compare_feat_b, dim=0)

    best_loss = -1
    best_iter = -1
    for t in range(steps):
        noise.requires_grad = True
        masked_noise = noise * mask
        advs = masked_noise + images

        disi_advs = [input_diversity(advs, resize_rate, diversity_prob) for i in range(n)]
        batch_advs = torch.cat(disi_advs, dim=0)
        aug_num = batch_advs.shape[0] // batch_size 

        origin_feat_w = []
        for model in models_w:   
            out = model(batch_advs)  
            origin_feat_w.append(out) 
        origin_logits_w = torch.concat(origin_feat_w, dim=0).reshape(n_model, aug_num, batch_size, -1)
        origin_logits_w = origin_logits_w.permute(1, 0, 2, 3)

        # todo: blackbox eval
        with torch.no_grad():
            origin_feat_b = []
            for model in models_b:   
                out = model(advs) 
                origin_feat_b.append(out) 
        origin_logits_b = torch.concat(origin_feat_b, dim=0).reshape(len(models_b), 1, batch_size, -1)
        origin_logits_b = origin_logits_b.permute(1, 0, 2, 3)
        
        # loss_w backward
        loss_w = criterion(origin_logits_w, compare_logits_w)
        loss_b = criterion(origin_logits_b, compare_logits_b)
        print("iter: {}, alpha: {}, loss_w: {}, mean_loss_w: {}".format(t, alpha, loss_w.detach().cpu().numpy(), loss_w.mean().item()))
        print("iter: {}, alpha: {}, loss_b: {}, mean_loss_b: {}".format(t, alpha, loss_b.detach().cpu().numpy(), loss_b.mean().item()))
        if best_loss < loss_b.mean().item():
            best_loss = loss_b.mean().item()
            best_iter = t
        loss_w = loss_w.sum()
        loss_w.backward()
        
        # noise = noise + alpha * torch.sign(noise.grad)
        cur_grad = noise.grad
        grad_norm = torch.norm(cur_grad.view(batch_size, -1), p=2, dim=1) + 1e-8
        grad = cur_grad / grad_norm.view(batch_size, 1, 1, 1)
        noise = noise + alpha * grad
        
        advs =  images + noise * mask
        advs = torch.clamp(advs, min=0, max=1)
        noise = (advs - images).detach()

    # prediction
    with torch.no_grad():
        advs_logits_w = []
        for model in models_w:
            advs_logits_w.append(model(advs).detach())
        advs_logits_w = torch.concat(advs_logits_w, dim=0)

        cos_after_w = criterion(advs_logits_w, compare_logits_w).detach()

        advs_logits_b = []
        for model in models_b:
            advs_logits_b.append(model(advs).detach())
        advs_logits_b = torch.concat(advs_logits_b, dim=0)

        cos_after_b = criterion(advs_logits_b, compare_logits_b).detach()
    print(f'cos_after_w: {cos_after_w}')
    print(f'cos_after_b: {cos_after_b}')

    print('best cos of blackbox: {}, idx: {}'.format(best_loss, best_iter))
    exit()
    return advs, cos_after_w.sum().item()


def main():
    preprocess = T.Compose([
        T.ToTensor()
    ])
    
    # images dataset
    lfw = KP_LFW(args.input_dir, preprocess, mtype=args.mtype)
    if args.hard_ctl:
        index = list(np.load("/data/projects/aisc_facecomp/hard_ctl.npy"))
        index = index[:args.hard_ctl_sample]
        lfw = torch.utils.data.Subset(lfw, index)
    lfw_loader = data.DataLoader(lfw, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)
    print("{}\n".format(os.path.basename(args.input_dir)))
    print("{}\n".format(args))
    print("loading whitebox models")
    models_w = get_fastmodel(args.model_w_idx)
    print("loading blackbox models")
    models_b = get_fastmodel(args.model_b_idx)
    # process images
    # mean_cossim = 0

    # advs dir path
    # save_path = args.output_dir
    # os.makedirs(save_path, exist_ok=True)
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

        origin_image = origin_image.cuda()
        compare_image = compare_image.cuda()
        keypoints_image = keypoints_image.cuda()
        
        advs, cossim = attack(
            origin_image,
            compare_image,
            models_w,
            models_b,
            # args.max_epsilon,
            args.steps,
            args.alpha/255.,
            # args.max_epsilon / 255 / args.steps,
        )
    
    
if __name__ == '__main__':
    main()


        