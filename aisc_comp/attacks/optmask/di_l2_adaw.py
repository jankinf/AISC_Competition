import torch
import torch.nn as nn
from util import *

def optmask_di_l2_adaw(images, 
        compare_images,
        keypoints_images,
        models, 
        steps, 
        alpha, 
        n,
        resize_rate, 
        diversity_prob,
        temperature=3,
        masktype='keypoints'
    ):

    images = images.clone().detach()
    noise = torch.zeros_like(images).clone().detach()   
    batch_size = images.shape[0]
    n_model = len(models)

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

    # todo: fill mask with compare image
    images = (1 - mask) * images + mask * compare_images
    advs = images.clone().detach()
    compare_feat = []
    with torch.no_grad():
        for model in models: 
            out = model(compare_images).detach() 
            compare_feat.append(out)
    compare_logits = torch.concat(compare_feat, dim=0).reshape(1, n_model, batch_size, -1) # (1, n_model, batch_size, 512)
    weight = torch.ones((n_model)).cuda()
    for i in range(steps):
        noise.requires_grad = True
        masked_noise = noise * mask
        advs = masked_noise + images
        
        disi_advs = [input_diversity(advs, resize_rate, diversity_prob) for i in range(n)]
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
        loss = loss.mean(dim=[0, 2]) * weight
        weight = F.softmax(-loss * temperature).detach() * n_model
        loss = loss.sum()
        # print("iter: {}, loss: {}".format(i, loss))
        
        loss.backward()

        cur_grad = noise.grad
        grad_norm = torch.norm(cur_grad.view(batch_size, -1), p=2, dim=1) + 1e-8
        grad = cur_grad / grad_norm.view(batch_size, 1, 1, 1)
        noise = noise + alpha * grad
        
        advs =  images + noise * mask
        advs = torch.clamp(advs, min=0, max=1)
        noise = (advs - images).detach()

    # prediction
    with torch.no_grad():
        advs_logits = 0
        for model in models:
            advs_logits += model(advs)
        advs_logits /= len(models)
        
        cos_after = criterion(advs_logits, compare_logits).detach()
    # print(f'advs  :{cos_after}')
    
    return advs, cos_after.sum().item()