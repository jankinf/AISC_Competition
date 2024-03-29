import torch
import torch.nn as nn

import numpy as np

from util import *

    

def optmask_dem(images, 
        compare_images,
        keypoints_images,
        models, 
        steps, 
        alpha, 
        bound=1.5,
        n=4,
        masktype='keypoints',
        use_mi=True,
    ):

    images = images.clone().detach()
    noise = torch.zeros_like(images).clone().detach()   
    batch_size = images.shape[0]
    if use_mi:
        momentum = torch.zeros_like(images, requires_grad=False)
    n_model = len(models)

    criterion = nn.CosineSimilarity(dim=1, eps=1e-8)
    
    if masktype == 'central':
        mask = central_mask().to(images.device)
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
    for i in range(steps):
        noise.requires_grad = True
        masked_noise = noise * mask
        advs = masked_noise + images
        
        origin_logits = 0
        x_d = [advs] + [ensemble_input_diversity_v2(advs, i, bound, n) for i in range(n)]
        advs = torch.cat(x_d, dim=0)
        aug_num = advs.shape[0] // batch_size 

        origin_feat = []
        for model in models:   
            out = model(advs) 
            origin_feat.append(out) 
        origin_logits = torch.concat(origin_feat, dim=0).reshape(len(models), aug_num, batch_size, -1)
        origin_logits = origin_logits.permute(1, 0, 2, 3)
       
        # loss backward
        loss = criterion(origin_logits, compare_logits)
        loss = loss.sum()
        # print("iter: {}, loss: {}".format(i, loss))
        loss.backward()
        
        if use_mi:
            momentum = 1.0 * momentum + noise.grad / torch.mean(torch.abs(noise.grad), dim=(1,2,3), keepdim=True)
            noise = noise + alpha * torch.sign(momentum)
        else:
            noise = noise + alpha * torch.sign(noise.grad)
        
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