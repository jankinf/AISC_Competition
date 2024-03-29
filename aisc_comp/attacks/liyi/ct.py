# mim update by masked noise

import torch
import torch.nn as nn

import numpy as np

from util import *

def ct(
    images, 
    compare_images,
    keypoints_images,
    models, 
    steps, 
    alpha, 
    decay=1.0,
    resize_rate=1.15, 
    diversity_prob=0.7,
    amplification_factor=2.0,
    use_di=False,
    use_ti=False,
    use_si=False,
    use_pi=False,
    use_mi=False,
    n_ens=1,
    ori_mask=False,
    init=False,
):

    images = images.clone().detach()
    advs = images.clone().detach()
    if init:
        noise = 2 * (torch.rand_like(images) - 1) * 0.1
        noise.requires_grad_(True)
    else:
        noise = torch.zeros_like(images, requires_grad=True).clone().detach()   
    grads = torch.zeros_like(images, requires_grad=False)
    
    n_model = len(models)
    batch_size = images.shape[0]
    augnum = n_ens
    
    criterion = nn.CosineSimilarity(dim=3, eps=1e-8)

    # initial di ti si pi
    if use_ti:
        raise NotImplementedError
        gkern = TI_kernel()

    # if use_pi:
    #     amplification = torch.zeros_like(images)      
    #     P_kern, kern_size = project_kern(3)
    #     alpha_beta = alpha * amplification_factor
    #     gamma = alpha_beta

    mask = torch.zeros_like(images).clone().detach()
    mask[keypoints_images == 0] = 1.0
    # mask[(images - keypoints_images) != 0] = 1.0
    print(mask[0, 0].sum())
    
    if not ori_mask:
        images = (1 - mask) * images + mask * compare_images
    
    compare_feat = []
    with torch.no_grad():
        for model in models: 
            out = model(compare_images)                            
            compare_feat.append(out)
    compare_logits = torch.concat(compare_feat, dim=0).reshape(1, n_model, batch_size, -1) # (1, n_model, batch_size, 512)

    for i in range(steps):
        noise.requires_grad = True
        advs_tmp = images + noise * mask
        if use_di and use_si:
            aug_advs = [input_diversity(advs_tmp / 2**i, resize_rate, diversity_prob) for i in range(augnum)]
        elif use_di:
            aug_advs = [advs_tmp] + [input_diversity(advs_tmp, resize_rate, diversity_prob) for i in range(augnum)]
        elif use_si:
            aug_advs = [advs_tmp / 2**i for i in range(augnum)]
        else:
            aug_advs = [advs_tmp]
        
        
        batch_advs = torch.cat(aug_advs, dim=0)
        aug_num = batch_advs.shape[0] // batch_size
        
        origin_logits = []
        for model in models:
            origin_out = model(batch_advs)
            origin_logits.append(origin_out)

        origin_logits = torch.concat(origin_logits, dim=0).reshape(n_model, aug_num, batch_size, -1)
        origin_logits = origin_logits.permute(1, 0, 2, 3)
  
        if i == 0:
            cos_before = criterion(origin_logits, compare_logits).mean(dim=(0,1)).detach()
            print(cos_before)

        # loss backward
        loss = criterion(origin_logits, compare_logits)
        loss_mean = loss.mean(dim=(0,1))
        print("iter: {}, loss mean: {}".format(i, loss_mean))
        loss = loss.sum()
        print("iter: {}, loss: {}".format(i, loss))
        loss.backward()
        cur_grad = noise.grad
        
        if use_ti:
            cur_grad = F.conv2d(cur_grad, gkern, bias=None, stride=1, groups=3, padding=(2, 2))

        if use_mi:
            print('using momentum')
            # momentum = decay*grads + cur_grad / torch.mean(torch.abs(cur_grad), dim=(1,2,3), keepdim=True)
            # grads = momentum
            # noise = noise + alpha*torch.sign(momentum)
  
            grad_norm = torch.norm(cur_grad.view(batch_size, -1), p=2, dim=1) + 1e-8
            grad = decay*grads +  cur_grad / grad_norm.view(batch_size, 1, 1, 1)
            grads = grad
            noise = noise.detach() + alpha * grad            
            
        else:
            # ord inf
            # noise = noise + alpha*torch.sign(cur_grad)     
            
            # ord 2
            grad_norm = torch.norm(cur_grad.view(batch_size, -1), p=2, dim=1) + 1e-8
            grad = cur_grad / grad_norm.view(batch_size, 1, 1, 1)
            noise = noise.detach() + alpha * grad
        
        noise = torch.clamp(images + noise, min=0, max=1) - images
        noise = noise.detach()
        
    advs_logits = 0
    clean_logits = 0
    advs = images + noise * mask
    criterion = nn.CosineSimilarity(dim=1, eps=1e-8)
    for model in models:
        advs_logits += model(advs)
        clean_logits += model(compare_images)
    
    advs_logits /= len(models)
    clean_logits /= len(models)
    
    cos_after = criterion(advs_logits, clean_logits).detach()
    
    print(f'clean :{cos_before}')
    print(f'advs  :{cos_after}')
    return advs, cos_after.sum().item()