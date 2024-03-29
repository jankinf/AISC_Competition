import torch
import torch.nn as nn

import numpy as np

from util import *

    

def dim(images, 
        compare_images,
        models, 
        eps, 
        steps, 
        alpha, 
        decay=1.0, 
        resize_rate=1.15, 
        diversity_prob=0.7
    ):

    images = images.clone().detach()
    advs = images.clone().detach()
    noise = torch.zeros_like(images).clone().detach()   
    momentum = torch.zeros_like(images, requires_grad=False)
    
    criterion = nn.CosineSimilarity(dim=1, eps=1e-8)
    
    mask = central_mask().to(advs.device)

    compare_logits = 0
    for model in models:   
        compare_logits += model(compare_images)
    compare_logits /= len(models)
    compare_logits = compare_logits.detach()
    for i in range(steps):
        noise.requires_grad = True
        masked_noise = noise * mask
        advs = masked_noise + images
        
        origin_logits = 0
        for model in models:   
            origin_logits += model(input_diversity(advs, resize_rate, diversity_prob))
        origin_logits /= len(models)
       
        # loss backward
        loss = criterion(origin_logits, compare_logits)
        print("iter: {}, loss: {}".format(i, loss))
        
        loss = loss.sum()
        loss.backward()
        
        momentum = decay * momentum + noise.grad / torch.mean(torch.abs(noise.grad), dim=(1,2,3), keepdim=True)
        noise = noise + alpha * torch.sign(momentum)
        
        advs =  images + noise * mask
        advs = torch.clamp(advs, min=0, max=1)
        noise = (advs - images).detach()

    # prediction
    advs_logits = 0
    for model in models:
        advs_logits += model(advs)
    advs_logits /= len(models)
    
    cos_after = criterion(advs_logits, compare_logits).detach()
    print(f'advs  :{cos_after}')
    
    return advs, cos_after.sum().item()