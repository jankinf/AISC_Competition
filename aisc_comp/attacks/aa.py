from logging import handlers
import torch
import torch.nn as nn

import numpy as np

from util import *

    

def aa(images, 
        labels,
        compare_images,
        compare_labels,
        models, 
        eps, 
        steps, 
        alpha, 
        decay=1.0
    ):

    images = images.clone().detach()
    advs = images.clone().detach()
    noise = torch.zeros_like(images).clone().detach()   
    grads = torch.zeros_like(images, requires_grad=False)
    
    criterion = nn.CosineSimilarity(dim=1, eps=1e-8)  
    
    mid_outputs = []
    def get_mid_output(m, i, o):
        mid_outputs.append(o)
    handles = [
        feature_layer(models[i]).register_forward_hook(get_mid_output)
        for i in range(len(models))
    ]
    mask = central_mask(images)
    for i in range(steps):
        advs.requires_grad = True
        origin_midoutput = 0
        compare_midoutput = 0
        
        for model in models:   
            mid_outputs = []
            origin_logits = model(advs)
            origin_midoutput += mid_outputs[0]
            compare_logits = model(compare_images)
            compare_midoutput += mid_outputs[1]

        origin_midoutput /= len(models)
        compare_midoutput /=  len(models)
        origin_logits /= len(models)
        compare_logits /= len(models)
        
        if i == 0:
            # img1, img2 diff
            loss_before = torch.norm(compare_midoutput - origin_midoutput, p=2, dim=(1, 2, 3)).detach()
            cos_before = criterion(origin_logits, compare_logits).detach()
            print(loss_before)

        # loss backward
        loss = torch.norm(compare_midoutput - origin_midoutput, p=2, dim=(1, 2, 3))
        print("iter: {}, loss: {}".format(i, loss))
        
        loss = loss.sum()
        loss.backward()
        
        momentum = decay*grads + advs.grad / torch.mean(torch.abs(advs.grad), dim=(1,2,3), keepdim=True)
        grads = momentum
        noise = noise + alpha*torch.sign(momentum)
        noise = noise * mask
        
        advs =  images - noise
        advs = torch.clamp(advs, min=0, max=1).detach()

    handles[0].remove()
    handles[1].remove()
    
    # prediction
    advs_logits = 0
    clean_logits = 0
    for model in models:
        # img1, img2 diff
        advs_logits += model(advs)
        clean_logits += model(compare_images)
    
    advs_logits /= len(models)
    clean_logits /= len(models)
    
    cos_after = criterion(advs_logits, clean_logits).detach()
    
    print(f'clean :{cos_before}')
    print(f'advs  :{cos_after}')

    return advs, cos_after.sum().item()