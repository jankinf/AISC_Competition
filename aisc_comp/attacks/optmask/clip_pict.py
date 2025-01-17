import torch
import torch.nn as nn
from util import *

def optmask_clip_pict(images, 
        compare_images,
        keypoints_images,
        gkern,
        models, 
        eps, 
        steps, 
        alpha, 
        pkern,
        padding_size,
        ampf,
        decay=1.0, 
        masktype='keypoints'
    ):

    images = images.clone().detach()
    advs = images.clone().detach()
    noise = torch.zeros_like(images).clone().detach()   
    # momentum = torch.zeros_like(images, requires_grad=False)
    
    # todo: pifgsm
    amplification = torch.zeros_like(images)      
    alpha_beta = alpha * ampf

    criterion = nn.CosineSimilarity(dim=1, eps=1e-8)
    
    if masktype == 'central':
        mask = central_mask().to(advs.device)
    elif masktype == 'patch5':
        mask = patch5_mask().to(advs.device)
    elif masktype == 'keypoints':
        mask = torch.zeros_like(images).to(advs.device)
        mask[(images - keypoints_images) != 0] = 1.0
        mask = mask.detach()
    else:
        raise NotImplementedError

    compare_logits = 0
    for model in models:   
        compare_logits += model(compare_images)
    compare_logits /= len(models)
    compare_logits = compare_logits.detach()

    for i in range(steps):
        noise.requires_grad = True
        masked_noise = noise * mask
        advs = masked_noise + images
        
        # todo: sifgsm
        si_advs = scale_transform(advs, m=4)
        # todo: difgsm
        di_advs = [ensemble_input_diversity(advs, i) for i in range(4)]    
        origin_logits = 0
        batch_advs = torch.cat(di_advs + si_advs, dim=0)

        for model in models:   
            origin_logits += model(batch_advs)
        origin_logits /= len(models) # shape (8 * batch, 512)
        origin_logits = origin_logits.reshape(8, *compare_logits.shape).mean(dim=0)
       
        # loss backward
        loss = criterion(origin_logits, compare_logits)
        print("iter: {}, loss: {}".format(i, loss))
        
        loss = loss.sum()
        loss.backward()
        
        # todo: tifgsm
        ngrad = F.conv2d(noise.grad, gkern, stride=1, padding='same', groups=3)

        # todo: pi+mifgsm
        amplification += alpha_beta * torch.sign(ngrad)
        cut_noise = torch.clamp(torch.abs(amplification) - eps, 0, 10000.0) * torch.sign(amplification)
        projection = alpha_beta * torch.sign(project_noise(cut_noise, pkern, padding_size))
        amplification += projection
        noise = noise + alpha_beta * torch.sign(ngrad) + projection

        # todo: mifgsm
        # momentum = decay * momentum + ngrad / torch.mean(torch.abs(ngrad), dim=(1,2,3), keepdim=True)
        # noise = noise + alpha * torch.sign(momentum)
        
        # todo: eps bound clamp
        noise = torch.clamp(noise, min=-eps, max=eps)
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