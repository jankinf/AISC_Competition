import torch
import torch.nn as nn
from util import *

def optmask_pict_catfeat_fill(images, 
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
    noise = torch.zeros_like(images).clone().detach()   
    # momentum = torch.zeros_like(images, requires_grad=False)
    batch_size = images.shape[0] # images (n, 3, 112, 112)
    n_model = len(models)
    # todo: pifgsm
    amplification = torch.zeros_like(images)      
    alpha_beta = alpha * ampf # alpha = eps / T

    criterion = nn.CosineSimilarity(dim=3, eps=1e-8)
    
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
    # Image.fromarray((images.detach().cpu().permute(0, 2, 3, 1)[0].numpy() * 255).astype(np.uint8)).save("./debug_out22.jpg")
    # exit()
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
        
        # todo: sifgsm
        # si_advs = scale_transform(advs, m=4)
        # todo: difgsm
        # di_advs = [ensemble_input_diversity(advs, i) for i in range(4)]    
        # todo: di+sifgsm
        disi_advs = [input_diversity(advs / 2**i) for i in range(4)]
        batch_advs = torch.cat(disi_advs, dim=0) # 4, 3, 112, 112
        aug_num = batch_advs.shape[0] // batch_size 

        origin_feat = []
        for model in models:   
            out = model(batch_advs) 
            origin_feat.append(out) 
        origin_logits = torch.concat(origin_feat, dim=0).reshape(len(models), aug_num, batch_size, -1)
        origin_logits = origin_logits.permute(1, 0, 2, 3)

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
        
        advs =  images + noise * mask
        advs = torch.clamp(advs, min=0, max=1)
        noise = (advs - images).detach()

    # prediction
    with torch.no_grad():
        advs_logits = []
        for model in models:
            advs_logits.append(model(advs).detach())
        advs_logits = torch.concat(advs_logits, dim=0).reshape(1, n_model, batch_size, -1) # (1, n_model, batch_size, 512)
        cos_after = criterion(advs_logits, compare_logits).detach()
    print(f'advs  :{cos_after}')

    return advs, cos_after.sum().item()