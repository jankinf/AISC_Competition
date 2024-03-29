import torch
import torch.nn as nn
from util import *

def ct_cos_catfeat_fill(images, 
        compare_images,
        keypoints_images,
        gkern,
        models_w, 
        models_b, 
        steps, 
        updata_alpha,
        masktype='keypoints'
    ):

    images = images.clone().detach()
    noise = torch.zeros_like(images).clone().detach()   
    # momentum = torch.zeros_like(images, requires_grad=False)
    batch_size = images.shape[0]
    n_model_w = len(models_w)
    n_model_b = len(models_b)

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

    compare_feat_w = []
    compare_feat_b = []
    with torch.no_grad():
        for model in models_w: 
            out = model(compare_images).detach() 
            compare_feat_w.append(out)
        for model in models_b: 
            out = model(compare_images).detach() 
            compare_feat_b.append(out)
    compare_logits_w = torch.concat(compare_feat_w, dim=0).reshape(1, n_model_w, batch_size, -1) # (1, n_model_w, batch_size, 512)
    compare_logits_b = torch.concat(compare_feat_b, dim=0).reshape(1, n_model_b, batch_size, -1) # (1, n_model_b, batch_size, 512)

    best_loss = -1
    best_iter = -1
    best_adv = None

    for t in range(steps):
        alpha = updata_alpha(t=t) / 255.
        noise.requires_grad = True
        masked_noise = noise * mask
        advs = masked_noise + images
        
        # todo: sifgsm
        # si_advs = scale_transform(advs, m=4)
        # todo: difgsm
        # di_advs = [ensemble_input_diversity(advs, i) for i in range(4)]    
        # todo: di+sifgsm
        disi_advs = [input_diversity(advs / 2**i) for i in range(4)]
        batch_advs = torch.cat(disi_advs, dim=0)
        aug_num = batch_advs.shape[0] // batch_size 

        origin_feat_w = []
        for model in models_w:   
            out = model(batch_advs) 
            origin_feat_w.append(out) 
        origin_logits_w = torch.concat(origin_feat_w, dim=0).reshape(len(models_w), aug_num, batch_size, -1)
        origin_logits_w = origin_logits_w.permute(1, 0, 2, 3)

        # todo: blackbox eval
        origin_feat_b = []
        with torch.no_grad():
            for model in models_b:   
                out = model(advs) 
                origin_feat_b.append(out) 
        origin_logits_b = torch.concat(origin_feat_b, dim=0).reshape(len(models_b), 1, batch_size, -1)
        origin_logits_b = origin_logits_b.permute(1, 0, 2, 3)

        # loss backward
        loss_w = criterion(origin_logits_w, compare_logits_w)
        loss_b = criterion(origin_logits_b, compare_logits_b)
        # print("iter: {}, alpha: {}, loss_w: {}, mean_loss_w: {}".format(t, alpha, loss_w.squeeze().detach().cpu().numpy(), loss_w.mean().item()))
        print("iter: {}, alpha: {}, loss_b: {}, mean_loss_b: {}".format(t, alpha, loss_b.squeeze().detach().cpu().numpy(), loss_b.mean().item()))

        if best_loss < loss_b.mean().item():
            best_loss = loss_b.mean().item()
            best_iter = t
            best_adv = advs.detach().clone()
        loss_w = loss_w.sum()
        loss_w.backward()
        
        # todo: tifgsm
        ngrad = F.conv2d(noise.grad, gkern, stride=1, padding='same', groups=3)

        noise = noise + alpha * torch.sign(ngrad)

        # todo: mifgsm
        # momentum = decay * momentum + ngrad / torch.mean(torch.abs(ngrad), dim=(1,2,3), keepdim=True)
        # noise = noise + alpha * torch.sign(momentum)
        
        advs =  images + noise * mask
        advs = torch.clamp(advs, min=0, max=1)
        noise = (advs - images).detach()

    print('best cos of blackbox: {}, idx: {}'.format(best_loss, best_iter))
    return best_adv, 0