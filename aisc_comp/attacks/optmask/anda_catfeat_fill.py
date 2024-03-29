import torch
import torch.nn as nn
from util import *

def optmask_anda_catfeat_fill(images, 
        compare_images,
        keypoints_images,
        gkern,
        models, 
        steps, 
        alpha, 
        thetas,
        masktype='keypoints'
    ):

    images = images.clone().detach()
    noise = torch.zeros_like(images).clone().detach()   
    # noise = torch.randn_like(images).clone().detach() * 0.006
    # momentum = torch.zeros_like(images, requires_grad=False)
    batch_size = images.shape[0]
    n_model = len(models)
    alpha /= 255.

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
    compare_feat = []
    with torch.no_grad():
        for model in models: 
            out = model(compare_images).detach() 
            compare_feat.append(out)
    compare_logits = torch.concat(compare_feat, dim=0).reshape(1, n_model, batch_size, -1) # (1, n_model, batch_size, 512)
    n_ens = thetas.shape[0]
    thetas = thetas.repeat(batch_size, 1, 1).reshape(batch_size, n_ens, 2, 3).permute(1, 0, 2, 3).reshape(batch_size * n_ens, 2, 3)

    for i in range(steps):
        noise.requires_grad = True
        masked_noise = noise * mask
        advs = masked_noise + images
        
        # todo: anda aug
        xt_batch = advs.repeat(n_ens, 1, 1, 1)
        batch_advs = translation(thetas, xt_batch)
        # for ii, item in enumerate(disi_advs):
        #     Image.fromarray((item.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)).save("debug_anda{}.jpg".format(ii))
        aug_num = batch_advs.shape[0] // batch_size 

        origin_feat = []
        for model in models:   
            out = model(batch_advs) 
            origin_feat.append(out) 
        origin_logits = torch.concat(origin_feat, dim=0).reshape(len(models), aug_num, batch_size, -1)
        origin_logits = origin_logits.permute(1, 0, 2, 3)

        # loss backward
        loss = criterion(origin_logits, compare_logits)
        loss = loss.sum()
        print("iter: {}, loss: {}".format(i, loss))
        
        loss.backward()
        # todo: tifgsm
        ngrad = F.conv2d(noise.grad, gkern, stride=1, padding='same', groups=3)
        noise = noise + alpha * torch.sign(ngrad)

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
    # import pdb; pdb.set_trace()
    
    return advs, cos_after.sum().item()