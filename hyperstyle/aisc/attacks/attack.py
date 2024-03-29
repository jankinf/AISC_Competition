import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
import torchvision.transforms as transforms
from torch import optim
from PIL import Image

def input_diversity(x, resize_rate=1.15, diversity_prob=0.7):
    assert resize_rate >= 1.0
    assert diversity_prob >= 0.0 and diversity_prob <= 1.0
    if torch.rand(1) >= diversity_prob:
        return x
    img_size = max(*list(x.shape[-2:]))
    img_resize = int(img_size * resize_rate)
    rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
    rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
    h_rem = img_resize - rnd
    w_rem = img_resize - rnd
    pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
    pad_bottom = h_rem - pad_top
    pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
    pad_right = w_rem - pad_left
    padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)
    padded = F.interpolate(padded, size=[img_size, img_size], mode='bilinear', align_corners=False)
    return padded

def attack(
    codes, 
    weights_deltas, 
    decoder,
    origin_images,
    compare_images,
    keypoints_images,
    models, 
    black_models,
    alpha=0.01, 
    steps=100, 
    resize_rate=1.15, 
    diversity_prob=0.7,
    n_ens=1,
    save_path=None,
):
    latent = codes.detach().clone()
    latent.requires_grad = True
    
    optimizer = optim.Adam(params=[latent], lr=alpha)
    face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
    criterion = nn.CosineSimilarity(dim=3, eps=1e-8)
    
    n_model = len(models)
    n_model_black = len(black_models) 
    batch_size = origin_images.shape[0]
    augnum = n_ens

    
    mask = torch.zeros_like(origin_images).clone().detach()
    mask[keypoints_images == 0] = 1.0
    # mask[(images - keypoints_images) != 0] = 1.0
    print(mask[0, 0].sum())
    
    compare_feat = []
    with torch.no_grad():
        for model in models: 
            out = model(compare_images)                            
            compare_feat.append(out)
    compare_logits = torch.concat(compare_feat, dim=0).reshape(1, n_model, batch_size, -1) # (1, n_model, batch_size, 512)

    black_compare_feat = []
    with torch.no_grad():
        for model in black_models: 
            out = model(compare_images)                            
            black_compare_feat.append(out)
    black_compare_logits = torch.concat(black_compare_feat, dim=0).reshape(1, n_model_black, batch_size, -1) # (1, n_model, batch_size, 512)

    print_freq = 10
    for i in range(steps):
        images, _ = decoder([latent], weights_deltas=weights_deltas, input_is_latent=True, randomize_noise=False, return_latents=False)
        images = face_pool(images) # (1, 3, 256, 256)
        
        # todo: transform needed
        images = ((images + 1) / 2)
        
        advs_tmp = origin_images * (1 - mask) + images * mask

        aug_advs = [advs_tmp] + [input_diversity(advs_tmp, resize_rate, diversity_prob) for i in range(augnum)]
        # aug_advs = [advs_tmp]
        
        batch_advs = torch.cat(aug_advs, dim=0)
        aug_num = batch_advs.shape[0] // batch_size
        
        origin_logits = []
        for model in models:
            origin_out = model(batch_advs)
            origin_logits.append(origin_out)

        origin_logits = torch.concat(origin_logits, dim=0).reshape(n_model, aug_num, batch_size, -1)
        origin_logits = origin_logits.permute(1, 0, 2, 3)

        loss = - criterion(origin_logits, compare_logits)
        loss = loss.mean()


        if i % print_freq == 0:
            black_origin_logits = []
            for model in black_models:
                origin_out = model(batch_advs)
                black_origin_logits.append(origin_out)
            black_origin_logits = torch.concat(black_origin_logits, dim=0).reshape(n_model_black, aug_num, batch_size, -1)
            black_origin_logits = black_origin_logits.permute(1, 0, 2, 3)
            
            black_loss = criterion(black_origin_logits, black_compare_logits)
            black_loss = black_loss.mean()
            print("{} iter white loss: {}".format(i, - loss.item()))
            print("{} iter black loss: {}".format(i, black_loss.item()))
            
            out = torch.clamp(advs_tmp, 0, 1).permute(0, 2, 3, 1)[0]
            out = out.detach().cpu().numpy() * 255
            Image.fromarray(out.astype(np.uint8)).save("{}/out{}_{}.png".format(save_path, i, round(black_loss.item(), 3)))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # import pdb; pdb.set_trace()
    images, _ = decoder([latent], weights_deltas=weights_deltas, input_is_latent=True, randomize_noise=False, return_latents=False)
    images = face_pool(images) # (1, 3, 256, 256)
    images = ((images + 1) / 2)
    advs = origin_images * (1 - mask) + images * mask
    advs = torch.clamp(advs, 0, 1)

    return advs

def attack_grad(
    origin_images,
    compare_images,
    keypoints_images,
    models, 
    black_models,
    alpha=0.01, 
    steps=100, 
    resize_rate=1.15, 
    diversity_prob=0.7,
    n_ens=1,
    save_path=None,
):
    criterion = nn.CosineSimilarity(dim=3, eps=1e-8)
    
    n_model = len(models)
    n_model_black = len(black_models) 
    batch_size = origin_images.shape[0]
    augnum = n_ens
    noise = torch.zeros_like(origin_images, requires_grad=True).clone().detach()   
    
    mask = torch.zeros_like(origin_images).clone().detach()
    mask[keypoints_images == 0] = 1.0
    print(mask[0, 0].sum() / mask[0, 0].numel())
    
    origin_images = (1 - mask) * origin_images + mask * compare_images

    compare_feat = []
    with torch.no_grad():
        for model in models: 
            out = model(compare_images)                            
            compare_feat.append(out)
    compare_logits = torch.concat(compare_feat, dim=0).reshape(1, n_model, batch_size, -1) # (1, n_model, batch_size, 512)

    black_compare_feat = []
    with torch.no_grad():
        for model in black_models: 
            out = model(compare_images)                            
            black_compare_feat.append(out)
    black_compare_logits = torch.concat(black_compare_feat, dim=0).reshape(1, n_model_black, batch_size, -1) # (1, n_model, batch_size, 512)

    print_freq = 10
    for i in range(steps):
        noise.requires_grad = True
        advs_tmp = origin_images + noise * mask
        aug_advs = [advs_tmp] + [input_diversity(advs_tmp, resize_rate, diversity_prob) for i in range(augnum)]
        
        # aug_advs = [advs_tmp]
        
        batch_advs = torch.cat(aug_advs, dim=0)
        aug_num = batch_advs.shape[0] // batch_size
        
        origin_logits = []
        for model in models:
            origin_out = model(batch_advs)
            origin_logits.append(origin_out)

        origin_logits = torch.concat(origin_logits, dim=0).reshape(n_model, aug_num, batch_size, -1)
        origin_logits = origin_logits.permute(1, 0, 2, 3)

        loss = criterion(origin_logits, compare_logits)
        loss = loss.mean()


        if i % print_freq == 0:
            black_origin_logits = []
            for model in black_models:
                origin_out = model(batch_advs)
                black_origin_logits.append(origin_out)
            black_origin_logits = torch.concat(black_origin_logits, dim=0).reshape(n_model_black, aug_num, batch_size, -1)
            black_origin_logits = black_origin_logits.permute(1, 0, 2, 3)
            
            black_loss = criterion(black_origin_logits, black_compare_logits)
            black_loss = black_loss.mean()
            print("{} iter white loss: {}".format(i, loss.item()))
            print("{} iter black loss: {}".format(i, black_loss.item()))

            out = torch.clamp(advs_tmp, 0, 1).permute(0, 2, 3, 1)[0]
            out = out.detach().cpu().numpy() * 255
            Image.fromarray(out.astype(np.uint8)).save("{}/out{}_{}.png".format(save_path, i, round(black_loss.item(), 3)))

        loss.backward()
        cur_grad = noise.grad
        grad_norm = torch.norm(cur_grad.view(batch_size, -1), p=2, dim=1) + 1e-8
        grad = cur_grad / grad_norm.view(batch_size, 1, 1, 1)
        noise = noise.detach() + alpha * grad
        
        noise = torch.clamp(origin_images + noise, min=0, max=1) - origin_images
        noise = noise.detach()

    
    advs = origin_images + noise * mask

    return advs