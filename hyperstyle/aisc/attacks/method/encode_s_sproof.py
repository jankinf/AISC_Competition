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

def encode_s(
    codes, 
    weights_deltas, 
    encoder,
    decoder,
    origin_images,
    compare_images,
    compare_images_encode,
    keypoints_images,
    as_model,
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

    target_codes = encoder.forward(compare_images_encode)
    
    mask = torch.zeros_like(origin_images).clone().detach()
    mask[keypoints_images == 0] = 1.0
    
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

    y = torch.zeros(1).type_as(compare_logits).long()

    print_freq = 10
    for i in range(steps):
        images, _ = decoder([latent], weights_deltas=weights_deltas, input_is_latent=True, randomize_noise=False, return_latents=False)
        images = face_pool(images) # (1, 3, 256, 256)
        
        # todo: transform needed
        images = ((images + 1) / 2)
        
        advs_tmp = origin_images * (1 - mask) + images * mask
        advs_tmp_encode = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(advs_tmp)
        adv_codes = encoder.forward(advs_tmp_encode)

        aug_advs = [advs_tmp] + [input_diversity(advs_tmp, resize_rate, diversity_prob) for i in range(augnum)]
        # aug_advs = [advs_tmp]
        
        batch_advs = torch.cat(aug_advs, dim=0)
        aug_num = batch_advs.shape[0] // batch_size
        
        out_as = as_model(batch_advs)
        if out_as.ndim == 1:
            out_as = out_as.unsqueeze(0)
        ys = y.repeat(batch_advs.shape[0])
        out_prob = F.softmax(out_as, dim=-1)[:, 0].mean()
        ce_loss = F.cross_entropy(out_as, ys, reduction="sum")

        origin_logits = []
        for model in models:
            origin_out = model(batch_advs)
            origin_logits.append(origin_out)

        origin_logits = torch.concat(origin_logits, dim=0).reshape(n_model, aug_num, batch_size, -1)
        origin_logits = origin_logits.permute(1, 0, 2, 3)

        l2_dis = torch.norm(adv_codes - target_codes.detach(), p=2) / adv_codes.numel()
        cos_dis = - criterion(origin_logits, compare_logits).mean()
        loss = 200 * l2_dis + cos_dis + ce_loss
        loss = cos_dis + ce_loss


        if i % print_freq == 0:
            with torch.no_grad():
                black_origin_logits = []
                for model in black_models:
                    origin_out = model(batch_advs)
                    black_origin_logits.append(origin_out)
                black_origin_logits = torch.concat(black_origin_logits, dim=0).reshape(n_model_black, aug_num, batch_size, -1)
                black_origin_logits = black_origin_logits.permute(1, 0, 2, 3)
                
                black_loss = criterion(black_origin_logits, black_compare_logits)
                black_loss = black_loss.mean()
                print("{} iter white loss: {}, ls_dis loss: {}, out_prob: {}".format(i, - cos_dis.item(), l2_dis.item(), out_prob.item()))
                print("{} iter black loss: {}".format(i, black_loss.item()))
                
                out = torch.clamp(advs_tmp, 0, 1).permute(0, 2, 3, 1)[0]
                out = out.detach().cpu().numpy() * 255
                Image.fromarray(out.astype(np.uint8)).save("{}/out{}_{}.png".format(save_path, i, round(black_loss.item(), 3)))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    with torch.no_grad():
        images, _ = decoder([latent], weights_deltas=weights_deltas, input_is_latent=True, randomize_noise=False, return_latents=False)
        images = face_pool(images) # (1, 3, 256, 256)
        images = ((images + 1) / 2)
        advs = origin_images * (1 - mask) + images * mask
        advs = torch.clamp(advs, 0, 1)

    return advs

