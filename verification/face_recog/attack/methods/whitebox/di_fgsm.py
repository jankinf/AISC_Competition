# coding=utf-8
import argparse
import os
import numpy as np
import torch
from torch import nn
from PIL import Image
from torchvision import transforms as T
from facenet_pytorch import InceptionResnetV1
from torch.nn import functional as F

# DI
def input_diversity(x, resize_rate=1.15, diversity_prob=0.7):
    assert resize_rate >= 1.0
    assert diversity_prob >= 0.0 and diversity_prob <= 1.0
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
    ret = padded if torch.rand(1) < diversity_prob else x
    return ret

def di_fgsm(feat1, x, model, num_iter, eps, alpha, fp=None, momentum=1, init=True, use_gpu=True):
    if use_gpu:
        feat1 = feat1.cuda()
        x = x.cuda()
        model = model.cuda()
    
    min_x = x - eps
    max_x = x + eps
    cos = nn.CosineSimilarity(dim=1, eps=1e-12)
    
    if init:
        adv = x + eps * 2 * (torch.rand_like(x).to(x.device) - 0.5) * 2
    else:
        adv = x.clone()

    with torch.enable_grad():
        for i in range(num_iter):
            adv.requires_grad = True
            feat2 = model(input_diversity(adv))
            assert feat1.ndim == feat2.ndim
            loss = cos(feat1, feat2)[0]
            print("cos dis: {}".format(loss.item()))
            if fp:
                fp.write("loss: {}\n".format(loss.item()))
            loss.backward()
            new_grad = adv.grad
            adv = adv - alpha * new_grad.sign()

            adv = torch.clamp(adv, 0.0, 1.0).detach()
            adv = torch.max(torch.min(adv, max_x), min_x).detach()
    return adv

def main():
    print(args)
    pretrain=args.pretrain
    normalize = T.Normalize(mean=[127.5/255, 127.5/255, 127.5/255], std=[128.0/255, 128.0/255, 128.0/255])
    model = InceptionResnetV1(pretrained=pretrain).eval()
    model = nn.Sequential(normalize, model)
    model = model.cuda()

    eps = args.max_epsilon / 255.
    iteration = args.num_iter
    os.makedirs(output_dir, exist_ok=True)
    img_list = os.listdir(input_dir)

    kwargs = {
        "num_iter": iteration,
        "eps": eps,
        "alpha": eps / iteration,
    }
    tsf = T.ToTensor()
    with open(logfile, 'a') as fp:
        reference_img = tsf(Image.open(os.path.join(input_dir, img_list[0]))).unsqueeze(0).cuda()
        # reference_img = tsf(Image.open("/data/projects/verification/face_recog/data/debug/3308203,1a5d0006f7799ed.jpg")).unsqueeze(0).cuda()
        feat1 = model(reference_img).detach()
        for idx in range(1, len(img_list)):
            img_file = os.path.join(output_dir, img_list[idx])
            if os.path.exists(img_file):
                continue    
            try:
                target_img = tsf(Image.open(os.path.join(input_dir, img_list[idx]))).unsqueeze(0).cuda()
                # feat2 = model(target_img).detach()
                # print((feat1 != feat2).sum())
                adv_x = di_fgsm(feat1=feat1, x=target_img, model=model, fp=fp, **kwargs)
                # import pdb; pdb.set_trace()
                out_eps = (adv_x - target_img).abs().max().item()
                logdata = "id:{}, file:{}, eps_check: {}\n\n".format(idx, img_list[idx], out_eps)
                print(logdata, end='')
                fp.write(logdata)
                fp.flush()
                adv_x = adv_x.detach().cpu().permute(0, 2, 3, 1).numpy()[0] * 255
                Image.fromarray(adv_x.astype(np.uint8)).save(img_file)

            except Exception as e:
                print(e)
                continue

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain", type=str, default='vggface2', choices=['vggface2', 'casia-webface'])
    parser.add_argument('--num_iter', default=10, type=int, help='max iteration')
    parser.add_argument("--max_epsilon", type=int, default=16)
    parser.add_argument('--output_dir', default='./data/adv', type=str,
                        help='directory of crafted adversarial examples')
    parser.add_argument("--log_dir", type=str, default="./logs")
    args = parser.parse_args()

    input_dir = "./data/video_img"
    method = "{}".format(os.path.splitext(__file__.split('/')[-1])[0])
    settings = f"iter{args.num_iter}_eps{args.max_epsilon}"
    output_dir = os.path.join(args.output_dir, args.pretrain, method, settings)
    logfile = os.path.join(args.log_dir, args.pretrain, method, settings, "out.log")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    print(output_dir)
    print(logfile)
    main()
    with open("toeval.out", "a") as fp:
        fp.write("{}\n".format(output_dir))