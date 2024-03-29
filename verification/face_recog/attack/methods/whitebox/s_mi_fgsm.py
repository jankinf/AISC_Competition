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

# SI
def scale_transform(input_tensor, m=5):
    shape = input_tensor.shape
    outs = [(input_tensor) / (2**i) for i in range(m)]
    x_batch = torch.concat(outs, dim=0)
    new_shape = x_batch.shape
    x_batch = x_batch.reshape(m, *shape).permute(1, 0, 2, 3, 4).reshape(*new_shape)
    return x_batch

def si_fgsm(feat1, x, model, num_iter, eps, alpha, fp=None, momentum=1, init=True, use_gpu=True):
    if use_gpu:
        feat1 = feat1.cuda()
        x = x.cuda()
        model = model.cuda()
    grads = torch.zeros_like(x, requires_grad=False)
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
            advs = scale_transform(adv)
            feat2 = model(advs)
            loss = cos(feat1, feat2).mean()
            print("cos dis: {}".format(loss.item()))
            if fp:
                fp.write("loss: {}\n".format(loss.item()))
            loss.backward()
            new_grad = adv.grad
            noise = momentum * grads + (new_grad) / torch.norm(new_grad, p=1)
            adv = adv - alpha * noise.sign()

            adv = torch.clamp(adv, 0.0, 1.0).detach()
            adv = torch.max(torch.min(adv, max_x), min_x).detach()
            grads = noise
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
                adv_x = si_fgsm(feat1=feat1, x=target_img, model=model, fp=fp, **kwargs)
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