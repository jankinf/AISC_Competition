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
import scipy.stats as st

# TI
class Translation_Kernel:
    def __init__(self, len_kernel=15, nsig=3, kernel_name='gaussian'):
        self.kernel_name = kernel_name
        self.len_kernel = len_kernel
        self.nsig = nsig

    def kernel_generation(self):
        if self.kernel_name == 'gaussian':
            kernel = self.gkern(self.len_kernel, self.nsig).astype(np.float32)
        elif self.kernel_name == 'linear':
            kernel = self.lkern(self.len_kernel).astype(np.float32)
        elif self.kernel_name == 'uniform':
            kernel = self.ukern(self.len_kernel).astype(np.float32)
        else:
            raise NotImplementedError

        stack_kernel = np.stack([kernel, kernel, kernel])
        stack_kernel = np.expand_dims(stack_kernel, 1)
        return stack_kernel

    def gkern(self, kernlen=15, nsig=3):
        """Returns a 2D Gaussian kernel array."""
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def ukern(self, kernlen=15):
        kernel = np.ones((kernlen,kernlen))* 1.0 /(kernlen*kernlen)
        return kernel

    def lkern(self, kernlen=15):
        kern1d = 1-np.abs(np.linspace((-kernlen+1)/2, (kernlen-1)/2, kernlen)/(kernlen+1)*2)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

def ti_fgsm(feat1, x, model, num_iter, eps, alpha, gkern, fp=None, momentum=1, init=True, use_gpu=True):
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
            feat2 = model(adv)
            assert feat1.ndim == feat2.ndim
            loss = cos(feat1, feat2)[0]
            print("cos dis: {}".format(loss.item()))
            if fp:
                fp.write("loss: {}\n".format(loss.item()))
            loss.backward()
            # todo: conv origin grad then store at momentum grads
            new_grad = adv.grad
            new_grad = F.conv2d(new_grad, gkern, stride=1, padding='same', groups=3)
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

    ti_kernel = Translation_Kernel(len_kernel=args.kernel_size, nsig=1, kernel_name='gaussian')
    gkern = torch.from_numpy(ti_kernel.kernel_generation()).cuda()
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
                adv_x = ti_fgsm(feat1=feat1, x=target_img, model=model, gkern=gkern, fp=fp, **kwargs)
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
    parser.add_argument('--kernel_size', default=5, type=int)
    parser.add_argument("--max_epsilon", type=int, default=16)
    parser.add_argument('--output_dir', default='./data/adv', type=str,
                        help='directory of crafted adversarial examples')
    parser.add_argument("--log_dir", type=str, default="./logs")
    args = parser.parse_args()

    input_dir = "./data/video_img"
    method = "{}".format(os.path.splitext(__file__.split('/')[-1])[0])
    settings = f"iter{args.num_iter}_eps{args.max_epsilon}_kernel{args.kernel_size}"
    output_dir = os.path.join(args.output_dir, args.pretrain, method, settings)
    logfile = os.path.join(args.log_dir, args.pretrain, method, settings, "out.log")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    print(output_dir)
    print(logfile)
    main()
    with open("toeval.out", "a") as fp:
        fp.write("{}\n".format(output_dir))