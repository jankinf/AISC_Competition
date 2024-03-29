# coding=utf-8
import argparse
import os
import numpy as np
import torch
from torch import nn
from PIL import Image
from torchvision import transforms as T
from facenet_pytorch import InceptionResnetV1, MTCNN
from facenet_pytorch.models.utils.detect_face import get_size
from torch.nn import functional as F

class Detector(MTCNN):
    def forward(self, ori_img, n=5):
        # Detect faces
        outs = []
        img = ori_img.permute(0, 2, 3, 1) * 255 # change to (NHWC)
        outs.append(img)
        batch_boxes, batch_probs, batch_points = self.detect(img, landmarks=True)
        # Select faces
        box, batch_probs, batch_points = self.select_boxes(
            batch_boxes, batch_probs, batch_points, img, method=self.selection_method
        )
        raw_image_size = img.shape[1:3]
        box = [
            int(max(box[0, 0, 0], 0)),
            int(max(box[0, 0, 1], 0)),
            int(min(box[0, 0, 2], raw_image_size[1])),
            int(min(box[0, 0, 3], raw_image_size[0])),
        ]
        out = img[:, box[1]:box[3], box[0]:box[2], :]
        out = F.interpolate(
            out.permute(0, 3, 1, 2).float(),
            (raw_image_size[0], raw_image_size[1])
        ).permute(0, 2, 3, 1)
        im_crop = Image.fromarray((out[0].detach().cpu().numpy()).astype(np.uint8))
        im_crop.save("./im0.jpg")
        outs.append(out)

        tops = torch.randint(0, box[1], (n,))
        bottoms = torch.randint(box[3], raw_image_size[0], (n,))
        lefts = torch.randint(0, box[0], (n,))
        rights = torch.randint(box[2], raw_image_size[1], (n,))

        i = 1
        for top, bottom, left, right in zip(tops, bottoms, lefts, rights):
            out = img[:, top:bottom, left:right, :]
            out = F.interpolate(
                out.permute(0, 3, 1, 2).float(),
                (raw_image_size[0], raw_image_size[1])
            ).permute(0, 2, 3, 1)
            im_crop = Image.fromarray((out[0].detach().cpu().numpy()).astype(np.uint8))
            im_crop.save("./im{}.jpg".format(i))
            outs.append(out)
            i += 1

        outs = torch.concat(outs, dim=0).permute(0, 3, 1, 2) / 255.
        return outs

    # def get_cropped_imgs(self, )
    def sanity_check(self, ori_img):
        # Detect faces
        img = ori_img.permute(0, 2, 3, 1) * 255 # change to (NHWC)
        batch_boxes, batch_probs, batch_points = self.detect(img, landmarks=True)
        # Select faces
        box, batch_probs, batch_points = self.select_boxes(
            batch_boxes, batch_probs, batch_points, img, method=self.selection_method
        )
        raw_image_size = img.shape[1:3]
        box = [
            int(max(box[0, 0, 0], 0)),
            int(max(box[0, 0, 1], 0)),
            int(min(box[0, 0, 2], raw_image_size[1])),
            int(min(box[0, 0, 3], raw_image_size[0])),
        ]
        img = img[:, box[1]:box[3], box[0]:box[2], :]
        out = F.interpolate(
            img.permute(0, 3, 1, 2).float(),
            (raw_image_size[0], raw_image_size[1])
        ).permute(0, 2, 3, 1)

        im = Image.fromarray((img[0].detach().cpu().numpy()).astype(np.uint8))
        im.save("./im_out.jpg")
        im_crop = Image.fromarray((out[0].detach().cpu().numpy()).astype(np.uint8))
        im_crop.save("./im_crop_out.jpg")
        return out

def rand_crop(feat1, x, model, num_iter, eps, alpha, detector, fp=None, momentum=1, init=False, use_gpu=True):
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
            if args.n:
                advs = detector(adv, n=args.n)
            else:
                advs = adv
            feat2 = model(advs)
            assert feat1.ndim == feat2.ndim
            loss = cos(feat1, feat2).mean()
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
    detector = Detector(device=torch.device('cuda'))
    with open(logfile, 'a') as fp:
        reference_img = tsf(Image.open(os.path.join(input_dir, img_list[0]))).unsqueeze(0).cuda()
        feat1 = model(reference_img).detach()
        for idx in range(1, len(img_list)):
            img_file = os.path.join(output_dir, img_list[idx])
            if os.path.exists(img_file):
                continue    
            target_img = tsf(Image.open(os.path.join(input_dir, img_list[idx]))).unsqueeze(0).cuda()
            adv_x = rand_crop(feat1=feat1, x=target_img, model=model, detector=detector, fp=fp, **kwargs)
            out_eps = (adv_x - target_img).abs().max().item()
            logdata = "id:{}, file:{}, eps_check: {}\n\n".format(idx, img_list[idx], out_eps)
            print(logdata, end='')
            fp.write(logdata)
            fp.flush()
            adv_x = adv_x.detach().cpu().permute(0, 2, 3, 1).numpy()[0] * 255
            Image.fromarray(adv_x.astype(np.uint8)).save(img_file)
            try:
                pass
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
    parser.add_argument("--n", type=int, default=5)
    args = parser.parse_args()

    input_dir = "./data/video_img"
    method = "{}".format(os.path.splitext(__file__.split('/')[-1])[0])
    settings = f"iter{args.num_iter}_eps{args.max_epsilon}_n{args.n}"
    output_dir = os.path.join(args.output_dir, args.pretrain, method, settings)
    logfile = os.path.join(args.log_dir, args.pretrain, method, settings, "out.log")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    print(output_dir)
    print(logfile)
    main()
    with open("toeval.out", "a") as fp:
        fp.write("{}\n".format(output_dir))