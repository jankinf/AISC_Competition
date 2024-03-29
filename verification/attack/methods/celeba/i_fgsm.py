# coding=utf-8
'''Implementation of I-FGSM attack in PyTorch'''
import argparse
import os.path
import torch
from torch.nn import functional as F
from attack.attack_dataset import get_dataloader, save_img
from attack.eval import verify_output
from src.tsn_predict import pretrain
from src.models import AENet
from src.tsn_predict import TSNPredictor as CelebASpoofDetector
import torchvision
from tqdm import tqdm
import numpy as np
parser = argparse.ArgumentParser(description='I-FGSM attack in PyTorch')
parser.add_argument('--batch_size', type=int, default=1, help='mini-batch size (default: 2)')
parser.add_argument('--workers', default=2, type=int, help='number of data loading workers (default: 2)')
parser.add_argument('--max_epsilon', default=16.0, type=float, help='max magnitude of adversarial perturbations')
parser.add_argument('--num_iter', default=10, type=int, help='max iteration')
parser.add_argument('--log_dir', default='./logs', type=str, help='dir of logs')
parser.add_argument('--output_dir', default='./attack/adv_data', type=str, help='directory of crafted adversarial examples')
parser.add_argument('--ckpt', type=str, help='path of ckpt to load')
parser.add_argument('--eval_ckpts', nargs='+', default=[], type=str, help='paths of ckpt')
parser.add_argument('--use_gpu', default=True, type=bool, help='use gpu or not')
parser.add_argument('--target', default=-1, type=int, help='target label: default live (0 for live; 1 for spoof; -1 for non-target attack)')
args = parser.parse_args()


def i_fgsm(x, y, model, num_iter, eps, alpha, target=True, use_gpu=True):
    '''
    craft adversarial examples
    :param x: clean images in batch in [0, 1]
    :param y: correct labels
    :return: adv in [0, 1]
    '''
    if use_gpu:
        x = x.cuda()
        y = y.cuda()
        model = model.cuda()

    min_x = x - eps
    max_x = x + eps

    adv = x.clone()

    with torch.enable_grad():
        for _ in range(num_iter):
            adv.requires_grad = True
            resize_adv = F.interpolate(adv, (224, 224), mode='bilinear', align_corners=False)
            output = model(resize_adv)
            if output.ndim == 1:
                output = output.unsqueeze(0)
            loss = F.cross_entropy(output, y, reduction="sum")
            loss.backward()
            noise = adv.grad
            if target:
                adv = adv - alpha * noise.sign()
            else:
                adv = adv + alpha * noise.sign()

            adv = torch.clamp(adv, 0.0, 1.0).detach()
            adv = torch.max(torch.min(adv, max_x), min_x).detach()
    
    return adv


transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])

def main():
    print(args)
    targeted = args.target != -1
    if targeted:
        method = "i_fgsm_t" + str(args.target)
    else:
        method = "i_fgsm_nt"
    settings = f"iter{args.num_iter}_eps{args.max_epsilon}"
    output_dir = os.path.join(args.output_dir, method, settings)
    log_dir = os.path.join(args.log_dir, method, settings)
    log_file = os.path.join(log_dir, 'log.log')
    os.makedirs(output_dir + "/attack", exist_ok=True)
    os.makedirs(output_dir + "/normal", exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    model = AENet(num_classes=2)
    checkpoint = torch.load(args.ckpt)
    pretrain(model, checkpoint['state_dict'])
    model.eval()
    clean_loader = get_dataloader(max_images=1000, transform=transform, batch_size=args.batch_size, num_workers=args.workers)
    kwargs = {
        "num_iter": args.num_iter,
        "eps": args.max_epsilon / 255,
        "alpha": args.max_epsilon / 255 / args.num_iter,
        "use_gpu": args.use_gpu,
        "target": targeted,
    }
    detectors = {ckpt: CelebASpoofDetector(ckpt=ckpt) for ckpt in args.eval_ckpts}
    scores = {ckpt: [] for ckpt in args.eval_ckpts}
    labels = []
    modes = ["normal", "attack"]
    for i, data in tqdm(enumerate(clean_loader)):
        img_path, x, y = data["filepath"], data["img"], data["label"]
        if args.target == 0:
            target_y = torch.zeros_like(y).type_as(y)
        elif args.target == 1:
            target_y = torch.ones_like(y).type_as(y)
        else:
            target_y = y
        name = list(map(lambda x: x.split('/')[-1], img_path))
        
        adv_x = i_fgsm(x, target_y, model, **kwargs)

        for ckpt in args.eval_ckpts:
            resize_adv = F.interpolate(adv_x, (224, 224), mode='bilinear', align_corners=False)
            scores[ckpt] += detectors[ckpt].predict(resize_adv)[:, 1].tolist()
        
        labels.append(y.numpy())
        for k in range(adv_x.shape[0]):
            save_img(os.path.join(output_dir, modes[y[k].item()], name[k]), adv_x[k].detach().permute(1, 2, 0).cpu())
        
    labels = np.concatenate(labels)
    with open(log_file, 'a') as f:
        f.write('-' * 30 + '\n')
        f.write(f'params: {str(args)}\n')
        for ckpt in args.eval_ckpts:
            scores[ckpt] = np.array(scores[ckpt])
            tpr_list = verify_output(scores[ckpt], labels)
            f.write('ckpt: {}\n'.format(ckpt))
            f.write('TPR@FPR=2E-1: {}\n'.format(tpr_list[0]))
            f.write('TPR@FPR=1E-1: {}\n'.format(tpr_list[1]))
            f.write('TPR@FPR=1E-2: {}\n'.format(tpr_list[2]))
            f.write('TPR@FPR=5E-3: {}\n'.format(tpr_list[3]))
            f.write('TPR@FPR=1E-3: {}\n'.format(tpr_list[4]))
            f.write('\n')
    print('attack finished')
    with open("to_eval.log", "a") as f:
        f.write('{}\n'.format(output_dir + "/attack"))
        f.write('{}\n'.format(output_dir + "/normal"))

if __name__ == '__main__':
    main()