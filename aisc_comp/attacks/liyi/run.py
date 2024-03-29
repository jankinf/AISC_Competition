import torchvision.transforms as T
from torch.utils import data

import argparse
import os

from attacks.liyi.ct import ct
from util import *

parser = argparse.ArgumentParser(description='attacks in PyTorch')

parser.add_argument('--input_dir', default='/data/public/data/2022aisc_facecomp/data/', type=str, help='directory of clean examples')
parser.add_argument('--output_dir', default='./results/', type=str, help='directory of crafted adversarial examples')
parser.add_argument('--keypoints_dir', default='/data/liyi/projects/aisc_facecomp/keypoints/show_masks/', type=str, help='directory of keypoints examples')
parser.add_argument('--attack_name', default='dim', type=str, help='attacks name')

parser.add_argument('--device', default='1', type=str, help='gpu device')
parser.add_argument('--use_gpu', default=True, type=bool, help='use gpu or not')
# parser.add_argument('--use_blocknum', default=-1, type=int, help='use blocks to accelerate or not')
# parser.add_argument('--total_blocks', default=-1, type=int, help='use total blocks to accelerate or not')
parser.add_argument('--ori_mask', action="store_true")

parser.add_argument('--speedup', action="store_true")
parser.add_argument('--sep', default=4, type=int, help='seperate data blocks to speed up')
parser.add_argument('--idx', default=0, type=int, help='index of current data block')
parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 0)')

parser.add_argument('--batch_size', type=int, default=30, help='mini-batch size (default: 1)')
parser.add_argument('--steps', default=10, type=int, help='max iteration')
parser.add_argument('--alpha', default=1.0, type=float, help='step size for each iter')
parser.add_argument('--decay', default=1.0, type=float, help='decay momentum of the attack')

parser.add_argument('--use_di', default=False, action='store_true', help='use di or not')
parser.add_argument('--use_ti', default=False, action='store_true', help='use ti or not')
parser.add_argument('--use_si', default=False, action='store_true', help='use si or not')
parser.add_argument('--use_pi', default=False, action='store_true', help='use pi or not')
parser.add_argument('--use_mi', default=False, action='store_true', help='use mi or not')
parser.add_argument('--n_ens', default=1, type=int)
parser.add_argument('--amplification_factor', default=2.0, type=float, help='PI amplification_factor')
parser.add_argument('--resize_rate', default=1.15, type=float, help='DIM resize_rate')
parser.add_argument('--diversity_prob', default=0.7, type=float, help='DIM diversity_prob')

parser.add_argument('--mtype', default="raw_keypoint", type=str)

parser.add_argument('--model_idx', nargs='+', default=-1, type=int)

args = parser.parse_args()

# pkern, kern_size = project_kern(3)
# gkern = TI_kernel()
# padding_size = kern_size

def main():
    # advs dir path
    save_path = args.output_dir
    os.makedirs(save_path, exist_ok=True)
    
    preprocess = T.Compose([
        T.ToTensor()
    ])
    
    #images dataset
    lfw = KP_LFW(args.input_dir, preprocess, mtype=args.mtype)
    if args.speedup:
        datalen = len(lfw) # 3000
        assert datalen % args.sep == 0
        bsize = datalen // args.sep
        start, end = bsize * args.idx, bsize * (args.idx + 1)
        index = list(range(start, end))
        lfw = torch.utils.data.Subset(lfw, index)
    lfw_loader = data.DataLoader(lfw, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    
    # models = get_fastmodel(idx=-1)
    models = get_fastmodel(args.model_idx)
    mean_cossim = 0
    for i, (origin_image, origin_file, compare_image, compare_file, keypoints_image, keypoints_file) in enumerate(lfw_loader):
        indice = []
        for k in range(origin_image.shape[0]):
            if not os.path.exists(os.path.join(save_path, os.path.basename(origin_file[k]))):
                indice.append(k)

        if not indice:
            continue
        else:
            origin_image = origin_image[indice]
            compare_image = compare_image[indice]
            keypoints_image = keypoints_image[indice]
            origin_file = [origin_file[i] for i in indice]

        if args.use_gpu:
            origin_image = origin_image.cuda()
            compare_image = compare_image.cuda()
            keypoints_image = keypoints_image.cuda()
        
        if args.attack_name == 'ct':
            advs, cossim = ct(
                origin_image,
                compare_image,
                keypoints_image,
                models,
                args.steps,
                args.alpha,
                decay=args.decay,
                resize_rate=args.resize_rate, 
                diversity_prob=args.diversity_prob,
                amplification_factor=args.amplification_factor,
                use_di=args.use_di,
                use_ti=args.use_ti,
                use_si=args.use_si,
                use_pi=args.use_pi,
                use_mi=args.use_mi,
                n_ens=args.n_ens,
                ori_mask=args.ori_mask,
            )
        else:
            raise NotImplementedError
        
        mean_cossim += cossim / len(lfw)

        for k in range(advs.shape[0]):
            save_img(os.path.join(save_path, os.path.basename(origin_file[k])), advs[k].detach().permute(1, 2, 0).cpu())
            print('attack in process, i = %d' % (i*args.batch_size+k))
    print(f'mean cosine similarty :{round(mean_cossim, 5)}')
    print('attack finished')
    
    with open('log.txt', 'a') as f:
        paras = save_path.split('/')[-1]
        f.write(f'{paras},inceptionresnetv1,mean cosine similarty,{round(mean_cossim, 5)}\n')    
    
if __name__ == '__main__':
    main()


        