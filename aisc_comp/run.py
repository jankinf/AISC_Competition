import torchvision.transforms as T
from torch.utils import data
import argparse
import os
from attacks import (
    dim, dem, mim, aa, optmask_dem, optmask_dem_v2,
    optmask_ct, optmask_pict, optmask_clip_pict,
    optmask_pict_catfeat, optmask_pict_catfeat_fill,
    optmask_ct_catfeat_fill, optmask_anda_catfeat_fill,
    ct_cos_catfeat_fill, ct_cos_catfeat_fill_multitgt,
    optmask_ds_l2, optmask_dem_v2_l2,
    optmask_di_l2, optmask_bim_l2, optmask_di_l2_adaw
)
from util import *
import functools

parser = argparse.ArgumentParser(description='attacks in PyTorch')
parser.add_argument('--input_dir', default='/data/projects/aisc_facecomp/data', type=str, help='directory of clean examples')
parser.add_argument('--output_dir', default='./results/', type=str, help='directory of crafted adversarial examples')
parser.add_argument('--attack_name', default='dim', type=str, help='attacks name')

parser.add_argument('--device', default='0', type=str, help='gpu device')
parser.add_argument('--use_gpu', default=True, type=bool, help='use gpu or not')
parser.add_argument('--workers', default=8, type=int, help='number of data loading workers (default: 8)')

parser.add_argument('--batch_size', type=int, default=30, help='mini-batch size (default: 1)')
parser.add_argument('--max_epsilon', default=16.0, type=float, help='max magnitude of adversarial perturbations')
parser.add_argument('--steps', default=10, type=int, help='max iteration')
parser.add_argument('--alpha', default=1.6, type=float, help='stepsize')
parser.add_argument('--masktype', default='keypoints', type=str)
parser.add_argument('--mtype', default="raw_keypoint", type=str)

parser.add_argument('--n', default=4, type=int)
parser.add_argument('--resize_rate', default=1.15, type=float)
parser.add_argument('--diversity_prob', default=0.7, type=float)
parser.add_argument('--bound', default=1.5, type=float)

parser.add_argument('--use_mi', action="store_true")

parser.add_argument('--hard_ctl', action="store_true")
parser.add_argument('--hard_ctl_sample', default=1500, type=int)

parser.add_argument('--temperature', default=3, type=int, help="temperature for softmax in adaw")

parser.add_argument('--sample', default=3000, type=int, help="number of data to attack")
parser.add_argument('--speedup', action="store_true")
parser.add_argument('--sep', default=4, type=int, help='seperate data blocks to speed up')
parser.add_argument('--idx', default=0, type=int, help='index of current data block')
parser.add_argument('--model_idx', nargs='+', default=-1, type=int)
parser.add_argument('--model_b_idx', nargs='+', default=[2], type=int)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.device
def main():
    # advs dir path
    save_path = args.output_dir
    os.makedirs(save_path, exist_ok=True)
    
    preprocess = T.Compose([
        T.ToTensor()
    ])
    
    # images dataset
    # lfw = LFW(args.input_dir, preprocess)
    if args.attack_name == 'beval_ct_cos_catfeat_fill_multitgt':
        assert args.batch_size == 1
        lfw = KP_LFW_MULTGT(args.input_dir, preprocess)
    else:
        lfw = KP_LFW(args.input_dir, preprocess, mtype=args.mtype, sample=args.sample)
    
    if args.hard_ctl:
        index = list(np.load("/data/projects/aisc_facecomp/hard_ctl.npy"))
        index = index[:args.hard_ctl_sample]
        lfw = torch.utils.data.Subset(lfw, index)

    if args.speedup:
        datalen = len(lfw) # 3000
        assert datalen % args.sep == 0
        bsize = datalen // args.sep
        start, end = bsize * args.idx, bsize * (args.idx + 1)
        index = list(range(start, end))
        lfw = torch.utils.data.Subset(lfw, index)
    lfw_loader = data.DataLoader(lfw, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    
    # models = get_model(args.model_idx)
    models = get_fastmodel(args.model_idx)
    # process images
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
        
        if args.attack_name == 'dim':
            advs, cossim = dim(
                origin_image,
                compare_image,
                models,
                args.max_epsilon,
                args.steps,
                args.max_epsilon / 255 / args.steps,
            )
        elif args.attack_name == 'dem':
            advs, cossim = dem(
                origin_image,
                compare_image,
                models,
                args.max_epsilon,
                args.steps,
                args.max_epsilon / 255 / args.steps,
            )
        elif args.attack_name == 'optmask_dem':
            # advs, cossim = optmask_dem(
            #     origin_image,
            #     compare_image,
            #     keypoints_image,
            #     models,
            #     args.max_epsilon,
            #     args.steps,
            #     args.max_epsilon / 255 / args.steps,
            # )
            advs, cossim = optmask_dem(
                origin_image,
                compare_image,
                keypoints_image,
                models,
                args.steps,
                args.alpha,
                args.bound,
                args.n,
                args.masktype,
                args.use_mi,
            )
        elif args.attack_name == 'optmask_dem_v2':
            advs, cossim = optmask_dem_v2(
                origin_image,
                compare_image,
                keypoints_image,
                models,
                args.max_epsilon,
                args.steps,
                args.max_epsilon / 255 / args.steps,
                n=args.n,
                bound=args.bound,
            )
        elif args.attack_name == 'optmask_ct':
            # ti_kernel = Translation_Kernel(len_kernel=7, nsig=3, kernel_name='gaussian')
            ti_kernel = Translation_Kernel(len_kernel=5, nsig=1, kernel_name='gaussian')
            gkern = torch.from_numpy(ti_kernel.kernel_generation())
            if args.use_gpu:
                gkern = gkern.cuda()
            advs, cossim = optmask_ct(
                origin_image,
                compare_image,
                keypoints_image,
                gkern,
                models,
                args.max_epsilon,
                args.steps,
                args.max_epsilon / 255 / args.steps,
            )
        elif args.attack_name == 'optmask_pict':
            # init some params for tifgsm
            ti_kernel = Translation_Kernel(len_kernel=5, nsig=1, kernel_name='gaussian')
            gkern = torch.from_numpy(ti_kernel.kernel_generation())

            # init some params for pifgsm
            ampf = 2.0 # amplification_factor
            pkern, padding_size = project_kern(3)
            if args.use_gpu:
                gkern = gkern.cuda()
                pkern = pkern.cuda()
            advs, cossim = optmask_pict(
                origin_image,
                compare_image,
                keypoints_image,
                gkern,
                models,
                args.max_epsilon,
                args.steps,
                args.max_epsilon / 255 / args.steps,
                pkern,
                padding_size,
                ampf,
            )
        elif args.attack_name == 'optmask_pict_catfeat':
            # init some params for tifgsm
            ti_kernel = Translation_Kernel(len_kernel=5, nsig=1, kernel_name='gaussian')
            gkern = torch.from_numpy(ti_kernel.kernel_generation())

            # init some params for pifgsm
            ampf = 2.0 # amplification_factor
            pkern, padding_size = project_kern(3)
            if args.use_gpu:
                gkern = gkern.cuda()
                pkern = pkern.cuda()
            advs, cossim = optmask_pict_catfeat(
                origin_image,
                compare_image,
                keypoints_image,
                gkern,
                models,
                args.max_epsilon,
                args.steps,
                args.max_epsilon / 255 / args.steps,
                pkern,
                padding_size,
                ampf,
            )
        elif args.attack_name == 'optmask_pict_catfeat_fill':
            # init some params for tifgsm
            ti_kernel = Translation_Kernel(len_kernel=5, nsig=1, kernel_name='gaussian')
            gkern = torch.from_numpy(ti_kernel.kernel_generation())

            # init some params for pifgsm
            ampf = 2.0 # amplification_factor
            pkern, padding_size = project_kern(3)
            if args.use_gpu:
                gkern = gkern.cuda()
                pkern = pkern.cuda()
            advs, cossim = optmask_pict_catfeat_fill(
                origin_image,
                compare_image,
                keypoints_image,
                gkern,
                models,
                args.max_epsilon,
                args.steps,
                args.max_epsilon / 255 / args.steps if args.steps != 0 else 0,
                pkern,
                padding_size,
                ampf,
            )
        elif args.attack_name == 'optmask_ct_catfeat_fill':
            # init some params for tifgsm
            ti_kernel = Translation_Kernel(len_kernel=5, nsig=1, kernel_name='gaussian')
            gkern = torch.from_numpy(ti_kernel.kernel_generation())

            if args.use_gpu:
                gkern = gkern.cuda()

            advs, cossim = optmask_ct_catfeat_fill(
                origin_image,
                compare_image,
                keypoints_image,
                gkern,
                models,
                args.steps,
                args.alpha,
                args.masktype,
            )
        elif args.attack_name == 'beval_ct_cos_catfeat_fill':
            # init some params for tifgsm
            ti_kernel = Translation_Kernel(len_kernel=5, nsig=1, kernel_name='gaussian')
            gkern = torch.from_numpy(ti_kernel.kernel_generation())
            
            def cos_alpha(t, tmin=0.1, tmax=5, T=50):
                # alpha = tmin + 0.5 * (tmax - tmin) * (1 + np.cos((t + 1) / T * np.pi))
                # alpha = tmin + 0.5 * (tmax - tmin) * (1 + np.cos(np.pi - 2 * (t + 1) / T * np.pi))
                # alpha = tmin + 0.5 * (tmax - tmin) * (1 + np.cos(np.pi - (t + 1) / T * np.pi))
                alpha = tmin + 0.5 * (tmax - tmin) * (1 + np.cos(np.pi - 4 * (t + 1) / T * np.pi))
                return alpha
            updata_alpha = functools.partial(cos_alpha, tmin=0.6, tmax=2 * args.alpha, T=args.steps)

            if args.use_gpu:
                gkern = gkern.cuda()

            models_blackbox = get_fastmodel(args.model_b_idx)
            advs, cossim = ct_cos_catfeat_fill(
                origin_image,
                compare_image,
                keypoints_image,
                gkern,
                models,
                models_blackbox,
                args.steps,
                updata_alpha
            )
        elif args.attack_name == 'beval_ct_cos_catfeat_fill_multitgt':
            # init some params for tifgsm
            ti_kernel = Translation_Kernel(len_kernel=5, nsig=1, kernel_name='gaussian')
            gkern = torch.from_numpy(ti_kernel.kernel_generation())
            
            def cos_alpha(t, tmin=0.1, tmax=5, T=50):
                # alpha = tmin + 0.5 * (tmax - tmin) * (1 + np.cos((t + 1) / T * np.pi))
                # alpha = tmin + 0.5 * (tmax - tmin) * (1 + np.cos(np.pi - 2 * (t + 1) / T * np.pi))
                # alpha = tmin + 0.5 * (tmax - tmin) * (1 + np.cos(np.pi - (t + 1) / T * np.pi))
                alpha = tmin + 0.5 * (tmax - tmin) * (1 + np.cos(np.pi - 4 * (t + 1) / T * np.pi))
                return alpha
            updata_alpha = functools.partial(cos_alpha, tmin=0.6, tmax=2 * args.alpha, T=args.steps)

            if args.use_gpu:
                gkern = gkern.cuda()

            models_blackbox = get_fastmodel(args.model_b_idx)
            advs, cossim = ct_cos_catfeat_fill_multitgt(
                origin_image,
                compare_image[0],
                keypoints_image,
                gkern,
                models,
                models_blackbox,
                args.steps,
                updata_alpha
            )
        elif args.attack_name == 'optmask_anda_catfeat_fill':
            # init some params for tifgsm
            ti_kernel = Translation_Kernel(len_kernel=5, nsig=1, kernel_name='gaussian')
            gkern = torch.from_numpy(ti_kernel.kernel_generation())
            thetas = get_thetas(int(math.sqrt(25)), -0.3, 0.3)
            if args.use_gpu:
                gkern = gkern.cuda()

            advs, cossim = optmask_anda_catfeat_fill(
                origin_image,
                compare_image,
                keypoints_image,
                gkern,
                models,
                args.steps,
                args.alpha,
                thetas,
            )
        elif args.attack_name == 'optmask_clip_pict':
            # init some params for tifgsm
            ti_kernel = Translation_Kernel(len_kernel=5, nsig=1, kernel_name='gaussian')
            gkern = torch.from_numpy(ti_kernel.kernel_generation())

            # init some params for pifgsm
            ampf = 2.0 # amplification_factor
            pkern, padding_size = project_kern(3)
            if args.use_gpu:
                gkern = gkern.cuda()
                pkern = pkern.cuda()
            advs, cossim = optmask_clip_pict(
                origin_image,
                compare_image,
                keypoints_image,
                gkern,
                models,
                args.max_epsilon,
                args.steps,
                args.max_epsilon / 255 / args.steps,
                pkern,
                padding_size,
                ampf,
            )
        elif args.attack_name == 'optmask_ds_l2':
            advs, cossim = optmask_ds_l2(
                origin_image,
                compare_image,
                keypoints_image,
                models,
                args.steps,
                args.alpha,
                args.masktype,
            )
        elif args.attack_name == 'optmask_di_l2':
            advs, cossim = optmask_di_l2(
                origin_image,
                compare_image,
                keypoints_image,
                models,
                args.steps,
                args.alpha,
                args.n,
                args.resize_rate, 
                args.diversity_prob,
                args.masktype,
            )
        elif args.attack_name == 'optmask_di_l2_adaw':
            advs, cossim = optmask_di_l2_adaw(
                origin_image,
                compare_image,
                keypoints_image,
                models,
                args.steps,
                args.alpha,
                args.n,
                args.resize_rate, 
                args.diversity_prob,
                args.temperature,
                args.masktype,
            )
        elif args.attack_name == 'optmask_bim_l2':
            advs, cossim = optmask_bim_l2(
                origin_image,
                compare_image,
                keypoints_image,
                models,
                args.steps,
                args.alpha,
                args.masktype,
            )
        elif args.attack_name == 'optmask_dem_v2_l2':
            advs, cossim = optmask_dem_v2_l2(
                origin_image,
                compare_image,
                keypoints_image,
                models,
                args.steps,
                args.alpha,
                args.bound,
                args.n,
                args.masktype,
            )
        elif args.attack_name == 'mim':
            advs, cossim = mim(
                origin_image,
                compare_image,
                models,
                args.max_epsilon,
                args.steps,
                args.max_epsilon / 255 / args.steps,
            )
        elif args.attack_name == 'aa':
            advs, cossim = aa(
                origin_image,
                compare_image,
                models,
                args.max_epsilon,
                args.steps,
                args.max_epsilon / 255 / args.steps,
            )
        else:
            raise NotImplementedError
        
        mean_cossim += cossim / len(lfw)
        for k in range(advs.shape[0]):
            img = advs[k].detach().permute(1, 2, 0).cpu()
            img = np.array(img * 255).astype('uint8')
            Image.fromarray(img).save(os.path.join(save_path, os.path.basename(origin_file[k])))

    print(f'mean cosine similarty: {round(mean_cossim, 5)}')
    print('attack finished')
    
    with open('log.txt', 'a') as f:
        paras = save_path.split('/')[-1]
        # f.write(f'{paras}, inceptionresnetv1, mean cosine similarty,{round(mean_cossim, 5)}\n')
        # f.write(f'{paras}, ens10, mean cosine similarty,{round(mean_cossim, 5)}\n')
        f.write(f'{paras}, ens1, mean cosine similarty,{round(mean_cossim, 5)}\n')
    
if __name__ == '__main__':
    main()


        