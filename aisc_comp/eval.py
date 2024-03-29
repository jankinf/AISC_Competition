import torchvision.transforms as T
from torch.utils import data
import argparse
import os
from util import *

parser = argparse.ArgumentParser(description='attacks in PyTorch')
parser.add_argument('--input_dir', default='/data/projects/aisc_facecomp/data', type=str, help='directory of clean examples')
parser.add_argument('--device', default='0', type=str, help='gpu device')
parser.add_argument('--use_gpu', default=True, type=bool, help='use gpu or not')
parser.add_argument('--workers', default=0, type=int, help='number of data loading workers (default: 0)')
parser.add_argument('--batch_size', type=int, default=30, help='mini-batch size (default: 30)')


args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.device

def main():
    preprocess = T.Compose([
        T.ToTensor()
    ])
    
    # images dataset
    lfw = KP_LFW(args.input_dir, preprocess)
    lfw_loader = data.DataLoader(lfw, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    
    criterion = nn.CosineSimilarity(dim=1, eps=1e-8)
    models = get_model()
    # process images
<<<<<<< HEAD
    mean_cossim = 0
=======
    n_model = len(models)

    results = [_ for _ in range(n_model + 1)]
    model_name = ["model_{}".format(i) for i in range(n_model)] + ['ensemble_{}'.format(n_model)]
>>>>>>> 4c3e810eeaef789e3fd57bb9506d1828d0e58cb0
    with torch.no_grad():
        for i, (adv_image, origin_file, compare_image, compare_file, keypoints_image, keypoints_file) in enumerate(lfw_loader):
            if args.use_gpu:
                adv_image = adv_image.cuda()
                compare_image = compare_image.cuda()
            
            adv_logits = 0
<<<<<<< HEAD
            for model in models:   
                adv_logits += model(adv_image)
            adv_logits /= len(models)

            compare_logits = 0
            for model in models:   
                compare_logits += model(compare_image)
            compare_logits /= len(models)

            cossim = criterion(adv_logits, compare_logits).detach().sum().item()
            
            mean_cossim += cossim / len(lfw)

        print(f'mean cosine similarty: {round(mean_cossim, 5)}')
    
    with open('eval.log', 'a') as f:
        paras = args.input_dir.split('/')[-1]
        f.write(f'{paras}, ens10, mean cosine similarty, {round(mean_cossim, 5)}\n')
=======
            compare_logits = 0
            for idx, model in enumerate(models):   
                logits1 = model(adv_image).detach()
                logits2 = model(compare_image).detach()
                sin_cos = criterion(logits1, logits2).detach().sum().item()
                results[idx] += sin_cos
                adv_logits += logits1
                compare_logits += logits2
            adv_logits /= len(models)
            compare_logits /= len(models)
            total_cos = criterion(adv_logits, compare_logits).detach().sum().item()
            results[-1] += total_cos
        
    n_sample = len(lfw)
    with open('eval.log', 'a') as f:
        paras = args.input_dir.split('/')[-1]
        f.write(paras + '\n')
        for name, result in zip(model_name, results):
            result /= n_sample
            f.write('{}: mean cosine similarty: {}\n'.format(name, round(result, 5)))
>>>>>>> 4c3e810eeaef789e3fd57bb9506d1828d0e58cb0
    
if __name__ == '__main__':
    main()


        