# coding=utf-8
'''Implementation of I-FGSM attack in PyTorch'''
import argparse
import os.path
from tqdm import tqdm
import scipy.stats as st
from scipy import ndimage
import numpy as np
import cv2
from inference import RunnableInference


class ScoreInference(RunnableInference):
    def get_grad(self, img, label):
        grad = self.grad_func(**{'data':img, 'label':np.array(label)})
        return grad
        
    def pred(self, input_batch):
        scores = self.pred_func(input_batch)[0]
        return scores

def scale_transform(input_tensor, m=5):
    shape = input_tensor.shape
    outs = [(input_tensor) / (2**i) for i in range(m)]
    x_batch = np.concatenate(outs, axis=0)
    new_shape = x_batch.shape
    x_batch = x_batch.reshape(m, *shape).transpose(1, 0).reshape(*new_shape)
    return x_batch

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

def input_diversity(input_tensor, diversity_prob=0.5):
    nsample = input_tensor.shape[0]
    indice = np.random.rand(nsample) >= diversity_prob
    image_height, image_width = input_tensor.shape[-2:]
    resize_width, resize_height = int(image_width * 1.1), int(image_height * 1.1)
    rnd_w = np.random.randint(image_width, resize_width, ())
    rnd_h = np.random.randint(image_height, resize_height, ())
    # rescaled = F.interpolate(input_tensor, size=[rnd_h, rnd_w], mode='bilinear', align_corners=True)
    rescaled = ndimage.zoom(input_tensor, (1, 1, rnd_h, rnd_w), order=1)[0].transpose(1, 2, 0)
    h_rem = resize_height - rnd_h
    w_rem = resize_width - rnd_w
    pad_top = np.random.randint(0, h_rem, ())
    pad_bottom = h_rem - pad_top
    pad_left = np.random.randint(0, w_rem, ())
    pad_right = w_rem - pad_left
    padded = np.pad((pad_left, pad_right, pad_top, pad_bottom), 0.)(rescaled)
    # padded_img = nn.functional.interpolate(padded, (224, 224), mode='bilinear', align_corners=False)
    # orig_img = nn.functional.interpolate(input_tensor, (224, 224), mode='bilinear', align_corners=False)
    padded_img = ndimage.zoom(padded, (1, 1, 224, 224), order=1)[0].transpose(1, 2, 0)
    orig_img = ndimage.zoom(input_tensor, (1, 1, 224, 224), order=1)[0].transpose(1, 2, 0)
    out_imgs = np.concatenate([padded_img[indice], orig_img[~indice]], axis=0)
    return out_imgs

def ct_fgsm_wb(x, y, model:ScoreInference, num_iter, eps, alpha, gaussian_kernel):
    '''
    craft adversarial examples
    :param x: clean images in batch in [0, 1]
    :param y: target labels
    :return: adv in [0, 1]
    '''
    x = x.astype(np.float32)
    min_x = x - eps
    max_x = x + eps

    adv = x.copy()

    for _ in range(num_iter):
        # todo: sifgsm
        si_xts = scale_transform(adv, m=args.m)
        # todo: difgsm
        resize_adv = input_diversity(si_xts, diversity_prob=0.5)
        noise = model.grad_func(**{'data':resize_adv, 'label':np.array(y)})
        # todo: tifgsm
        noise = F.conv2d(noise, gaussian_kernel, stride=1, padding='same', groups=3)
        adv = adv - alpha * noise.sign()
        
        adv = np.minimum(np.maximum(adv, min_x), max_x)
        adv = np.clip(adv, 0., 255.)
    
    return adv


def main():
    print(args)
    eps = args.max_epsilon
    iteration = args.num_iter
    y = np.array(args.label)

    model = ScoreInference(model_path=args.i_net, device="cpu0", alpha=3)

    os.makedirs(output_dir, exist_ok=True)
    # os.makedirs(os.path.dirname(logfile), exist_ok=True)
    # fp = open(logfile, 'a')

    ti_kernel = Translation_Kernel(len_kernel=args.kernel_size, nsig=args.sigma, kernel_name=args.kernel_name)
    gaussian_kernel = ti_kernel.kernel_generation()
    kwargs = {
        "num_iter": iteration,
        "eps": eps,
        "alpha": eps * 1.0 / iteration,
        "gaussian_kernel": gaussian_kernel,
    }

    for dtype in ['normal', 'attack']:
        cur_datadir = data_dir.format(dtype)
        img_list = [os.path.join(cur_datadir, v) for v in os.listdir(cur_datadir)]
        sorted(img_list)
    
        for img_path in img_list:
            name = list(map(lambda x: x.split('/')[-1], img_path))
            x = cv2.imread(name)
            x = x[None].transpose(0, 3, 1, 2)
            adv_x = ct_fgsm_wb(x, y, model, **kwargs)
            cv2.imwrite(os.path.join(output_dir, dtype, name), adv_x[0].tranpose(1, 2, 0).astype(np.uint8))
            
    # fp.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--i_net", type=str, help="image network path")
    parser.add_argument("--thres", type=float, default=0.5)
    parser.add_argument('--num_iter', default=10, type=int, help='max iteration')
    parser.add_argument("--max_epsilon", type=int, default=16)
    parser.add_argument("--label", type=int, default=0, choices=[0, 1], help="indicates target label")
    parser.add_argument('--output_dir', default='./attack/adv_data/whitebox', type=str,
                        help='directory of crafted adversarial examples')
    parser.add_argument("--log_dir", type=str, default="./logs/whitebox")

    parser.add_argument('--kernel_size', default=15, type=int, help='kernel_size')
    parser.add_argument('--sigma', default=3, type=int, help='sigma of kernel')
    parser.add_argument('--kernel_name', default='gaussian', type=str)
    parser.add_argument('--m', default=5, type=int, help='the number of scale copies')
    args = parser.parse_args()

    data_dir = '/data/projects/fmp_demo/attack/adv_data/preprocess224/{}'

    method = "{}_t{}".format(os.path.splitext(__file__.split('/')[-1])[0], str(args.label))
    settings = f"iter{args.num_iter}_eps{args.max_epsilon}"
    output_dir = os.path.join(args.output_dir, method, settings)
    logfile = os.path.join(args.log_dir, method, settings, "out.log")
    
    main()