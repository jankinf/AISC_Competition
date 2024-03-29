# coding=utf-8
'''Implementation of I-FGSM attack in PyTorch'''
import argparse
import os
import os.path
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
    x_batch = x_batch.reshape(m, *shape).transpose(1, 0, 2, 3, 4).reshape(*new_shape)
    return x_batch

def si_fgsm(x, y, model:ScoreInference, num_iter, eps, alpha, m):
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
    mul_coef =  np.array([(2**i) for i in range(m)]).reshape(-1, 1, 1, 1)
    for _ in range(num_iter):
        advs = scale_transform(adv, m)
        grads = model.grad_func(**{'data':advs, 'label':np.array([y for _ in range(m)])})
        noises = grads * mul_coef
        noise = np.sum(noises, axis=0, keepdims=True)
        adv = adv - alpha * np.sign(noise)
        adv = np.minimum(np.maximum(adv, min_x), max_x)
        adv = np.clip(adv, 0., 255.)
    
    return adv


def main():
    print(args)
    eps = args.max_epsilon * 1.0
    iteration = args.num_iter
    y = np.array(args.label)

    model = ScoreInference(model_path=args.i_net, device="cpu0", alpha=3)

    os.makedirs(output_dir, exist_ok=True)
    # os.makedirs(os.path.dirname(logfile), exist_ok=True)
    # fp = open(logfile, 'a')

    kwargs = {
        "num_iter": iteration,
        "eps": eps,
        "alpha": eps / iteration,
        "m": args.m,
    }

    for dtype in ['normal', 'attack']:
        cur_datadir = data_dir.format(dtype)
        img_list = [os.path.join(cur_datadir, v) for v in os.listdir(cur_datadir)]
        os.makedirs(os.path.join(output_dir, dtype), exist_ok=True)
        sorted(img_list)
    
        names = list(map(lambda x: x.split('/')[-1], img_list))
        for img_path, name in zip(img_list, names):
            save_path = os.path.join(output_dir, dtype, name)
            if os.path.exists(save_path):
                continue
            x = cv2.imread(img_path)
            x = x[None].transpose(0, 3, 1, 2)
            adv_x = si_fgsm(x, y, model, **kwargs)
            cv2.imwrite(save_path, adv_x[0].transpose(1, 2, 0).astype(np.uint8))
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
    parser.add_argument('--m', default=5, type=int, help='the number of scale copies')
    args = parser.parse_args()

    data_dir = '/data/projects/fmp_demo/attack/adv_data/preprocess224/{}'

    method = "{}_t{}".format(os.path.splitext(__file__.split('/')[-1])[0], str(args.label))
    settings = f"iter{args.num_iter}_eps{args.max_epsilon}"
    output_dir = os.path.join(args.output_dir, method, settings)
    logfile = os.path.join(args.log_dir, method, settings, "out.log")
    
    main()