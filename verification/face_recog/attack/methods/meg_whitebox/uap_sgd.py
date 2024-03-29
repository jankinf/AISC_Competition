# coding=utf-8
'''Implementation of I-FGSM attack in PyTorch'''
import argparse
import os
import os.path
import numpy as np
import cv2
from inference import RunnableInference_v2 as RunnableInference
from neupeak.utils.misc import set_mgb_default_device
import neupeak.utils.inference as inf
import megskull.opr.all as oprs
import neupeak


def grad_max_layernorm(net, layername, device=None):
    if device is not None:
        set_mgb_default_device(device)
    # layernames = list(net.all_oprs_dict.keys())
    # for layer in layernames: print(layer)
    # relu_indice = [idx for idx, layername in enumerate(layernames) if layername.startswith('ReLU')]
    # relu_layers = list(map(lambda idx: layernames[idx], relu_indice))
    # layername = 'ReLU2596'
    relu_layer = net.loss_visitor.all_oprs_dict[layername]
    fnorm = oprs.Sqrt((relu_layer.outputs[0]**2).sum())
    grad = oprs.Grad(fnorm, net.loss_visitor.all_oprs[0], name='fnorm_grad')
    grad_func = inf.Function(inf.get_fprop_env(fast_run=False))
    return grad_func.compile(grad)

def clamped_celoss(net, beta, device=None):
    if device is not None:
        set_mgb_default_device(device)
    clamped_loss = oprs.Min(net.loss_var, oprs.ConstProvider(beta, dtype=np.float32))
    grad_loss = oprs.Grad(clamped_loss, net.loss_visitor.all_oprs[0], name='grad_loss')
    gloss_func = inf.Function(inf.get_fprop_env(fast_run=False))
    return gloss_func.compile(grad_loss)

class ScoreInference(RunnableInference):
    def init_grad_max_layernorm(self, layername='ReLU2596'):
        self.grad_max_layernorm = grad_max_layernorm(self.net, layername)

    def init_grad_clamped_celoss(self, beta=12):
        self.clamped_celoss = clamped_celoss(self.net, beta)

    def get_grad_max_layernorm(self, img, label):
        grad = self.grad_max_layernorm(**{'data':img, 'label':np.array(label)})
        return grad

    def get_grad_loss(self, img, label):
        grad = self.clamped_celoss(**{'data':img, 'label':np.array(label)})
        return grad
        
    def pred(self, input_batch):
        scores = self.pred_func(input_batch)[0]
        return scores

def get_minibatch(x: np.ndarray, minibatch: int):
    nsize = x.shape[0]
    idx = np.arange(nsize)
    np.random.shuffle(idx)
    start = 0
    while start < nsize:
        end = min(nsize, start + minibatch)
        cur_idx = idx[start:end]
        yield x[cur_idx]
        start += minibatch

def uap_sgd(img_list, y, model:ScoreInference, get_grad, resume, epoch, eps, alpha, batch_size):
    """craft adversarial universal pattern

    Args:
        img_list (_type_): _description_
        y (_type_): target labels
        model (ScoreInference): _description_
        resume (_type_): _description_
        epoch (_type_): _description_
        eps (_type_): _description_
        alpha (_type_): _description_

    Returns:
        unipattern _type_: _description_
    """
    y = y.astype(np.int64)
    imgs = []
    for name in img_list:
        img = cv2.imread(name)
        imgs.append(img)
    x = np.array(imgs).astype(np.float32).transpose(0, 3, 1, 2)

    if resume and len(os.listdir(noise_saving)) > 0:
        # todo: load resume
        filenames = os.listdir(noise_saving)
        sorted(filenames, key=lambda x : float(x.split('@')[0][5:]))
        noise = np.load('{}/{}'.format(noise_saving, filenames[-1]))
        start_epoch = int(filenames[-1].split('@')[0][5:]) + 1
    else:
        noise = np.zeros(shape=(1, *x.shape[1:]))
        start_epoch = 1


    for cur_epoch in range(start_epoch - 1, epoch):
        for batch_x in get_minibatch(x, batch_size):
            perturbed = np.clip(batch_x + noise, 0., 255.)
            ys = y.repeat(perturbed.shape[0])
            grads = get_grad(**{'data':perturbed, 'label':ys})
            grad = np.mean(grads, axis=0)[None]
            noise = noise - alpha * np.sign(grad)
            noise = np.clip(noise, -eps, eps)
        
        suc = 0
        for batch_x in get_minibatch(x, batch_size):
            perturbed = np.clip(batch_x + noise, 0., 255.)
            scores = model.pred(perturbed)
            suc += (scores.argmax(-1) == y).sum()

        np.save("{}/epoch{}@suc{}.npy".format(noise_saving, cur_epoch, suc), noise)
    return noise

def get_img_list(path):
    if os.path.isfile(path):
        img_list = [path]
    else:
        img_list = [os.path.join(path, v) for v in os.listdir(path)]
    sorted(img_list)
    return img_list

def main():
    print(args)
    y = np.array(args.label)

    model = ScoreInference(model_path=args.i_net, device="cpu0", alpha=3)

    normal_list = get_img_list(data_dir.format('normal'))
    attack_list = get_img_list(data_dir.format('attack'))

    train_img_list = normal_list[:args.train_images // 2] + attack_list[:args.train_images // 2]
    val_img_list = normal_list[args.train_images // 2:] + attack_list[args.train_images // 2:]

    # os.makedirs(output_dir, exist_ok=True)
    os.makedirs(noise_saving, exist_ok=True)
    # os.makedirs(os.path.dirname(logfile), exist_ok=True)
    # fp = open(logfile, 'a')

    kwargs = {
        "epoch": args.epoch,
        "eps": args.max_epsilon * 1.0,
        "alpha": args.max_epsilon * args.step_decay,
        "batch_size": args.batch_size,
    }
    
    if args.loss_type == 'celoss':
        model.init_grad_clamped_celoss(beta=args.beta)
        get_grad = model.clamped_celoss
    elif args.loss_type == 'layernorm':
        model.init_grad_max_layernorm(layername=args.layername)
        get_grad = model.grad_max_layernorm

    uap_sgd(img_list=train_img_list, y=y, model=model, get_grad=get_grad, resume=True, **kwargs)
            
    # fp.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--i_net", type=str, help="image network path")
    parser.add_argument("--thres", type=float, default=0.5)
    # parser.add_argument('--output_dir', default='./attack/adv_data/whitebox', type=str,
    #                     help='directory of crafted adversarial examples')
    # parser.add_argument("--log_dir", type=str, default="./logs/whitebox")
    # --------------- params ---------------|
    parser.add_argument("--label", type=int, default=0, choices=[0, 1], help="indicates target label")
    parser.add_argument("--step_decay", type=float, default=0.6, help='0.6 for targeted attack and 0.7 for untargeted attack')
    parser.add_argument('--epoch', default=10, type=int, help='number of training epoch')
    parser.add_argument("--max_epsilon", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--train_images", type=int, default=500)
    
    parser.add_argument("--loss_type", type=str, default='celoss', choices=['celoss', 'layernorm'])
    parser.add_argument("--layername", type=str, default='ReLU2596', help='for layernorm loss')
    parser.add_argument("--beta", type=float, default=12, help='for clamped celoss')
    # --------------- params ---------------|
    args = parser.parse_args()

    data_dir = '/data/projects/fmp_demo/attack/adv_data/preprocess224/{}'

    method = "{}_t{}".format(os.path.splitext(__file__.split('/')[-1])[0], str(args.label))
    # if args.loss_type == 'celoss':
    #     old_settings = f"train{args.train_images}_epoch{args.epoch}_eps{args.max_epsilon}_{args.loss_type}_beta{args.beta}"
    # elif args.loss_type == 'layernorm':
    #     old_settings = f"train{args.train_images}_epoch{args.epoch}_eps{args.max_epsilon}_{args.loss_type}_{args.layername}"
    
    if args.loss_type == 'celoss':
        settings = f"train{args.train_images}_bs{args.batch_size}_epoch{args.epoch}_eps{args.max_epsilon}_step{args.step_decay}_{args.loss_type}_beta{args.beta}"
    elif args.loss_type == 'layernorm':
        settings = f"train{args.train_images}_bs{args.batch_size}_epoch{args.epoch}_eps{args.max_epsilon}_step{args.step_decay}_{args.loss_type}_{args.layername}"
    # output_dir = os.path.join(args.output_dir, method, settings)
    # logfile = os.path.join(args.log_dir, method, settings, "train{}.log".format(args.train_images))
    noise_saving = os.path.join("/data/projects/fmp_demo/attack/whitebox", method, settings)

    main()