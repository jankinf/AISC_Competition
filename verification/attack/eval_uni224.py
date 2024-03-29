import os
import cv2
import argparse
import numpy as np
from typing import List
from inference import RunnableInference, get_rect_from_ld, face_crop, resize_with_padding, i01c_to_ic01


class ScoreInference(RunnableInference):
    def pred(self, input_batch):
        scores = self.pred_func(input_batch)[0]
        return scores


def get_img_list(path):
    if os.path.isfile(path):
        img_list = [path]
    else:
        img_list = [os.path.join(path, v) for v in os.listdir(path)]
    sorted(img_list)
    return img_list


def get_minibatch(img_list: List[str], minibatch: int):
    nsize = len(img_list)
    start = 0
    while start < nsize:
        end = min(nsize, start + minibatch)
        data = []
        for idx in range(start, end):
            im = cv2.imread(img_list[idx])
            data.append(i01c_to_ic01(im))
        data = np.array(data, 'uint8')
        yield data
        start += minibatch


def run_imgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--i_net", type=str, help="image network path")
    parser.add_argument("--thres", type=float, default=0.5)
    parser.add_argument("--name", type=str)
    parser.add_argument("--pattern", type=str)
    args = parser.parse_args()

    r = ScoreInference(model_path=args.i_net, device="cpu0", alpha=3)
    normal_dir = '/data/projects/fmp_demo/attack/adv_data/preprocess224/normal'
    attack_dir = '/data/projects/fmp_demo/attack/adv_data/preprocess224/attack'
    normal_list = get_img_list(normal_dir)
    attack_list = get_img_list(attack_dir)

    train_images = int(args.pattern.split('train')[1].split('_')[0])
    train_img_list = normal_list[:train_images // 2] + attack_list[:train_images // 2]
    val_img_list = normal_list[train_images // 2:] + attack_list[train_images // 2:]
    
    train_size = len(train_img_list)
    val_size = len(val_img_list)

    # todo: get target label
    target = 1 if 't1' in args.pattern else 0

    # todo: load pattern
    pattern = np.load(args.pattern)

    # todo: generalization eval
    pred_train = []
    perturb_train = []
    for data in get_minibatch(train_img_list, 500):
        data = data + pattern
        data = np.clip(data, 0., 255.).astype(np.uint8)
        scores = r.pred(data)
        perturb_train.append(data)
        pred_train.append(scores.argmax(-1))
    pred_train = np.concatenate(pred_train).reshape(-1)
    perturb_train = np.concatenate(perturb_train)
    
    pred_val = []
    perturb_val = []
    for data in get_minibatch(val_img_list, 500):
        data = data + pattern
        data = np.clip(data, 0., 255.).astype(np.uint8)
        scores = r.pred(data)
        perturb_val.append(data)
        pred_val.append(scores.argmax(-1))
    pred_val = np.concatenate(pred_val).reshape(-1)
    perturb_val = np.concatenate(perturb_val)
    
    # todo: save perturb img
    normal_pimg = np.concatenate([perturb_train[:train_size // 2], perturb_val[:val_size // 2]]) 
    attack_pimg = np.concatenate([perturb_train[train_size // 2:], perturb_val[val_size // 2:]]) 
    root_dir = "/data/projects/fmp_demo/attack/adv_data/"
    attack_dir = os.path.join(root_dir, args.name, "attack")
    normal_dir = os.path.join(root_dir, args.name, "normal")
    os.makedirs(attack_dir, exist_ok=True)
    os.makedirs(normal_dir, exist_ok=True)
    for i, path in enumerate(normal_list):
        name = os.path.basename(path)
        save_path = os.path.join(normal_dir, name)
        cv2.imwrite(save_path, normal_pimg[i].transpose(1, 2, 0))

    for i, path in enumerate(attack_list):
        name = os.path.basename(path)
        save_path = os.path.join(attack_dir, name)
        cv2.imwrite(save_path, attack_pimg[i].transpose(1, 2, 0))

    normal_res = np.concatenate([pred_train[:train_size // 2], pred_val[:val_size // 2]])
    attack_res = np.concatenate([pred_train[train_size // 2:], pred_val[val_size // 2:]])

    tpr = attack_res.sum() / attack_res.shape[0]
    fpr = normal_res.sum() / normal_res.shape[0]

    suc_train = (pred_train == target)
    suc_val = (pred_val == target)

    print("suc rate on trainset: {}/{}={}".format(suc_train.sum(),
          train_size, suc_train.sum() * 1. / train_size))
    print("suc rate on valset: {}/{}={}".format(suc_val.sum(),
                                                val_size, suc_val.sum() * 1. / val_size))
    print("tpr: {}/{}={}".format(attack_res.sum().item(), attack_res.shape[0], tpr))
    print("fpr: 1 - {}/{}={}".format(normal_res.shape[0] - normal_res.sum().item(), normal_res.shape[0], fpr))

    logfile = os.path.join("logs/eval", "{}.log".format(args.name))
    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    with open(logfile, "a") as f:
        f.write(f'params: {str(args)}\n')
        f.write(f'eval {args.pattern}\n')
        f.write("suc rate on trainset: {}/{}={}\n".format(suc_train.sum(),
                train_size, suc_train.sum() * 1. / train_size))
        f.write("suc rate on valset: {}/{}={}\n".format(suc_val.sum(),
                val_size, suc_val.sum() * 1. / val_size))
        f.write("tpr: {}/{}={}\n".format(attack_res.sum().item(), attack_res.shape[0], tpr))
        f.write("fpr: 1 - {}/{}={}\n".format(normal_res.shape[0] - normal_res.sum().item(), normal_res.shape[0], fpr))
        f.write("{}\t{}\n".format(tpr, fpr))
        f.write('\n')


if __name__ == "__main__":
    run_imgs()
