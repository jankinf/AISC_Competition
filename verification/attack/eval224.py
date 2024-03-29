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
    parser.add_argument("--data_dir", type=str)
    args = parser.parse_args()

    r = ScoreInference(model_path=args.i_net, device="cpu0", alpha=3)
    data_dir = args.data_dir
    normal_dir = data_dir + '/normal'
    attack_dir = data_dir + '/attack'
    normal_list = get_img_list(normal_dir)
    attack_list = get_img_list(attack_dir)

    # todo: generalization eval
    pred_attack = []
    for data in get_minibatch(attack_list, 500):
        scores = r.pred(data)
        pred_attack.append(scores.argmax(-1))
    pred_attack = np.concatenate(pred_attack).reshape(-1)
    
    pred_normal = []
    for data in get_minibatch(normal_list, 500):
        scores = r.pred(data)
        pred_normal.append(scores.argmax(-1))
    pred_normal = np.concatenate(pred_normal).reshape(-1)
    
    tpr = pred_attack.sum() / pred_attack.shape[0]
    fpr = pred_normal.sum() / pred_normal.shape[0]
    print("tpr: {}/{}={}".format(pred_attack.sum().item(), pred_attack.shape[0], tpr))
    print("fpr: 1 - {}/{}={}".format(pred_normal.shape[0] - pred_normal.sum().item(), pred_normal.shape[0], fpr))

    logfile = os.path.join("logs/eval", "{}.log".format(args.name))
    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    with open(logfile, "a") as f:
        f.write(f'params: {str(args)}\n')
        f.write(f'eval {args.data_dir}\n')
        f.write("tpr: {}/{}={}\n".format(pred_attack.sum().item(), pred_attack.shape[0], tpr))
        f.write("fpr: 1 - {}/{}={}\n".format(pred_normal.shape[0] - pred_normal.sum().item(), pred_normal.shape[0], fpr))
        f.write("{}\t{}\n".format(tpr, fpr))
        f.write('\n')


if __name__ == "__main__":
    run_imgs()
