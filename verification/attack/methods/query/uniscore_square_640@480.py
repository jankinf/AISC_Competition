import time
import math
import random
import numpy as np
import os
import argparse
import cv2
from typing import List
from inference import RunnableInference, get_rect_from_ld, resize_with_padding

class ScoreInference(RunnableInference):
    def run_image(self, img):
        input_batch = self.preprocess(img)
        scores = self.pred(input_batch)
        return scores

    def get_rect(self, img, hwratio=1., alpha=3.):
        assert isinstance(img, np.ndarray)
        ld = self.face_detector.detect_face(img)
        ld = np.array(ld)
        rect = get_rect_from_ld(ld)
        assert rect is not None

        SH, SW = img.shape[:2]
        x0, y0, x1, y1 = rect
        fh, fw = y1-y0, x1-x0
        xc, yc = (x0+x1) / 2, (y0+y1) / 2
        unit = (fh+fw) / 2

        th, tw = alpha*unit*hwratio, alpha*unit
        xs, xe = xc-(tw/2), xc+(tw/2)
        ys, ye = yc-(th/2), yc+(th/2)
        if (xs < 0) and (xe > SW):
            xs, xe = 0, SW
        elif xs < 0:
            xs, xe = 0, min(tw, SW)
        elif xe > SW:
            xs, xe = max(SW-tw, 0), SW
        if (ys < 0) and (ye > SH):
            ys, ye = 0, SH
        elif ys < 0:
            ys, ye = 0, min(th, SH)
        elif ye > SH:
            ys, ye = max(SH-th, 0), SH

        ys, ye, xs, xe = map(int, (ys, ye, xs, xe))
        # rimg = img[ys:ye, xs:xe, :]
        return np.array([ys, ye, xs, xe]).reshape(1, -1)

    def get_batch_rect(self, imgs):
        rects = []
        for img in imgs:
            rects.append(self.get_rect(img))
        out_rects = np.concatenate(rects, axis=0)
        return out_rects
        
    def crop_resize_batch_img(self, imgs: np.ndarray, rects: np.ndarray):
        """_summary_

        Args:
            imgs (np.ndarray): (n, h, w, 3) or (n, 3, h, w)
            rects (np.ndarray): (n, 4)

        Returns:
            input_batch: (n, 3, 224, 224)
        """
        assert imgs.shape[0] == rects.shape[0] and imgs.ndim == 4 and (imgs.shape[1] == 3 or imgs.shape[-1] == 3)
        if imgs.shape[1] == 3:
            imgs = imgs.transpose(0, 2, 3, 1)
        input_batch = []
        
        for img, rect in zip(imgs, rects):
            crop_img = img[rect[0]:rect[1], rect[2]:rect[3], :].astype(np.uint8)
            img = resize_with_padding(crop_img, self.input_shape, resize_func=self.resize_func)
            input_batch.append(img)
        input_batch = np.array(input_batch, 'uint8').transpose(0, 3, 1, 2)
        return input_batch


    def pred(self, input_batch):
        scores = self.pred_func(input_batch)[0]
        return scores


class UniscoreAttack():
    def __init__(
            self,
            model: ScoreInference,
            norm='Linf',
            n_queries=5000,
            eps=None,
            p_init=.8,
            n_restarts=1,
            seed=0,
            targeted=False,
            loss='margin',
            fp=None):

        self.model = model
        self.predict = model.pred
        self.norm = norm
        self.n_queries = n_queries
        self.eps = eps * 1.
        self.p_init = p_init
        self.n_restarts = n_restarts
        self.seed = seed
        self.targeted = targeted
        self.loss = loss
        self.fp = fp
        self.rects = None
        assert targeted is True

    def cross_entropy(self, scores, y):
        num_class = scores.shape[-1]
        log_p = np.log(scores)
        y_onehot = np.eye(num_class)[y]
        loss = - np.sum(y_onehot * log_p, axis=1)
        return loss

    def margin_and_loss(self, x, y):
        """
        :param x: shape (n, 3, h, w), default: h=640, w=480
        :param y: correct labels if untargeted else target labels
        """
        # todo: marked by jankin, logits are not supported in megface, probabilities only: using probs in margin based loss instead.
        # todo: processing data
        x_crop = self.model.crop_resize_batch_img(x, self.rects) # (n, 3, 224, 224)
        scores = self.predict(x_crop.astype(np.uint8))
        xent = self.cross_entropy(scores, y)
        y_corr = scores[:, y].copy()
        scores[:, y] = -float('inf')
        y_others = scores.max(axis=-1)

        if not self.targeted:
            if self.loss == 'ce':
                return y_corr - y_others, -1. * xent
            elif self.loss == 'margin':
                return y_corr - y_others, y_corr - y_others
        else:
            return y_others - y_corr, xent

    def p_selection(self, it):
        """ schedule to decrease the parameter p """
        if 10 < it <= 50:
            p = self.p_init / 2
        elif 50 < it <= 200:
            p = self.p_init / 4
        elif 200 < it <= 500:
            p = self.p_init / 8
        elif 500 < it <= 1000:
            p = self.p_init / 16
        elif 1000 < it <= 2000:
            p = self.p_init / 32
        elif 2000 < it <= 4000:
            p = self.p_init / 64
        elif 4000 < it <= 6000:
            p = self.p_init / 128
        elif 6000 < it <= 8000:
            p = self.p_init / 256
        elif 8000 < it:
            p = self.p_init / 512
        else:
            p = self.p_init

        return p

    def init_hyperparam(self):
        assert self.norm in ['Linf']
        assert not self.eps is None
        assert self.loss in ['ce', 'margin']

        if self.seed is None:
            self.seed = time.time()

    def random_choice(self, shape):
        t = 2 * np.random.rand(*shape) - 1
        return np.sign(t)

    def random_int(self, low=0, high=1, shape=[1]):
        t = low + (high - low) * np.random.rand(*shape)
        return t.astype(np.int64).item()

    def get_seed(self):
        if self.seed is None:
            self.seed = time.time()

    def train(self, img_list, y, resume=False):
        # todo: get data
        y = y.astype(np.int64)
        imgs = []
        for name in img_list:
            img = cv2.imread(name)
            imgs.append(img)
        img_batch = np.array(imgs).astype(np.uint8) # (n, h, w, 3)
        rect_path = "/data/projects/fmp_demo/attack/adv_data/640x480rects/train{}.npy".format(len(img_list))
        if not os.path.exists(rect_path):
            self.rects = self.model.get_batch_rect(img_batch) # (n, 4)
            np.save(rect_path, self.rects)
        else:
            self.rects = np.load(rect_path)
        
        img_batch = img_batch.transpose(0, 3, 1, 2)

        # todo: setting seed    
        self.get_seed()
        self.init_hyperparam()
        np.random.seed(int(self.seed))
        random.seed(self.seed)

        # todo: eval, get init success rate & tnt score
        n, c, h, w = img_batch.shape
        if resume and len(os.listdir(noise_saving)) > 0:
            # todo: load resume
            filenames = os.listdir(noise_saving)
            sorted(filenames, key=lambda x : float(x.split('@')[0][2:]))
            load_path = '{}/{}'.format(noise_saving, filenames[-1])
            print("loading: ", load_path)
            noise = np.load(load_path)
            n_queries = float(filenames[-1].split('@')[0][2:]) + 1
            x_best = img_batch + noise
        else:
            n_queries = 1
            x_best = img_batch.copy()

        
        margin_min, loss_min = self.margin_and_loss(x_best, y)

        # todo: start attack
        for i_iter in range(int(n_queries - 1), self.n_queries):
            if (margin_min < 0.0).all():
                break
            
            x_best_curr = x_best.copy()
            margin_min_curr = margin_min.copy()
            loss_min_curr = loss_min.copy()

            # todo: noise
            p = self.p_selection(i_iter)
            s = max(int(round(math.sqrt(p) * min(h, w))), 1)
            vh = self.random_int(0, h - s)
            vw = self.random_int(0, w - s)

            new_deltas = np.zeros([1, c, h, w])
            new_deltas[:, :, vh:vh + s, vw:vw + s] = 2. * self.eps * self.random_choice([1, c, 1, 1])

            x_new = x_best_curr + new_deltas
            x_new = np.minimum(np.maximum(x_new, img_batch - self.eps), img_batch + self.eps)
            x_new = np.clip(x_new, 0., 255.)

            margin, loss = self.margin_and_loss(x_new, y)

            if loss.sum() < loss_min_curr.sum():
                logdata = "better >>> nquery:{}, loss_new:{}, loss_cur:{}, suc_new:{}, suc_cur:{}\n".format(
                    n_queries, loss.sum().item(), loss_min_curr.sum().item(), (margin < 0.0).sum().item(), (margin_min_curr < 0.0).sum().item())

                noise = x_best[0].astype(np.float32) - img_batch[0].astype(np.float32)
                noise = noise[None, :]
                print(noise.min(), noise.max())
                np.save("{}/nq{}@suc{}.npy".format(noise_saving, n_queries, (margin < 0.0).sum().item()), noise)

                loss_min = loss
                margin_min = margin 
                x_best = x_new
            else:
                logdata = "not better >>> nquery:{}, loss_new:{}, loss_cur:{}, suc_new:{}, suc_cur:{}\n".format(
                    n_queries, loss.sum().item(), loss_min_curr.sum().item(), (margin < 0.0).sum().item(), (margin_min_curr < 0.0).sum().item())
            
            print(logdata, end='')
            self.fp.write(logdata)
            self.fp.flush()

            n_queries += 1.
        noise = x_best[0].astype(np.float32) - img_batch[0].astype(np.float32)
        noise = noise[None, :]
        return n_queries, noise


def get_img_list(path):
    if os.path.isfile(path):
        img_list = [path]
    else:
        img_list = [os.path.join(path, v) for v in os.listdir(path)]
    sorted(img_list)
    return img_list


def eval(uniscore, img_list, fp, mode='train'):
    assert mode in ['train', 'val']
    for i, name in enumerate(img_list):
        img_file = os.path.join(output_dir, os.path.basename(name))
        try:
            img = cv2.imread(name)
            ori_score, ori_cimg = uniscore.predict(img)
            # todo: input wo y
            n_queries, adv_x = uniscore.perturb(x=img, y=y)
            adv_score, adv_cimg = uniscore.predict(adv_x)
            logdata = "mode:{}, id:{}, file:{}, ori_score: {}, adv_score: {}\n".format(
                mode, i, name, ori_score, adv_score)
            print(logdata, end='')
            fp.write(logdata)
            fp.flush()
            cv2.imwrite(img_file, adv_x)

        except Exception as e:
            print(e)
            continue


def main():
    eps = args.max_epsilon
    n_queries = args.n_queries
    y = np.array(args.label)

    model = ScoreInference(model_path=args.i_net, device="cpu0", alpha=3)

    normal_list = get_img_list(normal_dir)
    attack_list = get_img_list(attack_dir)

    train_img_list = normal_list[:args.train_images // 2] + attack_list[:args.train_images // 2]
    val_img_list = normal_list[args.train_images // 2:] + attack_list[args.train_images // 2:]

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(noise_saving, exist_ok=True)
    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    fp = open(logfile, 'a')

    uniscore = UniscoreAttack(model, p_init=.8, n_queries=n_queries, eps=eps, norm='Linf',
                              n_restarts=1, targeted=True, seed=None, fp=fp)


    # todo: train pattern
    n_queries, noise = uniscore.train(img_list=train_img_list, y=y, resume=True)

    fp.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--i_net", type=str, help="image network path")
    parser.add_argument("--thres", type=float, default=0.5)
    parser.add_argument("--train_images", type=int, default=500)
    parser.add_argument("--max_epsilon", type=int, default=16)
    parser.add_argument("--n_queries", type=int, default=2000)
    parser.add_argument("--label", type=int, default=0, choices=[0, 1], help="indicates target label")
    parser.add_argument('--output_dir', default='./attack/adv_data', type=str,
                        help='directory of crafted adversarial examples')
    parser.add_argument("--log_dir", type=str, default="./logs")
    args = parser.parse_args()

    normal_dir = '/data/projects/fmp_demo/attack/adv_data/resize640x480/normal'
    attack_dir = '/data/projects/fmp_demo/attack/adv_data/resize640x480/attack'

    method = "{}_t{}".format(os.path.splitext(__file__.split('/')[-1])[0], str(args.label))
    settings = f"train{args.train_images}_iter{args.n_queries}_eps{args.max_epsilon}"
    output_dir = os.path.join(args.output_dir, method, settings)
    logfile = os.path.join(args.log_dir, method, settings, "train{}.log".format(args.train_images))
    noise_saving = os.path.join("/data/projects/fmp_demo/attack/uniscore_noise", method, settings)

    main()
