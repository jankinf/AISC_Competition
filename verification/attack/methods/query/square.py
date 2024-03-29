import torch
import time
import math
import random
import numpy as np
import os
import argparse
import cv2
from inference import RunnableInference, get_rect_from_ld, face_crop, resize_with_padding, i01c_to_ic01

class ScoreInference(RunnableInference):
    def run_image(self, img, rect=None):
        assert isinstance(img, (list, np.ndarray))

        if isinstance(img, np.ndarray):
            img = [img]
            rect = [rect]
        else:
            assert len(img) == len(rect)
        img_list = img; rect_list = rect

        input_batch = []
        for idx, img in enumerate(img_list):
            rect = rect_list[idx]
            if rect is None:
                ld = self.face_detector.detect_face(img)
                ld = np.array(ld)
                rect = get_rect_from_ld(ld)
            if rect is not None:
                img = face_crop(img, rect, hwratio=self.hwratio, alpha=3.)
            img = resize_with_padding(img, self.input_shape, resize_func=self.resize_func)

            input_batch.append(i01c_to_ic01(img))

        input_batch = np.array(input_batch, 'uint8')
        scores = self.pred_func(input_batch)[0]
        if len(scores) == 1:
            return scores[0], img
        return scores, img


class SquareAttack():
    """
    Square Attack
    https://arxiv.org/abs/1912.00049

    :param predict:       forward pass function
    :param norm:          Lp-norm of the attack ('Linf', 'L2' supported)
    :param n_restarts:    number of random restarts
    :param n_queries:     max number of queries (each restart)
    :param eps:           bound on the norm of perturbations
    :param seed:          random seed for the starting point
    :param p_init:        parameter to control size of squares
    :param loss:          loss function optimized ('margin', 'ce' supported)
    :param resc_schedule  adapt schedule of p to n_queries
    """

    def __init__(
            self,
            predict: ScoreInference,
            norm='Linf',
            n_queries=5000,
            eps=None,
            p_init=.8,
            n_restarts=1,
            seed=0,
            verbose=False,
            targeted=False,
            loss='margin',
            resc_schedule=True):
        
        self.predict = predict
        self.norm = norm
        self.n_queries = n_queries
        self.eps = eps * 1.
        self.p_init = p_init
        self.n_restarts = n_restarts
        self.seed = seed
        self.verbose = verbose
        self.targeted = targeted
        self.loss = loss
        self.rescale_schedule = resc_schedule
    

    def cross_entropy(self, scores, y):
        num_class = scores.shape[-1]
        log_p = np.log(scores)
        y_onehot = np.eye(num_class)[y]
        loss = - np.sum(y_onehot * log_p)
        return loss

    def margin_and_loss(self, x, y):
        """
        :param y:        correct labels if untargeted else target labels
        """
        # todo: marked by jankin, logits are not supported in megface, probabilities only: using probs in margin based loss instead.
        scores, _ = self.predict(x.astype(np.uint8))
        xent = self.cross_entropy(scores, y)
        y_corr = scores[y].copy()
        scores[y] = -float('inf')
        y_others = scores.max()

        if not self.targeted:
            if self.loss == 'ce':
                return y_corr - y_others, -1. * xent
            elif self.loss == 'margin':
                return y_corr - y_others, y_corr - y_others
        else:
            return y_others - y_corr, xent

    def init_hyperparam(self, x):
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

    def normalize(self, x):
        t = x.abs().reshape(x.shape[0], -1).max(1)[0]
        return x / (t.reshape(-1, *([1] * self.ndims)) + 1e-12)
    
    
    def p_selection(self, it):
        """ schedule to decrease the parameter p """

        if self.rescale_schedule:
            it = int(it / self.n_queries * 10000)

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

    def attack_single_run(self, x, y):
        h, w, c = x.shape
        n_ex_total = 1

        x_best = np.clip(x + self.eps * self.random_choice([h, w, c]), 0., 255.)
        margin_min, loss_min = self.margin_and_loss(x_best, y)
        n_queries = 1
        
        if (margin_min < 0.0).all():
            return n_queries, x_best
        
        for i_iter in range(self.n_queries):
            acc = (margin_min > 0.0)
            if acc:
                x_curr = x.copy()
                x_best_curr = x_best.copy()
                y_curr = y.copy()
                margin_min_curr = margin_min.copy()
                loss_min_curr = loss_min.copy()
            
                p = self.p_selection(i_iter)
                s = max(int(round(math.sqrt(p) * min(h, w))), 1)
                vh = self.random_int(0, h - s)
                vw = self.random_int(0, w - s)
                new_deltas = np.zeros([h, w, c])
                new_deltas[vh:vh + s, vw:vw + s, :
                    ] = 2. * self.eps * self.random_choice([1, 1, c])
            
                x_new = x_best_curr + new_deltas
                x_new = np.minimum(np.maximum(x_new, x_curr - self.eps), x_curr + self.eps)
                x_new = np.clip(x_new, 0., 255.)
            
                margin, loss = self.margin_and_loss(x_new, y_curr)

                # update loss if new loss is better
                idx_improved = (loss < loss_min_curr).astype(np.float32)

                loss_min = idx_improved * loss + (
                    1. - idx_improved) * loss_min_curr

                # update margin and x_best if new loss is better
                # or misclassification
                idx_miscl = (margin <= 0.).astype(np.float32)
                idx_improved = np.maximum(idx_improved, idx_miscl)

                margin_min = idx_improved * margin + (
                    1. - idx_improved) * margin_min_curr

                idx_improved = idx_improved.reshape([-1, *[1]*len(x.shape[:-1])])
                x_best = idx_improved * x_new + (
                    1. - idx_improved) * x_best_curr
                n_queries += 1.

            ind_succ = int(margin_min <= 0.)

            if self.verbose:
                print('{}'.format(i_iter + 1),
                    '- success rate={}/{} ({:.2%})'.format(
                    ind_succ, n_ex_total,
                    ind_succ / n_ex_total),
                    '- # queries={:.1f}'.format(n_queries),
                    '- loss={:.3f}'.format(loss_min))

            if ind_succ:
                break
    
        return n_queries, x_best

    def get_seed(self):
        if self.seed is None:
            self.seed = time.time()

    def perturb(self, x, y=None):
        """
        :param x:           clean images
        :param y:           untargeted attack -> clean labels,
                            if None we use the predicted labels
                            targeted attack -> target labels, if None random classes,
                            different from the predicted ones, are sampled
        """
        
        self.get_seed()
        self.init_hyperparam(x)

        adv = x.copy()
        if y is None:
            output, _ = self.predict(x)
            y_pred = output.argmax()
            y = y_pred.astype(np.int64)
        else:
            y = y.astype(np.int64)


        if not self.targeted:
            acc = self.predict(x)[0].argmax() == y
        else:
            acc = self.predict(x)[0].argmax() != y

        startt = time.time()

        np.random.seed(int(self.seed))
        random.seed(self.seed)

        assert self.n_restarts == 1, "currently support n_restarts=1 only!"
        for counter in range(self.n_restarts):
            if acc:
                x_to_fool = x.copy()
                y_to_fool = y.copy()

                n_queries, adv_curr = self.attack_single_run(x_to_fool, y_to_fool)
                output_curr, _ = self.predict(adv_curr.astype(np.uint8))
                if not self.targeted:
                    acc = output_curr.argmax() == y_to_fool
                else:
                    acc = output_curr.argmax() != y_to_fool

                adv = adv_curr.copy().astype(np.uint8)
                if self.verbose:
                    print('restart {} - robust accuracy: {:.2%}'.format(
                        counter, acc.astype(np.float32).mean()),
                        '- cum. time: {:.1f} s'.format(
                        time.time() - startt))
            else:
                n_queries = 1

        return n_queries, adv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--i_net", type=str, help="image network path")
    parser.add_argument("--image", type=str)
    parser.add_argument("--thres", type=float, default=0.5)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--max_epsilon", type=int, default=16)
    parser.add_argument("--targeted", action="store_true")
    parser.add_argument("--label", type=int, default=0, choices=[0, 1], help="indicates target label when targeted is True; otherwise gt label")
    parser.add_argument('--output_dir', default='./attack/adv_data', type=str, help='directory of crafted adversarial examples')
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--mode", type=str, choices=['attack', 'normal'])
    parser.add_argument("--speedup", action="store_true")
    parser.add_argument("--start_id", type=int, default=0)
    parser.add_argument("--cnt", type=int, default=200)
    args = parser.parse_args()

    model = ScoreInference(model_path=args.i_net, device="cpu0", alpha=3)
    if os.path.isfile(args.image):
        img_list = [args.image]
    else:
        img_list = [os.path.join(args.image, v) for v in os.listdir(args.image)]
    sorted(img_list)

    if args.max_images > 0:
        img_list = img_list[:args.max_images]

    eps = args.max_epsilon
    square = SquareAttack(model.run_image, p_init=.8, n_queries=5000, eps=eps, norm='Linf',
                    n_restarts=1, targeted=args.targeted, seed=None, verbose=True, resc_schedule=False)
    
    y = np.array(args.label)

    if args.targeted:
        # method = "square_t{}_restart2".format(str(args.label))
        method = "square_t{}".format(str(args.label))
    else:
        method = "square_nt"
    settings = f"iter5000_eps{args.max_epsilon}"
    output_dir = os.path.join(args.output_dir, method, settings, args.mode)
    logfile = os.path.join(args.log_dir, method, settings, "{}{}.log".format(args.mode, args.max_images))
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    fp = open(logfile, 'a')

    if args.speedup:
        img_list = img_list[args.start_id: args.start_id + args.cnt]
        
    for i, name in enumerate(img_list[:]):
        img_file = os.path.join(output_dir, os.path.basename(name))
        if os.path.exists(img_file):
            continue    
        
        try:
            img = cv2.imread(name)
            ori_score, ori_cimg = model.run_image(img)
            n_queries, adv_x = square.perturb(x=img, y=y)
            adv_score, adv_cimg = square.predict(adv_x)
            logdata = "id:{}, file:{}, ori_score: {}, adv_score: {}, n_queries: {}\n".format(i, name, ori_score, adv_score, n_queries)
            print(logdata, end='')
            fp.write(logdata)
            fp.flush()
            cv2.imwrite(img_file, adv_x)

        except Exception as e:
            print(e)
            continue
    fp.close()

if __name__ == "__main__":
    main()