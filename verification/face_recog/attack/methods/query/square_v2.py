import numpy as np
import argparse
import random
import detect
import time
import math
import cv2
import os
'''
这个版本的更新策略不同于square
每一次query都更新样本: 看下能否提速
'''

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
    :param resc_schedule  adapt schedule of p to n_queries
    """

    def __init__(
            self,
            detector: detect.FaceFeaturer,
            norm='Linf',
            n_queries=5000,
            eps=None,
            p_init=.8,
            n_restarts=1,
            seed=0,
            verbose=False,
            threshold=60,
            resc_schedule=True):
        
        self.detector = detector
        self.norm = norm
        self.n_queries = n_queries
        self.eps = eps * 1.
        self.p_init = p_init
        self.n_restarts = n_restarts
        self.seed = seed
        self.verbose = verbose
        self.threshold = threshold
        self.rescale_schedule = resc_schedule

    def get_score(self, x1, x2):
        score = self.detector.compare_feat(x1, x2)
        return score

    def init_hyperparam(self):
        assert self.norm in ['Linf']
        assert not self.eps is None

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

    def attack_single_run(self, anchor, target_img, fp=None):
        h, w, c = target_img.shape

        x = target_img.copy()
        x_best = np.clip(x + self.eps * self.random_choice([h, w, c]), 0., 255.)
        # x_best = x.copy()
        target_feat = self.detector.get_feat(x_best)[0]
        score = self.get_score(anchor, target_feat)
        n_queries = 1
        if self.verbose:
            logdata = '{} - # queries={:.1f} - score={:.3f}\n'. \
                format(0, n_queries, score)
            print(logdata, end='')
            if fp:
                fp.write(logdata)
                fp.flush()

        if score < self.threshold:
            return n_queries, x_best
        
        for i_iter in range(self.n_queries):
            x_best_curr = x_best.copy()
        
            p = self.p_selection(i_iter)
            s = max(int(round(math.sqrt(p) * min(h, w))), 1)
            vh = self.random_int(0, h - s)
            vw = self.random_int(0, w - s)
            new_deltas = np.zeros([h, w, c])
            new_deltas[vh:vh + s, vw:vw + s, :
                ] = 2. * self.eps * self.random_choice([1, 1, c])
        
            x_new = x_best_curr + new_deltas
            x_new = np.minimum(np.maximum(x_new, x - self.eps), x + self.eps)
            x_new = np.clip(x_new, 0., 255.)

            target_feat = self.detector.get_feat(x_new)[0]
            score = self.get_score(anchor, target_feat)

            x_best = x_new

            if self.verbose:
                logdata = '{} - # queries={:.1f} - score={:.3f}\n'. \
                    format(i_iter + 1, n_queries, score)
                print(logdata, end='')
                if fp:
                    fp.write(logdata)
                    fp.flush()

            n_queries += 1.
            if score < self.threshold:
                break
    
        return n_queries, score, x_best

    def get_seed(self):
        if self.seed is None:
            self.seed = time.time()

    def perturb(self, x1, x2, fp=None):
        self.get_seed()
        self.init_hyperparam()

        adv = x2.copy()
        anchor = self.detector.get_feat(x1)[0]

        startt = time.time()

        np.random.seed(int(self.seed))
        random.seed(self.seed)

        assert self.n_restarts == 1, "currently support n_restarts=1 only!"
        for counter in range(self.n_restarts):
            x_to_fool = adv.copy()
            n_queries, score, adv_curr = self.attack_single_run(anchor, x_to_fool, fp)
            
            adv = adv_curr.copy().astype(np.float32)
            if self.verbose:
                logdata = 'restart {} - score: {:.2%} - cum. time: {:.1f} s\n'.format(counter, score, time.time() - startt)
                print(logdata, end='')
                if fp:
                    fp.write(logdata)
                    fp.flush()

        return n_queries, score, adv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--thres", type=float, default=60)
    parser.add_argument("--max_epsilon", type=int, default=4)
    parser.add_argument("--iter", type=int, default=5000)
    parser.add_argument('--output_dir', default='./data/adv', type=str, help='directory of crafted adversarial examples')
    parser.add_argument("--log_dir", type=str, default="./logs")
    args = parser.parse_args()


    feat_extractor = detect.FaceFeaturer()
    input_dir = "./data/video_img"
    img_list = sorted(os.listdir(input_dir), key=lambda x: int(x.split('.')[0]))
    square = SquareAttack(feat_extractor, p_init=.8, n_queries=args.iter, eps=args.max_epsilon, norm='Linf',
                    n_restarts=1, seed=None, verbose=True, threshold=args.thres, resc_schedule=False)
    method = "veri_square_v2"
    settings = f"iter{args.iter}_eps{args.max_epsilon}_thres{args.thres}"
    output_dir = os.path.join(args.output_dir, method, settings)
    logfile = os.path.join(args.log_dir, method, settings, "out.log")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    fp = open(logfile, 'a')
        
    reference_img = cv2.imread(os.path.join(input_dir, img_list[0])).astype(np.float32)

    for idx in range(1, len(img_list)):

        img_file = os.path.join(output_dir, img_list[idx])
        if os.path.exists(img_file):
            continue    
        
        try:
            target_img = cv2.imread(os.path.join(input_dir, img_list[idx])).astype(np.float32)
            n_queries, score, adv_x = square.perturb(x1=reference_img, x2=target_img, fp=fp)
            out_eps = np.abs(adv_x.astype(np.float32) - target_img.astype(np.float32)).max()
            logdata = "id:{}, file:{}, score: {}, n_queries: {}, eps_check: {}\n".format(idx, img_list[idx], score, n_queries, out_eps)
            print(logdata, end='')
            fp.write(logdata)
            fp.flush()
            cv2.imwrite(img_file, adv_x.astype(np.uint8))

        except Exception as e:
            print(e)
            continue
    fp.close()

if __name__ == "__main__":
    os.system('sudo /usr/sbin/CodeMeterLin')
    os.system('cmu --list-server')
    main()