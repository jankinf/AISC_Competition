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


def infer(x, model:ScoreInference):
    x = x.astype(np.float32)
    
    adv = x.copy()
    import pdb; pdb.set_trace()
    noise = model.grad_func(**{'data':adv, 'label':np.array(0)})
    


def main():

    net_file = "/data/projects/fmp_demo/panorama.i_epoch_202622.th_0.8308.neupeak"
    model = ScoreInference(model_path=net_file, device="cpu0", alpha=3)

    img_path = "/data/projects/verification/face_recog/data/video_img/0.jpg"
    x = cv2.imread(img_path)
    x = x[None].transpose(0, 3, 1, 2)
    infer(x, model)

if __name__ == '__main__':
    
    main()