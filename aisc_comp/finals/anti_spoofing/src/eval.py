"""
This script provides a local test routine so you can verify the algorithm works before pushing it to evaluation.

It runs your detector on several local images and verify whether they have obvious issues, e.g:
    - Fail to start
    - Wrong output format

It also prints out the runtime for the algorithms for your references.


The participants are expected to implement a face forgery detector class. The sample detector illustrates the interface.
Do not modify other part of the evaluation toolkit otherwise the evaluation will fail.

Author: Yuanjun Xiong, Zhengkui Guo, Yuanhan Zhang
Contact: celebaspoof@gmail.com

CelebA-Spoof 
"""

import logging
from src.client import get_image, verify_output
# from src.dataset import get_test_dataloader, jpeg_transform, png_transform
from src.dataset import get_test_dataloader
from src.tsn_predict import TSNPredictor as CelebASpoofDetector
logging.basicConfig(level=logging.INFO)

def run_test(detector, image_iter):
    """
    In this function we create the detector instance. And evaluate the wall time for performing CelebASpoofDetector.
    """

    # run the images one-by-one and get runtime
    output_probs = {}
    eval_cnt = 0

    logging.info("Starting runtime evaluation")
    
    for data in image_iter:
        # image_id, image
        image_id, image, label = data['filepath'], data['img'], data['label']
        image, label = image.cuda(), label.cuda()
        try:
            prob = detector.predict(image)
            for idx,i in enumerate(image_id):
                output_probs[i] = float(prob[idx][1])
        except:
            # send errors to the eval frontend
            logging.error("Image id failed: {}".format(image_id))
            raise


        eval_cnt += len(image)

        if eval_cnt % 10 == 0:
            logging.info("Finished {} images".format(eval_cnt))

    logging.info("""
    ================================================================================
    All images finished, showing verification info below:
    ================================================================================
    """)

    # verify the algorithm output
    verify_output(output_probs)


if __name__ == '__main__':
    # GLOBAL_IMAGE_LIST_PATH = '/data/datasets/celeba_spoof/metas/intra_test/test_label.json'
    # import json
    # with open(GLOBAL_IMAGE_LIST_PATH, 'r') as f:
    #     gts = json.load(f)
    # import pdb; pdb.set_trace()

    # todo: main entry
    # celebA_spoof_image_iter = get_image()
    celebA_spoof_image_iter = get_test_dataloader(batch_size=256, num_workers=8, jpeg=True, cropped=True)
    # celebA_spoof_image_iter = get_test_dataloader(batch_size=256, num_workers=8, jpeg=False, cropped=True)
    # celebA_spoof_image_iter = get_test_dataloader(batch_size=256, num_workers=8, jpeg=True, cropped=False)
    # celebA_spoof_image_iter = get_test_dataloader(batch_size=128, num_workers=8, jpeg=False, cropped=False)

    # celebA_spoof_image_iter = get_test_dataloader(batch_size=128, num_workers=8, cropped=False)
    # celebA_spoof_image_iter = get_test_dataloader(transform=png_transform, batch_size=128, num_workers=8, cropped=True)
    # celebA_spoof_image_iter = get_test_dataloader(transform=jpeg_transform, batch_size=128, num_workers=8, cropped=True)
    # celebA_spoof_image_iter = get_test_dataloader(transform=jpeg_transform, batch_size=128, num_workers=8, cropped=False)
    
    # initialize the detector
    logging.info("Initializing detector.")
    try:
        detector = CelebASpoofDetector(ckpt='/data/projects/CelebA-Spoof/ckpts/gpu8_bs1024_resnet18_sgd5e-3_wd5e-4_m9e-1_wo_cropped_jpeg_aug_corder_schedule15-30-45/14.pth.tar')
        # detector = CelebASpoofDetector(ckpt='/data/projects/CelebA-Spoof/ckpts/gpu8_bs1024_resnet18_sgd5e-2_wd5e-4_m9e-1_wo_cropped_aug_schedule30-45/39.pth.tar')
        # detector = CelebASpoofDetector(ckpt='/data/projects/CelebA-Spoof/ckpts/gpu8_bs1024_resnet18_sgd1e-2_wd2e-4_m9e-1_cropped_eval/12.pth.tar')
        # detector = CelebASpoofDetector(ckpt='/data/projects/CelebA-Spoof/ckpts/gpu8_bs1024_resnet18_sgd1e-2_wd2e-4_m9e-1_cropped/39.pth.tar')
        # detector = CelebASpoofDetector(ckpt='/data/projects/CelebA-Spoof/ckpts/gpu8_bs1024_resnet18_sgd1e-2_wd2e-4_m9e-1_cropped/40.pth.tar')
        # detector = CelebASpoofDetector(ckpt='/data/projects/CelebA-Spoof/src/ckpt_iter.pth.tar')
    except:
        # send errors to the eval frontend
        raise
    logging.info("Detector initialized.")

    run_test(detector, celebA_spoof_image_iter)
