import logging
from attack.attack_dataset import get_dataloader, get_cropped_dataloader
from src.client import get_thresholdtable_from_fpr, get_tpr_from_threshold
from src.tsn_predict import TSNPredictor as CelebASpoofDetector
logging.basicConfig(level=logging.INFO)

def verify_output(scores, labels):
    fpr_list = [0.2, 0.1, 0.01, 0.005, 0.001]
    threshold_list = get_thresholdtable_from_fpr(scores, labels, fpr_list)
    tpr_list = get_tpr_from_threshold(scores, labels, threshold_list)

    # Show the result into score_path/score.txt
    logging.info('TPR@FPR=2E-1: {}\n'.format(tpr_list[0]))
    logging.info('TPR@FPR=1E-1: {}\n'.format(tpr_list[1]))
    logging.info('TPR@FPR=1E-2: {}\n'.format(tpr_list[2]))
    logging.info('TPR@FPR=5E-3: {}\n'.format(tpr_list[3]))
    logging.info('TPR@FPR=1E-3: {}\n'.format(tpr_list[4]))

    logging.info("Done")
    return tpr_list

def run_test(detector, image_iter):
    """
    In this function we create the detector instance. And evaluate the wall time for performing CelebASpoofDetector.
    """

    # run the images one-by-one and get runtime
    eval_cnt = 0

    logging.info("Starting runtime evaluation")
    scores, labels = [], []
    # import pdb; pdb.set_trace()
    for data in image_iter:
        # image_id, image
        image_id, image, label = data['filepath'], data['img'], data['label']
        image = image.cuda()
        score = detector.predict(image)[:, 1].tolist()
        scores += score
        labels += label.tolist()
        
        eval_cnt += len(image)

        if eval_cnt % 10 == 0:
            logging.info("Finished {} images".format(eval_cnt))

    logging.info("""
    ================================================================================
    All images finished, showing verification info below:
    ================================================================================
    """)

    # verify the algorithm output
    verify_output(scores, labels)


if __name__ == '__main__':
    '''
    python -m attack.eval
    '''
    celebA_spoof_image_iter = get_dataloader(batch_size=128, num_workers=4)
    # celebA_spoof_image_iter = get_cropped_dataloader(batch_size=128, num_workers=8)
    print("dataset contains {} samples".format(len(celebA_spoof_image_iter.dataset)))
    # initialize the detector
    logging.info("Initializing detector.")
    try:
        # todo: rand cropped model with right aug order
        # detector = CelebASpoofDetector(ckpt='/data/projects/CelebA-Spoof/ckpts/gpu8_bs1024_resnet18_sgd5e-3_wd5e-4_m9e-1_wo_cropped_jpeg_aug_corder_schedule15-30-45/14.pth.tar')
        # todo: wo cropped model
        # detector = CelebASpoofDetector(ckpt='/data/projects/CelebA-Spoof/ckpts/gpu8_bs1024_resnet18_sgd1e-2_wd2e-4_m9e-1_wo_cropped/32.pth.tar')
        # todo: rand cropped augmentation
        # detector = CelebASpoofDetector(ckpt='/data/projects/CelebA-Spoof/ckpts/gpu8_bs1024_resnet18_sgd5e-2_wd5e-4_m9e-1_wo_cropped_aug_schedule30-45/39.pth.tar')
        # todo: cropped model
        # detector = CelebASpoofDetector(ckpt='/data/projects/CelebA-Spoof/ckpts/gpu8_bs1024_resnet18_sgd1e-2_wd2e-4_m9e-1_cropped_eval/12.pth.tar')
        # todo: official model
        # detector = CelebASpoofDetector(ckpt='./ckpt_iter.pth.tar')
        detector = CelebASpoofDetector(ckpt='/data/projects/CelebA-Spoof/src/ckpt_iter.pth.tar')
    except:
        # send errors to the eval frontend
        raise
    logging.info("Detector initialized.")

    run_test(detector, celebA_spoof_image_iter)
