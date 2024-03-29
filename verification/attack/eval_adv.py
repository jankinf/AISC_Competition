import os
import logging
import argparse
from torch.utils.data import Dataset, DataLoader
from attack.attack_dataset import get_dataloader
from src.client import get_thresholdtable_from_fpr, get_tpr_from_threshold
from src.tsn_predict import TSNPredictor as CelebASpoofDetector
logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description='eval perturbed dataset')
parser.add_argument('--batch_size', type=int, default=1, help='mini-batch size (default: 1)')
parser.add_argument('--workers', default=2, type=int, help='number of data loading workers (default: 2)')
parser.add_argument('--log_file', type=str, help='dir of logs')
parser.add_argument('--img_dir', type=str, help='dir of logs')
parser.add_argument('--ckpt', type=str, help='path of ckpt')
args = parser.parse_args()

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
    return verify_output(scores, labels)


if __name__ == '__main__':
    '''
    python -m attack.eval
    '''
    img_iter = get_dataloader(batch_size=1, num_workers=2, dir_path=args.img_dir)
    print("dataset contains {} samples".format(len(img_iter.dataset)))
    logging.info("Initializing detector.")
    try:
        detector = CelebASpoofDetector(ckpt=args.ckpt)
    except:
        raise
    logging.info("Detector initialized.")
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    tpr_list = run_test(detector, img_iter)
    with open(args.log_file, 'a') as f:
        f.write(f'params: {str(args)}\n')
        f.write(f'eval {args.img_dir} on {args.ckpt}\n')
        f.write('TPR@FPR=2E-1: {}\n'.format(tpr_list[0]))
        f.write('TPR@FPR=1E-1: {}\n'.format(tpr_list[1]))
        f.write('TPR@FPR=1E-2: {}\n'.format(tpr_list[2]))
        f.write('TPR@FPR=5E-3: {}\n'.format(tpr_list[3]))
        f.write('TPR@FPR=1E-3: {}\n\n'.format(tpr_list[4]))
    print('eval finished')
