import numpy as np
import argparse
import cv2
import os
import detect

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="./results")
    # parser.add_argument("--eval_dir", type=str, default="/data/projects/verification/face_recog/data/adv/veri_square/iter4000_eps3_thres60.0")
    parser.add_argument("--eval_dir", type=str, default="/data/projects/verification/face_recog/data/video_img")
    parser.add_argument("--exp_name", type=str)
    args = parser.parse_args()

    input_dir = args.eval_dir
    feat_extractor = detect.FaceFeaturer()
    img_list = sorted(os.listdir(input_dir), key=lambda x: int(x.split('.')[0]))
    clean_ref = cv2.imread("/data/projects/verification/face_recog/data/video_img/0.jpg").astype(np.uint8)
    feat1 = feat_extractor.get_feat(clean_ref)[0]

    logfile = os.path.join(args.log_dir, args.exp_name, "megface.log")
    os.makedirs(os.path.dirname(logfile), exist_ok=True)

    with open(logfile, 'a') as fp:        
        for idx in range(len(img_list)):        
            try:
                img_file = os.path.join(input_dir, img_list[idx])
                target_img = cv2.imread(img_file).astype(np.uint8)
                feat2 = feat_extractor.get_feat(target_img)[0]
                score = feat_extractor.compare_feat(feat1, feat2)
                logdata = "id:{}, file:{}, score: {}\n".format(idx, img_file, score)
                print(logdata, end='')
                fp.write(logdata)
                fp.flush()

            except Exception as e:
                print(e)
                continue

if __name__ == "__main__":
    os.system('sudo /usr/sbin/CodeMeterLin')
    os.system('cmu --list-server')
    main()