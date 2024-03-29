import numpy as np
import argparse
import cv2
import os
import detect
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", type=str)
    args = parser.parse_args()

    comp_dir = "/data/projects/aisc_facecomp/data"
    comp_files = [os.path.join(comp_dir, "{:04d}_compare.png".format(i + 1)) for i in range(3000)]
    adv_files = [os.path.join(args.eval_dir, "{:04d}.png".format(i + 1)) for i in range(3000)]
    
    feat_extractor = detect.FaceFeaturer()

    scores = []
    cnt = 0

    # adv_files, comp_files = adv_files[:2], comp_files[:2]
    # import pdb; pdb.set_trace()
    basename = os.path.basename(args.eval_dir)
    if len(os.listdir(args.eval_dir)) < 3000: 
        exit()
    with open("/data/projects/verification/aisc_csv/{}.csv".format(basename), 'w') as fp:
        fp.write("scores,dir_name\n")
        for file1, file2 in zip(adv_files, comp_files):    
            img1 = cv2.imread(file1).astype(np.float32)
            img2 = cv2.imread(file2).astype(np.float32)
            feat1 = feat_extractor.get_feat(img1)[0]
            feat2 = feat_extractor.get_feat(img2)[0]
            score = feat_extractor.compare_feat(feat1, feat2)
            scores.append(score)
            cnt += 1
            print(cnt)
            fp.write("{},{}\n".format(score, file1))

    # scores = np.array(scores)
    # data = {
    #     "scores": scores,
    #     "dir_name": adv_files,
    # }
    # df = pd.DataFrame.from_dict(data)
    # df.to_csv("/data/projects/verification/aisc_csv/{}.csv".format(basename), index=False)

if __name__ == "__main__":
    os.system('sudo /usr/sbin/CodeMeterLin')
    os.system('cmu --list-server')
    main()