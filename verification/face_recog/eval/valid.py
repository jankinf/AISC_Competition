import numpy as np
import cv2
import os
import detect

def main():
    feat_extractor = detect.FaceFeaturer()

    img1 = cv2.imread("/data/projects/verification/face_recog/data/video_img/0.jpg").astype(np.float32)
    img2 = cv2.imread("/data/projects/verification/face_recog/data/video_img/1.jpg").astype(np.float32)
    another = cv2.imread("/data/projects/verification/face_recog/data/debug/3308203,1a5d0006f7799ed.jpg").astype(np.float32)
    feat1 = feat_extractor.get_feat(img1)[0]
    feat2 = feat_extractor.get_feat(img2)[0]
    feat3 = feat_extractor.get_feat(another)[0]

    score1 = feat_extractor.compare_feat(feat1, feat2)
    score2 = feat_extractor.compare_feat(feat1, feat3)
    score3 = feat_extractor.compare_feat(feat2, feat3)

    print(score1, score2, score3)

if __name__ == "__main__":
    os.system('sudo /usr/sbin/CodeMeterLin')
    os.system('cmu --list-server')
    main()