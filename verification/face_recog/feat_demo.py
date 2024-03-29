import detect
import cv2
import os
def main():
    os.system('sudo /usr/sbin/CodeMeterLin')
    feat_extractor = detect.FaceFeaturer()
    img1 = cv2.imread("/data/projects/verification/face_recog/data/video_img/0.jpg")
    feat1 = feat_extractor.get_feat(img1)[0]
    print(feat1)
    # img2 = cv2.imread("/data/projects/verification/face_recog/data/video_img/1.jpg")
    # feat2 = feat_extractor.get_feat(img2)[0]
    # out = feat_extractor.compare_feat(feat1, feat2)
    # print(out)



main()