import cv2
import os
video_dir = "/data/projects/aisc_facecomp/finals/data/AISC_face"
paths = [
    "受害者1.mp4",
    "受害者2.mp4",
    "受害者3.mp4",
    "攻击者1.mp4",
    "攻击者2.mp4",
    "攻击者3.mp4",
]
for path in paths:
    name = os.path.splitext(path)[0]
    vidcap = cv2.VideoCapture(os.path.join(video_dir, path))
    success, image = vidcap.read()
    count = 0
    out_dir = "/data/projects/aisc_facecomp/finals/data/images/{}".format(name)
    os.makedirs(out_dir, exist_ok=True)
    while success:
        cv2.imwrite("{}/{}.png".format(out_dir, count), image)
        success, image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1