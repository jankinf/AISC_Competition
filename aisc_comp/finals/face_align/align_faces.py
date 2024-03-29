from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import argparse
import imutils
import dlib
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-r", "--root_dir", required=True, help="path to root directories of input images")
ap.add_argument("-d", "--des_dir", required=True, help="path to destination directories of output images")
args = vars(ap.parse_args())

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# fa = FaceAligner(predictor, desiredFaceWidth=112)
fa = FaceAligner(predictor, desiredFaceWidth=224)

root_dir = args["root_dir"]
des_dir = args["des_dir"]

os.makedirs(des_dir, exist_ok=True)

input_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(root_dir)) for f in fn]
for input_file in input_files:
    print(input_file)
    image = cv2.imread(input_file)
    image = imutils.resize(image, width=800)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    file_name = input_file.split("/")[-1]
    out_file = os.path.join(des_dir, file_name)

    rects = detector(gray, 2)
    for rect in rects:
        (x, y, w, h) = rect_to_bb(rect)
        x, y, w, h = int(x), int(y), int(w), int(h)
        # faceOrig = imutils.resize(image[y:y + h, x:x + w], width=112)
        faceAligned = fa.align(image, gray, rect)

        cv2.imwrite(out_file, faceAligned)
        cv2.waitKey(0)
        print(out_file)