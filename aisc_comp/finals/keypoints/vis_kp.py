import os
import cv2
from PIL import Image
import numpy as np
import pickle
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-r1", "--kpoint_file", required=True)
ap.add_argument("-r2", "--img_dir", required=True)
ap.add_argument("-d", "--out_dir", required=True)
# args = vars(ap.parse_args())
args = ap.parse_args()

def run_imgs():
    # kpoint_file = "/data/projects/aisc_facecomp/finals/keypoints/kpoints.pkl"
    # kpoint_file = "/data/projects/aisc_facecomp/finals/keypoints/kpoints224.pkl"
    kpoint_file = args.kpoint_file
    # out_dir = "/data/projects/aisc_facecomp/finals/keypoints/show_kp"
    # out_dir = "/data/projects/aisc_facecomp/finals/keypoints/show_kp224"
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    # img_files = ["/data/projects/aisc_facecomp/data/{:04d}.png".format(i + 1) for i in range(3000)]
    # img_dir = "/data/projects/aisc_facecomp/finals/keypoints/face_aligned"
    # img_dir = "/data/projects/aisc_facecomp/finals/keypoints/face_aligned224"
    img_dir = args.img_dir
    img_files = [os.path.join(img_dir, path) for path in os.listdir(img_dir)]
    with open(kpoint_file, "rb") as fp:
        pkl_data = pickle.load(fp)
    LANDMARKS = pkl_data[0]
    all_kpoints = pkl_data[1]
    for i, name in enumerate(img_files):
        try:
            kpoints = all_kpoints[i]
            out_file = os.path.join(out_dir, os.path.basename(name))
            if os.path.exists(out_file):
                continue
            img = cv2.imread(name)
            rimg = img[:, :, ::-1]
            for y, x in kpoints: 
                x, y = round(x), round(y)
                # if x >= 112 or y >= 112:
                #     continue
                if x >= 224 or y >= 224:
                    continue
                rimg[x, y, :] = np.ones(3) * 255
            origin_img = Image.fromarray(rimg)
            origin_img.save(out_file)

        except Exception as e:
            print("bug: ", e, name)
            continue

if __name__ == "__main__":
    run_imgs()


    """
    img[112, 112, 3]

    uint8, 8 bits, 0-255, 0(black) 1(white)
    """