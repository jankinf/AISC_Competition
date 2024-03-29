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
ap.add_argument("-m", "--mask_dir", required=True)
ap.add_argument("-l", "--log_file", required=True)
# args = vars(ap.parse_args())
args = ap.parse_args()

LANDMARKS = [
    "CONTOUR_CHIN",
    "CONTOUR_LEFT1",
    "CONTOUR_LEFT2",
    "CONTOUR_LEFT3",
    "CONTOUR_LEFT4",
    "CONTOUR_LEFT5",
    "CONTOUR_LEFT6",
    "CONTOUR_LEFT7",
    "CONTOUR_LEFT8",
    "CONTOUR_LEFT9",
    "CONTOUR_RIGHT1",
    "CONTOUR_RIGHT2",
    "CONTOUR_RIGHT3",
    "CONTOUR_RIGHT4",
    "CONTOUR_RIGHT5",
    "CONTOUR_RIGHT6",
    "CONTOUR_RIGHT7",
    "CONTOUR_RIGHT8",
    "CONTOUR_RIGHT9",
    "LEFTEYE_BOTTOM",
    "LEFTEYE_LEFTCORNER",
    "LEFTEYE_LOWERLEFTQUARTER",
    "LEFTEYE_LOWERRIGHTQUARTER",
    "LEFTEYE_PUPIL",
    "LEFTEYE_RIGHTCORNER",
    "LEFTEYE_TOP",
    "LEFTEYE_UPPERLEFTQUARTER",
    "LEFTEYE_UPPERRIGHTQUARTER",
    "LEFTEYEBROW_LEFTCORNER",
    "LEFTEYEBROW_LOWERLEFTQUARTER",
    "LEFTEYEBROW_LOWERMIDDLE",
    "LEFTEYEBROW_LOWERRIGHTQUARTER",
    "LEFTEYEBROW_RIGHTCORNER",
    "LEFTEYEBROW_UPPERLEFTQUARTER",
    "LEFTEYEBROW_UPPERMIDDLE",
    "LEFTEYEBROW_UPPERRIGHTQUARTER",
    "MOUTH_LEFTCORNER",
    "MOUTH_LOWERLIPBOTTOM",
    "MOUTH_LOWERLIPLEFTCONTOUR1",
    "MOUTH_LOWERLIPLEFTCONTOUR2",
    "MOUTH_LOWERLIPLEFTCONTOUR3",
    "MOUTH_LOWERLIPRIGHTCONTOUR1",
    "MOUTH_LOWERLIPRIGHTCONTOUR2",
    "MOUTH_LOWERLIPRIGHTCONTOUR3",
    "MOUTH_LOWERLIPTOP",
    "MOUTH_RIGHTCORNER",
    "MOUTH_UPPERLIPBOTTOM",
    "MOUTH_UPPERLIPLEFTCONTOUR1",
    "MOUTH_UPPERLIPLEFTCONTOUR2",
    "MOUTH_UPPERLIPLEFTCONTOUR3",
    "MOUTH_UPPERLIPRIGHTCONTOUR1",
    "MOUTH_UPPERLIPRIGHTCONTOUR2",
    "MOUTH_UPPERLIPRIGHTCONTOUR3",
    "MOUTH_UPPERLIPTOP",
    "NOSE_CONTOURLEFT1",
    "NOSE_CONTOURLEFT2",
    "NOSE_CONTOURLEFT3",
    "NOSE_CONTOURLOWERMIDDLE",
    "NOSE_CONTOURRIGHT1",
    "NOSE_CONTOURRIGHT2",
    "NOSE_CONTOURRIGHT3",
    "NOSE_LEFT",
    "NOSE_RIGHT",
    "NOSE_TIP",
    "RIGHTEYE_BOTTOM",
    "RIGHTEYE_LEFTCORNER",
    "RIGHTEYE_LOWERLEFTQUARTER",
    "RIGHTEYE_LOWERRIGHTQUARTER",
    "RIGHTEYE_PUPIL",
    "RIGHTEYE_RIGHTCORNER",
    "RIGHTEYE_TOP",
    "RIGHTEYE_UPPERLEFTQUARTER",
    "RIGHTEYE_UPPERRIGHTQUARTER",
    "RIGHTEYEBROW_LEFTCORNER",
    "RIGHTEYEBROW_LOWERLEFTQUARTER",
    "RIGHTEYEBROW_LOWERMIDDLE",
    "RIGHTEYEBROW_LOWERRIGHTQUARTER",
    "RIGHTEYEBROW_RIGHTCORNER",
    "RIGHTEYEBROW_UPPERLEFTQUARTER",
    "RIGHTEYEBROW_UPPERMIDDLE",
    "RIGHTEYEBROW_UPPERRIGHTQUARTER",
]

def get_bbox(kpoints, keyname, margin=0):
    assert keyname in ['EYE', 'NOSE']
    _name_points = {name: kpoints[idx] for idx, name in enumerate(LANDMARKS) if keyname in name}
    ld = np.array(list(_name_points.values()))
    y0, x0, y1, x1 = (
        ld[:, 0].min(), ld[:, 1].min(), ld[:, 0].max(), ld[:, 1].max()
    )
    y0 = max(0, round(y0) - margin)
    x0 = max(0, round(x0) - margin)
    y1 = min(224, round(y1) + margin)
    x1 = min(224, round(x1) + margin)
    return [x0, y0, x1, y1]

def run_imgs():
    
    # kpoint_file = "/data/projects/aisc_facecomp/finals/keypoints/kpoints224.pkl"
    kpoint_file = args.kpoint_file
    # out_dir = "/data/projects/aisc_facecomp/finals/keypoints/show_tmasks224"
    out_dir = args.out_dir
    # mask_dir = "/data/projects/aisc_facecomp/finals/keypoints/tmasks224"
    mask_dir = args.mask_dir
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    # img_dir = "/data/projects/aisc_facecomp/finals/keypoints/face_aligned224"
    img_dir = args.img_dir
    img_files = [os.path.join(img_dir, path) for path in os.listdir(img_dir)]
    with open(kpoint_file, "rb") as fp:
        pkl_data = pickle.load(fp)
    all_kpoints = pkl_data[1]

    best_ratio = 0.13
    # with open("/data/projects/aisc_facecomp/finals/keypoints/generate_tmasks224.out", "a") as fp:

    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    with open(args.log_file, "a") as fp:
        for i, name in enumerate(img_files):
            try:
                kpoints = all_kpoints[i]
                out_file = os.path.join(out_dir, os.path.basename(name))
                mask_file = os.path.join(mask_dir, os.path.basename(name))
                # if os.path.exists(out_file):
                #     continue
                img = cv2.imread(name) # BGR
                rimg = img[:, :, ::-1] # BGR->RGB

                margin = 0
                while True:
                    lds = []
                    for keyname in ['EYE', 'NOSE']:
                        ld = get_bbox(kpoints, keyname, margin=margin)
                        lds.append(ld)
                    lds = np.array(lds)
                    
                    # calculate ratio
                    mask = np.ones((224, 224, 3))
                    for ld in lds:
                        mask[ld[0]:ld[2], ld[1]:ld[3], :] = np.zeros(3)
                    ratio = (mask.size - mask.sum()) / mask.size
                    if ratio > best_ratio:
                        margin -= 1
                    else:
                        break

                while True:
                    lds = []
                    for keyname in ['EYE', 'NOSE']:
                        ld = get_bbox(kpoints, keyname, margin=margin)
                        lds.append(ld)
                    lds = np.array(lds)
                    
                    # calculate ratio
                    mask = np.ones((224, 224, 3))
                    for ld in lds:
                        mask[ld[0]:ld[2], ld[1]:ld[3], :] = np.zeros(3)
                    ratio = (mask.size - mask.sum()) / mask.size
                    if ratio < best_ratio:
                        margin += 1
                    else:
                        break

                print(ratio, name)
                rimg = rimg * mask
                origin_img = Image.fromarray(rimg.astype(np.uint8))
                origin_img.save(out_file)

                Image.fromarray((mask[:, :, 0] * 255).astype(np.uint8)).save(mask_file)

            except Exception as e:
                print("bug: ", e, name)
                continue

if __name__ == "__main__":
    run_imgs()
