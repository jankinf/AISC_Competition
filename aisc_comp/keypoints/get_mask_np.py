import os
import cv2
from PIL import Image
import numpy as np
import pickle

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

def get_bbox(kpoints, keyname):
    assert keyname in ['RIGHTEYE', 'LEFTEYE', 'NOSE', 'MOUTH']
    _name_points = {name: kpoints[idx] for idx, name in enumerate(LANDMARKS) if keyname in name}
    ld = np.array(list(_name_points.values()))
    y0, x0, y1, x1 = (
        ld[:, 0].min(), ld[:, 1].min(), ld[:, 0].max(), ld[:, 1].max()
    )
    y0 = max(0, round(y0))
    x0 = max(0, round(x0))
    y1 = min(112, round(y1))
    x1 = min(112, round(x1))
    return [x0, y0, x1, y1]

def run_imgs():
    # kpoint_file = "/data/projects/aisc_facecomp/keypoints/kpoints.pkl"
    # out_dir = "/data/projects/aisc_facecomp/keypoints/show_masks"
    kpoint_file = "/data/projects/aisc_facecomp/keypoints/kpoints_comp.pkl"
    # out_dir = "/data/projects/aisc_facecomp/keypoints/show_masks_comp_v2"
    # out_dir = "/data/projects/aisc_facecomp/keypoints/show_masks_comp_3p_2eye_1mouse"
    out_dir = "/data/projects/aisc_facecomp/keypoints/show_masks_comp_2p_2eye"
    # out_dir = "/data/projects/aisc_facecomp/keypoints/show_masks_comp_3p_2eye_1nose"
    # out_dir = "/data/projects/aisc_facecomp/keypoints/show_masks_comp_3p_2eye_1nose_nohole"
    os.makedirs(out_dir, exist_ok=True)
    # img_files = ["/data/projects/aisc_facecomp/data/{:04d}_compare.png".format(i + 1) for i in range(3000)]
    img_files = ["/data/projects/aisc_facecomp/data/{:04d}.png".format(i + 1) for i in range(3000)]
    with open(kpoint_file, "rb") as fp:
        pkl_data = pickle.load(fp)
    all_kpoints = pkl_data[1]

    # with open("debug.out", "a") as fp:
    # with open("debug_comp_debug.out", "a") as fp:
    # with open("debug_comp.out", "a") as fp:
    with open("debug_comp_3p.out", "w") as fp:
        for i, name in enumerate(img_files):
            # try:
            if True:
                kpoints = all_kpoints[i]
                out_file = os.path.join(out_dir, os.path.basename(name))
                if os.path.exists(out_file):
                    print("{} pass".format(out_file))
                    continue
                img = cv2.imread(name) # BGR
                rimg = img[:, :, ::-1] # BGR->RGB

                lds = []
                # for keyname in ['RIGHTEYE', 'LEFTEYE', 'NOSE', 'MOUTH']:
                # for keyname in ['RIGHTEYE', 'LEFTEYE', 'MOUTH']:
                for keyname in ['RIGHTEYE', 'LEFTEYE']:
                # for keyname in ['RIGHTEYE', 'LEFTEYE', 'NOSE']:
                    ld = get_bbox(kpoints, keyname)
                    lds.append(ld)
                lds = np.array(lds)
                idx = 0
                
                # calculate ratio
                mask = np.ones((112, 112, 3))
                for ld in lds:
                    mask[ld[0]:ld[2], ld[1]:ld[3], :] = np.zeros(3)
                ratio = (mask.size - mask.sum()) / mask.size

                n = lds.shape[0]
                while ratio > 0.10:
                    if np.random.rand() > 0.5:
                        if np.random.rand() > 0.5:
                            lds[idx][0] += 1
                        else:
                            lds[idx][2] -= 1
                    else:
                        if np.random.rand() > 0.5:
                            lds[idx][1] += 1
                        else:
                            lds[idx][3] -= 1
                    idx = (idx + 1) % n
                    
                    mask = np.ones((112, 112, 3))
                    for ld in lds:
                        mask[ld[0]:ld[2], ld[1]:ld[3], :] = np.zeros(3)
                    ratio = (mask.size - mask.sum()) / mask.size

                # import pdb; pdb.set_trace()
                while ratio < 0.10:
                    if np.random.rand() > 0.5:
                        if np.random.rand() > 0.5:
                            lds[idx][0] -= 1
                        else:
                            lds[idx][2] += 1
                    else:
                        if np.random.rand() > 0.5:
                            lds[idx][1] -= 1
                        else:
                            lds[idx][3] += 1
                    idx = (idx + 1) % n
                    
                    mask = np.ones((112, 112, 3))
                    for ld in lds:
                        mask[ld[0]:ld[2], ld[1]:ld[3], :] = np.zeros(3)
                    ratio = (mask.size - mask.sum()) / mask.size
                    
                # import pdb; pdb.set_trace()
                # todo: hole
                # if ratio > 0.10:
                #     mask = np.zeros((112, 112))
                #     for ld in lds:
                #         mask[ld[0]:ld[2], ld[1]:ld[3]] = 1

                #     flatten_mask = mask.reshape(-1)
                #     nz_idx = flatten_mask.nonzero()[0]
                #     cnt = mask.sum()
                #     while cnt > 1254:
                #         ridx = np.random.choice(nz_idx)
                #         xi = ridx // 112
                #         yi = ridx % 112
                #         mask[xi, yi] = 0
                #         cnt = mask.sum()

                # todo: no hole
                if ratio > 0.10:
                    mask = np.zeros((112, 112))
                    for ld in lds:
                        mask[ld[0]:ld[2], ld[1]:ld[3]] = 1

                    cnt = mask.sum()
                    while cnt > 1254:
                        idx_ld = np.random.choice(lds.shape[0])
                        bound_x = np.random.choice(4)
                        bound_y = (bound_x + 2) % 4

                        if lds[idx_ld][bound_x] > lds[idx_ld][bound_y]:
                            bound_x, bound_y = bound_y, bound_x

                        # import pdb; pdb.set_trace()
                        try:
                            if bound_x == 1:
                                ofs = min(lds[idx_ld][3] - lds[idx_ld][1], cnt - 1254)
                                ofs = int(ofs)
                                side = np.random.choice([0, 2])
                                mask[lds[idx_ld][side], lds[idx_ld][1]:lds[idx_ld][1]+ofs] = 0
                            else:
                                ofs = min(lds[idx_ld][2] - lds[idx_ld][0], cnt - 1254)
                                ofs = int(ofs)
                                side = np.random.choice([1, 3])
                                mask[lds[idx_ld][0]:lds[idx_ld][0]+ofs, lds[idx_ld][side]] = 0
                        except IndexError:
                            pass

                        cnt = mask.sum()

                if mask.sum() != 1254:
                    fp.write("bug mask size: {}, file: {}\n".format(mask.sum(), name))
                    continue
                else:
                    fp.write("mask size: {}, file: {}\n".format(mask.sum(), name))
                print(ratio, name)
                mask = np.expand_dims(mask, -1)
                assert mask.sum() == 1254
                mask = np.concatenate([mask] * 3, axis=2)
                mask = 1 - mask
                rimg = rimg * mask
                origin_img = Image.fromarray(rimg.astype(np.uint8))
                origin_img.save(out_file)

            # except Exception as e:
            #     print("bug: ", e, name)
            #     continue

if __name__ == "__main__":
    run_imgs()
