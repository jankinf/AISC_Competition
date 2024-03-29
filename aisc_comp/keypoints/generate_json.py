import json
import os
import cv2
from PIL import Image
import numpy as np
from base64 import b64encode, b64decode
import io
import pickle


LANDMARKS_FINEGRAINED = {
    "NOSE": [
        "NOSE_CONTOURLOWERMIDDLE",
        "NOSE_CONTOURLEFT3",
        "NOSE_LEFT",
        "NOSE_CONTOURLEFT2",
        "NOSE_CONTOURLEFT1",
        "LEFTEYE_RIGHTCORNER",
        "LEFTEYEBROW_RIGHTCORNER",
        "RIGHTEYEBROW_LEFTCORNER",
        "RIGHTEYE_LEFTCORNER",
        "NOSE_CONTOURRIGHT1",
        "NOSE_CONTOURRIGHT2",
        "NOSE_RIGHT",
        "NOSE_CONTOURRIGHT3",
    ],
    "LEFTEYE": [
        "LEFTEYE_LEFTCORNER",
        "LEFTEYEBROW_LEFTCORNER",
        "LEFTEYEBROW_UPPERLEFTQUARTER",
        "LEFTEYEBROW_UPPERMIDDLE",
        "LEFTEYEBROW_UPPERRIGHTQUARTER",
        "LEFTEYEBROW_RIGHTCORNER",
        "LEFTEYE_RIGHTCORNER",
        "LEFTEYE_LOWERRIGHTQUARTER",
        "LEFTEYE_BOTTOM",
        "LEFTEYE_LOWERLEFTQUARTER",
    ],
    "RIGHTEYE": [
        "RIGHTEYE_LEFTCORNER",
        "RIGHTEYEBROW_LEFTCORNER",
        "RIGHTEYEBROW_UPPERLEFTQUARTER",
        "RIGHTEYEBROW_UPPERMIDDLE",
        "RIGHTEYEBROW_UPPERRIGHTQUARTER",
        "RIGHTEYEBROW_RIGHTCORNER",
        "RIGHTEYE_RIGHTCORNER",
        "RIGHTEYE_LOWERRIGHTQUARTER",
        "RIGHTEYE_BOTTOM",
        "RIGHTEYE_LOWERLEFTQUARTER",
    ],
    "MOUTH": [
        "MOUTH_LEFTCORNER",
        "MOUTH_UPPERLIPLEFTCONTOUR2",
        "MOUTH_UPPERLIPTOP",
        "MOUTH_UPPERLIPRIGHTCONTOUR2",
        "MOUTH_RIGHTCORNER",
        "MOUTH_LOWERLIPRIGHTCONTOUR2",
        "MOUTH_LOWERLIPBOTTOM",
        "MOUTH_LOWERLIPLEFTCONTOUR2",
    ]
}

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

name2idx = {name: idx for idx, name in enumerate(LANDMARKS)}
# name_mapping = {
#     'RIGHTEYE': 'mask', 
#     'LEFTEYE': 'mask',  
#     'NOSE': 'mask', 
#     'MOUTH': 'mask',
# }
name_mapping = {
    'RIGHTEYE': 'right_eye', 
    'LEFTEYE': 'left_eye',  
    'NOSE': 'nose', 
    'MOUTH': 'mouth',
}

def get_bbox_v2(kpoints, keyname):
    assert keyname in ['RIGHTEYE', 'LEFTEYE', 'NOSE', 'MOUTH']
    _name_points = []
    for point in LANDMARKS_FINEGRAINED[keyname]:
        idx = name2idx[point]
        _name_points.append(kpoints[idx])
    ld = np.array(_name_points)
    if 'EYE' in keyname:
        x_min = np.min(ld[:, 0])
        x_max = np.max(ld[:, 0])
        y_min = np.min(ld[:, 1])
        y_max = np.max(ld[:, 1])
        ld = np.array([
            [x_min, y_min],
            [x_min, y_max],
            [x_max, y_max],
            [x_max, y_min],
        ])
    ld = np.maximum(0, ld)
    ld = np.minimum(112, ld)
    ld = [[ld_[0], ld_[1]] for ld_ in ld]

    return ld

def get_bbox(kpoints, keyname):
    assert keyname in ['RIGHTEYE', 'LEFTEYE', 'NOSE', 'MOUTH']
    _name_points = []
    for point in LANDMARKS_FINEGRAINED[keyname]:
        idx = name2idx[point]
        _name_points.append(kpoints[idx])
    # print(LANDMARKS_FINEGRAINED[keyname])
    ld = np.array(_name_points)
    ld = np.maximum(0, ld)
    ld = np.minimum(112, ld)
    ld = [[ld_[0], ld_[1]] for ld_ in ld]

    return ld

class BytesEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, bytes):
            return obj.decode('utf-8')
        return json.JSONEncoder.default(self, obj)

def run_imgs():
    # kpoint_file = "/data/projects/aisc_facecomp/keypoints/kpoints.pkl"
    kpoint_file = "/data/projects/aisc_facecomp/keypoints/kpoints_comp.pkl"
    
    with open(kpoint_file, "rb") as fp:
        pkl_data = pickle.load(fp)
    all_kpoints = pkl_data[1]

    # img_files = ["/data/projects/aisc_facecomp/data/{:04d}_compare.png".format(i + 1) for i in range(3000)]
    img_files = ["/data/projects/aisc_facecomp/data/{:04d}.png".format(i + 1) for i in range(3000)]
    output_dir = "/data/projects/aisc_facecomp/keypoints/comp_json_v3"
    # output_dir = "/data/projects/aisc_facecomp/keypoints/comp_json_v2"
    # output_dir = "/data/projects/aisc_facecomp/keypoints/comp_json"
    # output_dir = "/data/projects/aisc_facecomp/keypoints/origin_json"
    os.makedirs(output_dir, exist_ok=True)
    with open("/data/projects/aisc_facecomp/keypoints/mask.json", "r") as jfp:
        json_data = json.load(jfp)
    # with open("debug_comp_json.out", "w") as fp:
    with open("debug_json.out", "w") as fp:
        for i, name in enumerate(img_files):
            kpoints = all_kpoints[i]
            out_file = os.path.basename(name)
            if os.path.exists(out_file):
                print("{} pass".format(out_file))
                continue
            
            im = Image.open(name)
            buffered = io.BytesIO()
            im.save(buffered, format="PNG")
            img_str = b64encode(buffered.getvalue())

            im = Image.open(io.BytesIO(b64decode(json_data['imageData'])))
            shapes = []
            for keyname in ['RIGHTEYE', 'LEFTEYE', 'NOSE', 'MOUTH']:
                items = {"label": name_mapping[keyname]}
                # ld = get_bbox(kpoints, keyname)
                ld = get_bbox_v2(kpoints, keyname)
                items["points"] = ld
                items["shape_type"] = "polygon"
                shapes.append(items)
            
            json_data["shapes"] = shapes
            json_data["imagePath"] = out_file
            json_data["imageData"] = img_str
            json_outfile = os.path.join(output_dir, "{}.json".format(os.path.splitext(out_file)[0]))
            print(json_outfile)
            with open(json_outfile, "w") as lfp:
                json_str = json.dumps(json_data, cls=BytesEncoder)
                lfp.write(json_str)

if __name__ == "__main__":
    run_imgs()
