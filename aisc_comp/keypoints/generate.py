import os
import cv2
import inference
from PIL import Image, ImageDraw
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

def visualize_landmark(image_array, landmarks, file):
    """ plot landmarks on image
    :param image_array: numpy array of a single image
    :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
    :return: plots of images with landmarks on
    """
    origin_img = Image.fromarray(image_array)
    draw = ImageDraw.Draw(origin_img)
    for facial_feature in landmarks:
        draw.point(facial_feature)
    origin_img.save(file)

def run_imgs():
    r = inference.RunnableInference()
    img_dir = "/data/projects/aisc_facecomp/data"
    # img_files = ["/data/projects/aisc_facecomp/data/{:04d}.png".format(i + 1) for i in range(3000)]
    img_files = ["/data/projects/aisc_facecomp/data/{:04d}_compare.png".format(i + 1) for i in range(3000)]
    all_kpoints = []
    for i, name in enumerate(img_files):
        try:
            img = cv2.imread(name)
            kpoints = r.run_image(img)
            if not kpoints:
                kpoints = r.run_image(cv2.resize(img, (224, 224)))
                kpoints = [(v1/2, v2/2) for (v1, v2) in kpoints]
        except Exception as e:
            print("bug: ", e, name)
        all_kpoints.append(kpoints)
    all_res = [LANDMARKS, all_kpoints]
    # outfile = "/data/projects/aisc_facecomp/keypoints/kpoints.pkl"
    outfile = "/data/projects/aisc_facecomp/keypoints/kpoints_comp.pkl"
    with open(outfile, "wb") as fp:
        pickle.dump(all_res, fp)
    
    # with open(outfile, "rb") as fp:
    #     load_data = pickle.load(fp)
    # print(load_data)
    # print(type(load_data))
    # import pdb; pdb.set_trace()


if __name__ == "__main__":
    run_imgs()
