import os
import cv2
import inference
from PIL import Image, ImageDraw

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
    out_dir = "/data/projects/aisc_facecomp/keypoints/outs2"
    img_files = [
        "/data/projects/aisc_facecomp/data/0920.png",
        "/data/projects/aisc_facecomp/data/2958.png",
        "/data/projects/aisc_facecomp/data/1458.png",
    ]
    for i, name in enumerate(img_files):
        try:
            out_file = os.path.join(out_dir, os.path.basename(name))
            if os.path.exists(out_file):
                continue
            img = cv2.imread(name)
            kpoints = r.run_image(cv2.resize(img, (224, 224)))
            rimg = img[:, :, ::-1]
            kpoints = [(v1/2, v2/2) for (v1, v2) in kpoints]
            visualize_landmark(rimg, kpoints, out_file)
        except Exception as e:
            print("bug: ", e, name)
            continue

if __name__ == "__main__":
    run_imgs()
