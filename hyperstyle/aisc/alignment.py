import os
import dlib
import argparse
from scripts.align_faces_parallel import align_face

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', default="/data/projects/aisc_facecomp/finals/keypoints/aisc_self", type=str)
parser.add_argument('--output_dir', default="/data/projects/hyperstyle/aisc/aligned", type=str)
args = parser.parse_args()
print(args)

def run_alignment(image_path):
    if not os.path.exists("shape_predictor_68_face_landmarks.dat"):
        print('Downloading files for aligning face image...')
        os.system('wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2')
        os.system('bzip2 -dk shape_predictor_68_face_landmarks.dat.bz2')
        print('Done.')
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    aligned_image = align_face(filepath=image_path, predictor=predictor)
    print(f"Finished running alignment on image: {image_path}")
    return aligned_image

paths = [os.path.join(args.input_dir, path) for path in os.listdir(args.input_dir)]

os.makedirs(args.output_dir, exist_ok=True)
for path in paths:
    name = os.path.basename(path)
    out_file = os.path.join(args.output_dir, name)
    image = run_alignment(path)
    image.save(out_file)
