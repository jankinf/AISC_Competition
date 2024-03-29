# pip install imutils
# pip install dlib
# pip install opencv-python

# python align_faces.py \
# -r "/data/projects/aisc_facecomp/finals/keypoints/aisc_self" \
# -d "/data/projects/aisc_facecomp/finals/keypoints/face_aligned224"
# -d "/data/projects/aisc_facecomp/finals/keypoints/face_aligned"


python align_faces.py \
-r "/data/projects/aisc_facecomp/finals/data/filter40/受害者1" \
-d "/data/projects/aisc_facecomp/finals/data/filter40_aligned224/受害者1"

python align_faces.py \
-r "/data/projects/aisc_facecomp/finals/data/filter40/受害者2" \
-d "/data/projects/aisc_facecomp/finals/data/filter40_aligned224/受害者2"

python align_faces.py \
-r "/data/projects/aisc_facecomp/finals/data/filter40/受害者3" \
-d "/data/projects/aisc_facecomp/finals/data/filter40_aligned224/受害者3"

python align_faces.py \
-r "/data/projects/aisc_facecomp/finals/data/filter40/攻击者1" \
-d "/data/projects/aisc_facecomp/finals/data/filter40_aligned224/攻击者1"

python align_faces.py \
-r "/data/projects/aisc_facecomp/finals/data/filter40/攻击者2" \
-d "/data/projects/aisc_facecomp/finals/data/filter40_aligned224/攻击者2"

python align_faces.py \
-r "/data/projects/aisc_facecomp/finals/data/filter40/攻击者3" \
-d "/data/projects/aisc_facecomp/finals/data/filter40_aligned224/攻击者3"

