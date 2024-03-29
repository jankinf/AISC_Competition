cd /data/projects/aisc_facecomp

# 生成pkl文件
# python -m finals.keypoints.generate_kp \
# -r "/data/projects/aisc_facecomp/finals/keypoints/face_aligned224" \
# -d "/data/projects/aisc_facecomp/finals/data/pkl/face_aligned224.pkl"

# 生成json文件
python -m finals.keypoints.generate_tmode_mask224

# 生成mask
python -m finals.keypoints.labelme2voc