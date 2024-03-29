cd /data/projects/aisc_facecomp

# bash finals/keypoints/run.sh

exp_name="filter40_aligned224"
entities=(
    # 受害者1
    受害者2
    受害者3
    攻击者1
    攻击者2
    攻击者3
)

for entity in ${entities[@]}
do
    python -m finals.keypoints.generate_kp \
    -r "/data/projects/aisc_facecomp/finals/data/${exp_name}/${entity}" \
    -d "/data/projects/aisc_facecomp/finals/data/pkl/${exp_name}_${entity}.pkl"

    # python -m finals.keypoints.generate_tmask224 \
    # -r1 "/data/projects/aisc_facecomp/finals/data/pkl/${exp_name}_${entity}.pkl" \
    # -r2 "/data/projects/aisc_facecomp/finals/data/${exp_name}/${entity}" \
    # -d "/data/projects/aisc_facecomp/finals/data/show_tmask/${exp_name}/${entity}" \
    # -m "/data/projects/aisc_facecomp/finals/data/tmask/${exp_name}/${entity}" \
    # -l "/data/projects/aisc_facecomp/finals/data/logs/${exp_name}_${entity}.log"
done



# python -m finals.keypoints.generate_kp \
# -r "/data/projects/aisc_facecomp/finals/keypoints/face_aligned224" \
# -d "/data/projects/aisc_facecomp/finals/data/pkl/face_aligned224.pkl"