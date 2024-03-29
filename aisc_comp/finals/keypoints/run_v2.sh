cd /data/projects/aisc_facecomp

# bash finals/keypoints/run.sh

exp_name="aligned256"
entities=(
    受害者1
    受害者2
    受害者3
    攻击者1
    攻击者2
    攻击者3
)

for entity in ${entities[@]}
do
    # python -m finals.keypoints.generate_kp \
    # -r "/data/projects/hyperstyle/aisc/${exp_name}/${entity}" \
    # -d "/data/projects/hyperstyle/aisc/keypoints/${exp_name}_${entity}.pkl"

    # python -m finals.keypoints.generate_tmask256 \
    # -r1 "/data/projects/hyperstyle/aisc/keypoints/${exp_name}_${entity}.pkl" \
    # -r2 "/data/projects/hyperstyle/aisc/${exp_name}/${entity}" \
    # -d "/data/projects/hyperstyle/aisc/data/show_tmask/${exp_name}/${entity}" \
    # -m "/data/projects/hyperstyle/aisc/data/tmask/${exp_name}/${entity}" \
    # -l "/data/projects/hyperstyle/aisc/data/logs/${exp_name}_${entity}.log"

    # python -m finals.keypoints.generate_tmask256_wc \
    # -r1 "/data/projects/hyperstyle/aisc/keypoints/${exp_name}_${entity}.pkl" \
    # -r2 "/data/projects/hyperstyle/aisc/${exp_name}/${entity}" \
    # -d "/data/projects/hyperstyle/aisc/data/show_tmask/${exp_name}_wc_r1.3/${entity}" \
    # -m "/data/projects/hyperstyle/aisc/data/tmask/${exp_name}_wc_r1.3/${entity}" \
    # -l "/data/projects/hyperstyle/aisc/data/logs/${exp_name}_${entity}_wc_r1.3.log"

    python -m finals.keypoints.generate_tmask256_wc_final \
    -r1 "/data/projects/hyperstyle/aisc/keypoints/${exp_name}_${entity}.pkl" \
    -r2 "/data/projects/hyperstyle/aisc/${exp_name}/${entity}" \
    -d "/data/projects/hyperstyle/aisc/data/show_tmask/${exp_name}_m3/${entity}" \
    -m "/data/projects/hyperstyle/aisc/data/tmask/${exp_name}_m3/${entity}" \
    -l "/data/projects/hyperstyle/aisc/data/logs/${exp_name}_${entity}_m3.log"
done

# exp_name="aligned256_self"

# # python -m finals.keypoints.generate_kp \
# # -r "/data/projects/hyperstyle/aisc/aligned256" \
# # -d "/data/projects/hyperstyle/aisc/keypoints/${exp_name}.pkl"

# # python -m finals.keypoints.generate_tmask256 \
# # -r1 "/data/projects/hyperstyle/aisc/keypoints/${exp_name}.pkl" \
# # -r2 "/data/projects/hyperstyle/aisc/aligned256" \
# # -d "/data/projects/hyperstyle/aisc/data/show_tmask/${exp_name}" \
# # -m "/data/projects/hyperstyle/aisc/data/tmask/${exp_name}" \
# # -l "/data/projects/hyperstyle/aisc/data/logs/${exp_name}.log"

# # python -m finals.keypoints.generate_tmask256_wc \
# # -r1 "/data/projects/hyperstyle/aisc/keypoints/${exp_name}.pkl" \
# # -r2 "/data/projects/hyperstyle/aisc/aligned256" \
# # -d "/data/projects/hyperstyle/aisc/data/show_tmask/${exp_name}_wc_r0.1" \
# # -m "/data/projects/hyperstyle/aisc/data/tmask/${exp_name}_wc_r0.1" \
# # -l "/data/projects/hyperstyle/aisc/data/logs/${exp_name}_wc_r0.1.log"

# # python -m finals.keypoints.generate_tmask256_wc_final \
# # -r1 "/data/projects/hyperstyle/aisc/keypoints/${exp_name}.pkl" \
# # -r2 "/data/projects/hyperstyle/aisc/aligned256" \
# # -d "/data/projects/hyperstyle/aisc/data/show_tmask/${exp_name}_m3" \
# # -m "/data/projects/hyperstyle/aisc/data/tmask/${exp_name}_m3" \
# # -l "/data/projects/hyperstyle/aisc/data/logs/${exp_name}_m3.log"