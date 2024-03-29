# baseline clean
# python -m eval.megface_eval --exp_name baseline_clean \
# --eval_dir /data/projects/verification/face_recog/data/video_img &

# python -m eval.megface_eval --exp_name mi_fgsm/iter10_eps3 \
# --eval_dir /data/projects/verification/face_recog/data/adv/mi_fgsm/iter10_eps3 &

# python -m eval.megface_eval --exp_name veri_square/iter5000_eps3_thres60.0 \
# --eval_dir /data/projects/verification/face_recog/data/adv/veri_square/iter5000_eps3_thres60.0 &


python -m eval.megface_eval --exp_name mi_fgsm/iter10_eps16 \
--eval_dir /data/projects/verification/face_recog/data/adv/mi_fgsm/iter10_eps16 &

python -m eval.megface_eval --exp_name mi_fgsm/iter10_eps8 \
--eval_dir /data/projects/verification/face_recog/data/adv/mi_fgsm/iter10_eps8 &