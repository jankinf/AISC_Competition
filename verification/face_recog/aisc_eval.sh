# baseline clean
# python -m eval.aisc.megface_eval --eval_dir /data/projects/aisc_facecomp/results/liyi/ct_steps120_alpha0.3_rr1.3_dp1.0_d7_comp_tmode_ord2_model0134781315

python -m eval.aisc.megface_eval --eval_dir /data/projects/aisc_facecomp/results/liyi/ct_steps120_alpha0.3_rr1.3_dp1.0_d7_comp_tmode_ord2_model0134781315
python -m eval.aisc.megface_eval --eval_dir /data/projects/aisc_facecomp/results/liyi/ct_steps100_alpha0.3_rr1.3_dp1.0_d7_origin_tmode_v2_cmp_mask_ord2_model01347813
python -m eval.aisc.megface_eval --eval_dir /data/projects/aisc_facecomp/results/liyi/ct_steps120_alpha0.25_rr1.3_dp1.0_d7_fixed_mask_v1_ord2_model01347813
python -m eval.aisc.megface_eval --eval_dir /data/projects/aisc_facecomp/results/liyi/ct_steps120_alpha0.25_rr1.3_dp1.0_d7_fixed_mask_v1_ori_mask_ord2_model01347813
python -m eval.aisc.megface_eval --eval_dir /data/projects/aisc_facecomp/results/liyi/ct_steps120_alpha0.25_rr1.3_dp1.0_d7_fixed_mask_v2_ori_mask_ord2
python -m eval.aisc.megface_eval --eval_dir /data/projects/aisc_facecomp/results/liyi/ct_steps120_alpha0.25_rr1.3_dp1.0_d7_fixed_mask_v2_ori_mask_ord2_model01347813
python -m eval.aisc.megface_eval --eval_dir /data/projects/aisc_facecomp/results/liyi/ct_steps120_alpha0.25_rr1.3_dp1.0_d7_fixed_mask_v2_ori_mask_ord2_model0134781315
python -m eval.aisc.megface_eval --eval_dir /data/projects/aisc_facecomp/results/liyi/ct_steps100_alpha0.3_rr1.3_dp1.0_d7_comp_tmode_ord2_model01347813
python -m eval.aisc.megface_eval --eval_dir /data/projects/aisc_facecomp/results/liyi/ct_steps120_alpha0.3_rr1.3_dp1.0_d7_comp_tmode_v2_ord2_model01347813
python -m eval.aisc.megface_eval --eval_dir /data/projects/aisc_facecomp/results/liyi/ct_steps100_alpha0.3_rr1.3_dp1.0_d7_comp_tmode_v2_ord2_model01347813
python -m eval.aisc.megface_eval --eval_dir /data/projects/aisc_facecomp/results/liyi/ct_steps100_alpha0.3_rr1.3_dp1.0_d7_origin_tmode_ord2_model01347813
python -m eval.aisc.megface_eval --eval_dir /data/projects/aisc_facecomp/results/liyi/ct_steps100_alpha0.3_rr1.3_dp1.0_d7_origin_tmode_v2_ord2_model01347813
python -m eval.aisc.megface_eval --eval_dir /data/projects/aisc_facecomp/results/liyi/ct_steps100_alpha0.3_rr1.3_dp1.0_d4_comp_v2_ord2_model013478
python -m eval.aisc.megface_eval --eval_dir /data/projects/aisc_facecomp/results/liyi/ct_steps100_alpha0.3_rr1.3_dp1.0_d4_comp_v2_ord2_model0134578
python -m eval.aisc.megface_eval --eval_dir /data/projects/aisc_facecomp/results/liyi/ct_eps60_steps100_alpha0.45_rr1.3_dp0.8_d8_show_masks_comp_v2_ord2_ens0178
python -m eval.aisc.megface_eval --eval_dir /data/projects/aisc_facecomp/results/liyi/ct_eps60_steps100_alpha0.5_rr1.3_dp0.8_d8_show_masks_comp_v2_ord2_ens0178
python -m eval.aisc.megface_eval --eval_dir /data/projects/aisc_facecomp/results/liyi/ct_eps60_steps200_alpha0.35_rr1.3_dp0.8_d8_show_masks_comp_v2_ord2_ens0178
python -m eval.aisc.megface_eval --eval_dir /data/projects/aisc_facecomp/results/liyi/ct_eps60_steps200_alpha0.4_rr1.3_dp0.8_d8_show_masks_comp_v2_ord2_ens0178
python -m eval.aisc.megface_eval --eval_dir /data/projects/aisc_facecomp/results/liyi/ct_eps60_steps100_alpha0.5_rr1.3_dp0.8_d8_show_masks_comp_v2_ord2_ens0178
python -m eval.aisc.megface_eval --eval_dir /data/projects/aisc_facecomp/results/liyi/ct_eps60_steps200_alpha0.3_rr1.3_dp0.8_d8_show_masks_comp_v2_ord2_ens0178
python -m eval.aisc.megface_eval --eval_dir /data/projects/aisc_facecomp/results/liyi/ct_eps60_steps100_alpha0.3_rr1.3_dp0.8_d8_show_masks_comp_v2_ord2_ens0178
python -m eval.aisc.megface_eval --eval_dir /data/projects/aisc_facecomp/results/liyi/ct_eps60_steps100_alpha0.4_rr1.3_dp0.8_d8_show_masks_comp_v2_ord2_ens0178

# python -m eval.aisc.megface_eval --eval_dir /data/projects/aisc_facecomp/results/liyi/ct_steps120_alpha0.25_rr1.3_dp1.0_d7_fixed_mask_v1_ord2_model01347813
# python -m eval.aisc.megface_eval --eval_dir /data/projects/aisc_facecomp/results/liyi/ct_steps120_alpha0.25_rr1.3_dp1.0_d7_comp_tmode_v2_ord2_model01347813
# python -m eval.aisc.megface_eval --eval_dir /data/projects/aisc_facecomp/results/liyi/origin_mask_merge

# nohup rlaunch --cpu 2 --memory 16394 -- python -m eval.aisc.megface_eval --eval_dir /data/projects/aisc_facecomp/results/liyi/ct_steps100_alpha0.3_rr1.3_dp1.0_d7_origin_tmode_v2_cmp_mask_ord2_model01347813 &
# sleep 5s
# nohup rlaunch --cpu 2 --memory 16394 -- python -m eval.aisc.megface_eval --eval_dir /data/projects/aisc_facecomp/results/liyi/ct_steps120_alpha0.25_rr1.3_dp1.0_d7_fixed_mask_v1_ord2_model01347813 &
# sleep 5s
# nohup rlaunch --cpu 2 --memory 16394 -- python -m eval.aisc.megface_eval --eval_dir /data/projects/aisc_facecomp/results/liyi/ct_steps120_alpha0.25_rr1.3_dp1.0_d7_comp_tmode_v2_ord2_model01347813 &
# sleep 5s
# nohup rlaunch --cpu 2 --memory 16394 -- python -m eval.aisc.megface_eval --eval_dir /data/projects/aisc_facecomp/results/liyi/origin_mask_merge &
# sleep 5s