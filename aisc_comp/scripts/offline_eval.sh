# input_dir="/data/projects/aisc_facecomp/results/optmask_pict_catfeat_fill_eps40_steps50_ens4_model0178"
# input_dir="/data/projects/aisc_facecomp/results/optmask_pict_catfeat_fill_eps40_steps0_ens4_model0178"
# input_dir="/data/projects/aisc_facecomp/data_clean"
# input_dir="/data/projects/aisc_facecomp/results/beval_ct_cos_catfeat_fill_alpha1.6_steps50_ens4_model0178_bmodel2"

input_dirs=(
    # /data/projects/aisc_facecomp/results/optmask_ds_l2@comp_keypoint_alpha0.35_steps60_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_ds_l2@comp_keypoint_alpha0.35_steps70_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_ds_l2@comp_keypoint_alpha0.3_steps60_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_ds_l2@comp_keypoint_alpha0.3_steps70_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_ds_l2@comp_keypoint_alpha0.4_steps60_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_ds_l2@comp_keypoint_alpha0.4_steps70_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_ds_l2@hiltert_m5_alpha0.35_steps60_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_ds_l2@hiltert_m5_alpha0.35_steps70_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_ds_l2@hiltert_m5_alpha0.3_steps60_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_ds_l2@hiltert_m5_alpha0.3_steps70_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_ds_l2@hiltert_m5_alpha0.4_steps60_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_ds_l2@hiltert_m5_alpha0.4_steps70_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_ds_l2@hiltert_m7_alpha0.35_steps60_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_ds_l2@hiltert_m7_alpha0.35_steps70_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_ds_l2@hiltert_m7_alpha0.3_steps60_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_ds_l2@hiltert_m7_alpha0.3_steps70_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_ds_l2@hiltert_m7_alpha0.4_steps60_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_ds_l2@hiltert_m7_alpha0.4_steps70_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_ds_l2@raw_keypoint_alpha0.35_steps60_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_ds_l2@raw_keypoint_alpha0.35_steps70_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_ds_l2@raw_keypoint_alpha0.3_steps60_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_ds_l2@raw_keypoint_alpha0.3_steps70_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_ds_l2@raw_keypoint_alpha0.4_steps60_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_ds_l2@raw_keypoint_alpha0.4_steps70_ens4_model0178
    # /data/projects/aisc_facecomp/results/beval_ct_cos_catfeat_fill_alpha1.6_steps50_ens4_model0178_bmodel347
    # /data/projects/aisc_facecomp/results/optmask_pict_catfeat_fill_eps40_steps50_ens4_model0178
    # /data/projects/aisc_facecomp/results/ct_eps60_steps60_alpha0.3_keypoints_ds_imgmask_compare1_ord2_ens0178
    # /data/projects/aisc_facecomp/results/ct_eps60_steps60_alpha0.3_keypoints_comp_d_imgmask_compare1_ord2_ens0178
    # /data/projects/aisc_facecomp/results/optmask_ds_l2@comp_keypoint_alpha0.3_steps70_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_ds_l2@comp_keypoint_alpha0.31_steps60_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_ds_l2@comp_keypoint_alpha0.29_steps60_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_ds_l2@comp_keypoint_alpha0.32_steps60_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_ds_l2@comp_keypoint_alpha0.28_steps60_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_ds_l2@comp_keypoint_alpha0.25_steps60_ens4_model0178

    # /data/projects/aisc_facecomp/results/optmask_dem_v2_l2@comp_keypoint_alpha0.25_steps30_n5_bound2.5ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_dem_v2_l2@comp_keypoint_alpha0.25_steps30_n5_bound1.5ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_dem_v2_l2@comp_keypoint_alpha0.25_steps30_n5_bound3ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_dem_v2_l2@comp_keypoint_alpha0.25_steps30_n5_bound2ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_dem_v2_l2@comp_keypoint_alpha0.25_steps40_n5_bound2ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_dem_v2_l2@comp_keypoint_alpha0.25_steps40_n5_bound2.5ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_dem_v2_l2@comp_keypoint_alpha0.25_steps40_n5_bound1.5ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_dem_v2_l2@comp_keypoint_alpha0.25_steps40_n5_bound3ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_dem_v2_l2@comp_keypoint_alpha0.25_steps50_n5_bound1.5ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_dem_v2_l2@comp_keypoint_alpha0.25_steps50_n5_bound2ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_dem_v2_l2@comp_keypoint_alpha0.25_steps60_n5_bound3ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_dem_v2_l2@comp_keypoint_alpha0.25_steps50_n5_bound3ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_dem_v2_l2@comp_keypoint_alpha0.22_steps60_n5_bound2ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_dem_v2_l2@comp_keypoint_alpha0.28_steps60_n5_bound2.5ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_dem_v2_l2@comp_keypoint_alpha0.25_steps60_n5_bound2ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_dem_v2_l2@comp_keypoint_alpha0.25_steps60_n5_bound2ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_dem_v2_l2@comp_keypoint_alpha0.22_steps60_n5_bound3ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_dem_v2_l2@comp_keypoint_alpha0.25_steps50_n5_bound2.5ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_dem_v2_l2@comp_keypoint_alpha0.3_steps60_n5_bound2.5ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_dem_v2_l2@comp_keypoint_alpha0.25_steps60_n5_bound1.5ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_dem_v2_l2@comp_keypoint_alpha0.25_steps60_n5_bound1.5ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_dem_v2_l2@comp_keypoint_alpha0.25_steps60_n5_bound2.5ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_dem_v2_l2@comp_keypoint_alpha0.25_steps60_n5_bound2.5ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_dem_v2_l2@comp_keypoint_alpha0.28_steps60_n5_bound1.5ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_dem_v2_l2@comp_keypoint_alpha0.3_steps60_n5_bound2ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_dem_v2_l2@comp_keypoint_alpha0.28_steps60_n5_bound3ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_dem_v2_l2@comp_keypoint_alpha0.3_steps60_n5_bound3ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_dem_v2_l2@comp_keypoint_alpha0.28_steps60_n5_bound2ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_dem_v2_l2@comp_keypoint_alpha0.22_steps60_n5_bound1.5ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_dem_v2_l2@comp_keypoint_alpha0.22_steps60_n5_bound2.5ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_dem_v2_l2@comp_keypoint_alpha0.3_steps60_n5_bound1.5ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_dem_v2_l2@comp_keypoint_alpha0.25_steps70_n5_bound3ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_dem_v2_l2@comp_keypoint_alpha0.25_steps70_n5_bound2ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_dem_v2_l2@comp_keypoint_alpha0.25_steps70_n5_bound2.5ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_dem_v2_l2@comp_keypoint_alpha0.25_steps70_n5_bound1.5ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_dem_v2_l2@comp_keypoint_alpha0.25_steps90_n5_bound1.5ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_dem_v2_l2@comp_keypoint_alpha0.25_steps90_n5_bound2ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_dem_v2_l2@comp_keypoint_alpha0.25_steps90_n5_bound3ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_dem_v2_l2@comp_keypoint_alpha0.25_steps90_n5_bound2.5ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_dem_v2_l2@comp_keypoint_alpha0.25_steps120_n5_bound1.5ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_dem_v2_l2@comp_keypoint_alpha0.25_steps120_n5_bound3ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_dem_v2_l2@comp_keypoint_alpha0.25_steps150_n5_bound2ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_dem_v2_l2@comp_keypoint_alpha0.25_steps150_n5_bound1.5ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_dem_v2_l2@comp_keypoint_alpha0.25_steps150_n5_bound2.5ens4_model0178

    # /data/projects/aisc_facecomp/results/repo@08_19

    # /data/projects/aisc_facecomp/results/optmask_dem_v2_l2_worigin@comp_keypoint_alpha0.25_steps20_n5_bound1.5ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_dem_v2_l2_worigin@comp_keypoint_alpha0.25_steps20_n7_bound1.5ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_dem_v2_l2_worigin@comp_keypoint_alpha0.25_steps25_n5_bound1.5ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_dem_v2_l2_worigin@comp_keypoint_alpha0.25_steps20_n6_bound1.5ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_dem_v2_l2_worigin@comp_keypoint_alpha0.25_steps20_n4_bound1.5ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_dem_v2_l2_worigin@comp_keypoint_alpha0.25_steps25_n4_bound1.5ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_dem_v2_l2_worigin@comp_keypoint_alpha0.25_steps30_n4_bound1.5ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_dem_v2_l2_worigin@comp_keypoint_alpha0.25_steps25_n7_bound1.5ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_dem_v2_l2_worigin@comp_keypoint_alpha0.25_steps25_n6_bound1.5ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_dem_v2_l2_worigin@comp_keypoint_alpha0.25_steps20_n8_bound1.5ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_dem_v2_l2_worigin@comp_keypoint_alpha0.25_steps30_n6_bound1.5ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_dem_v2_l2_worigin@comp_keypoint_alpha0.25_steps30_n5_bound1.5ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_dem_v2_l2_worigin@comp_keypoint_alpha0.25_steps20_n9_bound1.5ens4_model0178

    # /data/projects/aisc_facecomp/results/optmask_dem_v2_l2_worigin@comp_keypoint_alpha0.25_steps30_n7_bound1.5ens4_model0178
    # /data/projects/aisc_facecomp/results/dem_mi@comp_keypoint_alpha0.00627_steps40_n4_bound1.5ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_dem_v2_l2_worigin@comp_keypoint_alpha0.25_steps30_n7_bound1.5ens4_model0178
    # /data/projects/aisc_facecomp/results/dem_mi@comp_keypoint_alpha0.00627_steps40_n4_bound1.5ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_dem_v2_l2_worigin@comp_keypoint_alpha0.25_steps25_n9_bound1.5ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_dem_v2_l2_worigin@comp_keypoint_alpha0.25_steps25_n8_bound1.5ens4_model0178
    # /data/projects/aisc_facecomp/results/dem_mi@comp_keypoint_alpha0.00627_steps40_n5_bound1.5ens4_model0178

    # /data/projects/aisc_facecomp/results/optmask_dem_mi@comp_keypoint_alpha0.00627_steps40_n4_bound1.5ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_dem@comp_keypoint_alpha0.00627_steps40_n4_bound1.5ens4_model0178
    # /data/projects/aisc_facecomp/results/dem_mi@comp_keypoint_alpha0.00627_steps50_n5_bound1.5ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_dem@comp_keypoint_alpha0.00627_steps40_n5_bound1.5ens4_model0178
    # /data/projects/aisc_facecomp/results/dem_mi@comp_keypoint_alpha0.00627_steps50_n4_bound1.5ens4_model0178
    # /data/projects/aisc_facecomp/results/dem_mi@comp_keypoint_alpha0.00627_steps60_n5_bound1.5ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_dem_v2_l2_worigin@comp_keypoint_alpha0.25_steps30_n9_bound1.5ens4_model0178
    # /data/projects/aisc_facecomp/results/dem_mi@comp_keypoint_alpha0.00627_steps60_n4_bound1.5ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_dem_v2_l2_worigin@comp_keypoint_alpha0.25_steps30_n8_bound1.5ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_dem_v2_l2_worigin@comp_keypoint_alpha0.25_steps60_n4_bound1.5ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_dem_mi@comp_keypoint_alpha0.00627_steps40_n5_bound1.5ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_dem_mi@comp_keypoint_alpha0.00627_steps50_n4_bound1.5ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_dem@comp_keypoint_alpha0.00627_steps50_n5_bound1.5ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_dem@comp_keypoint_alpha0.00627_steps60_n4_bound1.5ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_dem@comp_keypoint_alpha0.00627_steps50_n4_bound1.5ens4_model0178

    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.25_steps60_n1_rr1.1_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.3_steps60_n1_rr1.1_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.3_steps60_n1_rr1.15_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.25_steps60_n1_rr1.2_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.34_steps60_n1_rr1.15_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.34_steps60_n1_rr1.1_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.28_steps60_n1_rr1.2_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.28_steps60_n1_rr1.1_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.28_steps60_n1_rr1.15_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.34_steps60_n2_rr1.1_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.28_steps60_n2_rr1.15_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.25_steps60_n2_rr1.2_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.3_steps60_n1_rr1.2_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.3_steps60_n2_rr1.2_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.34_steps60_n2_rr1.2_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.3_steps60_n2_rr1.15_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.25_steps60_n2_rr1.1_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.25_steps60_n2_rr1.15_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.28_steps60_n2_rr1.2_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.28_steps60_n2_rr1.1_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.28_steps60_n3_rr1.1_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.25_steps60_n3_rr1.15_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.3_steps60_n3_rr1.1_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.3_steps60_n3_rr1.15_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.3_steps60_n3_rr1.2_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.34_steps60_n3_rr1.1_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.34_steps60_n3_rr1.2_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.25_steps60_n3_rr1.2_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.34_steps60_n3_rr1.15_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.25_steps60_n3_rr1.1_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.28_steps60_n3_rr1.15_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.28_steps60_n4_rr1.1_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.25_steps60_n4_rr1.1_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.34_steps60_n4_rr1.2_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.25_steps60_n4_rr1.15_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.28_steps60_n4_rr1.2_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.3_steps60_n4_rr1.15_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.34_steps60_n4_rr1.1_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.25_steps60_n4_rr1.2_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.3_steps60_n4_rr1.2_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.3_steps60_n4_rr1.1_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.3_steps50_n1_rr1.1_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.28_steps50_n1_rr1.1_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.25_steps50_n1_rr1.2_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.34_steps50_n1_rr1.1_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.34_steps50_n1_rr1.2_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.28_steps50_n1_rr1.15_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.28_steps50_n1_rr1.2_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.34_steps50_n1_rr1.15_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.25_steps50_n1_rr1.1_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.3_steps50_n1_rr1.15_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.25_steps50_n2_rr1.1_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.25_steps50_n2_rr1.15_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.28_steps50_n2_rr1.1_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.28_steps50_n2_rr1.2_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.34_steps50_n2_rr1.2_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.3_steps50_n1_rr1.2_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.25_steps50_n2_rr1.2_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.34_steps50_n2_rr1.1_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.34_steps50_n2_rr1.15_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.28_steps50_n2_rr1.15_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.3_steps50_n2_rr1.15_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.3_steps50_n2_rr1.1_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.28_steps50_n3_rr1.2_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.25_steps50_n3_rr1.2_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.3_steps50_n2_rr1.2_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.25_steps50_n3_rr1.15_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.25_steps50_n3_rr1.1_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.34_steps50_n3_rr1.2_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.34_steps50_n3_rr1.1_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.28_steps50_n3_rr1.15_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.3_steps50_n3_rr1.1_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.3_steps50_n3_rr1.15_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.3_steps50_n3_rr1.2_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.28_steps50_n3_rr1.1_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.28_steps50_n4_rr1.15_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.34_steps50_n3_rr1.15_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.3_steps50_n4_rr1.15_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.25_steps50_n4_rr1.15_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.28_steps50_n4_rr1.1_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.3_steps50_n4_rr1.1_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.3_steps50_n4_rr1.2_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.34_steps50_n4_rr1.15_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.34_steps50_n4_rr1.2_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.34_steps50_n4_rr1.1_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.25_steps50_n4_rr1.2_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.28_steps50_n4_rr1.2_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.25_steps50_n4_rr1.1_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.28_steps70_n1_rr1.2_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.28_steps70_n1_rr1.15_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.28_steps70_n1_rr1.1_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.34_steps70_n1_rr1.1_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.25_steps70_n1_rr1.1_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.34_steps70_n1_rr1.15_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.25_steps70_n1_rr1.2_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.3_steps70_n1_rr1.2_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.34_steps70_n1_rr1.2_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.3_steps70_n1_rr1.1_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.3_steps70_n1_rr1.15_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.34_steps70_n2_rr1.1_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.34_steps70_n2_rr1.15_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.34_steps70_n2_rr1.2_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.28_steps70_n2_rr1.15_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.25_steps70_n2_rr1.15_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.3_steps70_n2_rr1.15_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.25_steps70_n2_rr1.1_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.3_steps70_n2_rr1.2_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.3_steps70_n2_rr1.1_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.25_steps70_n2_rr1.2_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.28_steps70_n2_rr1.1_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.28_steps70_n2_rr1.2_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.3_steps70_n3_rr1.15_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.25_steps70_n3_rr1.15_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.25_steps70_n3_rr1.1_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.25_steps70_n3_rr1.2_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.28_steps70_n3_rr1.1_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.34_steps70_n3_rr1.1_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.28_steps70_n3_rr1.2_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.28_steps70_n3_rr1.15_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.3_steps70_n3_rr1.1_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.34_steps70_n3_rr1.15_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.34_steps70_n3_rr1.2_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.3_steps70_n3_rr1.2_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/repo@08_19_v2_shrink4
    # /data/projects/aisc_facecomp/results/ct_eps60_steps60_alpha0.3_keypoints_comp_d_imgmask_compare1_ord2_ens0178
    # /data/projects/aisc_facecomp/results/ct_eps60_steps60_alpha0.3_keypoints_ds_imgmask_compare1_ord2_ens0178
    # /data/projects/aisc_facecomp/results/optmask_ds_l2@comp_keypoint_alpha0.25_steps60_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_ds_l2@comp_keypoint_alpha0.3_steps60_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_dem_v2_l2@comp_keypoint_alpha0.25_steps40_n5_bound1.5ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_dem_v2_l2@comp_keypoint_alpha0.25_steps30_n5_bound1.5ens4_model0178

    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.28_steps70_n4_rr1.2_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.34_steps70_n4_rr1.2_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.3_steps70_n4_rr1.2_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.25_steps70_n4_rr1.2_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.34_steps70_n4_rr1.15_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.25_steps70_n4_rr1.1_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.3_steps70_n4_rr1.15_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.34_steps70_n4_rr1.1_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.25_steps70_n4_rr1.15_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.28_steps70_n4_rr1.1_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_dem_v2_l2_worigin@comp_keypoint_alpha0.25_steps60_n4_bound1.5ens7_model0126789

    # /data/projects/aisc_facecomp/results/optmask_dem_v2_l2_worigin@comp_keypoint_alpha0.25_steps20_n4_bound1.5ens7_model0126789
    # /data/projects/aisc_facecomp/results/optmask_dem_v2_l2_worigin@comp_keypoint_alpha0.25_steps25_n4_bound1.5ens7_model0126789
    # /data/projects/aisc_facecomp/results/optmask_dem_v2_l2_worigin@comp_keypoint_alpha0.25_steps30_n4_bound1.5ens7_model0126789
    # /data/projects/aisc_facecomp/results/optmask_dem_v2_l2_worigin@comp_keypoint_alpha0.25_steps60_n4_bound1.5ens7_model0126789

    # /data/projects/aisc_facecomp/results/optmask_bim_l2@comp_keypoint_alpha0.3_steps40_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_bim_l2@comp_keypoint_alpha0.28_steps40_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_bim_l2@comp_keypoint_alpha0.25_steps40_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_bim_l2@comp_keypoint_alpha0.25_steps50_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.25_steps60_n1_rr1.15_dp0.5_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_bim_l2@comp_keypoint_alpha0.28_steps60_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_bim_l2@comp_keypoint_alpha0.3_steps50_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.3_steps60_n1_rr1.15_dp0.9_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.25_steps60_n1_rr1.15_dp0.8_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_bim_l2@comp_keypoint_alpha0.3_steps60_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.25_steps60_n1_rr1.15_dp0.9_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_bim_l2@comp_keypoint_alpha0.25_steps60_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_bim_l2@comp_keypoint_alpha0.28_steps50_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.25_steps60_n1_rr1.15_dp0.6_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.3_steps60_n1_rr1.15_dp0.6_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.25_steps60_n2_rr1.15_dp0.6_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.25_steps60_n2_rr1.15_dp0.9_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.25_steps60_n2_rr1.15_dp0.8_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.3_steps60_n2_rr1.15_dp0.6_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_bim_l2@comp_keypoint_alpha0.28_steps90_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.25_steps60_n2_rr1.15_dp0.5_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.3_steps60_n2_rr1.15_dp0.8_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.3_steps60_n2_rr1.15_dp0.9_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_bim_l2@comp_keypoint_alpha0.28_steps80_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_bim_l2@comp_keypoint_alpha0.3_steps90_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_bim_l2@comp_keypoint_alpha0.25_steps80_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.3_steps60_n2_rr1.15_dp0.5_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.25_steps60_n3_rr1.15_dp0.6_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.3_steps60_n3_rr1.15_dp0.9_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.3_steps60_n3_rr1.15_dp0.5_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.3_steps60_n3_rr1.15_dp0.6_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.25_steps60_n3_rr1.15_dp0.5_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.25_steps60_n3_rr1.15_dp0.9_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.25_steps60_n3_rr1.15_dp0.8_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.3_steps60_n3_rr1.15_dp0.8_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_bim_l2@comp_keypoint_alpha0.25_steps90_ens4_model0178

    # /data/projects/aisc_facecomp/results/repo@08_19_v2_shrink4
    # /data/projects/aisc_facecomp/results/ct_eps60_steps60_alpha0.3_keypoints_comp_d_imgmask_compare1_ord2_ens0178
    # /data/projects/aisc_facecomp/results/ct_eps60_steps60_alpha0.3_keypoints_ds_imgmask_compare1_ord2_ens0178
    # /data/projects/aisc_facecomp/results/optmask_ds_l2@comp_keypoint_alpha0.25_steps60_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_ds_l2@comp_keypoint_alpha0.3_steps60_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_dem_v2_l2@comp_keypoint_alpha0.25_steps40_n5_bound1.5ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_dem_v2_l2@comp_keypoint_alpha0.25_steps30_n5_bound1.5ens4_model0178

    # /data/projects/aisc_facecomp/results/optmask_bim_l2@comp_keypoint_alpha0.3_steps40_ens7_model0126789
    # /data/projects/aisc_facecomp/results/optmask_bim_l2@comp_keypoint_alpha0.28_steps40_ens7_model0126789
    # /data/projects/aisc_facecomp/results/optmask_bim_l2@comp_keypoint_alpha0.25_steps40_ens7_model0126789
    # /data/projects/aisc_facecomp/results/optmask_bim_l2@comp_keypoint_alpha0.28_steps60_ens7_model0126789
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.25_steps60_n3_rr1.15_dp0.8_ens7_model0126789
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.25_steps60_n1_rr1.15_dp0.6_ens7_model0126789
    # /data/projects/aisc_facecomp/results/optmask_bim_l2@comp_keypoint_alpha0.25_steps60_ens7_model0126789
    # /data/projects/aisc_facecomp/results/optmask_bim_l2@comp_keypoint_alpha0.28_steps50_ens7_model0126789
    # /data/projects/aisc_facecomp/results/optmask_bim_l2@comp_keypoint_alpha0.3_steps50_ens7_model0126789
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_alpha0.25_steps60_n1_rr1.15_dp0.8_ens7_model0126789

    # /data/projects/aisc_facecomp/results/optmask_di_l2@raw_keypoint_alpha0.3_steps60_n4_rr1.15_dp0.8_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@raw_keypoint_alpha0.3_steps60_n4_rr1.15_dp0.9_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@raw_keypoint_alpha0.3_steps60_n4_rr1.15_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@raw_keypoint_alpha0.3_steps60_n5_rr1.15_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@raw_keypoint_alpha0.3_steps60_n5_rr1.15_dp0.8_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@raw_keypoint_alpha0.3_steps60_n5_rr1.15_dp0.9_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@raw_keypoint_alpha0.3_steps60_n6_rr1.15_dp0.9_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@raw_keypoint_alpha0.3_steps60_n6_rr1.15_dp0.8_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@raw_keypoint_alpha0.3_steps60_n6_rr1.15_dp0.7_ens4_model0178


    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_2p_2eye_alpha0.3_steps60_n4_rr1.15_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_2p_2eye_alpha0.3_steps60_n4_rr1.15_dp0.8_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_3p_2eye_1nose_alpha0.3_steps60_n4_rr1.15_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_3p_2eye_1mouse_alpha0.3_steps60_n4_rr1.15_dp0.8_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_v2_alpha0.3_steps60_n4_rr1.15_dp0.8_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_3p_2eye_1nose_alpha0.3_steps60_n4_rr1.15_dp0.8_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_3p_2eye_1mouse_alpha0.3_steps60_n4_rr1.15_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_3p_2eye_1nose_alpha0.3_steps60_n5_rr1.15_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_v2_alpha0.3_steps60_n5_rr1.15_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_3p_2eye_1mouse_alpha0.3_steps60_n5_rr1.15_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_3p_2eye_1nose_alpha0.3_steps60_n5_rr1.15_dp0.8_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_2p_2eye_alpha0.3_steps60_n5_rr1.15_dp0.8_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_2p_2eye_alpha0.3_steps60_n5_rr1.15_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_3p_2eye_1mouse_alpha0.3_steps60_n5_rr1.15_dp0.8_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_3p_2eye_1nose_alpha0.3_steps60_n6_rr1.15_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_3p_2eye_1nose_alpha0.3_steps60_n6_rr1.15_dp0.8_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_2p_2eye_alpha0.3_steps60_n6_rr1.15_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_v2_alpha0.3_steps60_n6_rr1.15_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_v2_alpha0.3_steps60_n6_rr1.15_dp0.8_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_3p_2eye_1mouse_alpha0.3_steps60_n6_rr1.15_dp0.8_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2_adaw@comp_keypoint_v2_alpha0.3_steps60_n4_rr1.15_dp0.7_tem3_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2_adaw@comp_keypoint_v2_alpha0.3_steps60_n4_rr1.15_dp0.7_tem1_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_3p_2eye_1mouse_alpha0.3_steps60_n6_rr1.15_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2_adaw@comp_keypoint_v2_alpha0.3_steps60_n4_rr1.15_dp0.7_tem2_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_2p_2eye_alpha0.3_steps60_n6_rr1.15_dp0.8_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2_adaw@comp_keypoint_v2_alpha0.3_steps60_n4_rr1.15_dp0.7_tem4_ens4_model0178

    # /data/projects/aisc_facecomp/results/ct_eps60_steps60_alpha0.3_d_show_masks_comp_3p_2eye_1nose_ord2_ens0178
    # /data/projects/aisc_facecomp/results/ct_eps60_steps60_alpha0.3_d_show_masks_comp_v2_ord2_ens0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_v2_bugfix_alpha0.3_steps60_n4_rr1.3_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_v2_bugfix_alpha0.25_steps60_n4_rr1.3_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_v2_bugfix_alpha0.28_steps60_n4_rr1.3_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_v2_bugfix_alpha0.28_steps60_n6_rr1.3_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_v2_bugfix_alpha0.25_steps60_n6_rr1.3_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_v2_bugfix_alpha0.3_steps60_n6_rr1.3_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_v2_bugfix_alpha0.28_steps100_n4_rr1.3_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_v2_bugfix_alpha0.25_steps60_n8_rr1.3_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_v2_bugfix_alpha0.3_steps60_n8_rr1.3_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_v2_bugfix_alpha0.28_steps60_n8_rr1.3_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_v2_bugfix_alpha0.25_steps120_n4_rr1.3_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_v2_bugfix_alpha0.3_steps100_n4_rr1.3_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_v2_bugfix_alpha0.25_steps100_n4_rr1.3_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_v2_bugfix_alpha0.28_steps120_n4_rr1.3_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_v2_bugfix_alpha0.3_steps120_n4_rr1.3_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_v2_bugfix_alpha0.3_steps100_n6_rr1.3_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_v2_bugfix_alpha0.28_steps100_n6_rr1.3_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_v2_bugfix_alpha0.25_steps100_n6_rr1.3_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_v2_bugfix_alpha0.28_steps100_n8_rr1.3_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_v2_bugfix_alpha0.3_steps120_n6_rr1.3_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_v2_bugfix_alpha0.28_steps120_n6_rr1.3_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_v2_bugfix_alpha0.3_steps100_n8_rr1.3_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_v2_bugfix_alpha0.3_steps120_n8_rr1.3_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/optmask_di_l2@comp_keypoint_v2_bugfix_alpha0.28_steps120_n8_rr1.3_dp0.7_ens4_model0178
    # /data/projects/aisc_facecomp/results/liyi/ct_steps100_alpha0.3_rr1.3_dp1.0_d4_comp_v2_ord2_model013478
    # /data/projects/aisc_facecomp/results/liyi/ct_steps100_alpha0.3_rr1.3_dp1.0_d4_comp_v2_ord2_model013478
    # /data/projects/aisc_facecomp/results/liyi/ct_steps100_alpha0.3_rr1.3_dp1.0_d7_comp_tmode_v2_ord2_model01347813
    # /data/projects/aisc_facecomp/results/liyi/ct_steps100_alpha0.3_rr1.3_dp1.0_d7_origin_tmode_ord2_model01347813
    # /data/projects/aisc_facecomp/results/liyi/ct_steps100_alpha0.3_rr1.3_dp1.0_d7_comp_tmode_ord2_model01347813
    # /data/projects/aisc_facecomp/results/liyi/ct_steps100_alpha0.3_rr1.3_dp1.0_d7_comp_tmode_ord2_model01347813
    # /data/projects/aisc_facecomp/results/liyi/ct_steps100_alpha0.3_rr1.3_dp1.0_d7_comp_tmode_ord2_model01347813
    # /data/projects/aisc_facecomp/results/liyi/ct_steps120_alpha0.3_rr1.3_dp1.0_d7_comp_tmode_v2_ord2_model01347813
    # /data/projects/aisc_facecomp/results/liyi/ct_steps120_alpha0.25_rr1.3_dp1.0_d7_comp_tmode_v2_ord2_model01347813
    # /data/projects/aisc_facecomp/results/liyi/ct_steps100_alpha0.3_rr1.3_dp1.0_d7_origin_tmode_v2_cmp_mask_ord2_model01347813

    /data/projects/aisc_facecomp/results/liyi/ct_steps120_alpha0.3_rr1.3_dp1.0_d7_comp_tmode_ord2_model0134781315
)
for input_dir in ${input_dirs[@]}
do
    # nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python offline_detail.py --model_idx 3 4 5 --input_dir $input_dir &
    nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python offline_detail.py --model_idx 5 6 9 --input_dir $input_dir &
    # nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python offline.py --model_idx 3 4 5 --input_dir $input_dir >> offline.log &
    sleep 5s
done


# nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python offline.py --model_idx 2 3 4 --input_dir $input_dir >> offline.log&
# sleep 5s
# nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python offline.py --model_idx 0 2 3 --input_dir $input_dir >> offline.log&
# sleep 5s
# nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python offline.py --model_idx 1 2 3 --input_dir $input_dir >> offline.log&
# sleep 5s
# nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python offline.py --model_idx 7 2 3 --input_dir $input_dir >> offline.log&
# sleep 5s
# nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python offline.py --model_idx 8 2 3 --input_dir $input_dir >> offline.log&
# sleep 5s
# nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python offline.py --model_idx 0 2 4 --input_dir $input_dir >> offline.log&
# sleep 5s
# nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python offline.py --model_idx 1 2 4 --input_dir $input_dir >> offline.log&
# sleep 5s
# nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python offline.py --model_idx 7 2 4 --input_dir $input_dir >> offline.log&
# sleep 5s
# nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python offline.py --model_idx 8 2 4 --input_dir $input_dir >> offline.log&
# sleep 5s
# nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python offline.py --model_idx 0 3 4 --input_dir $input_dir >> offline.log&
# sleep 5s
# nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python offline.py --model_idx 1 3 4 --input_dir $input_dir >> offline.log&
# sleep 5s
# nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python offline.py --model_idx 7 3 4 --input_dir $input_dir >> offline.log&
# sleep 5s
