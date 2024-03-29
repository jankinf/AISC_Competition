import os
import numpy as np
import pandas as pd
import shutil
from collections import Counter

cos_init = np.ones((3000, 1)) * -1
filename = None
all_files = []
scores = []
df = None
model_thresholds = {
    0: 0.33,
    1: 0.25,
    2: 0.23,
    # 3: 0.22,
    # 4: 0.22,
    # 5: 0.22,
    3: 0.34,
    4: 0.30,
    5: 0.28,
    6: 0.26,
    7: 0.23,
    8: 0.23,
    9: 0.21,
}
models = [3, 4, 5]
thresholds = [
    0.34,
    0.30,
    0.28,
]
def get_indice(file):
    global df, filename
    df_tmp = pd.read_csv(file)
    name = os.path.splitext(os.path.basename(file))[0]
    name = name[:name.rfind('@')]
    all_files.append(name)
    # score = (df_tmp.iloc[:] > 0.22).mean(0)[:3].mean()
    # score = np.mean([(df_tmp.iloc[:, i] > threshold).mean() for i, threshold in enumerate(thresholds)])

    score = np.mean([(df_tmp.iloc[:, i] > model_thresholds[models[i]]).mean() for i in range(len(models))])
    scores.append(score)
    print(name, score)
    if df is None:
        df = df_tmp
        filename = [name for _ in range(3000)]
    else:
        his_cur = np.zeros((df.shape[0], 1))
        suc_cur = np.zeros((df.shape[0], 1))

        for i in range(len(models)):
            his_cur = his_cur + (df.iloc[:, i] > model_thresholds[models[i]]).values.reshape(-1, 1)
            suc_cur = suc_cur + (df_tmp.iloc[:, i] > model_thresholds[models[i]]).values.reshape(-1, 1)
        # his_cur = (df.iloc[:, [0, 1, 2]] > 0.22).mean(1).values
        # suc_cur = (df_tmp.iloc[:, [0, 1, 2]] > 0.22).mean(1).values
        indice = (suc_cur > his_cur).nonzero()[0]
        df.iloc[indice, :] = df_tmp.iloc[indice, :]
        for idx in indice:
            filename[idx] = name
 

def generate_data(csv, to_dir):
    os.makedirs(to_dir, exist_ok=True)
    dir_names = pd.read_csv(csv).iloc[:, -1].values
    filenames = ["{:04d}.png".format(i + 1) for i in range(3000)]

    for dir_name, filename in zip(dir_names, filenames):
        src_file = os.path.join("/data/projects/aisc_facecomp/results", dir_name, filename)
        tgt_file = os.path.join(to_dir, filename)
        shutil.copyfile(src_file, tgt_file)

files = [
    # "/data/projects/aisc_facecomp/cos_stat/08-18/ct_eps60_steps60_alpha0.3_keypoints_comp_d_imgmask_compare1_ord2_ens0178@3:0.22-4:0.22-5:0.22.csv",
    # "/data/projects/aisc_facecomp/cos_stat/08-18/ct_eps60_steps60_alpha0.3_keypoints_ds_imgmask_compare1_ord2_ens0178@3:0.22-4:0.22-5:0.22.csv",
    # "/data/projects/aisc_facecomp/cos_stat/08-18/optmask_ds_l2@comp_keypoint_alpha0.25_steps60_ens4_model0178@3:0.22-4:0.22-5:0.22.csv",
    # "/data/projects/aisc_facecomp/cos_stat/08-18/optmask_ds_l2@comp_keypoint_alpha0.3_steps60_ens4_model0178@3:0.22-4:0.22-5:0.22.csv",
    # "/data/projects/aisc_facecomp/cos_stat/08-19/optmask_dem_v2_l2@comp_keypoint_alpha0.25_steps40_n5_bound1.5ens4_model0178@3:0.22-4:0.22-5:0.22.csv",
    # "/data/projects/aisc_facecomp/cos_stat/08-19/optmask_dem_v2_l2@comp_keypoint_alpha0.25_steps30_n5_bound1.5ens4_model0178@3:0.22-4:0.22-5:0.22.csv",
    
    # change model threshold
    # "/data/projects/aisc_facecomp/cos_stat/08-21/optmask_bim_l2@comp_keypoint_alpha0.25_steps60_ens7_model0126789@3:0.34-4:0.3-5:0.28-6:0.26-9:0.21.csv",
    # "/data/projects/aisc_facecomp/cos_stat/08-21/optmask_bim_l2@comp_keypoint_alpha0.28_steps40_ens7_model0126789@3:0.34-4:0.3-5:0.28-6:0.26-9:0.21.csv",
    # "/data/projects/aisc_facecomp/cos_stat/08-21/optmask_bim_l2@comp_keypoint_alpha0.28_steps50_ens7_model0126789@3:0.34-4:0.3-5:0.28-6:0.26-9:0.21.csv",
    # "/data/projects/aisc_facecomp/cos_stat/08-21/optmask_bim_l2@comp_keypoint_alpha0.28_steps60_ens7_model0126789@3:0.34-4:0.3-5:0.28-6:0.26-9:0.21.csv",
    # "/data/projects/aisc_facecomp/cos_stat/08-21/optmask_bim_l2@comp_keypoint_alpha0.3_steps40_ens7_model0126789@3:0.34-4:0.3-5:0.28-6:0.26-9:0.21.csv",
    # "/data/projects/aisc_facecomp/cos_stat/08-21/optmask_bim_l2@comp_keypoint_alpha0.3_steps50_ens7_model0126789@3:0.34-4:0.3-5:0.28-6:0.26-9:0.21.csv",
    # "/data/projects/aisc_facecomp/cos_stat/08-21/optmask_dem_v2_l2@comp_keypoint_alpha0.25_steps30_n5_bound1.5ens4_model0178@3:0.34-4:0.3-5:0.28-6:0.26-9:0.21.csv",
    # "/data/projects/aisc_facecomp/cos_stat/08-21/optmask_dem_v2_l2@comp_keypoint_alpha0.25_steps40_n5_bound1.5ens4_model0178@3:0.34-4:0.3-5:0.28-6:0.26-9:0.21.csv",
    # "/data/projects/aisc_facecomp/cos_stat/08-21/ct_eps60_steps60_alpha0.3_keypoints_comp_d_imgmask_compare1_ord2_ens0178@3:0.34-4:0.3-5:0.28-6:0.26-9:0.21.csv",
    # "/data/projects/aisc_facecomp/cos_stat/08-21/ct_eps60_steps60_alpha0.3_keypoints_ds_imgmask_compare1_ord2_ens0178@3:0.34-4:0.3-5:0.28-6:0.26-9:0.21.csv",
    # "/data/projects/aisc_facecomp/cos_stat/08-21/optmask_di_l2@comp_keypoint_alpha0.25_steps60_n1_rr1.15_dp0.6_ens7_model0126789@3:0.34-4:0.3-5:0.28-6:0.26-9:0.21.csv",
    # "/data/projects/aisc_facecomp/cos_stat/08-21/optmask_di_l2@comp_keypoint_alpha0.25_steps60_n1_rr1.15_dp0.8_ens7_model0126789@3:0.34-4:0.3-5:0.28-6:0.26-9:0.21.csv",
    # "/data/projects/aisc_facecomp/cos_stat/08-21/optmask_di_l2@comp_keypoint_alpha0.25_steps60_n3_rr1.15_dp0.8_ens7_model0126789@3:0.34-4:0.3-5:0.28-6:0.26-9:0.21.csv",
    # "/data/projects/aisc_facecomp/cos_stat/08-21/optmask_ds_l2@comp_keypoint_alpha0.25_steps60_ens4_model0178@3:0.34-4:0.3-5:0.28-6:0.26-9:0.21.csv",
    # "/data/projects/aisc_facecomp/cos_stat/08-21/optmask_ds_l2@comp_keypoint_alpha0.3_steps60_ens4_model0178@3:0.34-4:0.3-5:0.28-6:0.26-9:0.21.csv",
    # "/data/projects/aisc_facecomp/cos_stat/08-21/repo@08_19_v2_shrink4@3:0.34-4:0.3-5:0.28-6:0.26-9:0.21.csv",
    
    # "/data/projects/aisc_facecomp/cos_stat/08-22/optmask_di_l2@raw_keypoint_alpha0.3_steps60_n4_rr1.15_dp0.7_ens4_model0178@3:0.34-4:0.3-5:0.28-6:0.26-9:0.21.csv",
    # "/data/projects/aisc_facecomp/cos_stat/08-22/optmask_di_l2@raw_keypoint_alpha0.3_steps60_n4_rr1.15_dp0.8_ens4_model0178@3:0.34-4:0.3-5:0.28-6:0.26-9:0.21.csv",
    # "/data/projects/aisc_facecomp/cos_stat/08-22/optmask_di_l2@raw_keypoint_alpha0.3_steps60_n5_rr1.15_dp0.7_ens4_model0178@3:0.34-4:0.3-5:0.28-6:0.26-9:0.21.csv",
    # "/data/projects/aisc_facecomp/cos_stat/08-22/optmask_di_l2@raw_keypoint_alpha0.3_steps60_n5_rr1.15_dp0.8_ens4_model0178@3:0.34-4:0.3-5:0.28-6:0.26-9:0.21.csv",
    # "/data/projects/aisc_facecomp/cos_stat/08-22/optmask_di_l2@raw_keypoint_alpha0.3_steps60_n5_rr1.15_dp0.9_ens4_model0178@3:0.34-4:0.3-5:0.28-6:0.26-9:0.21.csv",
    # "/data/projects/aisc_facecomp/cos_stat/08-22/optmask_di_l2@raw_keypoint_alpha0.3_steps60_n6_rr1.15_dp0.7_ens4_model0178@3:0.34-4:0.3-5:0.28-6:0.26-9:0.21.csv",
    # "/data/projects/aisc_facecomp/cos_stat/08-22/optmask_di_l2@raw_keypoint_alpha0.3_steps60_n6_rr1.15_dp0.8_ens4_model0178@3:0.34-4:0.3-5:0.28-6:0.26-9:0.21.csv",
    # "/data/projects/aisc_facecomp/cos_stat/08-22/optmask_di_l2@raw_keypoint_alpha0.3_steps60_n6_rr1.15_dp0.9_ens4_model0178@3:0.34-4:0.3-5:0.28-6:0.26-9:0.21.csv",

    # 8.22
    "/data/projects/aisc_facecomp/cos_stat/08-22/ct_eps60_steps60_alpha0.3_d_show_masks_comp_3p_2eye_1nose_ord2_ens0178@3:0.34-4:0.3-5:0.28-6:0.26-9:0.21.csv",
    "/data/projects/aisc_facecomp/cos_stat/08-22/ct_eps60_steps60_alpha0.3_d_show_masks_comp_v2_ord2_ens0178@3:0.34-4:0.3-5:0.28-6:0.26-9:0.21.csv",
    "/data/projects/aisc_facecomp/cos_stat/08-22/optmask_di_l2@comp_keypoint_3p_2eye_1nose_alpha0.3_steps60_n4_rr1.15_dp0.7_ens4_model0178@3:0.34-4:0.3-5:0.28-6:0.26-9:0.21.csv",
    "/data/projects/aisc_facecomp/cos_stat/08-22/optmask_di_l2@comp_keypoint_3p_2eye_1nose_alpha0.3_steps60_n4_rr1.15_dp0.8_ens4_model0178@3:0.34-4:0.3-5:0.28-6:0.26-9:0.21.csv",
    "/data/projects/aisc_facecomp/cos_stat/08-22/optmask_di_l2@comp_keypoint_3p_2eye_1nose_alpha0.3_steps60_n5_rr1.15_dp0.7_ens4_model0178@3:0.34-4:0.3-5:0.28-6:0.26-9:0.21.csv",
    "/data/projects/aisc_facecomp/cos_stat/08-22/optmask_di_l2@comp_keypoint_3p_2eye_1nose_alpha0.3_steps60_n5_rr1.15_dp0.8_ens4_model0178@3:0.34-4:0.3-5:0.28-6:0.26-9:0.21.csv",
    "/data/projects/aisc_facecomp/cos_stat/08-22/optmask_di_l2@comp_keypoint_3p_2eye_1nose_alpha0.3_steps60_n6_rr1.15_dp0.7_ens4_model0178@3:0.34-4:0.3-5:0.28-6:0.26-9:0.21.csv",
    "/data/projects/aisc_facecomp/cos_stat/08-22/optmask_di_l2@comp_keypoint_3p_2eye_1nose_alpha0.3_steps60_n6_rr1.15_dp0.8_ens4_model0178@3:0.34-4:0.3-5:0.28-6:0.26-9:0.21.csv",
    "/data/projects/aisc_facecomp/cos_stat/08-22/optmask_di_l2@comp_keypoint_v2_alpha0.3_steps60_n4_rr1.15_dp0.8_ens4_model0178@3:0.34-4:0.3-5:0.28-6:0.26-9:0.21.csv",
    "/data/projects/aisc_facecomp/cos_stat/08-22/optmask_di_l2@comp_keypoint_v2_alpha0.3_steps60_n5_rr1.15_dp0.7_ens4_model0178@3:0.34-4:0.3-5:0.28-6:0.26-9:0.21.csv",
    "/data/projects/aisc_facecomp/cos_stat/08-22/optmask_di_l2@comp_keypoint_v2_alpha0.3_steps60_n6_rr1.15_dp0.7_ens4_model0178@3:0.34-4:0.3-5:0.28-6:0.26-9:0.21.csv",
    "/data/projects/aisc_facecomp/cos_stat/08-22/optmask_di_l2@comp_keypoint_v2_alpha0.3_steps60_n6_rr1.15_dp0.8_ens4_model0178@3:0.34-4:0.3-5:0.28-6:0.26-9:0.21.csv",

]

for file in files:
    get_indice(file)
# print(df)

# print(filename)
# print(set(filename))
# score = (df.iloc[:] > 0.22).mean(0)[:3].mean()
score = np.mean([(df.iloc[:, i] > model_thresholds[models[i]]).mean() for i in range(len(models))])

print("merge", score)
df = df.assign(filename=filename)
# print(df)
print(Counter(df.iloc[:, -1]))

# import pdb; pdb.set_trace()
# save_name = "08_21_ct_ens2"
# save_name = "08_21_ens6"
save_name = "08_22_multimask_ens12"
df.to_csv("cos_stat/repo/{}.csv".format(save_name), index=False)

generate_data("cos_stat/repo/{}.csv".format(save_name), "/data/projects/aisc_facecomp/results/repo@{}".format(save_name))