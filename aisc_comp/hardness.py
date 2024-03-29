import os
import numpy as np
import pandas as pd
import shutil

# csv_path = "/data/projects/aisc_facecomp/cos_stat/08-29/ct_steps100_alpha0.3_rr1.3_dp1.0_d4_comp_v2_ord2_model013478@5:0.28-6:0.26-9:0.21.csv"
# csv_path = "/data/projects/aisc_facecomp/cos_stat/08-29/ct_steps100_alpha0.3_rr1.3_dp1.0_d7_comp_tmode_v2_ord2_model01347813@5:0.28-6:0.26-9:0.21.csv"
csv_path = "/data/projects/aisc_facecomp/cos_stat/08-30/ct_steps100_alpha0.3_rr1.3_dp1.0_d7_origin_tmode_ord2_model01347813@5:0.28-6:0.26-9:0.21.csv"

df = pd.read_csv(csv_path)
# df2 = pd.read_csv(csv_path2)
ary = df.iloc[:, -1].values
# mid = np.median(ary)
# mid = 0.3
# indice = (ary < mid).nonzero()[0]
# print(len(indice))

# exit()
files = ["{:04d}.png".format(i + 1) for i in range(3000)]
comp_files = ["{:04d}_compare.png".format(i + 1) for i in range(3000)]


from collections import Counter
# print(Counter(df.iloc[indice, -1].values))

# import pdb; pdb.set_trace()
sorted_indice = df.sort_values(by=['mean_cos']).reset_index().iloc[:, 0].values
# sorted_indice = df2.sort_values(by=['mean_cos']).reset_index().iloc[:, 0].values
filtered_files = {}
# adv_dir = "/data/projects/aisc_facecomp/results/liyi/ct_steps100_alpha0.3_rr1.3_dp1.0_d4_comp_v2_ord2_model013478"
# adv_dir = "/data/projects/aisc_facecomp/results/liyi/ct_steps100_alpha0.3_rr1.3_dp1.0_d7_comp_tmode_v2_ord2_model01347813"
adv_dir = "/data/projects/aisc_facecomp/results/liyi/ct_steps100_alpha0.3_rr1.3_dp1.0_d7_origin_tmode_ord2_model01347813"

src_dir = "/data/projects/aisc_facecomp/data"
# out_dir = "./comp_data"
out_dir = "./comp_origin_data"

# mask_dir = "/data/projects/aisc_facecomp/keypoints/show_masks_seg_fill1254_v2"
# mask_dir = "/data/projects/aisc_facecomp/keypoints/show_masks_seg_fill1254_comp_json_tmode_v2"
mask_dir = "/data/projects/aisc_facecomp/keypoints/show_masks_seg_fill1254_origin_json_tmode_v2"
os.makedirs(out_dir, exist_ok=True)

sorted_indice = sorted_indice[:10]
# sorted_indice = sorted_indice[-10:]
for idx in sorted_indice:
    filtered_files[files[idx]] = df.iloc[idx, -1]
    # filtered_files[files[idx]] = df2.iloc[idx, -1]
    # print("{}: {}({})".format(files[idx], df.iloc[idx, -1], df.iloc[idx, -2]))
    # print("{}: {}({})".format(files[idx], df.iloc[idx, -1], df2.iloc[idx, -1]))

    base, ext = os.path.splitext(files[idx])
    shutil.copyfile(os.path.join(adv_dir, files[idx]), "{}/{}_adv{}".format(out_dir, base, ext))
    shutil.copyfile(os.path.join(src_dir, files[idx]), "{}/{}_src{}".format(out_dir, base, ext))
    shutil.copyfile(os.path.join(src_dir, comp_files[idx]), "{}/{}_cmp{}".format(out_dir, base, ext))
    shutil.copyfile(os.path.join(mask_dir, files[idx]), "{}/{}_cmpmask{}".format(out_dir, base, ext))
    print("{}: {}".format(files[idx], df.iloc[idx, -1]))
# print(filtered_files)
# print(len(filtered_files))

# for key, v in filtered_files.items():


# import pdb; pdb.set_trace()
# print(filtered_files.keys())

# cnt = []
# for file in filtered_files.keys():
#     tgt_dir = "/data/projects/aisc_facecomp/raw_data/game3000/{}_compare".format(os.path.splitext(file)[0])
#     cnt.append(len(os.listdir(tgt_dir)))


# for idx, c in zip(sorted_indice, cnt):
#     print("{}: {}({}), comp_cnt:{}".format(files[idx], df.iloc[idx, -1], df2.iloc[idx, -1], c))

# for idx in sorted_indice:
#     print(files[idx])
# import pdb; pdb.set_trace()