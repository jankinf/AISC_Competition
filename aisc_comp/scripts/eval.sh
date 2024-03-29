
paths=(
<<<<<<< HEAD
    /data/projects/aisc_facecomp/results/optmask_clip_pict_eps40_steps50
    /data/projects/aisc_facecomp/results/optmask_ct_eps40_steps50
    /data/projects/aisc_facecomp/results/optmask_dem_eps40_steps20
    /data/projects/aisc_facecomp/results/optmask_pict_eps40_steps20_ens10
    /data/projects/aisc_facecomp/results/optmask_pict_eps40_steps50
    /data/projects/aisc_facecomp/data
=======
    # /data/projects/aisc_facecomp/results/optmask_dem_eps40_steps20
    # /data/projects/aisc_facecomp/results/optmask_clip_pict_eps40_steps50
    # /data/projects/aisc_facecomp/results/optmask_ct_eps40_steps50
    # /data/projects/aisc_facecomp/data
    # /data/projects/aisc_facecomp/results/optmask_pict_eps40_steps20_ens10
    # /data/projects/aisc_facecomp/results/optmask_pict_eps40_steps50

    # /data/projects/aisc_facecomp/results/optmask_pict_eps40_steps20_ens4_model
    # /data/projects/aisc_facecomp/results/optmask_pict_eps40_steps20_ens3_model017
    # /data/projects/aisc_facecomp/results/optmask_pict_eps40_steps20_ens3_model018
    /data/projects/aisc_facecomp/results/optmask_pict_eps40_steps20_ens3_model078
    # /data/projects/aisc_facecomp/results/optmask_pict_eps40_steps20_ens3_model178
    # /data/projects/aisc_facecomp/results/optmask_pict_eps40_steps20_ens2_model78
    # /data/projects/aisc_facecomp/results/optmask_pict_eps40_steps20_ens2_model07
    # /data/projects/aisc_facecomp/results/optmask_pict_eps40_steps20_ens2_model08
    # /data/projects/aisc_facecomp/results/optmask_pict_eps40_steps20_ens2_model17
    # /data/projects/aisc_facecomp/results/optmask_pict_eps40_steps20_ens2_model18
>>>>>>> 4c3e810eeaef789e3fd57bb9506d1828d0e58cb0
)
for path in ${paths[@]}
do
    nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python eval.py \
    --input_dir $path \
    --device 0 &
done
<<<<<<< HEAD


# scp -r /data/projects/aisc_facecomp/results/optmask_clip_pict_eps40_steps50 jankinf.fangzhengwei.megvii-csg.ws@hh-d.brainpp.cn:/data/projects/aisc_facecomp/results/optmask_clip_pict_eps40_steps50 &
# scp -r /data/projects/aisc_facecomp/results/optmask_ct_eps40_steps50 jankinf.fangzhengwei.megvii-csg.ws@hh-d.brainpp.cn:/data/projects/aisc_facecomp/results/optmask_ct_eps40_steps50 &
# scp -r /data/projects/aisc_facecomp/results/optmask_dem_eps40_steps20 jankinf.fangzhengwei.megvii-csg.ws@hh-d.brainpp.cn:/data/projects/aisc_facecomp/results/optmask_dem_eps40_steps20 &
# scp -r /data/projects/aisc_facecomp/results/optmask_pict_eps40_steps20_ens10 jankinf.fangzhengwei.megvii-csg.ws@hh-d.brainpp.cn:/data/projects/aisc_facecomp/results/optmask_pict_eps40_steps20_ens10 &
# scp -r /data/projects/aisc_facecomp/results/optmask_pict_eps40_steps50 jankinf.fangzhengwei.megvii-csg.ws@hh-d.brainpp.cn:/data/projects/aisc_facecomp/results/optmask_pict_eps40_steps50 &
=======
>>>>>>> 4c3e810eeaef789e3fd57bb9506d1828d0e58cb0
