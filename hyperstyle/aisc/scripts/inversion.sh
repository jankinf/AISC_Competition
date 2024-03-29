
paths=(
    /data/projects/hyperstyle/aisc/aligned/ht_glasses.png
    # /data/projects/aisc_facecomp/finals/keypoints/aisc_self/ht_glasses.png
    # /data/projects/aisc_facecomp/finals/keypoints/aisc_self/ht.png
    # /data/projects/aisc_facecomp/finals/keypoints/aisc_self/liyi_glasses.png
    # /data/projects/aisc_facecomp/finals/keypoints/aisc_self/liyi.png
    # /data/projects/aisc_facecomp/finals/keypoints/aisc_self/qinyi02.png
    # /data/projects/aisc_facecomp/finals/keypoints/aisc_self/qinyi.png
    # /data/projects/aisc_facecomp/finals/keypoints/aisc_self/shibo2.png
    # /data/projects/aisc_facecomp/finals/keypoints/aisc_self/shibo.png
)
for path in ${paths[@]}
do
    # nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python -m scripts.attack --image_path $path --outputs_path "./outputs_hq_resize256" &
    # sleep 5s
    python -m aisc.inversion --image_path $path --outputs_path "/data/projects/hyperstyle/aisc/inversion"
done