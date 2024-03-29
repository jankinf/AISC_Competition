
paths=(
    /data/projects/hyperstyle/aisc/aligned/ht_glasses.png
    /data/projects/hyperstyle/aisc/aligned/ht.png
    /data/projects/hyperstyle/aisc/aligned/qinyi2.png
    /data/projects/hyperstyle/aisc/aligned/qinyi.png
    # /data/projects/hyperstyle/aisc/aligned/liyi_glasses.png
    # /data/projects/hyperstyle/aisc/aligned/liyi.png
    # /data/projects/hyperstyle/aisc/aligned/shibo2.png
    # /data/projects/hyperstyle/aisc/aligned/shibo.png
)
for path in ${paths[@]}
do
    # nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python -m scripts.attack --image_path $path --outputs_path "./outputs_hq_resize256" &
    # sleep 5s
    python -m aisc.decode --image_path $path --outputs_path "/data/projects/hyperstyle/aisc/decode"
done