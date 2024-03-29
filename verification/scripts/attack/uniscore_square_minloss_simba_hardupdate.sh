
# train_images=500
# train_images=1000


# sample_list=(200 300 400 600 700 800 900 1000)
# sample_list=(300 700 900)
sample_list=(300)
for train_images in "${sample_list[@]}"
do    
    label=1
    # nohup rlaunch --cpu 2 --memory 16394 -- python -m attack.methods.uniscore_square_minloss_simba_hardupdate \
    # --i_net panorama.i_epoch_202622.th_0.8308.neupeak \
    # --train_images ${train_images} \
    # --max_epsilon 4 \
    # --n_queries 10000 \
    # --label ${label} &
    # nohup rlaunch --cpu 2 --memory 16394 -- python -m attack.methods.uniscore_square_minloss_simba_hardupdate \
    # --i_net panorama.i_epoch_202622.th_0.8308.neupeak \
    # --train_images ${train_images} \
    # --max_epsilon 8 \
    # --n_queries 10000 \
    # --label ${label} &
    # nohup rlaunch --cpu 2 --memory 16394 -- python -m attack.methods.uniscore_square_minloss_simba_hardupdate \
    # --i_net panorama.i_epoch_202622.th_0.8308.neupeak \
    # --train_images ${train_images} \
    # --max_epsilon 16 \
    # --n_queries 10000 \
    # --label ${label} &

    label=0
    # nohup rlaunch --cpu 2 --memory 16394 -- python -m attack.methods.uniscore_square_minloss_simba_hardupdate \
    # --i_net panorama.i_epoch_202622.th_0.8308.neupeak \
    # --train_images ${train_images} \
    # --max_epsilon 4 \
    # --n_queries 10000 \
    # --label ${label} &
    nohup rlaunch --cpu 2 --memory 16394 -- python -m attack.methods.uniscore_square_minloss_simba_hardupdate \
    --i_net panorama.i_epoch_202622.th_0.8308.neupeak \
    --train_images ${train_images} \
    --max_epsilon 8 \
    --n_queries 10000 \
    --label ${label} &
    # nohup rlaunch --cpu 2 --memory 16394 -- python -m attack.methods.uniscore_square_minloss_simba_hardupdate \
    # --i_net panorama.i_epoch_202622.th_0.8308.neupeak \
    # --train_images ${train_images} \
    # --max_epsilon 16 \
    # --n_queries 10000 \
    # --label ${label} &
done