
# label=1
# sample_list=(200 300 400 500)
sample_list=(500)
#memory>>> 200:16394 300:20000 400: 26394; 500: 30000
for train_images in "${sample_list[@]}"
do    
    label=0
    nohup rlaunch --cpu 2 --memory 30000 -- python -m attack.methods.uniscore_square_640@480 \
    --i_net panorama.i_epoch_202622.th_0.8308.neupeak \
    --train_images ${train_images} \
    --max_epsilon 3 \
    --n_queries 10000 \
    --label ${label} &
    nohup rlaunch --cpu 2 --memory 30000 -- python -m attack.methods.uniscore_square_640@480 \
    --i_net panorama.i_epoch_202622.th_0.8308.neupeak \
    --train_images ${train_images} \
    --max_epsilon 4 \
    --n_queries 10000 \
    --label ${label} &
    nohup rlaunch --cpu 2 --memory 30000 -- python -m attack.methods.uniscore_square_640@480 \
    --i_net panorama.i_epoch_202622.th_0.8308.neupeak \
    --train_images ${train_images} \
    --max_epsilon 5 \
    --n_queries 10000 \
    --label ${label} &
    nohup rlaunch --cpu 2 --memory 30000 -- python -m attack.methods.uniscore_square_640@480 \
    --i_net panorama.i_epoch_202622.th_0.8308.neupeak \
    --train_images ${train_images} \
    --max_epsilon 6 \
    --n_queries 10000 \
    --label ${label} &

    label=1
    nohup rlaunch --cpu 2 --memory 30000 -- python -m attack.methods.uniscore_square_640@480 \
    --i_net panorama.i_epoch_202622.th_0.8308.neupeak \
    --train_images ${train_images} \
    --max_epsilon 3 \
    --n_queries 10000 \
    --label ${label} &
    nohup rlaunch --cpu 2 --memory 30000 -- python -m attack.methods.uniscore_square_640@480 \
    --i_net panorama.i_epoch_202622.th_0.8308.neupeak \
    --train_images ${train_images} \
    --max_epsilon 4 \
    --n_queries 10000 \
    --label ${label} &
    nohup rlaunch --cpu 2 --memory 30000 -- python -m attack.methods.uniscore_square_640@480 \
    --i_net panorama.i_epoch_202622.th_0.8308.neupeak \
    --train_images ${train_images} \
    --max_epsilon 5 \
    --n_queries 10000 \
    --label ${label} &
    nohup rlaunch --cpu 2 --memory 30000 -- python -m attack.methods.uniscore_square_640@480 \
    --i_net panorama.i_epoch_202622.th_0.8308.neupeak \
    --train_images ${train_images} \
    --max_epsilon 6 \
    --n_queries 10000 \
    --label ${label} &
done