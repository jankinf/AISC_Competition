# conda activate py36_megbrain

# nohup rlaunch --cpu 2 --memory 16394 -- python3 preprocess.py \
# 	--i_net panorama.i_epoch_202622.th_0.8308.neupeak \
# 	--image /data/projects/fmp_demo/attack/adv_data/preprocess224/attack \
#     --mode attack \
#     --name preprocess224_loop \
#     --max_images 1000 & 

# nohup rlaunch --cpu 2 --memory 16394 -- python3 preprocess.py \
#  	--i_net panorama.i_epoch_202622.th_0.8308.neupeak \
#  	--image /data/projects/fmp_demo/attack/adv_data/preprocess224/normal \
#     --mode normal \
#     --name preprocess224_loop \
#     --max_images 1000 & 


# nohup rlaunch --cpu 2 --memory 16394 -- python3 preprocess.py \
# 	--i_net panorama.i_epoch_202622.th_0.8308.neupeak \
# 	--image /data/datasets/fmp/imgs/attack/2019-07-22_2019-07-28_attack.nori.list \
#     --max_images 1000 & 

# nohup rlaunch --cpu 2 --memory 16394 -- python3 preprocess.py \
#  	--i_net panorama.i_epoch_202622.th_0.8308.neupeak \
#  	--image /data/datasets/fmp/imgs/normal/2019-07-22_2019-07-28_normal.nori.list \
#     --max_images 1000 & 


nohup rlaunch --cpu 2 --memory 16394 -- python3 preprocess.py \
	--i_net panorama.i_epoch_202622.th_0.8308.neupeak \
	--image /data/projects/fmp_demo/attack/adv_data/square_t0/iter5000_eps8/attack \
    --mode attack \
    --name preprocess224_square_t0_eps8_attack \
    --max_images 1000 & 

nohup rlaunch --cpu 2 --memory 16394 -- python3 preprocess.py \
 	--i_net panorama.i_epoch_202622.th_0.8308.neupeak \
 	--image /data/projects/fmp_demo/attack/adv_data/square_t1/iter5000_eps8/normal \
    --mode normal \
    --name preprocess224_square_t1_eps8_normal \
    --max_images 1000 & 