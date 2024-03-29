# conda activate py36_megbrain

nohup rlaunch --cpu 2 --memory 16394 -- python3 main.py \
	--i_net panorama.i_epoch_202622.th_0.8308.neupeak \
	--image /data/datasets/fmp/imgs/attack/2019-07-22_2019-07-28_attack.nori.list \
	--logfile 1000.log \
	--max_images 1000 &

nohup rlaunch --cpu 2 --memory 16394 -- python3 main.py \
 	--i_net panorama.i_epoch_202622.th_0.8308.neupeak \
 	--image /data/datasets/fmp/imgs/normal/2019-07-22_2019-07-28_normal.nori.list \
 	--logfile 1000.log \
 	--max_images 1000 &

nohup rlaunch --cpu 2 --memory 16394 -- python3 main.py \
	--i_net panorama.i_epoch_202622.th_0.8308.neupeak \
	--image /data/datasets/fmp/imgs/attack/2019-07-22_2019-07-28_attack.nori.list \
	--logfile 500.log \
	--max_images 500 &

nohup rlaunch --cpu 2 --memory 16394 -- python3 main.py \
 	--i_net panorama.i_epoch_202622.th_0.8308.neupeak \
 	--image /data/datasets/fmp/imgs/normal/2019-07-22_2019-07-28_normal.nori.list \
 	--logfile 500.log \
 	--max_images 500 &

