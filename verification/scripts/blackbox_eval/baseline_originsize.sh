# conda activate py36_megbrain

# whitebox panorama.i_epoch_202622.th_0.8308.neupeak
# blackbox panorama.i_epoch_208528.th_0.0162.neupeak
# python3 main.py \
# 	--i_net panorama.i_epoch_208528.th_0.0162.neupeak \
# 	--image /data/datasets/fmp/imgs/attack/2019-07-22_2019-07-28_attack.nori.list \
# 	--logfile 1000_blackbox.log \
# 	--max_images 1000

nohup rlaunch --cpu 1 --memory 8394 -- python3 main.py \
	--i_net panorama.i_epoch_208528.th_0.0162.neupeak \
	--image /data/datasets/fmp/imgs/attack/2019-07-22_2019-07-28_attack.nori.list \
	--logfile 1000_blackbox.log \
	--max_images 1000 &

nohup rlaunch --cpu 1 --memory 8394 -- python3 main.py \
 	--i_net panorama.i_epoch_208528.th_0.0162.neupeak \
 	--image /data/datasets/fmp/imgs/normal/2019-07-22_2019-07-28_normal.nori.list \
 	--logfile 1000_blackbox.log \
 	--max_images 1000 &

# nohup rlaunch --cpu 1 --memory 8394 -- python3 main.py \
# 	--i_net panorama.i_epoch_208528.th_0.0162.neupeak \
# 	--image /data/datasets/fmp/imgs/attack/2019-07-22_2019-07-28_attack.nori.list \
# 	--logfile 500_blackbox.log \
# 	--max_images 500 &

# nohup rlaunch --cpu 1 --memory 8394 -- python3 main.py \
#  	--i_net panorama.i_epoch_208528.th_0.0162.neupeak \
#  	--image /data/datasets/fmp/imgs/normal/2019-07-22_2019-07-28_normal.nori.list \
#  	--logfile 500_blackbox.log \
#  	--max_images 500 &

