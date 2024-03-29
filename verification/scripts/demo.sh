# conda activate py36_megbrain

python3 demo.py \
	--i_net panorama.i_epoch_202622.th_0.8308.neupeak \
	--image /data/datasets/fmp/imgs/attack/2019-07-22_2019-07-28_attack.nori.list \
	--max_images 5

# python3 demo.py \
#  	--i_net panorama.i_epoch_202622.th_0.8308.neupeak \
#  	--image /data/datasets/fmp/imgs/normal/2019-07-22_2019-07-28_normal.nori.list \
#  	--max_images 0