attack=mim
eps_list=(512)
steps=50
for eps in ${eps_list[@]}
do
    CUDA_VISIBLE_DEVICES=3 python run.py \
    --attack_name $attack --batch_size 120 --max_epsilon $eps --steps $steps \
    --input_dir /data/public/data/2022aisc_facecomp/data/ \
    --output_dir ./results/${attack}_eps${eps}_steps${steps}_patch5mask
done
