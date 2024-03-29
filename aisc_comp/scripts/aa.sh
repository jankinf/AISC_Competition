attack=aa
eps_list=(1 2 4 8)
steps=30
for eps in ${eps_list[@]}
do
    CUDA_VISIBLE_DEVICES=4 python run.py \
    --attack_name $attack --batch_size 60 --max_epsilon $eps --steps $steps \
    --input_dir /data/public/data/2022aisc_facecomp/data/ \
    --output_dir ./results/${attack}_eps${eps}_steps${steps}_centralmask
done
