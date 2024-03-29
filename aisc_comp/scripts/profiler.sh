# --alpha 0.2 \ 

# for alpha in 0.01 0.05 0.1 0.2 0.5 1.0 1.6 2.0 2.5 3 
# do
#     python profiler.py \
#     --steps 50 \
#     --alpha $alpha \
#     --model_w_idx 0 \
#     --model_b_idx 2 \
#     --device 0 >> out_profiler.log
# done

# for alpha in 0.01 0.05 0.1 0.2 0.5 1.0 1.6 2.0 2.5 3 
# do
#     python profiler.py \
#     --steps 50 \
#     --alpha $alpha \
#     --model_w_idx 0 \
#     --model_b_idx 3 \
#     --device 0 >> out_profiler.log
# done

# for alpha in 0.8 1.6 3.2 4.8
# for alpha in 3.2
# do
#     python profiler.py \
#     --steps 50 \
#     --alpha $alpha \
#     --model_w_idx 4 5 6 \
#     --model_b_idx 2 \
#     --device 0 >> out_profiler.log
# done
# for alpha in 3.2
# do
#     python profiler.py \
#     --steps 50 \
#     --alpha $alpha \
#     --model_w_idx 7 8 9 \
#     --model_b_idx 2 \
#     --device 0 >> out_profiler.log
# done
# for alpha in 3.2
# do
#     python profiler.py \
#     --steps 50 \
#     --alpha $alpha \
#     --model_w_idx 6 7 8 \
#     --model_b_idx 2 \
#     --device 0 >> out_profiler.log
# done
# for alpha in 3.2
# do
#     python profiler.py \
#     --steps 50 \
#     --alpha $alpha \
#     --model_w_idx 7 8 \
#     --model_b_idx 2 \
#     --device 0 >> out_profiler.log
# done
# for alpha in 3.2
# do
#     for idx in 0 1 3 4 5 6 7 8 9
#     do
#         python profiler.py \
#         --steps 50 \
#         --alpha $alpha \
#         --model_w_idx $idx \
#         --model_b_idx 2 \
#         --device 0 >> out_profiler.log
#     done
# done

# for alpha in 0.8 1.6 3.2 10
# do
#     python profiler.py \
#     --steps 30 \
#     --alpha $alpha \
#     --model_w_idx 0 \
#     --model_b_idx 2 \
#     --device 0 >> out_profiler.log
# done


# alpha=0.2 不用dim 效果最好



for alpha in 0.3
do
    python profiler.py \
    --steps 60 \
    --alpha $alpha \
    --mtype comp_keypoint \
    --hard_ctl \
    --model_w_idx 0 1 7 8 \
    --model_b_idx 3 4 5 \
    --device 0 >> out_profiler.log
done