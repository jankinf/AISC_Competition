

# for model_idx in 9
# do
#     nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python -m raw_data.raw_lfw_eval \
#     --model_idx $model_idx \
#     --feature_save_path ./raw_data/mat/model${model_idx}_lfw_result.mat \
#     --outlog ./raw_data/logs/model${model_idx}_lfw_result.log &
# done
# for model_idx in 0 1 2 3 4 5 6 7 8 9
# for model_idx in 10 11 12 13
for model_idx in 14 15
do
    # CUDA_VISIBLE_DEVICES=0 python -m raw_data.raw_lfw_eval \
    nohup rlaunch --cpu 2 --memory 16394 --gpu 1 -- python -m raw_data.raw_lfw_eval \
    --model_idx $model_idx \
    --feature_save_path ./raw_data/mat/model${model_idx}_lfw_result.mat \
    --outlog ./raw_data/logs/model${model_idx}_lfw_result.log &
    sleep 5s
done
