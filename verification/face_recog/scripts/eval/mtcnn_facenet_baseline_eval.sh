
for m in 0 1 2 3 4 5
do
    for size in 112 160 224 299 350
    do
        python -m eval.mtcnn_facenet_baseline_eval \
        --margin $m \
        --img_size $size
    done
done