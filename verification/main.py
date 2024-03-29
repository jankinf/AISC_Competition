import os
import cv2
import argparse
import inference


def run_imgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--i_net", type=str, help="image network path")
    parser.add_argument("--image", type=str)
    parser.add_argument("--thres", type=float, default=0.5)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--logfile", type=str)
    args = parser.parse_args()
    
    r = inference.RunnableInference(model_path=args.i_net, device="cpu0", alpha=3)
    if os.path.isfile(args.image):
        img_list = [args.image]
    else:
        img_list = [os.path.join(args.image, v) for v in os.listdir(args.image)]
    cnt = 0
    err = 0
    sorted(img_list)

    if args.max_images > 0:
        img_list = img_list[:args.max_images]

    attack_id, normal_id = [], []
    for i, name in enumerate(img_list[:]):
        try:
            img = cv2.imread(name)
            print("img shape:", img.shape)
            score, cimg = r.run_image(img)
            print("score:", score, name)

            # print(img.shape, cimg.shape)
            # print(img.min(), img.max(), cimg.min(), cimg.max())
            # cv2.imwrite("img.jpg", img)
            # cv2.imwrite("cimg.jpg", cimg)
            if score > args.thres:
                cnt += 1
                attack_id.append(i)
            else:
                normal_id.append(i)
        except Exception as e:
            err += 1
            print(e)
            continue
    
    if args.image.split('/')[-1] in ["attack", "normal"]:
        if args.image.split('/')[-1] == "attack":
            attack = True
        else:
            attack = False
    else:
        attack = "attack" in args.image
        
    metrics = "tpr" if attack else "fpr"
    acc = cnt / (i + 1 - err)
    info = "total:{}, attack:{}, error:{}, {}:{}".format(
        i + 1, cnt, err, metrics, acc
    )
    print(info)
    logfile = os.path.join("logs/eval", args.logfile)
    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    with open(logfile, "a") as f:
        f.write(f'params: {str(args)}\n')
        f.write(f'eval {args.image}\n')
        f.write(f'attack ids: {attack_id}\n')
        f.write(f'normal ids: {normal_id}\n')
        f.write(f'{info}\n\n')


if __name__ == "__main__":
    run_imgs()
