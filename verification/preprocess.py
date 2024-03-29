import os
import cv2
import argparse
import inference



def run_imgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--i_net", type=str, default="panorama.i_epoch_202622.th_0.8308.neupeak", help="image network path")
    parser.add_argument("--image", type=str)
    parser.add_argument("--mode", default="", type=str)
    parser.add_argument("--name", default="preprocess224", type=str)
    parser.add_argument("--thres", type=float, default=0.5)
    parser.add_argument("--max_images", type=int, default=0)
    args = parser.parse_args()
    
    r = inference.RunnableInference(model_path=args.i_net, device="cpu0", alpha=3)
    if os.path.isfile(args.image):
        img_list = [args.image]
    else:
        img_list = [os.path.join(args.image, v) for v in os.listdir(args.image)]
    
    err = 0
    sorted(img_list)

    if args.max_images > 0:
        img_list = img_list[:args.max_images]
    
    if args.mode:
        mode = args.mode
    else:
        mode = "atack" if "attack" in args.image else "normal"
    output_dir = "/data/projects/fmp_demo/attack/adv_data/{}/{}".format(args.name, mode)
    print(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    for i, name in enumerate(img_list[:]):
        img_file = os.path.join(output_dir, os.path.basename(name))
        try:
            img = cv2.imread(name)
            print("img shape:", img.shape)
            cimg = r.preprocess(img)

            print(img.shape, cimg.shape)
            print(img.min(), img.max(), cimg.min(), cimg.max())
            cv2.imwrite(img_file, cimg)
            
        except Exception as e:
            err += 1
            print(e)
            continue

if __name__ == "__main__":
    run_imgs()
