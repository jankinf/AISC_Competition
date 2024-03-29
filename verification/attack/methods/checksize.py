from cmath import inf
import os
import cv2

def get_img_list(path):
    if os.path.isfile(path):
        img_list = [path]
    else:
        img_list = [os.path.join(path, v) for v in os.listdir(path)]
    sorted(img_list)
    return img_list


def main():
    normal_list = get_img_list(normal_dir)
    attack_list = get_img_list(attack_dir)

    logfile = '/data/projects/fmp_demo/attack/methods/size.log'
    fp = open(logfile, 'a')
    min_w, min_h = inf, inf
    for i, name in enumerate(normal_list):
        img = cv2.imread(name)
        logdata = "id:{}, file:{}, size:{}\n".format(i, name, img.shape)
        min_h = min(min_h, img.shape[0])
        min_w = min(min_w, img.shape[1])
        print(logdata, end='')
        fp.write(logdata)
        fp.flush()

    for i, name in enumerate(attack_list):
        img = cv2.imread(name)
        logdata = "id:{}, file:{}, size:{}\n".format(i, name, img.shape)
        min_h = min(min_h, img.shape[0])
        min_w = min(min_w, img.shape[1])
        print(logdata, end='')
        fp.write(logdata)
        fp.flush()

    print(min_h, min_w)    

    fp.close()


if __name__ == "__main__":
    normal_dir = '/data/datasets/fmp/imgs/normal/2019-07-22_2019-07-28_normal.nori.list'
    attack_dir = '/data/datasets/fmp/imgs/attack/2019-07-22_2019-07-28_attack.nori.list'
    main()
