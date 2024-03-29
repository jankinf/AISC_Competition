from util import central_mask, img_loader, patch5_mask, square_patch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
# img1 = central_mask()
# img2 = patch5_mask()
# im = np.array(Image.open("/data/projects/aisc_facecomp/data/0001.png")).astype(np.float32)
# for dtype in ['center', 'top', 'bottom']:
#     mask = square_patch(loc=dtype)
#     print(mask.sum() / mask.numel())
#     mask = mask[0].permute(1,2,0).numpy()
#     mask = 1 - mask
#     masked_im = (im * mask).astype(np.uint8)

#     Image.fromarray(masked_im).save("./mask/{}_0001.png".format(dtype))

# import pdb; pdb.set_trace()

def _hilbert(direction, rotation, order, move):
    if order == 0:
        return

    direction += rotation
    _hilbert(direction, -rotation, order - 1, move)
    step1(direction, move)

    direction -= rotation
    _hilbert(direction, rotation, order - 1, move)
    step1(direction, move)
    _hilbert(direction, rotation, order - 1, move)

    direction -= rotation
    step1(direction, move)
    _hilbert(direction, -rotation, order - 1, move)

def step1(direction, move=5):
    next = {0: (move, 0), 1: (0, move), 2: (-move, 0), 3: (0, -move)}[direction & 0x3]#取后两位？

    global x, y
    x.append(x[-1] + next[0])
    y.append(y[-1] + next[1])

def hilbert(order, move=5):
    global x, y
    x = [0,]
    y = [0,]
    _hilbert(0, 1, order, move)
    return (x, y)
  
    
# # https://www.pythonf.cn/read/173791
def move5():
    # x, y = hilbert(4, move=5)
    x = [0, 5, 5, 0, 0, 0, 5, 5, 10, 10, 15, 15, 15, 10, 10, 15, 20, 20, 25, 25, 30, 35, 35, 30, 30, 35, 35, 30, 25, 25, 20, 20, 20, 20, 25, 25, 30, 35, 35, 30, 30, 35, 35, 30, 25, 25, 20, 20, 15, 10, 10, 15, 15, 15, 10, 10, 5, 5, 0, 0, 0, 5, 5, 0, 0, 0, 5, 5, 10, 15, 15, 10, 10, 15, 15, 10, 5, 5, 0, 0, 0, 5, 5, 0, 0, 0, 5, 5, 10, 10, 15, 15, 15, 10, 10, 15, 20, 25, 25, 20, 20, 20, 25, 25, 30, 30, 35, 35, 35, 30, 30, 35, 35, 35, 30, 30, 25, 20, 20, 25, 25, 20, 20, 25, 30, 30, 35, 35, 40, 40, 45, 45, 50, 55, 55, 50, 50, 55, 55, 50, 45, 45, 40, 40, 40, 45, 45, 40, 40, 40, 45, 45, 50, 50, 55, 55, 55, 50, 50, 55, 60, 65, 65, 60, 60, 60, 65, 65, 70, 70, 75, 75, 75, 70, 70, 75, 75, 75, 70, 70, 65, 60, 60, 65, 65, 60, 60, 65, 70, 70, 75, 75, 75, 70, 70, 75, 75, 75, 70, 70, 65, 65, 60, 60, 60, 65, 65, 60, 55, 55, 50, 50, 45, 40, 40, 45, 45, 40, 40, 45, 50, 50, 55, 55, 55, 55, 50, 50, 45, 40, 40, 45, 45, 40, 40, 45, 50, 50, 55, 55, 60, 65, 65, 60, 60, 60, 65, 65, 70, 70, 75, 75, 75, 70, 70, 75]
    y = [0, 0, 5, 5, 10, 15, 15, 10, 10, 15, 15, 10, 5, 5, 0, 0, 0, 5, 5, 0, 0, 0, 5, 5, 10, 10, 15, 15, 15, 10, 10, 15, 20, 25, 25, 20, 20, 20, 25, 25, 30, 30, 35, 35, 35, 30, 30, 35, 35, 35, 30, 30, 25, 20, 20, 25, 25, 20, 20, 25, 30, 30, 35, 35, 40, 45, 45, 40, 40, 40, 45, 45, 50, 50, 55, 55, 55, 50, 50, 55, 60, 60, 65, 65, 70, 75, 75, 70, 70, 75, 75, 70, 65, 65, 60, 60, 60, 60, 65, 65, 70, 75, 75, 70, 70, 75, 75, 70, 65, 65, 60, 60, 55, 50, 50, 55, 55, 55, 50, 50, 45, 45, 40, 40, 40, 45, 45, 40, 40, 45, 45, 40, 40, 40, 45, 45, 50, 50, 55, 55, 55, 50, 50, 55, 60, 60, 65, 65, 70, 75, 75, 70, 70, 75, 75, 70, 65, 65, 60, 60, 60, 60, 65, 65, 70, 75, 75, 70, 70, 75, 75, 70, 65, 65, 60, 60, 55, 50, 50, 55, 55, 55, 50, 50, 45, 45, 40, 40, 40, 45, 45, 40, 35, 35, 30, 30, 25, 20, 20, 25, 25, 20, 20, 25, 30, 30, 35, 35, 35, 30, 30, 35, 35, 35, 30, 30, 25, 25, 20, 20, 20, 25, 25, 20, 15, 10, 10, 15, 15, 15, 10, 10, 5, 5, 0, 0, 0, 5, 5, 0, 0, 0, 5, 5, 10, 15, 15, 10, 10, 15, 15, 10, 5, 5, 0, 0]
    xs = np.array(x)
    ys = np.array(y)

    bd = xs.max()
    print(bd)
    img = np.zeros((bd + 1, bd + 1))
    for i in range(1, len(xs)):
        if xs[i] == xs[i-1]:
            if ys[i] > ys[i-1]:
                for j in range(ys[i-1], ys[i]+1):
                    img[xs[i], j] = 1
            else:
                for j in range(ys[i], ys[i-1]+1):
                    img[xs[i], j] = 1
        elif ys[i] == ys[i-1]:
            if xs[i] > xs[i-1]:
                for j in range(xs[i-1], xs[i]+1):
                    img[j, ys[i]] = 1
            else:
                for j in range(xs[i], xs[i-1]+1):
                    img[j, ys[i]] = 1

    img[:6, 0] = 0
    img[-6:, 0] = 0
    img[:5, 75] = 0
    img[-5:, 75] = 0

    print(img.sum())
    im = (img * 255).astype(np.uint8)
    Image.fromarray(im).save("move5.png")
    return img

def move7():
    n = 7
    # x, y = hilbert(4, move=7)
    x = [0, 7, 7, 0, 0, 0, 7, 7, 14, 14, 21, 21, 21, 14, 14, 21, 28, 28, 35, 35, 42, 49, 49, 42, 42, 49, 49, 42, 35, 35, 28, 28, 28, 28, 35, 35, 42, 49, 49, 42, 42, 49, 49, 42, 35, 35, 28, 28, 21, 14, 14, 21, 21, 21, 14, 14, 7, 7, 0, 0, 0, 7, 7, 0, 0, 0, 7, 7, 14, 21, 21, 14, 14, 21, 21, 14, 7, 7, 0, 0, 0, 7, 7, 0, 0, 0, 7, 7, 14, 14, 21, 21, 21, 14, 14, 21, 28, 35, 35, 28, 28, 28, 35, 35, 42, 42, 49, 49, 49, 42, 42, 49, 49, 49, 42, 42, 35, 28, 28, 35, 35, 28, 28, 35, 42, 42, 49, 49, 56, 56, 63, 63, 70, 77, 77, 70, 70, 77, 77, 70, 63, 63, 56, 56, 56, 63, 63, 56, 56, 56, 63, 63, 70, 70, 77, 77, 77, 70, 70, 77, 84, 91, 91, 84, 84, 84, 91, 91, 98, 98, 105, 105, 105, 98, 98, 105, 105, 105, 98, 98, 91, 84, 84, 91, 91, 84, 84, 91, 98, 98, 105, 105, 105, 98, 98, 105, 105, 105, 98, 98, 91, 91, 84, 84, 84, 91, 91, 84, 77, 77, 70, 70, 63, 56, 56, 63, 63, 56, 56, 63, 70, 70, 77, 77, 77, 77, 70, 70, 63, 56, 56, 63, 63, 56, 56, 63, 70, 70, 77, 77, 84, 91, 91, 84, 84, 84, 91, 91, 98, 98, 105, 105, 105, 98, 98, 105]
    y = [0, 0, 7, 7, 14, 21, 21, 14, 14, 21, 21, 14, 7, 7, 0, 0, 0, 7, 7, 0, 0, 0, 7, 7, 14, 14, 21, 21, 21, 14, 14, 21, 28, 35, 35, 28, 28, 28, 35, 35, 42, 42, 49, 49, 49, 42, 42, 49, 49, 49, 42, 42, 35, 28, 28, 35, 35, 28, 28, 35, 42, 42, 49, 49, 56, 63, 63, 56, 56, 56, 63, 63, 70, 70, 77, 77, 77, 70, 70, 77, 84, 84, 91, 91, 98, 105, 105, 98, 98, 105, 105, 98, 91, 91, 84, 84, 84, 84, 91, 91, 98, 105, 105, 98, 98, 105, 105, 98, 91, 91, 84, 84, 77, 70, 70, 77, 77, 77, 70, 70, 63, 63, 56, 56, 56, 63, 63, 56, 56, 63, 63, 56, 56, 56, 63, 63, 70, 70, 77, 77, 77, 70, 70, 77, 84, 84, 91, 91, 98, 105, 105, 98, 98, 105, 105, 98, 91, 91, 84, 84, 84, 84, 91, 91, 98, 105, 105, 98, 98, 105, 105, 98, 91, 91, 84, 84, 77, 70, 70, 77, 77, 77, 70, 70, 63, 63, 56, 56, 56, 63, 63, 56, 49, 49, 42, 42, 35, 28, 28, 35, 35, 28, 28, 35, 42, 42, 49, 49, 49, 42, 42, 49, 49, 49, 42, 42, 35, 35, 28, 28, 28, 35, 35, 28, 21, 14, 14, 21, 21, 21, 14, 14, 7, 7, 0, 0, 0, 7, 7, 0, 0, 0, 7, 7, 14, 21, 21, 14, 14, 21, 21, 14, 7, 7, 0, 0]
    xs = np.array(x)
    ys = np.array(y)

    bd = xs.max()
    print(bd)
    img = np.zeros((bd + 1, bd + 1))
    for i in range(1, len(xs)):
        if xs[i] == xs[i-1]:
            if ys[i] > ys[i-1]:
                for j in range(ys[i-1], ys[i]+1):
                    img[xs[i], j] = 1
            else:
                for j in range(ys[i], ys[i-1]+1):
                    img[xs[i], j] = 1
        elif ys[i] == ys[i-1]:
            if xs[i] > xs[i-1]:
                for j in range(xs[i-1], xs[i]+1):
                    img[j, ys[i]] = 1
            else:
                for j in range(xs[i], xs[i-1]+1):
                    img[j, ys[i]] = 1

    margin = n * 4
    img[:margin, :margin] = 0
    img[-margin:, :margin] = 0
    img[:margin, -margin:] = 0
    img[-margin:, -margin:] = 0

    blockh, blockw = n * 5, n
    img[:blockh, :blockw] = 0
    img[-blockh:, :blockw] = 0
    img[:blockw, -blockh:] = 0
    img[-blockw:, -blockh:] = 0

    blockh, blockw = n * 5 + 1, n + 1
    img[:blockh+1, :blockw] = 0
    img[-blockh:, :blockw] = 0
    img[:blockw, -blockh:] = 0
    img[-blockw-1:, -blockh:] = 0

    img[49:56, 49] = 1
    img[49:56, 56] = 1
    img[49, 49:56] = 1
    img[56, 49:56] = 1

    print(img.sum())
    im = (img * 255).astype(np.uint8)
    Image.fromarray(im).save("move7.png")
    return img

if __name__=='__main__':
    # move7()
    # _mask = move5()
    _mask = move7()

    h, w = _mask.shape
    start_x, start_y = (112 - h) // 2, (112 - w) // 2 

    
    mask = np.zeros((112, 112, 3))
    

    mask[start_x:start_x+h, start_y:start_y+w, :] = np.concatenate([np.expand_dims(_mask, -1)] * 3, 2)
    mask = 1 - mask

    for i in range(3000):
        im = np.array(Image.open("/data/projects/aisc_facecomp/data/{:04d}.png".format(i + 1))).astype(np.float32)
        masked_im = (im * mask).astype(np.uint8)
        Image.fromarray(masked_im).save("masks/hilbert@m7/{:04d}.png".format(i + 1))

    # Image.fromarray(masked_im).save("./masks/hilbert@m5_0001.png")
    # Image.fromarray(masked_im).save("./masks/hilbert@m7_0001.png")
    # import pdb; pdb.set_trace()

