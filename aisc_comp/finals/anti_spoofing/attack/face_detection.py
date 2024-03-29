from facenet_pytorch import MTCNN
from PIL import Image, ImageDraw
import os
'''
https://github.com/timesler/facenet-pytorch
pip install facenet-pytorch
'''

class GetFrame(MTCNN):
    def draw_rectangle(self, img:Image.Image, box, save_path=None):
        raw_image_size = img.size
        box = [
            int(max(box[1], 0)),
            int(max(box[0], 0)),
            int(min(box[3], raw_image_size[1])),
            int(min(box[2], raw_image_size[0])),
        ]

        shape = [box[1], box[0], box[3], box[2]]
        pen = ImageDraw.Draw(img)
        pen.rectangle(shape, outline="red")

        if save_path is not None:
            os.makedirs(os.path.dirname(save_path) + "/", exist_ok=True)
            img.save(save_path)

        return img

    def forward(self, img, save_path=None, return_prob=False):
        """only support single PIL.Image.Image object"""
        assert isinstance(img, Image.Image)
        # Detect faces
        batch_boxes, batch_probs, batch_points = self.detect(img, landmarks=True)
        # Select faces
        if not self.keep_all:
            batch_boxes, batch_probs, batch_points = self.select_boxes(
                batch_boxes, batch_probs, batch_points, img, method=self.selection_method
            )
        # Extract faces
        faces = self.draw_rectangle(img, batch_boxes[0], save_path)

        if return_prob:
            return faces, batch_boxes[0], batch_probs
        else:
            return faces, batch_boxes[0]

def _test():
    # mtcnn = MTCNN(image_size=160, margin=0)
    mtcnn = GetFrame()

    img_path = "/data/projects/Megvii-spoof/out_tnt/1_1.jpg"
    img = Image.open(img_path)
    result = mtcnn(img, save_path="./debug.jpg")


if __name__ == '__main__':
    # _test()
    # mtcnn = MTCNN(image_size=160, margin=0)
    mtcnn = GetFrame()

    inp_dir = "/data/projects/CelebA-Spoof/attack/spoof_data/out_png"
    out_dir = "/data/projects/CelebA-Spoof/attack/spoof_data/out_png_detect"
    files = os.listdir(inp_dir)
    for file in files:
        img_path = os.path.join(inp_dir, file)
        img = Image.open(img_path)
        try:
            mtcnn(img, save_path=os.path.join(out_dir, file))
        except Exception:
            import traceback
            traceback.print_exc()
            print(file)