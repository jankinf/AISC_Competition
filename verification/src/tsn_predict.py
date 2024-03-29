from src.detector import CelebASpoofDetector
from src.models import AENet
import numpy as np
import torchvision
import torch
import os


def save_checkpoint(state, dirname, arch):
    filepath = os.path.join(dirname, arch + '.pth.tar')
    torch.save(state, filepath)


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def pretrain(model, state_dict):
    own_state = model.state_dict()
    for name, param in state_dict.items():
        realname = name.replace('module.', '')
        if realname in own_state:
            if isinstance(param, torch.nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            try:
                own_state[realname].copy_(param)
            except:
                print('While copying the parameter named {}, '
                      'whose dimensions in the model are {} and '
                      'whose dimensions in the checkpoint are {}.'
                      .format(realname, own_state[name].size(), param.size()))
                print("But don't worry about it. Continue pretraining.")


class TSNPredictor(CelebASpoofDetector):
    def __init__(self, ckpt='./ckpt_iter.pth.tar'):
        self.num_class = 2
        self.net = AENet(num_classes=self.num_class)
        checkpoint = torch.load(ckpt)

        pretrain(self.net, checkpoint['state_dict'])

        self.new_width = self.new_height = 224

        self.transform = torchvision.transforms.Compose([
            # torchvision.transforms.Resize((self.new_width, self.new_height)),
            torchvision.transforms.ToTensor(),
        ])

        self.net.cuda()
        self.net.eval()

    def preprocess_data(self, image):
        # processed_data = Image.fromarray(image)
        # processed_data = self.transform(image)
        processed_data = image.float() / 255.
        return processed_data

    def eval_image(self, image):
        with torch.no_grad():
            rst = self.net(image).detach()
        return rst.reshape(-1, self.num_class)

    def predict(self, images):
        # real_data = []
        # for image in images:
        #     data = self.preprocess_data(image)
        #     real_data.append(data)
        # real_data = self.preprocess_data(images)
        # rst = self.eval_image(real_data)
        rst = self.eval_image(images)
        rst = torch.nn.functional.softmax(rst, dim=1).cpu().numpy().copy()
        probability = np.array(rst)
        return probability
