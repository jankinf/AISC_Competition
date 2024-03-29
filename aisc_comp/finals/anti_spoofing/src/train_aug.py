"""
This script provides a local test routine so you can verify the algorithm works before pushing it to evaluation.

It runs your detector on several local images and verify whether they have obvious issues, e.g:
    - Fail to start
    - Wrong output format

It also prints out the runtime for the algorithms for your references.


The participants are expected to implement a face forgery detector class. The sample detector illustrates the interface.
Do not modify other part of the evaluation toolkit otherwise the evaluation will fail.

Author: Yuanjun Xiong, Zhengkui Guo, Yuanhan Zhang
Contact: celebaspoof@gmail.com

CelebA-Spoof
"""

import torchvision.transforms as transforms
from torch.nn import CrossEntropyLoss
from torchvision.models import resnet18
from attack.attack_dataset import show
from torch.optim import SGD, Adam
import numpy as np
import logging
import random
import torch
import argparse
import os
from tqdm import tqdm
from PIL import Image
from io import BytesIO
import pandas as pd
from src.client import verify_output
from src.dataset import get_test_dataloader, get_train_dataloader
from src.tsn_predict import AverageMeter
from src.models import AENet
logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description='AENet training in PyTorch')
parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0, help='learning rate')
parser.add_argument('--momentum', default=0, type=float, help='momentum of optim')
parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size')
parser.add_argument('--num_workers', type=int, default=2, help='number of workers of dataloader')
parser.add_argument('--optim_type', type=str, choices=['adam', 'sgd'], default='sgd')
parser.add_argument('--ckpt_dir', type=str, default='/data/projects/CelebA-Spoof/ckpts')
parser.add_argument('--schedule', type=int, nargs='+', default=[])
parser.add_argument('--arch', type=str, default='resnet18')
parser.add_argument('--train_cropped', action="store_true")
parser.add_argument('--test_cropped', action="store_true")
args = parser.parse_args()

def pretrain(model, state_dict):
    own_state = model.state_dict()
    for name, param in state_dict.items():
        realname = name
        # realname = name.replace('module.', '')
        if realname in own_state:
            if isinstance(param, torch.nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            try:
                own_state[realname].copy_(param)
            except BaseException:
                print('While copying the parameter named {}, '
                      'whose dimensions in the model are {} and '
                      'whose dimensions in the checkpoint are {}.'
                      .format(realname, own_state[name].size(), param.size()))
                print("But don't worry about it. Continue pretraining.")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def run_train(image_iter, model, optimizer, criterion):
    losses = AverageMeter()
    accs = AverageMeter()
    model.train()

    for data in tqdm(image_iter):
        image_id, image, label = data['filepath'], data['img'], data['label']
        image, label = image.cuda(), label.cuda()
        # show(image, "jpeg_debug")
        # import pdb; pdb.set_trace()
        rst = model(image)
        loss = criterion(rst, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = (rst.argmax(-1) == label).float().mean()
        losses.update(loss.item(), image.size(0))
        accs.update(acc.item(), image.size(0))

    return losses.avg, accs.avg


def run_test(model, image_iter):
    output_probs = {}
    logging.info("Starting runtime evaluation")
    model.eval()
    with torch.no_grad():
        for data in tqdm(image_iter):
            # image_id, image
            image_id, image, label = data['filepath'], data['img'], data['label']
            image, label = image.cuda(), label.cuda()
            try:
                rst = model(image).detach()
                prob = torch.nn.functional.softmax(rst, dim=1).cpu().numpy()
                for idx, i in enumerate(image_id):
                    output_probs[i] = float(prob[idx][1])
            except BaseException:
                # send errors to the eval frontend
                logging.error("Image id failed: {}".format(image_id))
                raise

    logging.info("""
    ================================================================================
    All images finished, showing verification info below:
    ================================================================================
    """)

    # verify the algorithm output
    tpr_list = verify_output(output_probs)
    return tpr_list

def schedule_lr(optimizer):
    for params in optimizer.param_groups:
        params['lr'] /= 10.

    print(optimizer)

def randomJPEGcompression(image, p=0.4):
    if np.random.rand() > p: 
        return image
    qf = random.randrange(40, 100)
    outputIoStream = BytesIO()
    image.save(outputIoStream, "JPEG", quality=qf, optimice=True)
    outputIoStream.seek(0)
    return Image.open(outputIoStream)

def main():
    dirname = os.path.join(args.ckpt_dir, args.arch)
    os.makedirs(dirname, exist_ok=True)
    model = AENet()
    model.cuda()

    # todo: load pretrain params from official resnet18
    pretrained_model = resnet18(pretrained=True)
    state_dict = pretrained_model.state_dict()
    pretrain(model, state_dict)
    model = torch.nn.DataParallel(model)
    criterion = CrossEntropyLoss()
    if args.optim_type == 'sgd':
        optimizer = SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    elif args.optim_type == 'adam':
        optimizer = Adam(model.parameters(), lr=args.lr)
    else:
        raise TypeError
    
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=(0.9, 1.5), hue=.1),
        transforms.Lambda(randomJPEGcompression),
        transforms.Resize([int(128 * INPUT_SIZE[0] / 112), int(128 * INPUT_SIZE[0] / 112)]), # smaller side resized
        transforms.RandomCrop([INPUT_SIZE[0], INPUT_SIZE[1]]),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
    ])
    train_iter = get_train_dataloader(transform=transform_train, batch_size=args.batch_size, num_workers=args.num_workers, cropped=args.train_cropped)
    test_iter = get_test_dataloader(transform=transform_test, batch_size=args.batch_size, num_workers=args.num_workers, cropped=args.test_cropped)
    tpr_res = []
    logging.info("Starting training")
    for epoch in range(50):
        if epoch == 0:
            # todo: resume
            filepath = os.path.join(dirname, f"{epoch + 1}.pth.tar")
            state_dict = torch.load(filepath)
            model.load_state_dict(state_dict['state_dict'])
            print('epoch: {}, loss: {}, acc: {}'.format(epoch, state_dict['loss'], state_dict['acc']))
        else:
            if epoch + 1 in STAGES:
                schedule_lr(optimizer)
            train_loss, train_acc = run_train(train_iter, model, optimizer, criterion)
            print('epoch: {}, loss: {}, acc: {}'.format(epoch, train_loss, train_acc))

            filepath = os.path.join(dirname, f"{epoch + 1}.pth.tar")
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'acc': train_acc,
                'loss': train_loss,
                'optimizer': optimizer.state_dict(),
            }, filepath)
        # todo: update csv per epoch
        tpr_list = run_test(model, test_iter)
        item = [epoch + 1] + tpr_list
        tpr_res.append(item)
        df = pd.DataFrame(np.array(tpr_res), columns=[
            "Epoch",
            "TPR@FPR=10E-3",
            "TPR@FPR=5E-3",
            "TPR@FPR=10E-4",
        ])
        df.to_csv(os.path.join(dirname, "tpr_res.csv"), index=False)

    logging.info("Finished training")


if __name__ == '__main__':
    INPUT_SIZE = (224, 224)
    STAGES = args.schedule
    set_seed(0)
    main()
