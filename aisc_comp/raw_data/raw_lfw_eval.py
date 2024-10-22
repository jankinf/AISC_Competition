from ast import arg
import numpy as np
import scipy.io
import os
import torch.utils.data
from util import RAW_LFW as LFW
import torchvision.transforms as transforms
import argparse
# from util import get_model
from util import get_fastmodel

def getAccuracy(scores, flags, threshold):
    p = np.sum(scores[flags == 1] > threshold)
    n = np.sum(scores[flags == -1] < threshold)
    return 1.0 * (p + n) / len(scores)

def getThreshold(scores, flags, thrNum):
    accuracys = np.zeros((2 * thrNum + 1, 1))
    thresholds = np.arange(-thrNum, thrNum + 1) * 1.0 / thrNum # (-1, 1) 10000
    for i in range(2 * thrNum + 1):
        accuracys[i] = getAccuracy(scores, flags, thresholds[i])
    max_index = np.squeeze(accuracys == np.max(accuracys))
    bestThreshold = np.mean(thresholds[max_index])
    return bestThreshold

def evaluation_10_fold(feature_path='./cur_epoch_result.mat'):
    ACCs = np.zeros(10)
    Thresholds = np.zeros(10)
    result = scipy.io.loadmat(feature_path)
    for i in range(10):
        fold = result['fold']
        flags = result['flag']
        featureLs = result['fl']
        featureRs = result['fr']

        valFold = fold != i
        testFold = fold == i
        flags = np.squeeze(flags)


        # if args.model_idx != 10 and args.model_idx != 11:
        #     mu = np.mean(np.concatenate((featureLs[valFold[0], :], featureRs[valFold[0], :]), 0), 0)
        #     mu = np.expand_dims(mu, 0)
        #     featureLs = featureLs - mu
        #     featureRs = featureRs - mu
        featureLs = featureLs / np.expand_dims(np.sqrt(np.sum(np.power(featureLs, 2), 1)), 1)
        featureRs = featureRs / np.expand_dims(np.sqrt(np.sum(np.power(featureRs, 2), 1)), 1)

        scores = np.sum(np.multiply(featureLs, featureRs), 1)
        threshold = getThreshold(scores[valFold[0]], flags[valFold[0]], 10000)
        ACCs[i] = getAccuracy(scores[testFold[0]], flags[testFold[0]], threshold)
        Thresholds[i] = threshold

    return ACCs, Thresholds

def loadModel(data_root, file_list):
    if isinstance(args.model_idx, list):
        nets = get_fastmodel(args.model_idx)
    else:
        nets = get_fastmodel([args.model_idx])

    net = nets[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = net.to(device)

    transform = transforms.Compose([
        transforms.ToTensor(),  
    ])
    lfw_dataset = LFW(data_root, file_list, transform=transform)
    lfw_loader = torch.utils.data.DataLoader(lfw_dataset, batch_size=128,
                                             shuffle=False, num_workers=2, drop_last=False)

    return net.eval(), device, lfw_dataset, lfw_loader

def getFeatureFromTorch(feature_save_dir, net, device, data_set, data_loader):
    featureLs = None
    featureRs = None
    count = 0

    for data in data_loader:
        for i in range(len(data)):
            data[i] = data[i].to(device)
        count += data[0].size(0)
        #print('extracing deep features from the face pair {}...'.format(count))
        with torch.no_grad():
            res = [net(d).data.cpu().numpy() for d in data]
        featureL = np.concatenate((res[0], res[1]), 1)
        featureR = np.concatenate((res[2], res[3]), 1)
        # print(featureL.shape, featureR.shape)
        if featureLs is None:
            featureLs = featureL
        else:
            featureLs = np.concatenate((featureLs, featureL), 0)
        if featureRs is None:
            featureRs = featureR
        else:
            featureRs = np.concatenate((featureRs, featureR), 0)
        # print(featureLs.shape, featureRs.shape)

    result = {'fl': featureLs, 'fr': featureRs, 'fold': data_set.folds, 'flag': data_set.flags}
    scipy.io.savemat(feature_save_dir, result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--root', type=str, default='/data/projects/aisc_facecomp/raw_data/LFW/lfw_align_112', help='The path of lfw data')
    parser.add_argument('--file_list', type=str, default='/data/projects/aisc_facecomp/raw_data/LFW/pairs.txt', help='The path of lfw data')
    parser.add_argument('--feature_dim', type=int, default=512, help='feature dimension')
    parser.add_argument('--model_idx', type=int, default=0)
    parser.add_argument('--feature_save_path', type=str, default='./cur_epoch_lfw_result.mat',
                        help='The path of the extract features save, must be .mat file')
    parser.add_argument('--outlog', type=str)
    args = parser.parse_args()
    
    net, device, lfw_dataset, lfw_loader = loadModel(args.root, args.file_list)
    getFeatureFromTorch(args.feature_save_path, net, device, lfw_dataset, lfw_loader)
    ACCs, Thresholds = evaluation_10_fold(args.feature_save_path)
    os.makedirs(os.path.dirname(args.outlog), exist_ok=True)
    with open(args.outlog, 'a') as fp:
        for i in range(len(ACCs)):
            fp.write('{}    acc: {:.2f}, threshold: {:.2f}\n'.format(i+1, ACCs[i] * 100, Thresholds[i]))
        fp.write('--------\n')
        fp.write('AVE    {:.4f}\n'.format(np.mean(ACCs) * 100))
