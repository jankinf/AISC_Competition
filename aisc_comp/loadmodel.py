import os
import torch
from torch import nn
from torchvision import transforms as T
from third_party.FaceX_Zoo.backbone.backbone_def import BackboneFactory
# from third_party.FaceX_Zoo.training_mode.conventional_training.train import FaceModel
# from third_party.FaceX_Zoo.addition_module.face_lightning.KDF.test_protocol.utils.model_loader import ModelLoader

class ModelLoader:
    """Load a model by network and weights file.

    Attributes: 
        model(object): the model definition file.
    """
    def __init__(self, backbone_factory):
        self.model = backbone_factory.get_backbone()

    def load_model_default(self, model_path):
        """The default method to load a model.
        
        Args:
            model_path(str): the path of the weight file.
        
        Returns:
            model(object): initialized model.
        """
        self.model.load_state_dict(torch.load(model_path)['state_dict'], strict=True) 
        model = self.model.cuda()
        return model

    def load_model(self, model_path):
        """The custom method to load a model.
        
        Args:
            model_path(str): the path of the weight file.
        
        Returns:
            model(object): initialized model.
        """
        model_dict = self.model.state_dict()
        pretrained_dict = torch.load(model_path)['state_dict']
        new_pretrained_dict = {}

        for k in model_dict:
            new_pretrained_dict[k] = pretrained_dict['backbone.'+k] # tradition training
            #new_pretrained_dict[k] = pretrained_dict['feat_net.'+k] # tradition training
            #new_pretrained_dict[k] = pretrained_dict['module.'+k]
            #new_pretrained_dict[k] = pretrained_dict['module.backbone.'+k]
            #new_pretrained_dict[k] = pretrained_dict[k] # co-mining
        
        model_dict.update(new_pretrained_dict)
        self.model.load_state_dict(model_dict)
        model = self.model.cuda()
        return model


MEAN = [127.5 / 255] * 3
STD = [128 / 255] * 3
CONFIG_FILE = "/data/projects/aisc_facecomp/third_party/FaceX_Zoo/training_mode/backbone_conf.yaml"
CKPT_DIR = "/data/projects/aisc_facecomp/third_party/FaceX_Zoo/ckpt/backbone"

def get_fastmodel(idx=-1):
    models = []
    if 10 in idx:
        print("loading model {}".format(10))
        backbone_type = 'AttentionNet56'
        backbone_conf_file = CONFIG_FILE
        model_path = os.path.join(CKPT_DIR, 'attention56.pt')
        backbone_factory =  BackboneFactory(backbone_type, backbone_conf_file)
        model_loader = ModelLoader(backbone_factory)
        model = model_loader.load_model(model_path)
        model.eval()
        models.append(nn.Sequential(
            T.Normalize(mean=MEAN, std=STD),
            model
        ))
        models.append(model)

    if 11 in idx:
        print("loading model {}".format(11))
        backbone_type = 'AttentionNet92'
        backbone_conf_file = CONFIG_FILE
        model_path = os.path.join(CKPT_DIR, 'attention92.pt')
        backbone_factory =  BackboneFactory(backbone_type, backbone_conf_file)
        model_loader = ModelLoader(backbone_factory)
        model = model_loader.load_model(model_path)
        model.eval()
        models.append(nn.Sequential(
            T.Normalize(mean=MEAN, std=STD),
            model
        ))
    if 12 in idx:
        print("loading model {}".format(12))
        backbone_type = 'ResNet'
        backbone_conf_file = CONFIG_FILE
        model_path = os.path.join(CKPT_DIR, 'resnet152_irse.pt')
        backbone_factory =  BackboneFactory(backbone_type, backbone_conf_file)
        model_loader = ModelLoader(backbone_factory)
        model = model_loader.load_model(model_path)
        model.eval()
        models.append(nn.Sequential(
            T.Normalize(mean=MEAN, std=STD),
            model
        ))
    if 13 in idx:
        print("loading model {}".format(13))
        backbone_type = 'SwinTransformer'
        backbone_conf_file = CONFIG_FILE
        model_path = os.path.join(CKPT_DIR, 'swin_s.pt')
        backbone_factory =  BackboneFactory(backbone_type, backbone_conf_file)
        model_loader = ModelLoader(backbone_factory)
        model = model_loader.load_model(model_path)
        model.eval()
        models.append(nn.Sequential(
            T.Resize(224),
            T.Normalize(mean=MEAN, std=STD),
            model
        ))
    if 14 in idx:
        print("loading model {}".format(14))
        backbone_type = 'SwinTransformer'
        backbone_conf_file = CONFIG_FILE
        model_path = "/data/projects/aisc_facecomp/third_party/FaceX_Zoo/training_mode/conventional_training/out_dir/lr0.01/Epoch_49.pt"
        backbone_factory =  BackboneFactory(backbone_type, backbone_conf_file)
        model_loader = ModelLoader(backbone_factory)
        model = model_loader.load_model(model_path)
        model.eval()
        models.append(nn.Sequential(
            T.Resize(224),
            T.Normalize(mean=MEAN, std=STD),
            model
        ))
    if 15 in idx:
        print("loading model {}".format(14))
        backbone_type = 'SwinTransformer'
        backbone_conf_file = CONFIG_FILE
        model_path = "/data/projects/aisc_facecomp/third_party/FaceX_Zoo/training_mode/conventional_training/out_dir/adv_lr0.1_l2_eps0.06_alpha0.3_iter10/Epoch_20.pt"
        backbone_factory =  BackboneFactory(backbone_type, backbone_conf_file)
        model_loader = ModelLoader(backbone_factory)
        model = model_loader.load_model(model_path)
        model.eval()
        models.append(nn.Sequential(
            T.Resize(224),
            T.Normalize(mean=MEAN, std=STD),
            model
        ))

    return models
if __name__ == '__main__':
    # models = get_fastmodel([10, 11, 12, 13])
    # models = get_fastmodel([12, 13])
    models = get_fastmodel([13, 14])
