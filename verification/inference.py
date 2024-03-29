# -*- coding: utf-8 -*-

import os
import sys
import cv2
import functools
from pathlib import Path
import numpy as np

import megbrain as mgb
import neupeak
from megskull.network.visitor import NetworkVisitor
from megskull.opr.image_process import Resize
from megskull.opr.netsrc import DataProvider
from meghair.utils.misc import i01c_to_ic01
from neupeak.utils.misc import set_mgb_default_device
from neupeak.utils.imgproc import resize_ensure_longest_edge
from neupeak.utils.imgproc import pad_image_to_shape
import neupeak.utils.inference as inf
import megskull.opr.all as oprs
#import torch
import pickle
import pdb

INTERP = cv2.INTER_LINEAR
# INPUT_SHAPE = (224, 224)
MODEL_PATH = ""

def get_all_func(model_path, device=None):
    if device is not None:
        set_mgb_default_device(device)
    net = neupeak.utils.cli.load_network(model_path)
    # grad_loss = oprs.Grad(net.loss_var, net.loss_visitor.all_oprs[0], name='grad_loss')
    # gloss_func = inf.Function(inf.get_fprop_env(fast_run=True))
    # gloss = gloss_func.compile(grad_loss)

    pred_func = inf.Function(inf.get_fprop_env(fast_run=False))
    mgb.config.set_comp_graph_option(pred_func.comp_graph, "log_level", 0)
    pred = pred_func.compile(net.outputs)

    # return net, gloss, pred
    return net, pred

def get_grad_func(model_path, device=None):
    if device is not None:
        set_mgb_default_device(device)
    net = neupeak.utils.cli.load_network(model_path)
    grad = oprs.Grad(net.loss_var, net.loss_visitor.all_oprs[0], name='grad')
    grad_func = inf.Function(inf.get_fprop_env(fast_run=True))
    return grad_func.compile(grad)

def get_pred_func(model_path, device=None):
    if device is not None:
        set_mgb_default_device(device)
    
    n = neupeak.utils.cli.load_network(model_path)
    f = inf.Function(inf.get_fprop_env(fast_run=False))
    mgb.config.set_comp_graph_option(f.comp_graph, "log_level", 0)
    return f.compile(n.outputs)
    # if device is None:
    #     return f.compile(n.outputs)
    # else:
    #     # reset comp_node by network_visitor
    #     orig_output = network.outputs[0]
    #     nv = NetworkVisitor(orig_output)
    #     nv_copied = nv.reset_comp_node(device)
    #     dest_outputs = nv_copied.dest_vars
    #     return f.compile(dest_outputs)

def get_megcv_resize_func(inp_size=(224, 224)):
    input_image = DataProvider('img', (1, ) + inp_size + (3, ), dtype='uint8', comp_node='cpux')
    input_shape = DataProvider('img_shape', (2, ), dtype='uint8', comp_node='cpux')
    output_image = Resize('output_img', input_image, input_shape)
    f = inf.Function(inf.get_fprop_env(fast_run=False))
    mgb.config.set_comp_graph_option(f.comp_graph, "log_level", 0)
    func = f.compile(output_image)

    def resize_func(img, shape):
        return func(img=np.array([img], 'uint8'), img_shape=shape)[0]
    return resize_func

@functools.lru_cache(maxsize=1)
def get_mgfpy():
    import mgfpy
    return mgfpy

def get_face_detector(uid=0):
    return FaceDetector('middle.v3')

def resize_with_padding(img, shape, resize_func=None):
    ih, iw = img.shape[:2]
    th, tw = shape
    if ih / iw < th / tw:
        rw, rh = tw, tw / iw * ih
    else:
        rw, rh = th / ih * iw, th

    rw, rh = int(rw), int(rh)
    # img = cv2.resize(img, (rw, rh), interpolation=INTERP)
    img = resize_func(img, (rh, rw))
    img = pad_image_to_shape(img, shape)
    return img

class MGFLoader:
    # ROOT_DIR = Path('/home/group-csg_algorithm//megface—v2.10')
    ROOT_DIR = Path('/home/fangzhengwei/megface—v2.10')

    def __init__(self, version):
        # to keep other parts of the library clean
        self.version = version
        sys.path.insert(0, self.python_dir)
        self.mgfpy = get_mgfpy()
        self.mgfpy.MegFace.init(self.library_path)
        sys.path.pop(0)

    @property
    def base_dir(self) -> Path:
        return self.ROOT_DIR

    @property
    def python_dir(self) -> str:
        return str(self.base_dir / "python")

    @property
    def library_dir(self) -> Path:
        return self.base_dir / "lib"

    @property
    def data_dir(self) -> Path:
        return self.base_dir / "data"

    @property
    def library_path(self) -> str:
        return str(self.library_dir / "libmegface.so")

    def get_conf(self, name, module='detector'):
        if module.lower() == 'feature':
            module = 'feat'  # XXX blame megface
        #return str(self.data_dir / 'detector.retinanet.mobile.ao.face.grayC.conf')
        #return str(self.data_dir / 'detector.retina.faceplusplus.x14.acc.x14.contour.weights.20190116.conf')
        #return str(self.data_dir / 'detector.legacy.conf')
        # return str('/data/special_attack/venv/megfacemb8/data/detector.legacy.conf')
        # return str('/home/wangpeng/software/2521_sdk_detector/data/detector.middle.v3.conf')
        return str("/home/fangzhengwei/megface—v2.10/data/detector.middle.v3.conf")

    def load_module(self, name, module='detector', **kwargs):
        ctx = getattr(self.mgfpy, '{}Context'.format(module.capitalize()), None)
        if ctx is None:
            return None
        return ctx(config_path=self.get_conf(name, module=module), **kwargs)

def get_landmark_81p(mgf_loader: MGFLoader):
    LandmarkTag = mgf_loader.mgfpy.LandmarkTag

    return [
        LandmarkTag.CONTOUR_CHIN,
        LandmarkTag.CONTOUR_LEFT1,
        LandmarkTag.CONTOUR_LEFT2,
        LandmarkTag.CONTOUR_LEFT3,
        LandmarkTag.CONTOUR_LEFT4,
        LandmarkTag.CONTOUR_LEFT5,
        LandmarkTag.CONTOUR_LEFT6,
        LandmarkTag.CONTOUR_LEFT7,
        LandmarkTag.CONTOUR_LEFT8,
        LandmarkTag.CONTOUR_LEFT9,
        LandmarkTag.CONTOUR_RIGHT1,
        LandmarkTag.CONTOUR_RIGHT2,
        LandmarkTag.CONTOUR_RIGHT3,
        LandmarkTag.CONTOUR_RIGHT4,
        LandmarkTag.CONTOUR_RIGHT5,
        LandmarkTag.CONTOUR_RIGHT6,
        LandmarkTag.CONTOUR_RIGHT7,
        LandmarkTag.CONTOUR_RIGHT8,
        LandmarkTag.CONTOUR_RIGHT9,
        LandmarkTag.LEFTEYE_BOTTOM,
        LandmarkTag.LEFTEYE_LEFTCORNER,
        LandmarkTag.LEFTEYE_LOWERLEFTQUARTER,
        LandmarkTag.LEFTEYE_LOWERRIGHTQUARTER,
        LandmarkTag.LEFTEYE_PUPIL,
        LandmarkTag.LEFTEYE_RIGHTCORNER,
        LandmarkTag.LEFTEYE_TOP,
        LandmarkTag.LEFTEYE_UPPERLEFTQUARTER,
        LandmarkTag.LEFTEYE_UPPERRIGHTQUARTER,
        LandmarkTag.LEFTEYEBROW_LEFTCORNER,
        LandmarkTag.LEFTEYEBROW_LOWERLEFTQUARTER,
        LandmarkTag.LEFTEYEBROW_LOWERMIDDLE,
        LandmarkTag.LEFTEYEBROW_LOWERRIGHTQUARTER,
        LandmarkTag.LEFTEYEBROW_RIGHTCORNER,
        LandmarkTag.LEFTEYEBROW_UPPERLEFTQUARTER,
        LandmarkTag.LEFTEYEBROW_UPPERMIDDLE,
        LandmarkTag.LEFTEYEBROW_UPPERRIGHTQUARTER,
        LandmarkTag.MOUTH_LEFTCORNER,
        LandmarkTag.MOUTH_LOWERLIPBOTTOM,
        LandmarkTag.MOUTH_LOWERLIPLEFTCONTOUR1,
        LandmarkTag.MOUTH_LOWERLIPLEFTCONTOUR2,
        LandmarkTag.MOUTH_LOWERLIPLEFTCONTOUR3,
        LandmarkTag.MOUTH_LOWERLIPRIGHTCONTOUR1,
        LandmarkTag.MOUTH_LOWERLIPRIGHTCONTOUR2,
        LandmarkTag.MOUTH_LOWERLIPRIGHTCONTOUR3,
        LandmarkTag.MOUTH_LOWERLIPTOP,
        LandmarkTag.MOUTH_RIGHTCORNER,
        LandmarkTag.MOUTH_UPPERLIPBOTTOM,
        LandmarkTag.MOUTH_UPPERLIPLEFTCONTOUR1,
        LandmarkTag.MOUTH_UPPERLIPLEFTCONTOUR2,
        LandmarkTag.MOUTH_UPPERLIPLEFTCONTOUR3,
        LandmarkTag.MOUTH_UPPERLIPRIGHTCONTOUR1,
        LandmarkTag.MOUTH_UPPERLIPRIGHTCONTOUR2,
        LandmarkTag.MOUTH_UPPERLIPRIGHTCONTOUR3,
        LandmarkTag.MOUTH_UPPERLIPTOP,
        LandmarkTag.NOSE_CONTOURLEFT1,
        LandmarkTag.NOSE_CONTOURLEFT2,
        LandmarkTag.NOSE_CONTOURLEFT3,
        LandmarkTag.NOSE_CONTOURLOWERMIDDLE,
        LandmarkTag.NOSE_CONTOURRIGHT1,
        LandmarkTag.NOSE_CONTOURRIGHT2,
        LandmarkTag.NOSE_CONTOURRIGHT3,
        LandmarkTag.NOSE_LEFT,
        LandmarkTag.NOSE_RIGHT,
        LandmarkTag.NOSE_TIP,
        LandmarkTag.RIGHTEYE_BOTTOM,
        LandmarkTag.RIGHTEYE_LEFTCORNER,
        LandmarkTag.RIGHTEYE_LOWERLEFTQUARTER,
        LandmarkTag.RIGHTEYE_LOWERRIGHTQUARTER,
        LandmarkTag.RIGHTEYE_PUPIL,
        LandmarkTag.RIGHTEYE_RIGHTCORNER,
        LandmarkTag.RIGHTEYE_TOP,
        LandmarkTag.RIGHTEYE_UPPERLEFTQUARTER,
        LandmarkTag.RIGHTEYE_UPPERRIGHTQUARTER,
        LandmarkTag.RIGHTEYEBROW_LEFTCORNER,
        LandmarkTag.RIGHTEYEBROW_LOWERLEFTQUARTER,
        LandmarkTag.RIGHTEYEBROW_LOWERMIDDLE,
        LandmarkTag.RIGHTEYEBROW_LOWERRIGHTQUARTER,
        LandmarkTag.RIGHTEYEBROW_RIGHTCORNER,
        LandmarkTag.RIGHTEYEBROW_UPPERLEFTQUARTER,
        LandmarkTag.RIGHTEYEBROW_UPPERMIDDLE,
        LandmarkTag.RIGHTEYEBROW_UPPERRIGHTQUARTER,
    ]

class FaceDetector():

    def __init__(self, model_config, use_gpu=False):
        mgf_loader = MGFLoader('v2.2.1.6-cpu')
        self.mgfpy = mgf_loader.mgfpy
        self.mgfpy.MegFace.set_log_level(self.mgfpy.LogLevel.MGF_LOG_ERROR)
        self.landmark_81p_tags = get_landmark_81p(mgf_loader)

        settings = {
            'device': {
                'dev_type': self.mgfpy.DeviceType.MGF_CPU,
                'stream_id': 0,
                'dev_id': 0
            }
        }
        self.detector = mgf_loader.load_module(
            model_config, module='detector', settings=settings,
        )

        self.det_opts = {
            'roi': {'left': 0, 'top': 0, 'right': 0, 'bottom': 0},
            'min_face': 50,
        }

    def detect_face(self, img, ratio=1):

        score, face = -1, None
        orients = [
            self.mgfpy.Orient.MGF_UP, self.mgfpy.Orient.MGF_DOWN,
            self.mgfpy.Orient.MGF_LEFT, self.mgfpy.Orient.MGF_RIGHT,
        ]
        mgf_img = self.mgfpy.MegFace.make_image_from_cv2_image(img)
        # mgf_img = self.mgfpy.Image.from_cv2_image(img)
        for orient in orients:
            self.det_opts['orient'] = orient
            res = self.detector.detect(mgf_img, options=self.det_opts)
            for _face in res['items']:
                if _face['confidence'] > score:
                    score = _face['confidence']
                    face = _face

        if score > 0.2:
            ld = dict(zip(face['landmark']['tags'], face['landmark']['points']))
            # pdb.set_trace()
            return [
                (ld[tag]['x'] / ratio, ld[tag]['y'] / ratio)
                for tag in self.landmark_81p_tags
            ]

        return None

def get_rect_from_ld(ld):
    if ld.ndim == 0:
        return None
    x0, y0, x1, y1 = (
        ld[:, 0].min(), ld[:, 1].min(), ld[:, 0].max(), ld[:, 1].max()
    )
    return [x0, y0, x1, y1]

def face_crop(img, rect, hwratio=1., alpha=3.):
    SH, SW = img.shape[:2]

    x0, y0, x1, y1 = rect
    fh, fw = y1-y0, x1-x0
    xc, yc = (x0+x1) / 2, (y0+y1) / 2
    unit = (fh+fw) / 2

    th, tw = alpha*unit*hwratio, alpha*unit

    xs, xe = xc-(tw/2), xc+(tw/2)
    ys, ye = yc-(th/2), yc+(th/2)

    if (xs < 0) and (xe > SW):
        xs, xe = 0, SW
    elif xs < 0:
        xs, xe = 0, min(tw, SW)
    elif xe > SW:
        xs, xe = max(SW-tw, 0), SW

    if (ys < 0) and (ye > SH):
        ys, ye = 0, SH
    elif ys < 0:
        ys, ye = 0, min(th, SH)
    elif ye > SH:
        ys, ye = max(SH-th, 0), SH

    ys, ye, xs, xe = map(int, (ys, ye, xs, xe))
    rimg = img[ys:ye, xs:xe, :]

    return rimg

def face_crop_rect(img, rect, hwratio=1., alpha=3.):
    SH, SW = img.shape[:2]

    x0, y0, x1, y1 = rect
    
    rimg = img[int(y0):int(y1), int(x0):int(x1), :]

    return rimg

def face_crop_rect0308(img, rect, hwratio=1., alpha=3.):
        x0, y0, x1, y1 = rect
        fh, fw = y1-y0, x1-x0
        x0 = max(0, x0-(fw*0.3))
        y0 = max(0, y0-(fh*0.8))
        x1 = min(img.shape[1], x1+(fw*0.3))
        y1 = min(img.shape[0], y1+(fh*0.3))
        ys, ye, xs, xe = map(int, (y0, y1, x0, x1))
        rimg = img[ys:ye, xs:xe, :]
        return rimg

def face_crop_rect0206(img, rect, hwratio=1., alpha=3.):
        x0, y0, x1, y1 = rect
        fh, fw = y1-y0, x1-x0
        x0 = max(0, x0-(fw*0.2))
        y0 = max(0, y0-(fh*0.6))
        x1 = min(img.shape[1], x1+(fw*0.2))
        y1 = min(img.shape[0], y1+(fh*0.2))
        ys, ye, xs, xe = map(int, (y0, y1, x0, x1))
        rimg = img[ys:ye, xs:xe, :]
        return rimg

def face_crop_rect061(img, rect, hwratio=1., alpha=3.):
        x0, y0, x1, y1 = rect
        fh, fw = y1-y0, x1-x0
        x0 = max(0, x0-(fw*0.6))
        y0 = max(0, y0-(fh*1.))
        x1 = min(img.shape[1], x1+(fw*0.6))
        y1 = min(img.shape[0], y1+(fh*0.6))
        ys, ye, xs, xe = map(int, (y0, y1, x0, x1))
        rimg = img[ys:ye, xs:xe, :]
        return rimg

class RunnableInference:
    def __init__(self, model_path=MODEL_PATH, device=None, alpha=3., input_shape=(224, 224)):
        self.model_path = model_path
        self.input_shape = input_shape
        self.hwratio = input_shape[0] / input_shape[1]
        self.device = device
        self.alpha = alpha

        self.resize_func = get_megcv_resize_func(inp_size=input_shape)
        self.face_detector = get_face_detector()
        self.pred_func = get_pred_func(model_path, device=device)
        self.grad_func = get_grad_func(model_path, device=device)

    def run_image(self, img, rect=None):
        # todo: run_image: preprocess + pred
        assert isinstance(img, (list, np.ndarray))

        if isinstance(img, np.ndarray):
            img = [img]
            rect = [rect]
        else:
            assert len(img) == len(rect)
        img_list = img; rect_list = rect

        input_batch = []
        for idx, img in enumerate(img_list):
            rect = rect_list[idx]
            if rect is None:
                # print('find_one')
                # if img.shape[0] * img.shape[1] > 800 ** 2:
                #     img = resize_ensure_longest_edge(img, 800)
                ld = self.face_detector.detect_face(img)
                ld = np.array(ld)
                rect = get_rect_from_ld(ld)
            # frect = [0.3893, 0.1903, 0.1854, 0.5191];  h,w = img.shape[:2]
            # rect=[0,0,0,0];  rect[0]=frect[1]*w;  rect[1]=frect[2]*h; rect[2]=(frect[1]+frect[3])*w; rect[3]=(frect[0]+frect[2])*h
            if rect is not None:
                img = face_crop(img, rect, hwratio=self.hwratio, alpha=3.)
                # img = face_crop_rect0206(img, rect, hwratio=self.hwratio, alpha=3.)
                # img = face_crop_rect0308(img, rect, hwratio=self.hwratio, alpha=3.)
                # img = face_crop_rect061(img, rect, hwratio=self.hwratio, alpha=3.)
                # img = face_crop_rect(img, rect, hwratio=self.hwratio, alpha=3.)
            img = resize_with_padding(img, self.input_shape, resize_func=self.resize_func)

            input_batch.append(i01c_to_ic01(img))

        input_batch = np.array(input_batch, 'uint8')
        scores = self.pred_func(input_batch)[0]
        pred_score = [scores[i][1] for i in range(len(img_list))]
        # print(pred_score)
        if len(pred_score) == 1:
            return pred_score[0], img
        return pred_score, img

    def pred(self, img):
        assert isinstance(img, (list, np.ndarray))

        if isinstance(img, np.ndarray):
            img = [img]
        img_list = img

        input_batch = []
        for img in img_list:
            input_batch.append(i01c_to_ic01(img))
        input_batch = np.array(input_batch, 'uint8')
        scores = self.pred_func(input_batch)[0]
        pred_score = scores[:, 1]
        
        if len(pred_score) == 1:
            return pred_score[0]
        return pred_score

    def preprocess(self, img, rect=None):
        assert isinstance(img, (list, np.ndarray))

        if isinstance(img, np.ndarray):
            img = [img]
            rect = [rect]
        else:
            assert len(img) == len(rect)
        img_list = img; rect_list = rect

        for idx, img in enumerate(img_list):
            rect = rect_list[idx]
            if rect is None:
                ld = self.face_detector.detect_face(img)
                ld = np.array(ld)
                rect = get_rect_from_ld(ld)
            if rect is not None:
                img = face_crop(img, rect, hwratio=self.hwratio, alpha=3.)
            img = resize_with_padding(img, self.input_shape, resize_func=self.resize_func)
        return img


class RunnableInference_v2:
    def __init__(self, model_path=MODEL_PATH, device=None, alpha=3., input_shape=(224, 224)):
        self.model_path = model_path
        self.input_shape = input_shape
        self.hwratio = input_shape[0] / input_shape[1]
        self.device = device
        self.alpha = alpha

        self.resize_func = get_megcv_resize_func(inp_size=input_shape)
        self.face_detector = get_face_detector()
        # self.net, self.grad_func, self.pred_func = get_all_func(model_path, device=device)
        self.net, self.pred_func = get_all_func(model_path, device=device)

    def run_image(self, img, rect=None):
        # todo: run_image: preprocess + pred
        assert isinstance(img, (list, np.ndarray))

        if isinstance(img, np.ndarray):
            img = [img]
            rect = [rect]
        else:
            assert len(img) == len(rect)
        img_list = img; rect_list = rect

        input_batch = []
        for idx, img in enumerate(img_list):
            rect = rect_list[idx]
            if rect is None:
                # print('find_one')
                # if img.shape[0] * img.shape[1] > 800 ** 2:
                #     img = resize_ensure_longest_edge(img, 800)
                ld = self.face_detector.detect_face(img)
                ld = np.array(ld)
                rect = get_rect_from_ld(ld)
            # frect = [0.3893, 0.1903, 0.1854, 0.5191];  h,w = img.shape[:2]
            # rect=[0,0,0,0];  rect[0]=frect[1]*w;  rect[1]=frect[2]*h; rect[2]=(frect[1]+frect[3])*w; rect[3]=(frect[0]+frect[2])*h
            if rect is not None:
                img = face_crop(img, rect, hwratio=self.hwratio, alpha=3.)
                # img = face_crop_rect0206(img, rect, hwratio=self.hwratio, alpha=3.)
                # img = face_crop_rect0308(img, rect, hwratio=self.hwratio, alpha=3.)
                # img = face_crop_rect061(img, rect, hwratio=self.hwratio, alpha=3.)
                # img = face_crop_rect(img, rect, hwratio=self.hwratio, alpha=3.)
            img = resize_with_padding(img, self.input_shape, resize_func=self.resize_func)

            input_batch.append(i01c_to_ic01(img))

        input_batch = np.array(input_batch, 'uint8')
        scores = self.pred_func(input_batch)[0]
        pred_score = [scores[i][1] for i in range(len(img_list))]
        # print(pred_score)
        if len(pred_score) == 1:
            return pred_score[0], img
        return pred_score, img

    def pred(self, img):
        assert isinstance(img, (list, np.ndarray))

        if isinstance(img, np.ndarray):
            img = [img]
        img_list = img

        input_batch = []
        for img in img_list:
            input_batch.append(i01c_to_ic01(img))
        input_batch = np.array(input_batch, 'uint8')
        scores = self.pred_func(input_batch)[0]
        pred_score = scores[:, 1]
        
        if len(pred_score) == 1:
            return pred_score[0]
        return pred_score

    def preprocess(self, img, rect=None):
        assert isinstance(img, (list, np.ndarray))

        if isinstance(img, np.ndarray):
            img = [img]
            rect = [rect]
        else:
            assert len(img) == len(rect)
        img_list = img; rect_list = rect

        for idx, img in enumerate(img_list):
            rect = rect_list[idx]
            if rect is None:
                ld = self.face_detector.detect_face(img)
                ld = np.array(ld)
                rect = get_rect_from_ld(ld)
            if rect is not None:
                img = face_crop(img, rect, hwratio=self.hwratio, alpha=3.)
            img = resize_with_padding(img, self.input_shape, resize_func=self.resize_func)
        return img
