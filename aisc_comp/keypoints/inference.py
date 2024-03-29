# -*- coding: utf-8 -*-

import sys
import functools
from pathlib import Path
import numpy as np


@functools.lru_cache(maxsize=1)
def get_mgfpy():
    import mgfpy
    return mgfpy

def get_face_detector(uid=0):
    return FaceDetector('middle.v3')

class MGFLoader:
    ROOT_DIR = Path('/home/fangzhengwei/megface—v2.10')

    def __init__(self, version):
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
            return [(ld[tag]['x'] / ratio, ld[tag]['y'] / ratio)
                for tag in self.landmark_81p_tags]

        return None

def get_rect_from_ld(ld):
    if ld.ndim == 0:
        return None
    x0, y0, x1, y1 = (
        ld[:, 0].min(), ld[:, 1].min(), ld[:, 0].max(), ld[:, 1].max()
    )
    return [x0, y0, x1, y1]


class RunnableInference:
    def __init__(self):
        self.face_detector = get_face_detector()

    def run_image(self, img):
        assert isinstance(img, np.ndarray)
        ld = self.face_detector.detect_face(img)
        # ld = np.array(ld)

        return ld
