import functools
import os
import sys
from pathlib import Path


@functools.lru_cache(maxsize=1)
def get_mgfpy():
    import mgfpy

    return mgfpy


class MGFLoader:
    ROOT_DIR = Path(os.path.realpath("./megface-maps_834"))

    def __init__(self):
        sys.path.insert(0, self.python_dir)
        self.mgfpy = get_mgfpy()
        self.mgfpy.MegFace.init(self.library_path)
        sys.path.pop(0)

    @property
    def base_dir(self) -> Path:
        return self.ROOT_DIR

    @property
    def commom_dir(self) -> str:
        return str(self.base_dir / "python/mgfpy/megcommon")

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

    def get_conf(self, module, config_file):
        return str(self.data_dir / config_file)

    def load_module(self, module, config_file, **kwargs):
        ctx = getattr(self.mgfpy, "{}Context".format(module.capitalize()), None)
        if ctx is None:
            return None
        return ctx(
            config_path=self.get_conf(module=module, config_file=config_file), **kwargs
        )


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


class FaceDetector:
    def __init__(self, config_file="detector.middle.v3.conf", use_gpu=False):
        mgf_loader = MGFLoader()
        self.mgfpy = mgf_loader.mgfpy
        self.mgfpy.MegFace.set_log_level(self.mgfpy.LogLevel.MGF_LOG_ERROR)
        self.landmark_81p_tags = get_landmark_81p(mgf_loader)

        settings = {
            "device": {
                "dev_type": self.mgfpy.DeviceType.MGF_CPU,
                "stream_id": 0,
                "dev_id": 0,
            }
        }
        self.detector = mgf_loader.load_module(
            module="detector", config_file=config_file, settings=settings,
        )

        self.det_opts = {
            "roi": {"left": 0, "top": 0, "right": 0, "bottom": 0},
            "min_face": 50,
        }

    def detect_face(self, img, ratio=1):

        score, face = -1, None
        orients = [
            self.mgfpy.Orient.MGF_UP,
            self.mgfpy.Orient.MGF_DOWN,
            self.mgfpy.Orient.MGF_LEFT,
            self.mgfpy.Orient.MGF_RIGHT,
        ]

        # mgf_img = self.mgfpy.Image.from_cv2_image(img)
        # This method is deprecated, please use MegFace.make_image_from_cv2_image() instead.
        mgf_img = self.mgfpy.MegFace.make_image_from_cv2_image(img)

        for orient in orients:
            self.det_opts["orient"] = orient
            res = self.detector.detect(mgf_img, options=self.det_opts)
            for _face in res["items"]:
                if _face["confidence"] > score:
                    score = _face["confidence"]
                    face = _face

        if score > 0.2:
            ld = dict(zip(face["landmark"]["tags"], face["landmark"]["points"]))
            return (
                face["landmark"]["points"],
                [
                    (ld[tag]["x"] / ratio, ld[tag]["y"] / ratio)
                    for tag in self.landmark_81p_tags
                ],
            )

        return None, None

    def run_imgs(self, imgs):
        ld_marks = list()
        for i, img in enumerate(imgs):
            # mgf_img = self.mgfpy.Image.from_cv2_image(img)
            # This method is deprecated, please use MegFace.make_image_from_cv2_image() instead.
            mgf_img = self.mgfpy.MegFace.make_image_from_cv2_image(img)
            res = self.detector.detect(mgf_img)
            face, score = None, -1
            for _face in res["items"]:
                if _face["confidence"] > score:
                    score = _face["confidence"]
                    face = _face

            if face is None:
                raise ValueError("found no face.")

            ld = dict(zip(face["landmark"]["tags"], face["landmark"]["points"]))
            ld_marks.append(
                (
                    face["landmark"]["points"],
                    [(ld[tag]["x"], ld[tag]["y"]) for tag in self.landmark_81p_tags],
                )
            )
        return ld_marks


class FaceTracker:
    def __init__(self, config_file="tracker.middle.v3.conf", use_gpu=False):
        mgf_loader = MGFLoader()
        self.mgfpy = mgf_loader.mgfpy
        self.mgfpy.MegFace.set_log_level(self.mgfpy.LogLevel.MGF_LOG_ERROR)
        self.landmark_81p_tags = get_landmark_81p(mgf_loader)

        detector_options = {
            "device": {
                "dev_type": self.mgfpy.DeviceType.MGF_CPU,
                "stream_id": 0,
                "dev_id": 0,
            }
        }
        self.detector = mgf_loader.load_module(
            module="detector",
            config_file="detector.middle.v3.conf",
            settings=detector_options,
        )

        tracker_options = {
            "type": self.mgfpy.TrackerType.MGF_SYNC_DETECTOR_TRACKER,
            "sync": {
                "tracker_device": {
                    "dev_type": self.mgfpy.DeviceType.MGF_CPU,
                    "dev_id": 0,
                    "stream_id": 0,
                },
                "detector_options": {
                    "roi": {"left": 0, "top": 0, "right": 500, "bottom": 600},
                    "min_face": 50,
                    "orient": self.mgfpy.Orient.MGF_UP,
                    "work_load": 0.5,
                },
                "missing_tolerance": 3,
                "grid_num_row": 3,
                "grid_num_column": 2,
                "max_num_faces": 1,
                "manually_detect": False,
            },
        }
        self.tracker = mgf_loader.load_module(
            module="tracker", config_file=config_file, settings=tracker_options,
        )

    def run_imgs(self, imgs):
        ld_marks = list()
        for i, img in enumerate(imgs):
            # img = self.mgfpy.Image.from_cv2_image(img)
            # This method is deprecated, please use MegFace.make_image_from_cv2_image() instead.
            img = self.mgfpy.MegFace.make_image_from_cv2_image(img)
            result = self.tracker.track_frame(img)
            size = result["size"]
            if size == 0:
                if i == 0:
                    result = self.detector.detect(img)
                    if result["size"] == 0:
                        raise ValueError("Found no face.")
                else:
                    ld_marks.append(ld_marks[-1])
                    continue

            face, score = None, -1
            for _face in result["items"]:
                if _face["confidence"] > score:
                    score = _face["confidence"]
                    face = _face

            ld = dict(zip(face["landmark"]["tags"], face["landmark"]["points"]))
            ld_marks.append(
                (
                    face["landmark"]["points"],
                    [(ld[tag]["x"], ld[tag]["y"]) for tag in self.landmark_81p_tags],
                )
            )

        assert len(ld_marks) == len(imgs)
        return ld_marks


class FaceFeaturer:
    def __init__(self):
        mgf_loader = MGFLoader()
        self.mgfpy = mgf_loader.mgfpy
        # self.mgfpy.MegFace.set_log_level(self.mgfpy.LogLevel.MGF_LOG_ERROR)

        settings = {
            "device": {
                "dev_type": self.mgfpy.DeviceType.MGF_CPU,
                "stream_id": 0,
                "dev_id": 0,
            }
        }
        self.detector = mgf_loader.load_module(
            module="detector", config_file="detector.middle.v3.conf", settings=settings
        )
        self.featurer = mgf_loader.load_module(
            module="feature", config_file="feat.large.v2.conf", settings=settings
        )
        self.det_opts = {
            "roi": {"left": 0, "top": 0, "right": 0, "bottom": 0},
            "min_face": 50,
        }

    def get_feat(self, img):
        score, face = -1, None
        orients = [
            self.mgfpy.Orient.MGF_UP,
            # self.mgfpy.Orient.MGF_DOWN,
            # self.mgfpy.Orient.MGF_LEFT,
            # self.mgfpy.Orient.MGF_RIGHT,
        ]
        # mgf_img = self.mgfpy.MegFace.make_image_from_cv2_image(img)
        mgf_img = self.mgfpy.Image.from_cv2_image(img)
        for orient in orients:
            self.det_opts["orient"] = orient
            res = self.detector.detect(mgf_img, options=self.det_opts)
            for _face in res["items"]:
                if _face["confidence"] > score:
                    score = _face["confidence"]
                    face = _face

        if score > 0.0:
            ld = face["landmark"]
            feat = self.featurer.extract([mgf_img], [ld])
            return feat
        else:
            rect = {"left": 0, "top": 0, "right": img.shape[1], "bottom": img.shape[0]}
            feat = self.featurer.extract_from_rect([mgf_img], [rect])
            return feat

    def compare_feat(self, fa, fb):
        score = self.featurer.compare_features(fa, fb)
        return score

    def compare_feat_mxn(self, fa, fb):
        score = self.featurer.compare_features_MxN(fa, fb)
        return score
