from typing import Tuple
import cv2
import numpy as np
import mxnet as mx
from mxnet.module import Module
from config.constants import SCALES, THRESHOLD
from lib.rcnn.retinaface import RetinaFace
from config.constants import RETINA_MODEL_PATH_25, MODEL_AGE_PATH, MODEL_GENDER_PATH
from lib.rcnn import face_preprocess
from config.constants import GPU_CTX
from service.utils.image_processing_utils import ImageProcessingUtils
from model.gender_age_model.gender_age_model import GenderAgeResponse
from service.utils.logger import Logger


def get_model(ctx, image_size, model_str, layer)-> Module:
    context = mx.gpu(ctx) if ctx > 0 else mx.cpu()
    _vec = model_str.split(',')
    prefix = _vec[0]
    epoch = int(_vec[1])
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
    all_layers = sym.get_internals()
    sym = all_layers[layer + '_output']
    model = mx.mod.Module(symbol=sym,
                          data_names=('data', 'stage_num0', 'stage_num1', 'stage_num2'),
                          context=context,
                          label_names=None)
    model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1])),
                            ('stage_num0', (1, 3)), ('stage_num1', (1, 3)),
                            ('stage_num2', (1, 3))])
    model.set_params(arg_params, aux_params)
    return model


class GenderAgeService:
    def __init__(self):
        self.retina_detector = RetinaFace(prefix=RETINA_MODEL_PATH_25, epoch=0, ctx_id=GPU_CTX, network='net3')
        self.model_age = get_model(ctx=GPU_CTX, image_size=(64, 64), model_str=MODEL_AGE_PATH, layer='_mulscalar16')
        self.model_gender = get_model(ctx=GPU_CTX, image_size=(64, 64), model_str=MODEL_GENDER_PATH,
                                      layer='_mulscalar16')
        self.img_proc_utils = ImageProcessingUtils()
        self.logger = Logger()

    def detect_center_face_retina(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        det, landmarks = self.retina_detector.detect(img, THRESHOLD, scales=SCALES)
        bindex = 0

        if det.shape[0] > 1:
            img_size = np.asarray(img.shape)[0:2]
            bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            img_center = img_size / 2
            offsets = np.vstack([(det[:, 0] + det[:, 2]) / 2 - img_center[1],
                                 (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            bindex = np.argmax(bounding_box_size - offset_dist_squared * 2.0)
        landmark = landmarks[bindex, :, :]
        warped = face_preprocess.norm_crop(img, landmark=landmark)
        warped = cv2.resize(warped, (64, 64))
        bbox = det[bindex, :]
        return warped, bbox

    def crop_face(self, x1: int, y1: int, x2: int, y2: int, img: np.ndarray) -> np.ndarray:
        cropped_image = img[x1: y1, x2: y2]
        dimensions = cropped_image.shape
        min_shape = min(dimensions[0], dimensions[1]) * 0.3
        subx = x1 - int(min_shape)
        subw = x2 - int(min_shape)
        if subx < 0: subx = x1
        if subw < 0: subw = x2
        cropped_image = img[subx: y1 + int(min_shape), subw: y2 + int(min_shape)]
        return cropped_image

    def detect_gender_age(self, img: np.ndarray) -> GenderAgeResponse:
        response = GenderAgeResponse()
        try:
            nimg, bbox = self.detect_center_face_retina(img=img)
            nimg = nimg[:, :, ::-1]
            nimg = np.transpose(nimg, (2, 0, 1))
            input_blob = np.expand_dims(nimg, axis=0)
            data = mx.nd.array(input_blob)
            db = mx.io.DataBatch(data=(data, mx.nd.array([[0, 1, 2]]),
                                       mx.nd.array([[0, 1, 2]]), mx.nd.array([[0, 1, 2]])))
            self.model_age.forward(db, is_train=False)
            age = self.model_age.get_outputs()[0].asnumpy()
            self.model_gender.forward(db, is_train=False)
            gender = self.model_gender.get_outputs()[0].asnumpy()
            gender_str = 'Male' if gender[0] > 0.5 else 'Female'
            x1, y1, x2, y2 = int(bbox[1]), int(bbox[3]), int(bbox[0]), int(bbox[2])
            cropped_face = self.crop_face(x1=x1, y1=y1, x2=x2, y2=y2, img=img)
            face_base64 = self.img_proc_utils.cv_image_to_base64(cv_img=cropped_face)
            response = GenderAgeResponse(age=age, gender=gender_str, face_base64=face_base64)
        except Exception as exception:
            self.logger.info("[EXCEPTION]  {}".format(exception))
        return response
