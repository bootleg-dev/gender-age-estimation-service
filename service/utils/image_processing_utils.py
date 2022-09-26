import io
import cv2
import numpy
from PIL import Image
import numpy as np
import base64
from service.utils.logger import Logger

class ImageProcessingUtils:
    def __init__(self):
        self.logger = Logger()

    def convert_base64_to_image(self, file: str) -> numpy.ndarray:
        try:
            image_base = self.split_image(file)
            image_base = base64.b64decode(image_base.encode())
            mat_image = cv2.cvtColor(np.array(Image.open(io.BytesIO(image_base))),
                                     cv2.COLOR_BGR2RGB)
            return mat_image
        except Exception as exception:
            self.logger.info(exception)

    def split_image(self, image: str) -> str:
        img_list = image.split(",")
        if len(img_list) == 1:
            return img_list[0]
        elif len(img_list) == 2:
            return img_list[1]
        else:
            return ""

    def cv_image_to_base64(self, cv_img: numpy.ndarray) -> str:
        _, buffer_img = cv2.imencode('.jpg', cv_img)
        data = (base64.b64encode(buffer_img)).decode('utf-8')
        img_base64 = "data:image/jpeg;base64," + str(data)
        return img_base64
