import cv2
import numpy as np
from skimage import transform as trans

src1 = np.array([[51.642, 50.115], [57.617, 49.990], [35.740, 69.007],
                 [51.157, 89.050], [57.025, 89.702]],
                dtype=np.float32)
# <--left
src2 = np.array([[45.031, 50.118], [65.568, 50.872], [39.677, 68.111],
                 [45.177, 86.190], [64.246, 86.758]],
                dtype=np.float32)

# ---frontal
src3 = np.array([[39.730, 51.138], [72.270, 51.138], [56.000, 68.493],
                 [42.463, 87.010], [69.537, 87.010]],
                dtype=np.float32)

# -->right
src4 = np.array([[46.845, 50.872], [67.382, 50.118], [72.737, 68.111],
                 [48.167, 86.758], [67.236, 86.190]],
                dtype=np.float32)

# -->right profile
src5 = np.array([[54.796, 49.990], [60.771, 50.115], [76.673, 69.007],
                 [55.388, 89.702], [61.257, 89.050]],
                dtype=np.float32)

src = np.array([src1, src2, src3, src4, src5])
src_map = {112: src, 224: src * 2}

arcface_src = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32)

arcface_src = np.expand_dims(arcface_src, axis=0)


# lmk is prediction; src is template
def estimate_norm(lmk, image_size=112, mode='arcface'):
    assert lmk.shape == (5, 2)
    tform = trans.SimilarityTransform()
    lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
    min_M = []
    min_index = []
    min_error = float('inf')
    if mode == 'arcface':
        if image_size == 112:
            src = arcface_src
        else:
            src = float(image_size) / 112 * arcface_src
    else:
        src = src_map[image_size]
    for i in np.arange(src.shape[0]):
        tform.estimate(lmk, src[i])
        M = tform.params[0:2, :]
        results = np.dot(M, lmk_tran.T)
        results = results.T
        error = np.sum(np.sqrt(np.sum((results - src[i]) ** 2, axis=1)))
        #         print(error)
        if error < min_error:
            min_error = error
            min_M = M
            min_index = i
    return min_M, min_index


def norm_crop(img, landmark, image_size=112, mode='arcface'):
    M, pose_index = estimate_norm(landmark, image_size, mode)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped


# def preprocess(img, bbox=None, landmark=None, **kwargs):
#     M = None
#     image_size = []
#     str_image_size = kwargs.get('image_size', '')
#     if len(str_image_size) > 0:
#         image_size = [int(x) for x in str_image_size.split(',')]
#         if len(image_size) == 1:
#             image_size = [image_size[0], image_size[0]]
#         assert len(image_size) == 2
#         assert image_size[0] == 112
#         assert image_size[0] == 112 or image_size[1] == 96
#
#     if landmark is not None:
#         assert len(image_size) == 2
#
#         src = np.array([
#             [30.2946, 51.6963],
#             [65.5318, 51.5014],
#             [48.0252, 71.7366],
#             [33.5493, 92.3655],
#             [62.7299, 92.2041]], dtype=np.float32)
#
#         # for RetinaFace adjuct +=degree
#         if image_size[1] == 112:
#             src[:, 0] += 8.0
#         dst = landmark.astype(np.float32)
#         tform = trans.SimilarityTransform()
#         tform.estimate(dst, src)
#         M = tform.params[0:2, :]
#
#     if M is None:
#         if bbox is None:
#             det = np.zeros(4, dtype=np.int32)
#             det[0] = int(img.shape[1] * 0.0625)
#             det[1] = int(img.shape[0] * 0.0625)
#             det[2] = img.shape[1] - det[0]
#             det[3] = img.shape[0] - det[1]
#         else:
#             det = bbox
#         margin = kwargs.get('margin', 44)
#         bb = np.zeros(4, dtype=np.int32)
#         bb[0] = np.maximum(det[0] - margin / 2, 0)
#         bb[1] = np.maximum(det[1] - margin / 2, 0)
#         bb[2] = np.minimum(det[2] + margin / 2, img.shape[1])
#         bb[3] = np.minimum(det[3] + margin / 2, img.shape[0])
#         ret = img[bb[1]:bb[3], bb[0]:bb[2], :]
#         if len(image_size) > 0:
#             ret = cv2.resize(ret, (image_size[1], image_size[0]))
#         return ret
#     else:
#         assert len(image_size) == 2
#         warped = cv2.warpAffine(img, M, (image_size[1], image_size[0]), borderValue=0.0)
#         return warped
