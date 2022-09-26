import os

HOST_API = '0.0.0.0'
PORT_API = 8000
RETINA_MODEL_PATH_25 = "./lib/ml_models/retina_face/mnet.25"
MODEL_AGE_PATH='./lib/ml_models/ssr2_megaage/model,0'
MODEL_GENDER_PATH='./lib/ml_models/ssr2_imdb_gender/model,0'
# GPU_CTX = int(os.getenv('GPU_CTX'))
GPU_CTX = -1

THRESHOLD = .9
SCALES = [1]