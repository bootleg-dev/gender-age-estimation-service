# Gender age estimation service using MXNet(SSR-Net)


![GitHub](https://img.shields.io/github/license/mashape/apistatus.svg)
![PyPI - Python Version](https://img.shields.io/badge/python-v3.7-blue)

   This service can estimate gender and age of a person whose face is located closer to the center of the image
   

## How to Execute
* **Install Dependencies:**
```sh
pip3 install -r requirements.txt
```
```sh
python3 app.py
```
* Then you can test the service using Swagger by submitting input image as base64 format:
**http://0.0.0.0:8000/docs**


## Run as Docker container
```sh
docker build -t gender-age-service:latest . -f Dockerfile
```
```sh
docker run --name ga-service -d -p 8000:8000 --env-file env.list gender-age-service:latest
```
```sh
docker logs -f ga-service
```

## Thanks to:
*  [**SSR-Net: A Compact Soft Stagewise Regression Network for Age Estimation**](https://github.com/wayen820/gender_age_estimation_mxnet) 
*  [**RetinaFace: Single-stage Dense Face Localisation in the Wild**](https://github.com/deepinsight/insightface/tree/master/detection/retinaface) 

## License
MIT LICENSE
