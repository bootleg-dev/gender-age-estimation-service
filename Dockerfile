FROM python:3.7-slim

RUN apt-get -y update
RUN apt-get -y -qq install build-essential
RUN apt-get -y -qq install gcc
RUN apt-get -y -qq install libpq-dev
RUN apt-get -y -qq install libgtk2.0-dev
RUN apt-get -y -qq install cmake
RUN apt-get -y -qq install libgl1-mesa-glx


ENV INSTALL_PATH /gender-age-estimation-service

RUN mkdir -p $INSTALL_PATH

WORKDIR $INSTALL_PATH
COPY . .

RUN pip3 install -r requirements.txt

CMD gunicorn 'app:app' -w1 -b 0.0.0.0:8000 -k uvicorn.workers.UvicornWorker --timeout 120
