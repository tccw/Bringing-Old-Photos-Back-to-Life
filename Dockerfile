FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime

COPY requirements.txt /tmp
WORKDIR /tmp
RUN apt-get update
RUN apt-get -y install make cmake gcc g++
RUN pip install -r requirements.txt