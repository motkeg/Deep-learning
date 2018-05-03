# Docker image for running examples in Tensorflow models.
# base_image depends on whether we are running on GPUs or non-GPUs
FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    build-essential \
    git \
    python \
    python-pip \
    python-setuptools

RUN pip install tf-nightly-gpu
EXPOSE 6006
# Checkout tensorflow/models at HEAD
RUN git clone https://github.com/tensorflow/models.git /tensorflow_models
RUN git clone https://github.com/motkeg/Msc-Project.git

WORKDIR /Msc-Project