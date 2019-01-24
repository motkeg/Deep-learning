FROM  motkeg/deeplearning:gcp

RUN apt-get update && apt-get install -y 
EXPOSE 6006
# Checkout tensorflow/models at HEAD
RUN git clone https://github.com/tensorflow/models.git /tensorflow_models
RUN git clone https://github.com/motkeg/Deep-learning.git

WORKDIR /Deep-learning