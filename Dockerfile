FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-runtime

LABEL maintainer='crapthings@gmail.com'

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONBUFFERED 1
ENV DEBIAN_FRONTEND noninteractive

WORKDIR /workspace

COPY scripts ./scripts
COPY transparent-background ./transparent-background
COPY runpod_app.py .

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN chmod +x ./scripts/install.sh
RUN ./scripts/install.sh
RUN python ./scripts/cache.py

RUN rm -rf ./scripts

CMD python -u ./runpod_app.py
