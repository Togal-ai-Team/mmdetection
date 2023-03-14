ARG PYTORCH="1.10.0"
ARG CUDA="11.3"
ARG CUDNN="8"
FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV PYTHONUNBUFFERED TRUE
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    ca-certificates \
    g++ \
    openjdk-11-jre-headless \
    # MMDet Requirements
    ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/conda/bin:$PATH"
RUN export FORCE_CUDA=1

# TORCHSEVER
RUN pip install torchserve==0.6.0 torch-model-archiver nvgpu sklearn numpy openmim opencv-python

# MMLAB
ARG PYTORCH
ARG CUDA

RUN ["/bin/bash", "-c", "mim install mmcv-full==1.7.0"]
COPY mmdetection mmdetection
RUN pip install -e mmdetection

RUN useradd -m model-server \
    && mkdir -p /home/model-server/tmp

COPY mmdetection/docker/serve/entrypoint.sh /usr/local/bin/entrypoint.sh

RUN chmod +x /usr/local/bin/entrypoint.sh \
    && chown -R model-server /home/model-server

COPY mmdetection/docker/serve/config.properties /home/model-server/config.properties
RUN mkdir /home/model-server/model-store && chown -R model-server /home/model-server/model-store
COPY mmdetection/docker/serve/ml_objects.mar /home/model-server/model-store

EXPOSE 8080 8081 8082

USER model-server
WORKDIR /home/model-server
ENV TEMP=/home/model-server/tmp
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["serve"]
