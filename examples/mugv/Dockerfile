# Base image: NGC PyTorch 25.02
FROM nvcr.io/nvidia/pytorch:25.02-py3
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace
COPY . /workspace/

ARG REQUIREMENTS=./examples/mugv/requirements.txt
ARG REQUIREMENTS_NODEPS=./examples/mugv/requirements-nodeps.txt
ARG NINJA_FLAGS="-j 1"

RUN apt update \
    && apt install -y sudo ffmpeg libsm6 libxext6 libgl1 tmux git net-tools iputils-ping \
    && python3 -m pip install --no-cache-dir -r ${REQUIREMENTS} \
    && python3 -m pip install --no-cache-dir --no-deps -r ${REQUIREMENTS_NODEPS} \
    && umask 0000 \
    && ls -alh /workspace


# NOTE: AIS set work as default user to run experiments, so make it happy
RUN userdel -rf ubuntu || true
RUN adduser --gecos '' --disabled-password work && \
    echo "work ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/nopasswd
ENV PATH=${PATH}:/home/work/.local/bin
RUN chown -R work:work /usr/local/* /home/work /workspace
USER work


CMD ["bash", "examples/mugv/pretrain_notebook.sh"]
