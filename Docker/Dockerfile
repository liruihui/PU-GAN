
From nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

LABEL version="1.0" maintainer="Unknownue <unknownue@outlook.com>"

ARG DEBIAN_FRONTEND=noninteractive

# COPY mirror-ubuntu1604.txt /etc/apt/sources.list

RUN apt update && \
    apt install -y software-properties-common && \
    apt install -y wget git && \
    add-apt-repository ppa:deadsnakes/ppa && \
    add-apt-repository ppa:ubuntu-toolchain-r/test

# Install python and pip
RUN apt update && \
    apt install -y python3.6 python3.6-tk && \
    wget https://bootstrap.pypa.io/get-pip.py && \
    python3.6 get-pip.py && rm get-pip.py && \
    ln -sf /usr/bin/python3.6 /usr/bin/python && \
    ln -sf /usr/bin/python3.6 /usr/bin/python3 && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# RUN pip install pqi && \
#     pqi use aliyun

# Install tensorflow
RUN pip install --no-cache-dir tensorflow-gpu==1.11.0
RUN pip install --no-cache-dir numpy==1.15.4 matplotlib==2.2.0 scikit-learn==0.20.0 && \
    pip install --no-cache-dir plyfile Pillow tqdm

RUN pip install --no-cache-dir open3d==0.9.0 && \
    apt install -y libgl1-mesa-glx python-apt && \
    cp /usr/lib/python3/dist-packages/apt_pkg.cpython-35m-x86_64-linux-gnu.so /usr/lib/python3/dist-packages/apt_pkg.cpython-36m-x86_64-linux-gnu.so && \
    apt update && \
    apt upgrade -y libstdc++6 && \
    rm /usr/bin/python && \
    ln -sf /usr/bin/python3.6 /usr/bin/python

CMD ["bash"]
