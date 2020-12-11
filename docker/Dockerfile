FROM cuda9_ubuntu1804

ENV TENSORFLOW_VERSION 1.12.0
ENV CUDNN_VERSION 7.1.4.18
LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"

RUN apt-get update && apt-get install -y --no-install-recommends --allow-downgrades --allow-change-held-packages \
            libcudnn7=$CUDNN_VERSION-1+cuda9.0 \
            libcudnn7-dev=$CUDNN_VERSION-1+cuda9.0 && \
    apt-mark hold libcudnn7 && \
    rm -rf /var/lib/apt/lists/*
RUN apt-get update

RUN apt-get -y install curl vim wget unzip screen libc6-dev-i386 linux-libc-dev

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get -y install python3 python3-pip libopencv-dev
RUN echo 'alias python=python3' >> ~/.bashrc

RUN pip3 install --upgrade pip
RUN pip install numpy==1.16.4 matplotlib==2.2.4 scikit-image pillow scipy pyyaml cython

# DeepLab V3+
RUN pip install tensorflow-gpu==$TENSORFLOW_VERSION
RUN pip3 install scipy==1.3.0 numpy==1.15.0 matplotlib==2.2.4 pillow==6.1.0
RUN pip3 install scikit-image==0.15.0 pyyaml cython opencv-python==4.1.0.25 futures==3.1.1
RUN pip3 install tensorflow-gpu==$TENSORFLOW_VERSION

# download the object detection git
RUN apt-get -y install git

# install the dependencies
RUN pip install jupyter matplotlib PrettyTable

# Add Libraries to PYTHONPATH
ENV LD_LIBRARY_PATH=/usr/local/cuda-9.0/extras/CUPTI/lib64:$LD_LIBRARY_PATH
ENV PYTHONPATH $PYTHONPATH:/home/crack/models/deeplab/research:/home/crack/models/deeplab/research/slim

# DeepSegmentor
RUN pip install https://download.pytorch.org/whl/cu90/torch-0.4.1-cp36-cp36m-linux_x86_64.whl
RUN pip install torchvision==0.2.1 visdom==0.1.7 dominate==2.3.1

# for generating low quality annotations
RUN apt install -y python3-tk
RUN pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python
RUN pip install albumentations sklearn opencv-contrib-python pandas
