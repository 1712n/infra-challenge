FROM rust:1.69.0-bullseye

RUN apt-get update --yes && \
    apt-get install --yes --no-install-recommends \
    wget \
    unzip \
    python3-pip \
    g++ \
    git \
    git-lfs \
    jq \
    pkg-config \
    libssl-dev && \
    git lfs install && \
    rm -rf /var/lib/apt/lists/*


WORKDIR /tmp
RUN wget https://download.pytorch.org/libtorch/cu117/libtorch-cxx11-abi-shared-with-deps-1.13.1%2Bcu117.zip && \
    unzip -d /usr/lib libtorch-cxx11-abi-shared-with-deps-1.13.1+cu117.zip && \
    rm libtorch-cxx11-abi-shared-with-deps-1.13.1+cu117.zip

RUN pip install numpy==1.24.3 torch==1.13.1 
    
# instructions copied from pytorch dockerfile in hope that it'll make same effect as using pytorch image istself
LABEL com.nvidia.volumes.needed=nvidia_driver
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    LIBTORCH=/usr/lib/libtorch \
    LD_LIBRARY_PATH=/usr/lib/libtorch/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

WORKDIR /usr/src
# cloning release 0.20.0 this way; why tf they don't have tags?!
RUN git clone https://github.com/guillaume-be/rust-bert.git && \
    cd rust-bert && \
    git checkout f1b8409 && \
    cargo build --bin=convert-tensor 

WORKDIR /usr/src/infra-challenge
COPY . .
RUN cargo install --target-dir /usr/local/cargo/registry/infra-challenge --path .

CMD ["./cmd.sh"]