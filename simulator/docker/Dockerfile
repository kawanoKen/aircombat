ARG ACAC4TH_BASE_IMAGE=acac-4th:base-cli-ubuntu22.04
FROM ${ACAC4TH_BASE_IMAGE}

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get upgrade -y\
    && DEBIAN_FRONTEND=noninteractive apt-get dist-upgrade -y\
    && DEBIAN_FRONTEND=noninteractive apt-get autoremove \
    && DEBIAN_FRONTEND=noninteractive apt-get autoclean \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
      apt-offline \
      cifs-utils \
      gdebi \
      net-tools \
      iputils-ping \
      curl \
      wget \
      build-essential \
      cmake \
      gedit \
      git \
      nano \
      python3.10-dev \
      python3.10-venv \
      python3-pip \
      python3-pip-whl \
      libboost-all-dev \
    && apt-get clean \
    && rm -rf /var/cache/apt/archives/* \
    && rm -rf /var/lib/apt/lists/*

COPY ASRCAISim1 /opt/data/ASRCAISim1
COPY ASRCAISim1/docker/build_sim.sh /opt/data
RUN mkdir /opt/data/venv \
    && cd /opt/data/venv \
    && python3 -m venv acac_4th
RUN cd /opt/data \
    && bash ./build_sim.sh

RUN touch /etc/profile \
    && sed -i '$asource /opt/data/venv/acac_4th/bin/activate' /etc/profile

#CUDA11.8.0ベースのイメージだとgroupaddがpermission deniedで失敗する。
RUN chmod u+s /usr/sbin/useradd \
&& chmod u+s /usr/sbin/groupadd
