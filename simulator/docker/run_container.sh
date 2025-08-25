#!/bin/bash -e

prev=$pwd
current=$(cd $(dirname $0); pwd)
cd $current

GUI_TYPE=$1
UBUNTU_VER=22.04

if [ $GUI_TYPE = vnc ]; then
    docker run --rm -it \
        -p 5900:5900 \
        -p 8080:80 \
        -u $(id -u):$(id -g) \
        -e USER=test \
        -e PASSWD=test \
        -e RESOLUTION=1920x1080x24 \
        -v ..:/opt/data/workspace \
        acac-4th:env-$GUI_TYPE-ubuntu$UBUNTU_VER
elif [ $GUI_TYPE = xrdp ]; then
    docker run --rm -it \
        -p 3389:3389 \
        -u $(id -u):$(id -g) \
        -e USER=test \
        -e PASSWD=test \
        -v ..:/opt/data/workspace \
        acac-4th:env-$GUI_TYPE-ubuntu$UBUNTU_VER
else
    GUI_TYPE=cli
    docker run --rm -it \
        -v ..:/opt/data/workspace \
        acac-4th:env-$GUI_TYPE-ubuntu$UBUNTU_VER \
        bash
fi

cd $prev
