#!/bin/bash -e

prev=$pwd
current=$(cd $(dirname $0); pwd)
cd $current

GUI_TYPE=$1
UBUNTU_VER=22.04

if [ $GUI_TYPE = vnc ]; then
    # nothing
    GUI_TYPE=vnc
elif [ $GUI_TYPE = xrdp ]; then
    # nothing
    GUI_TYPE=xrdp
else
    GUI_TYPE=cli
fi

cd $GUI_TYPE
docker build . -f ./Dockerfile.ubuntu$UBUNTU_VER -t acac-4th:base-$GUI_TYPE-ubuntu$UBUNTU_VER
cd ..

docker build ../.. -f ./Dockerfile -t acac-4th:env-$GUI_TYPE-ubuntu$UBUNTU_VER --build-arg="ACAC4TH_BASE_IMAGE=acac-4th:base-$GUI_TYPE-ubuntu$UBUNTU_VER"

cd $prev