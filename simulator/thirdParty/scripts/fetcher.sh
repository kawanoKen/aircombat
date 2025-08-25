#!/bin/bash -e
# Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)

prev=$(pwd)
current=$(cd $(dirname $0); pwd)
cd $current/../deps

if [ -e tmpfile ]; then
    rm -f tmpfile
fi

#pybind11
url=https://github.com/pybind/pybind11/archive/refs/tags/v2.13.6.tar.gz
filename=pybind11-2.13.6.tar.gz
removerstr=pybind11-*.tar.gz*
if [ -e $filename ]; then
    mv $filename tmpfile
    rm -f $removerstr
    mv tmpfile $filename
else
    rm -f $removerstr
    curl $url -o $filename -L
fi

#eigen
url=https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
filename=eigen-3.4.0.tar.gz
removerstr=eigen-*.tar.gz*
if [ -e $filename ]; then
    mv $filename tmpfile
    rm -f $removerstr
    mv tmpfile $filename
else
    rm -f $removerstr
    curl $url -o $filename -L
fi

#nlopt
url=https://github.com/stevengj/nlopt/archive/refs/tags/v2.7.1.tar.gz
filename=nlopt-2.7.1.tar.gz
removerstr=nlopt-*.tar.gz*
if [ -e $filename ]; then
    mv $filename tmpfile
    rm -f $removerstr
    mv tmpfile $filename
else
    rm -f $removerstr
    curl $url -o $filename -L
fi

#nlohmann json
url=https://github.com/nlohmann/json/archive/refs/tags/v3.12.0.tar.gz
filename=json-3.12.0.tar.gz
removerstr=json-*.tar.gz*
if [ -e $filename ]; then
    mv $filename tmpfile
    rm -f $removerstr
    mv tmpfile $filename
else
    rm -f $removerstr
    curl $url -o $filename -L
fi

#magic_enum
url=https://github.com/Neargye/magic_enum/archive/refs/tags/v0.9.7.tar.gz
filename=magic_enum-0.9.7.tar.gz
removerstr=magic_enum-*.tar.gz*
if [ -e $filename ]; then
    mv $filename tmpfile
    rm -f $removerstr
    mv tmpfile $filename
else
    rm -f $removerstr
    curl $url -o $filename -L
fi

#thread-pool
url=https://github.com/bshoshany/thread-pool/archive/refs/tags/v5.0.0.tar.gz
filename=thread-pool-5.0.0.tar.gz
removerstr=thread-pool-*.tar.gz*
if [ -e $filename ]; then
    mv $filename tmpfile
    rm -f $removerstr
    mv tmpfile $filename
else
    rm -f $removerstr
    curl $url -o $filename -L
fi

#boost
url=https://archives.boost.io/release/1.88.0/source/boost_1_88_0.tar.gz
filename=boost_1_88_0.tar.gz
removerstr=boost_*.tar.gz*
if [ -e $filename ]; then
    mv $filename tmpfile
    rm -f $removerstr
    mv tmpfile $filename
else
    rm -f $removerstr
    curl $url -o $filename -L
fi

#cereal
url=https://github.com/USCiLab/cereal/archive/refs/tags/v1.3.2.tar.gz
filename=cereal-1.3.2.tar.gz
removerstr=cereal*.tar.gz*
if [ -e $filename ]; then
    mv $filename tmpfile
    rm -f $removerstr
    mv tmpfile $filename
else
    rm -f $removerstr
    curl $url -o $filename -L
fi

cd $prev
