#!/bin/bash -e
# Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)

prev=$(pwd)
current=$(cd $(dirname $0); pwd)
cd $current/../deps

#pybind11
rm -rf pybind11
mkdir pybind11
tar xzf $(find . -maxdepth 1 -type f -name "pybind11-*.tar.gz") --strip-components 1 -C pybind11

#eigen
rm -rf eigen
mkdir eigen
tar xzf $(find . -maxdepth 1 -type f -name "eigen*.tar.gz") --strip-components 1 -C eigen

#nlopt
rm -rf nlopt
mkdir nlopt
tar xzf $(find . -maxdepth 1 -type f -name "nlopt*.tar.gz") --strip-components 1 -C nlopt

#nlohmann json
rm -rf json
mkdir json
tar xzf $(find . -maxdepth 1 -type f -name "json*.tar.gz") --strip-components 1 -C json

#magic_enum
rm -rf magic_enum
mkdir magic_enum
tar xzf $(find . -maxdepth 1 -type f -name "magic_enum*.tar.gz") --strip-components 1 -C magic_enum

#thread-pool
rm -rf thread-pool
mkdir thread-pool
tar xzf $(find . -maxdepth 1 -type f -name "thread-pool*.tar.gz") --strip-components 1 -C thread-pool

#boost (builder.shで展開する)
#rm -rf boost
#mkdir boost
#tar xzf $(find . -maxdepth 1 -type f -name "boost*.tar.gz") --strip-components 1 -C boost

#cereal
rm -rf cereal && mkdir cereal && tar xzf $(find . -maxdepth 1 -type f -name "cereal*.tar.gz") --strip-components 1 -C cereal

cd $prev
