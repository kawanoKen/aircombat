#!/bin/bash -e
# Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)

prev=$(pwd)
current=$(cd $(dirname $0); pwd)
cd $current/../deps

#pybind11
rm -rf pybind11

#eigen
rm -rf eigen

#nlopt
rm -rf nlopt

#nlohmann json
rm -rf json

#magic_enum
rm -rf magic_enum

#thread-pool
rm -rf thread-pool

#boost
rm -rf boost

#cereal
rm -rf cereal

cd $prev
