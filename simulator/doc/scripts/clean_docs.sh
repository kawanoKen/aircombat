#!/bin/bash
# Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)

prev=$(pwd)
current=$(cd $(dirname $0); pwd)
cd $current/../..

rm -rf doc/html
rm -rf doc/tags

cd $prev
