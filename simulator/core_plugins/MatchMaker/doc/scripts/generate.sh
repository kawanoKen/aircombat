#!/bin/bash
# Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)

prev=$(pwd)
current=$(cd $(dirname $0); pwd)
cd $current/.. #doc

export PackageName=$(basename $(cd ..; pwd))

export Python3_INCLUDE_DIRS=$(python scripts/get_python_include_path.py)
export Python3_Numpy_INCLUDE_DIRS=$(python scripts/get_numpy_include_path.py)

echo "Python3_INCLUDE_DIRS=${Python3_INCLUDE_DIRS}"
echo "Python3_Numpy_INCLUDE_DIRS=${Python3_Numpy_INCLUDE_DIRS}"

export RepoRoot=$(cd ../../..; pwd)

doxygen

cd $prev
