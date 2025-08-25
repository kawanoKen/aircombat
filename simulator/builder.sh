#!/bin/bash
# Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)

PROJECT_ROOT_DIR=$(cd $(dirname $0); pwd)
CMAKE_BUILD_TYPE=$1

if [ $CMAKE_BUILD_TYPE = Release ]; then
    BUILD_DIR="$PROJECT_ROOT_DIR/build/Release"
elif [ $CMAKE_BUILD_TYPE = Debug ]; then
    BUILD_DIR="$PROJECT_ROOT_DIR/build/Debug"
else
    echo "Invalid build type. Expected [Release|Debug], but given " $CMAKE_BUILD_TYPE
    exit /b 1
fi
mkdir -p "$BUILD_DIR"

# check whether MSYS or not
if type cygpath > /dev/null 2>&1; then
    MSYS_ROOT_DIR=$(cygpath -m /)
fi

# check third-party dependencies
if [ -d $PROJECT_ROOT_DIR/thirdParty/scripts ]; then
    WITH_THIRDPARTIES=ON
    mkdir -p "$PROJECT_ROOT_DIR/thirdParty/deps"
    cd "$PROJECT_ROOT_DIR/thirdParty/scripts"
    if [ -e fetcher.sh ]; then
        source fetcher.sh
    fi
    if [ -e extractor.sh ]; then
        source extractor.sh
    fi
    if [ -e builder.sh ]; then
        source builder.sh
    fi
    if [ -e thirdPartyConfigPath.cmake ]; then
        mkdir -p "$BUILD_DIR/thirdParty/install"
        cp thirdPartyConfigPath.cmake "$BUILD_DIR/thirdParty/install/thirdPartyConfigPath.cmake"
    fi
else
    WITH_THIRDPARTIES=OFF
fi
cd $BUILD_DIR

# build and install
cmake_args=(
    -S $PROJECT_ROOT_DIR
    -B $BUILD_DIR
    -DTARGET_NAME=ASRCAISim1
    -DCMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE
    -DWITH_THIRDPARTIES=$WITH_THIRDPARTIES
    -DPython3_INCLUDE_DIRS="$2"
    -DPython3_LIBRARY_DIRS="$3"
    -DPython3_Numpy_INCLUDE_DIRS="$4"
)
if [ -n "$ASRC_CMAKE_PREFIX_PATH" ]; then
    cmake_args+=(
        -DCMAKE_PREFIX_PATH="$ASRC_CMAKE_PREFIX_PATH"
    )
fi

if [ -n "$MSYS_ROOT_DIR" ]; then
    cmake_args+=(
        -G "MSYS Makefiles"
        -DMSYS_ROOT_DIR="$MSYS_ROOT_DIR"
    )
fi
cmake "${cmake_args[@]}"

if [ "$(uname)" == 'Darwin' ]; then
    make -j $(sysctl -n hw.physicalcpu)
else
    make -j $(nproc --all)
fi
make install
