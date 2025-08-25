#!/bin/bash
# Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)

PROJECT_ROOT_DIR=$(cd $(dirname $0); pwd)
CMAKE_BUILD_TYPE=$1
export ASRC_CMAKE_PREFIX_PATH="$3"
export ASRC_PATH_INFO="$4"
export ASRC_CORE_PLUGIN_NAMES=$5
export ASRC_USER_PLUGIN_DIRS="$6"

if [ $# -lt 7 ]; then
    if [ $CMAKE_BUILD_TYPE = Release ]; then
        BUILD_DIR="$PROJECT_ROOT_DIR/build/Release"
    elif [ $CMAKE_BUILD_TYPE = Debug ]; then
        BUILD_DIR="$PROJECT_ROOT_DIR/build/Debug"
    else
        echo "Invalid build type. Expected [Release|Debug], but given " $CMAKE_BUILD_TYPE
        exit /b 1
    fi
else
    if [ $CMAKE_BUILD_TYPE = "Release" ]; then
        BUILD_DIR="$PROJECT_ROOT_DIR/build/ReleaseB"
    elif [ $CMAKE_BUILD_TYPE = "Debug" ]; then
        BUILD_DIR="$PROJECT_ROOT_DIR/build/DebugB"
    else
        echo "Invalid build type. Expected [Release|Debug], but given " $CMAKE_BUILD_TYPE
        exit /b 1
    fi
fi
mkdir -p "$BUILD_DIR"

# check whether MSYS or not
if type cygpath > /dev/null 2>&1; then
    MSYS_ROOT_DIR=$(cygpath -m /)
fi

# check third-party dependencies
if [ -d $PROJECT_ROOT_DIR/thirdParty ]; then
    WITH_THIRDPARTIES=ON

    mkdir -p "$PROJECT_ROOT_DIR/thirdParty/deps"
    cd "$PROJECT_ROOT_DIR/thirdParty"
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
    -DTARGET_NAME=$2
    -DCMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE
    -DWITH_THIRDPARTIES=$WITH_THIRDPARTIES
    -DASRC_PATH_INFO="$ASRC_PATH_INFO"
    -DASRC_CORE_PLUGIN_NAMES=$ASRC_CORE_PLUGIN_NAMES
    -DASRC_USER_PLUGIN_DIRS="$ASRC_USER_PLUGIN_DIRS"
)
if [ $# -ge 7 ]; then
    cmake_args+=(
        -DBUILD_IDENTIFIER=$7
    )
fi
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
