#!/bin/bash -e
# Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#
# === directory specification of third-party dependencies ===
# source directory : ${PROJECT_ROOT_DIR}/thirdParty/deps/<each dependency>
# build directory : ${BUILD_DIR}/thirdParty/<each dependency>
# build destination : ${BUILD_DIR}/thirdParty/install/[bin|include|lib|share]
# install destination : ${PROJECT_ROOT_DIR}/${PACKAGE_NAME}/thirdParty/[bin|include|lib|share]

prev=$(pwd)
current=$(cd $(dirname $0); pwd)
cd $current/../deps

# pybind11
mkdir -p $BUILD_DIR/thirdParty/pybind11
cmake_args=(
    -S $PROJECT_ROOT_DIR/thirdParty/deps/pybind11
    -B $BUILD_DIR/thirdParty/pybind11
    -DCMAKE_INSTALL_PREFIX="$BUILD_DIR/thirdParty/install"
    -DPYBIND11_TEST=OFF
)
if [ -n "$ASRC_CMAKE_PREFIX_PATH" ]; then
    echo "ASRC_CMAKE_PREFIX_PATH detected! '$ASRC_CMAKE_PREFIX_PATH'"
    cmake_args+=(
        -DCMAKE_PREFIX_PATH="$ASRC_CMAKE_PREFIX_PATH"
    )
else
    echo "ASRC_CMAKE_PREFIX_PATH not detected!"
fi
if [ -n "$MSYS_ROOT_DIR" ]; then
    echo "MSYS detected! '$MSYS_ROOT_DIR'"
    cmake_args+=(
        -G "MSYS Makefiles"
        -DMSYS_ROOT_DIR="$MSYS_ROOT_DIR"
    )
else
    echo "MSYS not detected!"
fi
cmake "${cmake_args[@]}"
cd $BUILD_DIR/thirdParty/pybind11
make install
cd $BUILD_DIR
# apply modification
cp -r $PROJECT_ROOT_DIR/thirdParty/modification/pybind11/include/pybind11 $BUILD_DIR/thirdParty/install/include

# eigen
mkdir -p $BUILD_DIR/thirdParty/eigen
cmake_args=(
    -S $PROJECT_ROOT_DIR/thirdParty/deps/eigen
    -B $BUILD_DIR/thirdParty/eigen
    -DCMAKE_INSTALL_PREFIX="$BUILD_DIR/thirdParty/install"
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
cd $BUILD_DIR/thirdParty/eigen
make install
cd $BUILD_DIR

# nlopt
mkdir -p $BUILD_DIR/thirdParty/nlopt
cmake_args=(
    -S $PROJECT_ROOT_DIR/thirdParty/deps/nlopt
    -B $BUILD_DIR/thirdParty/nlopt
    -DCMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE
    -DCMAKE_INSTALL_PREFIX="$BUILD_DIR/thirdParty/install"
    -DCMAKE_INSTALL_LIBDIR=lib
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

cd $BUILD_DIR/thirdParty/nlopt
if [ "$(uname)" == 'Darwin' ]; then
    make -j $(sysctl -n hw.physicalcpu)
else
    make -j $(nproc --all)
fi
make install
cd $BUILD_DIR

#nlohmann json
mkdir -p $BUILD_DIR/thirdParty/json
cmake_args=(
    -S $PROJECT_ROOT_DIR/thirdParty/deps/json
    -B $BUILD_DIR/thirdParty/json
    -DCMAKE_INSTALL_PREFIX="$BUILD_DIR/thirdParty/install"
    -DJSON_BuildTests=OFF
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
cd $BUILD_DIR/thirdParty/json
make install
cd $BUILD_DIR

#magic_enum
mkdir -p $BUILD_DIR/thirdParty/magic_enum
cmake_args=(
    -S $PROJECT_ROOT_DIR/thirdParty/deps/magic_enum
    -B $BUILD_DIR/thirdParty/magic_enum
    -DCMAKE_INSTALL_PREFIX="$BUILD_DIR/thirdParty/install"
    -DMAGIC_ENUM_OPT_BUILD_EXAMPLES=OFF
    -DMAGIC_ENUM_OPT_BUILD_TESTS=OFF
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
cd $BUILD_DIR/thirdParty/magic_enum
make install
cd $BUILD_DIR

#thread-pool
mkdir -p $BUILD_DIR/thirdParty/install/include/thread-pool
cp $PROJECT_ROOT_DIR/thirdParty/deps/thread-pool/include/BS_thread_pool.hpp $BUILD_DIR/thirdParty/install/include/thread-pool/BS_thread_pool.hpp

# boost
cd $BUILD_DIR/thirdParty
mkdir boost
tar xzf $(find $PROJECT_ROOT_DIR/thirdParty/deps/ -maxdepth 1 -type f -name "boost*.tar.gz") --strip-components 1 -C boost
cd boost
if [ -n "$MSYS_ROOT_DIR" ]; then
    ./bootstrap.sh --prefix=$(cygpath -m $BUILD_DIR/thirdParty/install) --includedir=$(cygpath -m $BUILD_DIR/thirdParty/install/include)
else
    ./bootstrap.sh --prefix=$BUILD_DIR/thirdParty/install
fi
./b2 install --with-headers
cd $BUILD_DIR

#cereal
mkdir -p $BUILD_DIR/thirdParty/cereal
cmake_args=(
    -S $PROJECT_ROOT_DIR/thirdParty/deps/cereal
    -B $BUILD_DIR/thirdParty/cereal
    -DCMAKE_INSTALL_PREFIX="$BUILD_DIR/thirdParty/install/"
    -DJUST_INSTALL_CEREAL=ON
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
cd $BUILD_DIR/thirdParty/cereal
make install
cd $BUILD_DIR

cd $prev
