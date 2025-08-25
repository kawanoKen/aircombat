rem Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)

set prev=%cd%
set current=%~dp0
cd %current%\deps

rem pybind11
mkdir %BUILD_DIR%\thirdParty\pybind11
set cmake_args=^
 -S %PROJECT_ROOT_DIR%\thirdParty\deps\pybind11^
 -B %BUILD_DIR%\thirdParty\pybind11^
 -DCMAKE_INSTALL_PREFIX=%BUILD_DIR%\thirdParty\install^
 -DPYBIND11_TEST=OFF
if defined ASRC_CMAKE_PREFIX_PATH (
    if [%ASRC_CMAKE_PREFIX_PATH%] neq [] (
        set cmake_args=%cmake_args% -DCMAKE_PREFIX_PATH=%ASRC_CMAKE_PREFIX_PATH%
    )
)
cmake %cmake_args%
cd %BUILD_DIR%\thirdParty\eigen
cmake --install .
set ASRC_CMAKE_PREFIX_PATH=%BUILD_DIR%\thirdParty\install\share\eigen3\cmake;%ASRC_CMAKE_PREFIX_PATH%
cd %BUILD_DIR%
rem apply modification
cp -r %PROJECT_ROOT_DIR%\thirdParty\modification\pybind11\include\pybind11 %BUILD_DIR%\thirdParty\install\include

rem eigen
mkdir %BUILD_DIR%\thirdParty\eigen
set cmake_args=^
 -S %PROJECT_ROOT_DIR%\thirdParty\deps\eigen^
 -B %BUILD_DIR%\thirdParty\eigen^
 -DCMAKE_INSTALL_PREFIX=%BUILD_DIR%\thirdParty\install
if defined ASRC_CMAKE_PREFIX_PATH (
    if [%ASRC_CMAKE_PREFIX_PATH%] neq [] (
        set cmake_args=%cmake_args% -DCMAKE_PREFIX_PATH=%ASRC_CMAKE_PREFIX_PATH%
    )
)
cmake %cmake_args%
cd %BUILD_DIR%\thirdParty\eigen
cmake --install .
set ASRC_CMAKE_PREFIX_PATH=%BUILD_DIR%\thirdParty\install\share\eigen3\cmake;%ASRC_CMAKE_PREFIX_PATH%
cd %BUILD_DIR%

rem nlopt
mkdir %BUILD_DIR%\thirdParty\nlopt
set cmake_args=^
 -S %PROJECT_ROOT_DIR%\thirdParty\deps\nlopt^
 -B %BUILD_DIR%\thirdParty\nlopt^
 -DCMAKE_BUILD_TYPE=%CMAKE_BUILD_TYPE%^
 -DCMAKE_INSTALL_PREFIX=%BUILD_DIR%\thirdParty\install^
 -DCMAKE_INSTALL_LIBDIR=lib
if defined ASRC_CMAKE_PREFIX_PATH (
    if [%ASRC_CMAKE_PREFIX_PATH%] neq [] (
        set cmake_args=%cmake_args% -DCMAKE_PREFIX_PATH=%ASRC_CMAKE_PREFIX_PATH%
    )
)
cmake %cmake_args%
cd %BUILD_DIR%\thirdParty\nlopt
cmake --build . -j %NUMBER_OF_PROCESSORS% --config %CMAKE_BUILD_TYPE%
cmake --install .
set ASRC_CMAKE_PREFIX_PATH=%BUILD_DIR%\thirdParty\install\lib\cmake\nlopt;%ASRC_CMAKE_PREFIX_PATH%
cd %BUILD_DIR%

rem nlohmann json
mkdir %BUILD_DIR%\thirdParty\json
set cmake_args=^
 -S %PROJECT_ROOT_DIR%\thirdParty\deps\json^
 -B %BUILD_DIR%\thirdParty\json^
 -DCMAKE_INSTALL_PREFIX=%BUILD_DIR%\thirdParty\install^
 -DJSON_BuildTests=OFF
if defined ASRC_CMAKE_PREFIX_PATH (
    if [%ASRC_CMAKE_PREFIX_PATH%] neq [] (
        set cmake_args=%cmake_args% -DCMAKE_PREFIX_PATH=%ASRC_CMAKE_PREFIX_PATH%
    )
)
cmake %cmake_args%
cd %BUILD_DIR%\thirdParty\json
cmake --install .
set ASRC_CMAKE_PREFIX_PATH=%BUILD_DIR%\thirdParty\install\share\cmake\nlohmann_json;%ASRC_CMAKE_PREFIX_PATH%
cd %BUILD_DIR%

rem magic_enum
mkdir %BUILD_DIR%\thirdParty\magic_enum
set cmake_args=^
 -S %PROJECT_ROOT_DIR%\thirdParty\deps\magic_enum^
 -B %BUILD_DIR%\thirdParty\magic_enum^
 -DCMAKE_INSTALL_PREFIX=%BUILD_DIR%\thirdParty\install^
 -DMAGIC_ENUM_OPT_BUILD_EXAMPLES=OFF^
 -DMAGIC_ENUM_OPT_BUILD_TESTS=OFF
if defined ASRC_CMAKE_PREFIX_PATH (
    if [%ASRC_CMAKE_PREFIX_PATH%] neq [] (
        set cmake_args=%cmake_args% -DCMAKE_PREFIX_PATH=%ASRC_CMAKE_PREFIX_PATH%
    )
)
cmake %cmake_args%
cd %BUILD_DIR%\thirdParty\magic_enum
cmake --install .
set ASRC_CMAKE_PREFIX_PATH=%BUILD_DIR%\thirdParty\install\lib\cmake\magic_enum;%ASRC_CMAKE_PREFIX_PATH%
cd %BUILD_DIR%

rem thread-pool
mkdir %BUILD_DIR%\thirdParty\include/thread-pool
copy /y %PROJECT_ROOT_DIR%\thirdParty\deps\thread-pool\include\BS_thread_pool.hpp %BUILD_DIR%\thirdParty\install\include\thread-pool\BS_thread_pool.hpp

rem boost
xcopy /e /y %PROJECT_ROOT_DIR%\thirdParty\deps\boost\ %BUILD_DIR%\thirdParty\boost\
cd %BUILD_DIR%\thirdParty\boost
.\bootstrap.bat --prefix=%BUILD_DIR%\thirdParty\install
.\b2 install --with-headers
cd %BUILD_DIR%
set BOOST_ROOT=%BUILD_DIR%\thirdParty\install

rem cereal
mkdir %BUILD_DIR%\thirdParty\cereal
set cmake_args=^
 -S %PROJECT_ROOT_DIR%\thirdParty\deps\cereal^
 -B %BUILD_DIR%\thirdParty\cereal^
 -C %PROJECT_ROOT_DIR%\thirdParty\scripts\preloader.cmake^
 -DCMAKE_INSTALL_PREFIX=%BUILD_DIR%\thirdParty
if defined ASRC_CMAKE_PREFIX_PATH (
    if [%ASRC_CMAKE_PREFIX_PATH%] neq [] (
        set cmake_args=%cmake_args% -DCMAKE_PREFIX_PATH=%ASRC_CMAKE_PREFIX_PATH%
    )
)
cmake %cmake_args%
cd %BUILD_DIR%\thirdParty\cereal
cmake --install .
cd %BUILD_DIR%

cd %prev%
