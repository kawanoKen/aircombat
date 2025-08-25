rem Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)

set PROJECT_ROOT_DIR=%~dp0
set CMAKE_BUILD_TYPE=%1

if  %CMAKE_BUILD_TYPE% == Release (
    set BUILD_DIR=%PROJECT_ROOT_DIR%\build\Release
) else if  %CMAKE_BUILD_TYPE% == Debug (
    set BUILD_DIR=%PROJECT_ROOT_DIR%\build\Debug
) else (
    echo "Invalid build type. Expected [Release|Debug], but given " %CMAKE_BUILD_TYPE%
    exit /b 1
)
mkdir %BUILD_DIR% > NUL 2>&1

if exist %PROJECT_ROOT_DIR%\thirdParty\scripts (
    set WITH_THIRDPARTIES=ON
    mkdir %PROJECT_ROOT_DIR%\thirdParty\deps
    cd %PROJECT_ROOT_DIR%\thirdParty\scripts
    if exist fetcher.bat (
        call fetcher.bat
    )
    if exist extractor.bat (
        call extractor.bat
    )
    if exist builder.bat (
        call builder.bat
    )
    if exist thirdPartyConfigPath.cmake (
        mkdir %BUILD_DIR%\thirdParty\install
        copy thirdPartyConfigPath.cmake %BUILD_DIR%\thirdParty\install\thirdPartyConfigPath.cmake
    )
) else (
    set WITH_THIRDPARTIES=OFF
)
cd %BUILD_DIR%

set cmake_args=^
 -S %PROJECT_ROOT_DIR%^
 -B %BUILD_DIR%^
 -DTARGET_NAME=ASRCAISim1^
 -DCMAKE_BUILD_TYPE=%CMAKE_BUILD_TYPE%^
 -DWITH_THIRDPARTIES=%WITH_THIRDPARTIES%^
 -DPython3_INCLUDE_DIRS=%2^
 -DPython3_LIBRARY_DIRS=%3^
 -DPython3_Numpy_INCLUDE_DIRS=%4
if defined ASRC_CMAKE_PREFIX_PATH (
    if [%ASRC_CMAKE_PREFIX_PATH%] neq [] (
        set cmake_args=%cmake_args% -DCMAKE_PREFIX_PATH=%ASRC_CMAKE_PREFIX_PATH%
    )
)
cmake %cmake_args%
cmake --build . -j %NUMBER_OF_PROCESSORS% --config %CMAKE_BUILD_TYPE%
cmake --install .
