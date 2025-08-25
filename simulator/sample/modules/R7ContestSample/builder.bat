rem Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)

set PROJECT_ROOT_DIR=%~dp0
set CMAKE_BUILD_TYPE=%1
set ASRC_CMAKE_PREFIX_PATH=%3
set ASRC_PATH_INFO=%4
set ASRC_CORE_PLUGIN_NAMES=%5
set ASRC_USER_PLUGIN_DIRS=%6

if [%7] == [] (
    if  %CMAKE_BUILD_TYPE% == Release (
        set BUILD_DIR=%PROJECT_ROOT_DIR%\build\Release
    ) else if  %CMAKE_BUILD_TYPE% == Debug (
        set BUILD_DIR=%PROJECT_ROOT_DIR%\build\Debug
    ) else (
        echo "Invalid build type. Expected [Release|Debug], but given " %CMAKE_BUILD_TYPE%
        exit /b 1
    )
) else (
    if  %CMAKE_BUILD_TYPE% == Release (
        set BUILD_DIR=%PROJECT_ROOT_DIR%\build\ReleaseB
    ) else if  %CMAKE_BUILD_TYPE% == Debug (
        set BUILD_DIR=%PROJECT_ROOT_DIR%\build\DebugB
    ) else (
        echo "Invalid build type. Expected [Release|Debug], but given " %CMAKE_BUILD_TYPE%
        exit /b 1
    )
)
mkdir %BUILD_DIR% > NUL 2>&1

if exist %PROJECT_ROOT_DIR%\thirdParty (
    set WITH_THIRDPARTIES=ON

    cd %PROJECT_ROOT_DIR%\thirdParty
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

if exist CMakeCache.txt (
    del CMakeCache.txt
)
set cmake_args=^
 -S %PROJECT_ROOT_DIR%^
 -B %BUILD_DIR%^
 -DTARGET_NAME=%2 ^
 -DCMAKE_BUILD_TYPE=%CMAKE_BUILD_TYPE%^
 -DWITH_THIRDPARTIES=%WITH_THIRDPARTIES%^
 -DASRC_PATH_INFO=%ASRC_PATH_INFO%^
 -DASRC_CORE_PLUGIN_NAMES=%ASRC_CORE_PLUGIN_NAMES%^
 -DASRC_USER_PLUGIN_DIRS=%ASRC_USER_PLUGIN_DIRS%
if [%7] neq [] (
    set cmake_args=%cmake_args% -DBUILD_IDENTIFIER=%7
)
if defined ASRC_CMAKE_PREFIX_PATH (
    if [%ASRC_CMAKE_PREFIX_PATH%] neq [] (
        set cmake_args=%cmake_args% -DASRC_CMAKE_PREFIX_PATH=%ASRC_CMAKE_PREFIX_PATH%
    )
)
cmake %cmake_args%
cmake --build . -j %NUMBER_OF_PROCESSORS% --config %CMAKE_BUILD_TYPE%
cmake --install .
