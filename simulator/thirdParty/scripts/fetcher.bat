rem Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)

set prev=%cd%
set current=%~dp0
cd %current%\deps

if exist tmpfile (
    del tmpfile
)

rem pybind11
set url=https://github.com/pybind/pybind11/archive/refs/tags/v2.13.6.tar.gz
set filename=pybind11-2.13.6.tar.gz
set removerstr=pybind11-*.tar.gz*
if exist %filename% (
    move %filename% tmpfile
    del %removerstr%
    move tmpfile %filename%
) else (
    del %removerstr%
)
curl %url% -o %filename% -L

rem eigen
set url=https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
set filename=eigen-3.4.0.tar.gz
set removerstr=eigen*.tar.gz*
if exist %filename% (
    move %filename% tmpfile
    del %removerstr%
    move tmpfile %filename%
) else (
    del %removerstr%
)
curl %url% -o %filename% -L

rem nlopt
set url=https://github.com/stevengj/nlopt/archive/refs/tags/v2.7.1.tar.gz
set filename=nlopt-2.7.1.tar.gz
set removerstr=nlopt*.tar.gz*
if exist %filename% (
    move %filename% tmpfile
    del %removerstr%
    move tmpfile %filename%
) else (
    del %removerstr%
    curl %url% -o %filename% -L
)

rem nlohmann json
set url=https://github.com/nlohmann/json/archive/refs/tags/v3.12.0.tar.gz
set filename=json-3.12.0.tar.gz
set removerstr=json*.tar.gz*
if exist %filename% (
    move %filename% tmpfile
    del %removerstr%
    move tmpfile %filename%
) else (
    del %removerstr%
    curl %url% -o %filename% -L
)

rem magic_enum
set url=https://github.com/Neargye/magic_enum/archive/refs/tags/v0.9.7.tar.gz
set filename=magic_enum-0.9.7.tar.gz
set removerstr=magic_enum*.tar.gz*
if exist %filename% (
    move %filename% tmpfile
    del %removerstr%
    move tmpfile %filename%
) else (
    del %removerstr%
    curl %url% -o %filename% -L
)

rem thread-pool
set url=https://github.com/bshoshany/thread-pool/archive/refs/tags/v5.0.0.tar.gz
set filename=thread-pool-5.0.0.tar.gz
set removerstr=thread-pool*.tar.gz*
if exist %filename% (
    move %filename% tmpfile
    del %removerstr%
    move tmpfile %filename%
) else (
    del %removerstr%
    curl %url% -o %filename% -L
)

rem boost
set url=https://archives.boost.io/release/1.88.0/source/boost_1_88_0.zip
set filename=boost_1_88_0.zip
set removerstr=boost*.zip*
if exist %filename% (
    move %filename% tmpfile
    del %removerstr%
    move tmpfile %filename%
) else (
    del %removerstr%
    curl %url% -o %filename% -L
)

rem cereal
set url=https://github.com/USCiLab/cereal/archive/refs/tags/v1.3.2.tar.gz
set filename=cereal-1.3.2.tar.gz
set removerstr=cereal*.tar.gz*
if exist %filename% (
    move %filename% tmpfile
    del %removerstr%
    move tmpfile %filename%
) else (
    del %removerstr%
    curl %url% -o %filename% -L
)

cd %prev%
