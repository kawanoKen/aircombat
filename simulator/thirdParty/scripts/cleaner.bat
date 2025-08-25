rem Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)

set prev=%cd%
set current=%~dp0
cd %current%\deps

rem pybind11
rmdir /s /q pybind11

rem eigen
rmdir /s /q eigen

rem nlopt
rmdir /s /q nlopt

rem nlohmann json
rmdir /s /q json

rem magic_enum
rmdir /s /q magic_enum

rem thread-pool
rmdir /s /q thread-pool

rem boost
rmdir /s /q boost

rem cereal
rmdir /s /q cereal

cd %prev%
