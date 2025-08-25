rem Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)

set prev=%cd%
set current=%~dp0
cd %current%\..\deps

rem pybind11
rmdir /s /q pybind11
mkdir pybind11
for %%f in (pybind11-*.tar.gz) do (
    tar xzf %%f --strip-components 1 -C pybind11
)

rem eigen
rmdir /s /q eigen
mkdir eigen
for %%f in (eigen*.tar.gz) do (
    tar xzf %%f --strip-components 1 -C eigen
)

rem nlopt
rmdir /s /q nlopt
mkdir nlopt
for %%f in (eigen*.tar.gz) do (
    tar xzf %%f --strip-components 1 -C nlopt
)

rem nlohmann json
rmdir /s /q json
mkdir json
for %%f in (json*.tar.gz) do (
    tar xzf %%f --strip-components 1 -C json
)

rem magic_enum
rmdir /s /q magic_enum
mkdir magic_enum

for %%f in (magic_enum*.tar.gz) do (
    tar xzf %%f --strip-components 1 -C magic_enum
)

rem thread-pool
rmdir /s /q thread-pool
mkdir thread-pool

for %%f in (thread-pool*.tar.gz) do (
    tar xzf %%f --strip-components 1 -C thread-pool
)

rem boost
rmdir /s /q boost
mkdir boost
for %%f in (boost*.zip) do (
    tar xzf %%f --strip-components 1 -C boost
)

rem cereal
rmdir /s /q cereal
mkdir cereal
for %%f in (cereal*.tar.gz) do (
    tar xzf %%f --strip-components 1 -C cereal
)

cd %prev%
