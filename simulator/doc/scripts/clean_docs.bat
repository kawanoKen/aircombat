rem Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)

set prev=%cd%
set current=%~dp0
cd %current%\deps

rmdir /s /q doc/html
rmdir /s /q doc/tags

cd %prev%
