@echo off
rem Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)

set prev=%cd%
set current=%~dp0
cd %current%..\

set PackageName=%current%..\..\
for %%i in ("%PackageName:~0,-1%") do set PackageName=%%~nxi

for /f "usebackq" %%i in (python scripts\get_python_include_path.py) do set Python3_INCLUDE_DIRS=%%i
for /f "usebackq" %%i in (python scripts\get_numpy_include_path.py) do set Python3_Numpy_INCLUDE_DIRS=%%i

echo Python3_INCLUDE_DIRS= %Python3_INCLUDE_DIRS%
echo Python3_Numpy_INCLUDE_DIRS= %Python3_Numpy_INCLUDE_DIRS%

set RepoRoot=%current%..\..\..\..\..\
for %%i in ("%RepoRoot:~0,-1%") do set RepoRoot=%%~nxi

doxygen

cd %prev%
