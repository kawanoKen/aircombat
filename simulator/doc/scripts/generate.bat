@echo off
rem Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)

set prev=%cd%
set current=%~dp0
cd %current%..\

mkdir html > NUL 2>&1
mkdir tags > NUL 2>&1

for /f "usebackq" %%i in (python scripts\get_python_include_path.py) do set Python3_INCLUDE_DIRS=%%i
for /f "usebackq" %%i in (python scripts\get_numpy_include_path.py) do set Python3_Numpy_INCLUDE_DIRS=%%i

echo Python3_INCLUDE_DIRS= %Python3_INCLUDE_DIRS%
echo Python3_Numpy_INCLUDE_DIRS= %Python3_Numpy_INCLUDE_DIRS%

set RepoRoot=%current%..\..\

rem 1巡目はタグファイルの生成
rem 本体部分
doxygen
rem core plugin
..\core_plugins\HandyRLUtility\doc\scripts\generate.bat
..\core_plugins\MatchMaker\doc\scripts\generate.bat
..\core_plugins\rayUtility\doc\scripts\generate.bat
..\core_plugins\Supervised\doc\scripts\generate.bat
..\core_plugins\torch_truncnorm\doc\scripts\generate.bat
rem user plugin
..\user_plugins\BasicAirToAirCombatModels01\doc\scripts\generate.bat
..\user_plugins\BasicAgentUtility\doc\scripts\generate.bat
..\user_plugins\R7ContestModels\doc\scripts\generate.bat
rem sample
..\sample\modules\cpp_template\doc\scripts\generate.bat
..\sample\modules\py_template\doc\scripts\generate.bat
..\sample\modules\R7ContestSample\doc\scripts\generate.bat

rem 2巡目はタグファイルを利用して相互参照の解決
rem 本体部分
doxygen
rem core plugin
..\core_plugins\HandyRLUtility\doc\scripts\generate.bat
..\core_plugins\MatchMaker\doc\scripts\generate.bat
..\core_plugins\rayUtility\doc\scripts\generate.bat
..\core_plugins\Supervised\doc\scripts\generate.bat
..\core_plugins\torch_truncnorm\doc\scripts\generate.bat
rem user plugin
..\user_plugins\BasicAirToAirCombatModels01\doc\scripts\generate.bat
..\user_plugins\BasicAgentUtility\doc\scripts\generate.bat
..\user_plugins\R7ContestModels\doc\scripts\generate.bat
rem sample
..\sample\modules\cpp_template\doc\scripts\generate.bat
..\sample\modules\py_template\doc\scripts\generate.bat
..\sample\modules\R7ContestSample\doc\scripts\generate.bat

cd %prev%
