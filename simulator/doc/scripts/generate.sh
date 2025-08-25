#!/bin/bash
# Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)

prev=$(pwd)
current=$(cd $(dirname $0); pwd)
cd $current/.. #doc

mkdir -p html
mkdir -p tags

only_core=0
while (( $# > 0 ))
do 
    case $1 in
        # ...
        -c | --core)
            only_core=1
            ;;
        # ...
    esac
    shift
done

export Python3_INCLUDE_DIRS=$(python scripts/get_python_include_path.py)
export Python3_Numpy_INCLUDE_DIRS=$(python scripts/get_numpy_include_path.py)

echo "Python3_INCLUDE_DIRS=${Python3_INCLUDE_DIRS}"
echo "Python3_Numpy_INCLUDE_DIRS=${Python3_Numpy_INCLUDE_DIRS}"

export RepoRoot=$(cd $current/../..; pwd)

if [ $only_core = 1 ]; then
    # 本体部分のみ
    doxygen
else
    # 1巡目はタグファイルの生成
    ## 本体部分
    doxygen
    ## core plugin
    bash ../core_plugins/HandyRLUtility/doc/scripts/generate.sh
    bash ../core_plugins/MatchMaker/doc/scripts/generate.sh
    bash ../core_plugins/rayUtility/doc/scripts/generate.sh
    bash ../core_plugins/Supervised/doc/scripts/generate.sh
    bash ../core_plugins/torch_truncnorm/doc/scripts/generate.sh
    ## user plugin
    bash ../user_plugins/BasicAirToAirCombatModels01/doc/scripts/generate.sh
    bash ../user_plugins/BasicAgentUtility/doc/scripts/generate.sh
    bash ../user_plugins/R7ContestModels/doc/scripts/generate.sh
    ## sample
    bash ../sample/modules/cpp_template/doc/scripts/generate.sh
    bash ../sample/modules/py_template/doc/scripts/generate.sh
    bash ../sample/modules/R7ContestSample/doc/scripts/generate.sh

    # 2巡目はタグファイルを利用して相互参照の解決
    ## 本体部分
    doxygen
    ## core plugin
    bash ../core_plugins/HandyRLUtility/doc/scripts/generate.sh
    bash ../core_plugins/MatchMaker/doc/scripts/generate.sh
    bash ../core_plugins/rayUtility/doc/scripts/generate.sh
    bash ../core_plugins/Supervised/doc/scripts/generate.sh
    bash ../core_plugins/torch_truncnorm/doc/scripts/generate.sh
    ## user plugin
    bash ../user_plugins/BasicAirToAirCombatModels01/doc/scripts/generate.sh
    bash ../user_plugins/BasicAgentUtility/doc/scripts/generate.sh
    bash ../user_plugins/R7ContestModels/doc/scripts/generate.sh
    ## sample
    bash ../sample/modules/cpp_template/doc/scripts/generate.sh
    bash ../sample/modules/py_template/doc/scripts/generate.sh
    bash ../sample/modules/R7ContestSample/doc/scripts/generate.sh
fi

cd $prev
