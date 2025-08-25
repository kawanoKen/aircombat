// Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#pragma once
#include <ASRCAISim1/Common.h>
#include <BasicAirToAirCombatModels01/Common.h>

//このモジュールで定義するクラスのメンバ変数のうちSTLコンテナ型のものについて、
//Python側から書き換えたいものがある場合は、
//ヘッダ側でASRC_PYBIND11_MAKE_OPAQUEを、ソース側でasrc::core::bind_stl_container又はbind_stl_container_nameを呼ぶ必要がある。

//R7ContestMassPointFighter
ASRC_PYBIND11_MAKE_OPAQUE(std::map<std::string,Eigen::Tensor<double,2>>);

ASRC_PLUGIN_NAMESPACE_BEGIN

// 自作クラスではない要素型を持つSTLコンテナのバインドを行う。
// py::module_local(true)とするものはモジュールごとにコンパイルする必要があるため、各プラグインのエントリポイントからも呼び出す。
// 各モジュール上で実行されるようにするために、テンプレート関数として実装している。
template<typename Dummy=void>
void PYBIND11_EXPORT bindSTLContainer(py::module &m){

    ::asrc::core::bindSTLContainer(m); //bind standard STL containers from <ASRCAISim1/Common.h>

    //依存プラグインがbindSTLContainerを定義していればそれも呼び出す。
    BasicAirToAirCombatModels01::bindSTLContainer(m);

    using namespace asrc::core;
    using namespace util;

    //R7ContestMassPointFighter
    bind_stl_container<std::map<std::string,Eigen::Tensor<double,2>>>(m,py::module_local(true));

}

ASRC_PLUGIN_NAMESPACE_END
