// Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#pragma once
#include <ASRCAISim1/Common.h>
#include <ASRCAISim1/TimeSystem.h>
#include <ASRCAISim1/Fighter.h>
#include <ASRCAISim1/Missile.h>
#include <BasicAirToAirCombatModels01/Common.h>
#include <BasicAgentUtility/Common.h>
#include <R7ContestModels/Common.h>


//このモジュールで定義するクラスのメンバ変数のうちSTLコンテナ型のものについて、
//Python側から書き換えたいものがある場合は、
//ヘッダ側でASRC_PYBIND11_MAKE_OPAQUEを、ソース側でasrc::core::bind_stl_container又はbind_stl_container_nameを呼ぶ必要がある。

//R7ContestAgentSample01M,S
ASRC_PYBIND11_MAKE_OPAQUE(std::map<boost::uuids::uuid,asrc::core::Time>);

//R7ContestRewardSample01
ASRC_PYBIND11_MAKE_OPAQUE(std::map<std::string,Eigen::VectorX<bool>>);
ASRC_PYBIND11_MAKE_OPAQUE(std::vector<std::weak_ptr<asrc::core::Fighter>>);
ASRC_PYBIND11_MAKE_OPAQUE(std::map<std::string,std::vector<std::weak_ptr<asrc::core::Fighter>>>);
//ASRC_PYBIND11_MAKE_OPAQUE(std::vector<std::weak_ptr<asrc::core::Missile>>);
ASRC_PYBIND11_MAKE_OPAQUE(std::map<std::string,std::vector<std::weak_ptr<asrc::core::Missile>>>);
//R7ContestRewardSample02
ASRC_PYBIND11_MAKE_OPAQUE(std::unordered_map<std::string,bool>);
ASRC_PYBIND11_MAKE_OPAQUE(std::unordered_map<std::string,std::string>);

ASRC_PLUGIN_NAMESPACE_BEGIN

// 自作クラスではない要素型を持つSTLコンテナのバインドを行う。
// py::module_local(true)とするものはモジュールごとにコンパイルする必要があるため、各プラグインのエントリポイントからも呼び出す。
// 各モジュール上で実行されるようにするために、テンプレート関数として実装している。
template<typename Dummy=void>
void PYBIND11_EXPORT bindSTLContainer(py::module &m){

    ::asrc::core::bindSTLContainer(m); //bind standard STL containers from <ASRCAISim1/Common.h>

    //依存プラグインがbindSTLContainerを定義していればそれも呼び出す。
    BasicAirToAirCombatModels01::bindSTLContainer(m);
    BasicAgentUtility::bindSTLContainer(m);
    R7ContestModels::bindSTLContainer(m);

    using namespace asrc::core;
    using namespace util;

    //R7ContestAgentSample01M,S
    bind_stl_container<std::map<boost::uuids::uuid,Time>>(m,py::module_local(true));

    //R7ContestRewardSample01
    bind_stl_container<std::map<std::string,Eigen::VectorX<bool>>>(m,py::module_local(true));
    bind_stl_container<std::vector<std::weak_ptr<Fighter>>>(m,py::module_local(true));
    bind_stl_container<std::map<std::string,std::vector<std::weak_ptr<Fighter>>>>(m,py::module_local(true));
    //bind_stl_container<std::vector<std::weak_ptr<Missile>>>(m,py::module_local(true));
    bind_stl_container<std::map<std::string,std::vector<std::weak_ptr<Missile>>>>(m,py::module_local(true));

    bind_stl_container<std::unordered_map<std::string,bool>>(m,py::module_local(true));
    bind_stl_container<std::unordered_map<std::string,std::string>>(m,py::module_local(true));

}

ASRC_PLUGIN_NAMESPACE_END
