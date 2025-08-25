// Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#pragma once
#include <ASRCAISim1/Common.h>

//このモジュールで定義するクラスのメンバ変数のうちSTLコンテナ型のものについて、
//Python側から書き換えたいものがある場合は、
//ヘッダ側でASRC_PYBIND11_MAKE_OPAQUEを、ソース側でasrc::core::bind_stl_container又はbind_stl_container_nameを呼ぶ必要がある。

//BasicAACRuler01, BasicAACReward01
ASRC_PYBIND11_MAKE_OPAQUE(std::map<std::string,std::vector<std::string>>);
// ASRC_PYBIND11_MAKE_OPAQUE(std::map<std::string,std::map<std::string,double>>); //asrc::coreでMAKE_OPAQUE済
ASRC_PYBIND11_MAKE_OPAQUE(std::map<std::string,std::map<std::string,int>>);

ASRC_PLUGIN_NAMESPACE_BEGIN

// 自作クラスではない要素型を持つSTLコンテナのバインドを行う。
// py::module_local(true)とするものはモジュールごとにコンパイルする必要があるため、各プラグインのエントリポイントからも呼び出す。
// 各モジュール上で実行されるようにするために、テンプレート関数として実装している。
template<typename Dummy=void>
void PYBIND11_EXPORT bindSTLContainer(py::module &m){

    ::asrc::core::bindSTLContainer(m); //bind standard STL containers from <ASRCAISim1/Common.h>
    using namespace asrc::core;
    using namespace util;

    //BasicAACRuler01, BasicAACReward01
    bind_stl_container<std::map<std::string,std::vector<std::string>>>(m,py::module_local(true));
    //bind_stl_container<std::map<std::string,std::map<std::string,double>>>(m,,py::module_local(true)); //asrc::coreでbind済
    bind_stl_container<std::map<std::string,std::map<std::string,int>>>(m,py::module_local(true));
}

ASRC_PLUGIN_NAMESPACE_END
