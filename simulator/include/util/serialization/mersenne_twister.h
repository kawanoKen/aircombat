// Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
//
//    std::mt19937 (std::mersenne_twister_engine)をシリアライズしたりpy::objectと相互変換する機能を提供する。
//    内部状態にアクセスする手段が文字列ストリームのみなのでそれを用いる。
//
#pragma once
#include <random>
#include <sstream>
#include <cereal/cereal.hpp>
#include <pybind11/pybind11.h>
#include "../../util/macros/common_macros.h"

namespace py=pybind11;

namespace cereal{
    template<class Archive,
        class UIntType, size_t w, size_t n, size_t m, size_t r,
        UIntType a, size_t u, UIntType d, size_t s,
        UIntType b, size_t t,
        UIntType c, size_t l, UIntType f
    >
    std::string save_minimal(const Archive & archive, const std::mersenne_twister_engine<UIntType,w,n,m,r,a,u,d,s,b,t,c,l,f>& mt){
        std::ostringstream oss;
        oss<<mt;
        return oss.str();
    }
    template<class Archive,
        class UIntType, size_t w, size_t n, size_t m, size_t r,
        UIntType a, size_t u, UIntType d, size_t s,
        UIntType b, size_t t,
        UIntType c, size_t l, UIntType f
    >
    void load_minimal(const Archive & archive, std::mersenne_twister_engine<UIntType,w,n,m,r,a,u,d,s,b,t,c,l,f>& mt, const std::string& value){
        std::istringstream iss(value);
        iss>>mt;
    }
    template<class Archive,
        class UIntType, size_t w, size_t n, size_t m, size_t r,
        UIntType a, size_t u, UIntType d, size_t s,
        UIntType b, size_t t,
        UIntType c, size_t l, UIntType f
    >
    struct specialize<Archive, std::mersenne_twister_engine<UIntType,w,n,m,r,a,u,d,s,b,t,c,l,f>, cereal::specialization::non_member_load_save_minimal> {};

}

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)
ASRC_INLINE_NAMESPACE_BEGIN(util)

void exportStdMersenneTwister(py::module &m);

ASRC_NAMESPACE_END(util)
ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
