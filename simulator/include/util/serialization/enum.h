// Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
//
// enum classを文字列として扱うためのユーティリティ。
//
#pragma once
#include <pybind11/pybind11.h>
#include <nlohmann/json.hpp>
#include <magic_enum/magic_enum.hpp>
#include <cereal/cereal.hpp>
#include <cereal/archives/portable_binary.hpp>
#include "../macros/common_macros.h"
#include "../../traits/json.h"

namespace py=pybind11;
namespace nl=nlohmann;

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)
ASRC_INLINE_NAMESPACE_BEGIN(util)

template<class E, std::enable_if_t<std::is_enum_v<E>,std::nullptr_t> = nullptr>
E strToEnum(const std::string& s){
    auto ret=magic_enum::enum_cast<E>(s);
    if(ret.has_value()){
        return ret.value();
    }else{
        throw std::runtime_error("'"+s+"' is not an element of '"+py::type_id<E>()+"'.");
    }
}
template<class E, std::enable_if_t<std::is_enum_v<E>,std::nullptr_t> = nullptr>
std::string enumToStr(const E& e){
    return std::string(magic_enum::enum_name(e));
}

template<class E, traits::basic_json BasicJsonType=nl::json, std::enable_if_t<std::is_enum_v<E>,std::nullptr_t> = nullptr>
E jsonToEnum(const BasicJsonType& j){
    return j.template get<E>();
}
template<class E, traits::basic_json BasicJsonType=nl::json, std::enable_if_t<std::is_enum_v<E>,std::nullptr_t> = nullptr>
BasicJsonType enumToJson(const E& e){
    return e;
}

template<class E, std::enable_if_t<std::is_enum_v<E>,std::nullptr_t> = nullptr>
E binaryToEnum(const std::string& s){
    E ret;
    {std::istringstream iss(s);
        cereal::PortableBinaryInputArchive ar(iss);
        ar(ret);
    }
    return ret;
}
template<class E, std::enable_if_t<std::is_enum_v<E>,std::nullptr_t> = nullptr>
std::string enumToBinary(const E& e){
    std::stringstream oss;
    {
        cereal::PortableBinaryOutputArchive ar(oss);
        ar(e);
    }
    return oss.str();
}

ASRC_NAMESPACE_END(util)
ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)

/*
    enum classを文字列としてシリアライズするための関数を追加するためのマクロ。
    各enum classを定義した名前空間中に記述する。これを記述しなかった場合には数値型による変換となる。
*/
#define DEFINE_SERIALIZE_ENUM_AS_STRING(className) \
template<::asrc::core::traits::output_archive Archive> \
std::string save_minimal(const Archive & archive, const className& m){ \
    return ::asrc::core::util::enumToStr(m); \
} \
template<::asrc::core::traits::input_archive Archive> \
void load_minimal(const Archive & archive, className& m, const std::string& value){ \
    m=::asrc::core::util::strToEnum<className>(value); \
}
