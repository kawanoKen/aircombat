// Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
//
// ある型から他の型への変換を行う関数オブジェクトであるかどうかを判定するためのtraits
//
#pragma once
#include <type_traits>
#include <concepts>
#include <string>
#include "../util/macros/common_macros.h"
#include "is_string.h"

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)
ASRC_NAMESPACE_BEGIN(traits)

// TがSrcを引数にとりDst(の互換型)を返す関数オブジェクトかどうかを判定
template<class T, class Src, class Dst>
concept converter = requires (T& t, Src ins){
	{ t(std::forward<Src>(ins)) } -> std::convertible_to<Dst>;
};

// TがSrcを引数にとり文字列を返す関数オブジェクトかどうかを判定
template<class T, class Src>
concept stringifier = requires (T& t, Src ins){
	{ t(std::forward<Src>(ins)) } -> string_like;
};

ASRC_NAMESPACE_END(traits)
ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)

