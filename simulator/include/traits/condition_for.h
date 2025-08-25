// Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
//
// 複数の型に対して条件式(メタ関数)を適用して論理積・論理和を取るためのメタ関数
//
#pragma once
#include <type_traits>
#include "../util/macros/common_macros.h"

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)
ASRC_NAMESPACE_BEGIN(traits)

//
// 任意の数のテンプレートパラメータを一つのパラメータとしてパックして他のテンプレートに渡すための構造体。
//
template<class ... Types>
struct pack {};

//
// 第2引数のasrc::core::traits::packのパラメータとして与えた全ての型Tに対して条件式Cond<T, OtherArgs ...>の論理積をとる。
//
template<template<class...> class Cond, class CheckedTypePack, class ... OtherArgs>
struct conjunction_for;
template<template<class...> class Cond, class Head, class ... Others, class ... OtherArgs>
struct conjunction_for<Cond, pack<Head, Others...>, OtherArgs...> : std::conjunction<
    Cond<Head, OtherArgs ...>,
    conjunction_for<Cond, pack<Others ...>, OtherArgs...>
> {};
template<template<class...> class Cond, class Head, class ... OtherArgs>
struct conjunction_for<Cond, pack<Head>, OtherArgs ...> : Cond<Head, OtherArgs ...> {};

//
// 第2引数のasrc::core::traits::packのパラメータとして与えた全ての型Tに対して条件式Cond<T, OtherArgs ...>の論理和をとる。
//
template<template<class...> class Cond, class CheckedTypePack, class ... OtherArgs>
struct disjunction_for;
template<template<class...> class Cond, class Head, class ... Others, class ... OtherArgs>
struct disjunction_for<Cond, pack<Head, Others...>, OtherArgs...> : std::disjunction<
    Cond<Head, OtherArgs ...>,
    disjunction_for<Cond, pack<Others ...>, OtherArgs...>
> {};
template<template<class...> class Cond, class Head, class ... OtherArgs>
struct disjunction_for<Cond, pack<Head>, OtherArgs ...> : Cond<Head, OtherArgs ...> {};

ASRC_NAMESPACE_END(traits)
ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)

