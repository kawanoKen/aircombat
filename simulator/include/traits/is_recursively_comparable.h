// Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
//
// コンテナの要素に対する再帰的な等値比較の可否に関するtraits
//
#pragma once
#include <concepts>
#include "../util/macros/common_macros.h"
#include "pointer_like.h"
#include "is_stl_container.h"

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)
ASRC_NAMESPACE_BEGIN(traits)

//
// std::equality_comparable<T>の再帰
//
template<typename T>
struct is_recursively_comparable; // forward declaration

template<typename T>
struct is_recursively_comparable_impl : std::bool_constant<std::equality_comparable<T>> {};

template<typename T>
requires (
    stl_set<T>
    || stl_multiset<T>
    || stl_unordered_set<T>
    || stl_unordered_multiset<T>
)
struct is_recursively_comparable_impl<T> : std::conjunction<
    std::bool_constant<std::equality_comparable<T>>,
    is_recursively_comparable<typename T::key_type>
> {};

template<typename T>
requires (
    stl_array<T>
    || stl_deque<T>
    || stl_forward_list<T>
    || stl_list<T>
    || stl_vector<T>
    || stl_valarray<T>
)
struct is_recursively_comparable_impl<T> : std::conjunction<
    std::bool_constant<std::equality_comparable<T>>,
    is_recursively_comparable<typename T::value_type>
> {};

template<typename T>
requires (
    stl_map<T>
    || stl_multimap<T>
    || stl_unordered_map<T>
    || stl_unordered_multimap<T>
)
struct is_recursively_comparable_impl<T> : std::conjunction<
    std::bool_constant<std::equality_comparable<T>>,
    is_recursively_comparable_impl<typename T::key_type>,
    is_recursively_comparable_impl<typename T::mapped_type>
> {};

template<template <typename...> class Tuple_, typename... Ts>
requires (
    stl_tuple<Tuple_<Ts...>>
    || stl_pair<Tuple_<Ts...>>
)
struct is_recursively_comparable_impl<Tuple_<Ts...>> {
    static constexpr bool value = (
        std::equality_comparable<Tuple_<Ts...>>
        && std::conjunction_v<is_recursively_comparable<Ts>...>
    );
};

template<typename T>
struct is_recursively_comparable {
    static constexpr bool value = is_recursively_comparable_impl<T>::value;
};
template<typename T>
concept recursively_comparable = is_recursively_comparable<T>::value;

//
// std::equality_comparable_with<T,U>の再帰
//
template<typename T,typename U>
struct is_recursively_comparable_with; // forward declaration

template<typename T,typename U>
struct is_recursively_comparable_with_impl : std::bool_constant<std::equality_comparable_with<T,U>> {};

template<typename T,typename U>
requires (
    (
        stl_set<T>
        || stl_multiset<T>
        || stl_unordered_set<T>
        || stl_unordered_multiset<T>
    )
    && (
        stl_set<U>
        || stl_multiset<U>
        || stl_unordered_set<U>
        || stl_unordered_multiset<U>
    )
)
struct is_recursively_comparable_with_impl<T,U> : std::conjunction<
    std::bool_constant<std::equality_comparable_with<T,U>>,
    is_recursively_comparable_with<typename T::key_type,typename U::key_type>
> {};

template<typename T,typename U>
requires (
    (
        stl_array<T>
        || stl_deque<T>
        || stl_forward_list<T>
        || stl_list<T>
        || stl_vector<T>
        || stl_valarray<T>
    )
    && (
        stl_array<U>
        || stl_deque<U>
        || stl_forward_list<U>
        || stl_list<U>
        || stl_vector<U>
        || stl_valarray<U>
    )
)
struct is_recursively_comparable_with_impl<T,U> : std::conjunction<
    std::bool_constant<std::equality_comparable_with<T,U>>,
    is_recursively_comparable_with<typename T::value_type,typename U::value_type>
> {};

template<typename T,typename U>
requires (
    (
        stl_map<T>
        || stl_multimap<T>
        || stl_unordered_map<T>
        || stl_unordered_multimap<T>
    )
    && (
        stl_map<U>
        || stl_multimap<U>
        || stl_unordered_map<U>
        || stl_unordered_multimap<U>
    )
)
struct is_recursively_comparable_with_impl<T,U> : std::conjunction<
    std::bool_constant<std::equality_comparable_with<T,U>>,
    is_recursively_comparable_with_impl<typename T::key_type,typename U::key_type>,
    is_recursively_comparable_with_impl<typename T::mapped_type,typename U::mapped_type>
> {};

template<template <typename...> class TTuple_, template <typename...> class UTuple_, typename... Ts, typename... Us>
requires (
    (
        stl_tuple<TTuple_<Ts...>>
        || stl_pair<TTuple_<Ts...>>
    )
    && (
        stl_tuple<UTuple_<Us...>>
        || stl_pair<UTuple_<Us...>>
    )
)
struct is_recursively_comparable_with_impl<TTuple_<Ts...>,UTuple_<Us...>> {
    static constexpr bool value = (
        std::equality_comparable_with<TTuple_<Ts...>,UTuple_<Us...>>
        && std::conjunction_v<is_recursively_comparable_with<Ts,Us>...>
    );
};

template<typename T,typename U>
struct is_recursively_comparable_with {
    static constexpr bool value = is_recursively_comparable_with_impl<T,U>::value;
};
template<typename T,typename U>
concept recursively_comparable_with = is_recursively_comparable_with<T,U>::value;

ASRC_NAMESPACE_END(traits)
ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)

