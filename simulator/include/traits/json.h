// Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
//
// nlohmann's jsonの利用に関するtraits
//
#pragma once
#include <nlohmann/json.hpp>
#include <concepts>
#include "../util/macros/common_macros.h"
#include "pointer_like.h"
#include "forwarder.h"
#include "is_stl_container.h"
#include "is_string.h"
namespace nl=nlohmann;

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)
ASRC_NAMESPACE_BEGIN(traits)

//
// concept version of nlohmann json's traits
//
template<typename T>
using is_basic_json = nl::detail::is_basic_json<get_true_value_type_t<T>>;
template<typename T>
concept basic_json =is_basic_json<T>::value;


template<typename BasicJsonType, typename T>
concept has_to_json_by_default = requires (BasicJsonType& j, T&& val) {
    ::nl::to_json(j, std::forward<T>(val));
};
template<typename BasicJsonType, typename T>
concept has_from_json_by_default = requires (BasicJsonType&& j, T& val) {
    ::nl::from_json(std::forward<BasicJsonType>(j), val);
};
template<typename BasicJsonType, typename T>
concept has_non_default_from_json_by_default = requires (BasicJsonType&& j, nl::detail::identity_tag<T>& tag) {
    ::nl::from_json(std::forward<BasicJsonType>(j), tag);
};
template<typename BasicJsonType, typename T>
concept is_jsonable_by_default = has_to_json_by_default<BasicJsonType,T> && (has_from_json_by_default<BasicJsonType,T> || has_non_default_from_json_by_default<BasicJsonType,T>);

template<typename BasicJsonType, typename T>
concept has_to_json = std::conjunction_v<
    std::negation<nl::detail::is_basic_json<get_true_value_type_t<T>>>,
    nl::detail::has_to_json<BasicJsonType,nl::detail::uncvref_t<T>>
>;
template<typename BasicJsonType, typename T>
concept has_from_json = std::conjunction_v<
    std::negation<nl::detail::is_basic_json<get_true_value_type_t<T>>>,
    nl::detail::has_from_json<BasicJsonType,nl::detail::uncvref_t<T>>
>;
template<typename BasicJsonType, typename T>
concept has_non_default_from_json = std::conjunction_v<
    std::negation<nl::detail::is_basic_json<get_true_value_type_t<T>>>,
    nl::detail::has_non_default_from_json<BasicJsonType,nl::detail::uncvref_t<T>>
>;
template<typename BasicJsonType, typename T>
concept is_jsonable = basic_json<T> || (has_to_json<BasicJsonType,T> && (has_from_json<BasicJsonType,T> || has_non_default_from_json<BasicJsonType,T>));

//
// nlohmann's jsonのordered_mapかどうかを判定する
//
template<class T>
struct is_ordered_map_impl : std::false_type {};
template<class Key, class T, class IgnoredLess, class Allocator>
struct is_ordered_map_impl<nl::ordered_map<Key,T,IgnoredLess,Allocator>> : std::true_type {};
template<class T>
struct is_ordered_map {
    private:
    using Type = get_true_value_type_t<T>;
    public:
    static constexpr bool value = is_ordered_map_impl<Type>::value;
};
template<class T>
concept ordered_map = is_ordered_map<T>::value;

//
// nlohmann's jsonの標準のto_json/from_jsonを使用してシリアライズする型の判定
// 数値型、文字列型、basic_json、nlohmann::detail::value_t及びそれらを要素とするSTLコンテナ(コンテナのコンテナ及びコンテナの派生クラスを含む)が該当する。
//
template<typename T>
struct is_direct_serializable_with_json; // forward declaration

template<typename T>
struct is_direct_serializable_with_json_impl : std::disjunction<
    std::is_arithmetic<T>,
    is_string_like<T>,
    is_basic_json<T>,
    std::is_same<T,nlohmann::detail::value_t>
> {};

template<typename T>
requires (
    stl_set<T>
    || stl_multiset<T>
    || stl_unordered_set<T>
    || stl_unordered_multiset<T>
)
struct is_direct_serializable_with_json_impl<T> : is_direct_serializable_with_json<typename T::key_type> {};

template<typename T>
requires (
    stl_array<T>
    || stl_deque<T>
    || stl_forward_list<T>
    || stl_list<T>
    || stl_vector<T>
    || stl_valarray<T>
)
struct is_direct_serializable_with_json_impl<T> : is_direct_serializable_with_json<typename T::value_type> {};

template<typename T>
requires (
    stl_map<T>
    || stl_multimap<T>
    || stl_unordered_map<T>
    || stl_unordered_multimap<T>
)
struct is_direct_serializable_with_json_impl<T> : std::conjunction<
    is_direct_serializable_with_json_impl<typename T::key_type>,
    is_direct_serializable_with_json_impl<typename T::mapped_type>
> {};

template<stl_pair T>
struct is_direct_serializable_with_json_impl<T> {
    private:
    template<typename ... Args>
    requires (std::conjunction_v<is_direct_serializable_with_json<Args>...>)
    static std::true_type test(const std::pair<Args...>* v);
    static std::false_type test(...);

    public:
    static constexpr bool value = decltype(test(std::declval<T*>()))::value;
};

template<stl_tuple T>
struct is_direct_serializable_with_json_impl<T> {
    private:
    template<typename ... Args>
    requires (std::conjunction_v<is_direct_serializable_with_json<Args>...>)
    static std::true_type test(const std::tuple<Args...>* v);
    static std::false_type test(...);

    public:
    static constexpr bool value = decltype(test(std::declval<T*>()))::value;
};

template<typename T>
struct is_direct_serializable_with_json {
    using Type = get_true_value_type_t<T>;
    static constexpr bool value = is_direct_serializable_with_json_impl<Type>::value;
};
template<typename T>
concept direct_serializable_with_json = is_direct_serializable_with_json<T>::value;

ASRC_NAMESPACE_END(traits)
ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)

