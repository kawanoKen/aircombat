// Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
//
// ある型が文字列型であるかどうかを判定するためのtraits
//
#pragma once
#include <type_traits>
#include <concepts>
#include <string>
#include "../util/macros/common_macros.h"
#include "pointer_like.h"

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)
ASRC_NAMESPACE_BEGIN(traits)

// std::basic_stringの判定
template<class T>
struct is_std_basic_string : std::false_type {};

template<class charT, class traits, class Allocator>
struct is_std_basic_string<std::basic_string<charT,traits,Allocator>> : std::true_type {};

template<class T>
concept std_basic_string = is_std_basic_string<T>::value;

// 文字列リテラル(const char[N]等)の判定
template<class T>
struct is_string_literal : std::conjunction<
    std::is_array<T>,
    std::is_constructible<std::basic_string<get_true_value_type_t<std::decay_t<T>>>,std::decay_t<T>>
> {};

template<class T>
concept string_literal = is_string_literal<T>::value;

// 文字列型(std::basic_stringまたは文字列リテラル)の判定
template<class T>
struct is_string_like : std::disjunction<
    is_std_basic_string<T>,
    is_string_literal<T>
> {};

template<class T>
concept string_like = std_basic_string<T> || string_literal<T>;

// ストリームへの読み書き可否
template<class T>
concept writeable_to_stream = requires (std::ostream& s, T& t) {
    s << t;
};
template<class T>
struct is_writeable_to_stream {
    static constexpr bool value = writeable_to_stream<T>;
};
template<class T>
concept readable_from_stream = requires (std::istream& s, T& t) {
    s >> t;
};
template<class T>
struct is_readable_from_stream {
    static constexpr bool value = readable_from_stream<T>;
};

ASRC_NAMESPACE_END(traits)
ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)

