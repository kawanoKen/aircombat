// Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
//
// ある型がSTLコンテナ又はその派生型であるかどうかを判定するためのtraits
//
#pragma once
#include <type_traits>
#include <concepts>
#include <array> //array
#include <bitset> //bitset
#include <deque> //deque
#include <forward_list> //forward_list
#include <list> //list
#include <map> //map,multi_map
#include <queue> //queue,priority_queue
#include <set> //set,multiset
#include <stack> //stack
#include <tuple> //tuple
#include <unordered_map> //unordered_map,unordered_multimap
#include <unordered_set> //unordered_set,unordered_multiset
#include <utility> //pair
#include <valarray> //valarray
#include <vector> //vector
#include "../util/macros/common_macros.h"
#include "is_derived_from_template.h"

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)
ASRC_NAMESPACE_BEGIN(traits)

template<class T>
struct is_stl_array {
    private:
    using Type=get_true_value_type_t<T>;
    template<class U, std::size_t N>
    static std::true_type test(const std::array<U,N>*);
    static std::false_type test(...);

    public:
    static constexpr bool value = decltype(test(std::declval<Type*>()))::value;
};
template<class T>
concept stl_array = is_stl_array<T>::value;

template<class T>
struct is_stl_bitset {
    private:
    using Type=get_true_value_type_t<T>;
    template<std::size_t N>
    static std::true_type test(const std::bitset<N>*);
    static std::false_type test(...);

    public:
    static constexpr bool value = decltype(test(std::declval<Type*>()))::value;
};
template<class T>
concept stl_bitset = is_stl_bitset<T>::value;

template<class T>
struct is_stl_deque : is_derived_from_template<T,std::deque> {};
template<class T>
concept stl_deque = derived_from_template<T,std::deque>;

template<class T>
struct is_stl_forward_list : is_derived_from_template<T,std::forward_list> {};
template<class T>
concept stl_forward_list = derived_from_template<T,std::forward_list>;

template<class T>
struct is_stl_list : is_derived_from_template<T,std::list> {};
template<class T>
concept stl_list = derived_from_template<T,std::list>;

template<class T>
struct is_stl_map : is_derived_from_template<T,std::map> {};
template<class T>
concept stl_map = derived_from_template<T,std::map>;

template<class T>
struct is_stl_multimap : is_derived_from_template<T,std::multimap> {};
template<class T>
concept stl_multimap = derived_from_template<T,std::multimap>;

template<class T>
struct is_stl_queue : is_derived_from_template<T,std::queue> {};
template<class T>
concept stl_queue = derived_from_template<T,std::queue>;

template<class T>
struct is_stl_priority_queue : is_derived_from_template<T,std::priority_queue> {};
template<class T>
concept stl_priority_queue = derived_from_template<T,std::priority_queue>;

template<class T>
struct is_stl_set : is_derived_from_template<T,std::set> {};
template<class T>
concept stl_set = derived_from_template<T,std::set>;

template<class T>
struct is_stl_multiset : is_derived_from_template<T,std::multiset> {};
template<class T>
concept stl_multiset = derived_from_template<T,std::multiset>;

template<class T>
struct is_stl_stack : is_derived_from_template<T,std::stack> {};
template<class T>
concept stl_stack = derived_from_template<T,std::stack>;

template<class T>
struct is_stl_tuple : is_derived_from_template<T,std::tuple> {};
template<class T>
concept stl_tuple = derived_from_template<T,std::tuple>;

template<class T>
struct is_stl_unordered_map : is_derived_from_template<T,std::unordered_map> {};
template<class T>
concept stl_unordered_map = derived_from_template<T,std::unordered_map>;

template<class T>
struct is_stl_unordered_multimap : is_derived_from_template<T,std::unordered_multimap> {};
template<class T>
concept stl_unordered_multimap = derived_from_template<T,std::unordered_multimap>;

template<class T>
struct is_stl_unordered_set : is_derived_from_template<T,std::unordered_set> {};
template<class T>
concept stl_unordered_set = derived_from_template<T,std::unordered_set>;

template<class T>
struct is_stl_unordered_multiset : is_derived_from_template<T,std::unordered_multiset> {};
template<class T>
concept stl_unordered_multiset = derived_from_template<T,std::unordered_multiset>;

template<class T>
struct is_stl_pair : is_derived_from_template<T,std::pair> {};
template<class T>
concept stl_pair = derived_from_template<T,std::pair>;

template<class T>
struct is_stl_valarray : is_derived_from_template<T,std::valarray> {};
template<class T>
concept stl_valarray = derived_from_template<T,std::valarray>;

template<class T>
struct is_stl_vector : is_derived_from_template<T,std::vector> {};
template<class T>
concept stl_vector = derived_from_template<T,std::vector>;

template<class T>
struct is_stl_container : std::disjunction<
    is_stl_array<T>,
    is_stl_bitset<T>,
    is_stl_deque<T>,
    is_stl_forward_list<T>,
    is_stl_list<T>,
    is_stl_map<T>,
    is_stl_multimap<T>,
    is_stl_queue<T>,
    is_stl_priority_queue<T>,
    is_stl_set<T>,
    is_stl_multiset<T>,
    is_stl_stack<T>,
    is_stl_tuple<T>,
    is_stl_unordered_map<T>,
    is_stl_unordered_multimap<T>,
    is_stl_unordered_set<T>,
    is_stl_unordered_multiset<T>,
    is_stl_pair<T>,
    is_stl_valarray<T>,
    is_stl_vector<T>
> {};
template<class T>
concept stl_container = (
    stl_array<T>
    || stl_bitset<T>
    || stl_deque<T>
    || stl_forward_list<T>
    || stl_list<T>
    || stl_map<T>
    || stl_multimap<T>
    || stl_queue<T>
    || stl_priority_queue<T>
    || stl_set<T>
    || stl_multiset<T>
    || stl_stack<T>
    || stl_tuple<T>
    || stl_unordered_map<T>
    || stl_unordered_multimap<T>
    || stl_unordered_set<T>
    || stl_unordered_multiset<T>
    || stl_pair<T>
    || stl_valarray<T>
    || stl_vector<T>
);

ASRC_NAMESPACE_END(traits)
ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)

