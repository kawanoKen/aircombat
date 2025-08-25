// Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
//
// pointer-like typeの識別と、その参照先の値の型の取得
// https://stackoverflow.com/questions/49904809/
// ↑をC++20のconceptで書き直したもの
//
#pragma once
#include <type_traits>
#include <concepts>
#include "../util/macros/common_macros.h"

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)
ASRC_NAMESPACE_BEGIN(traits)

//
// pointer-like typeの識別
// https://stackoverflow.com/questions/49904809/
// ↑をC++20のconceptで書き直したもの
template<typename T>
concept dereferenceable = requires (T& t) {
    *t;
};
template<typename T>
struct is_dereferenceable : std::bool_constant<dereferenceable<T>> {};

template<typename T>
concept arrow_dereferenceable = requires (T& t) {
    t.operator->();
};
template<typename T>
struct is_arrow_dereferenceable : std::bool_constant<arrow_dereferenceable<T>> {};

template<typename T>
concept pointer = std::is_pointer<T>::value;

template<typename T>
struct is_pointer_like_dereferenceable : is_dereferenceable<T> {};
template<typename T>
concept pointer_like_dereferenceable = dereferenceable<T>;

template<typename T>
struct is_pointer_like_arrow_dereferenceable : std::disjunction<
    std::is_pointer<T>,
    is_arrow_dereferenceable<T>
> {};
template<typename T>
concept pointer_like_arrow_dereferenceable = is_pointer_like_arrow_dereferenceable<T>::value;
template<typename T>
struct is_pointer_like : std::conjunction<
    is_dereferenceable<T>,
    is_arrow_dereferenceable<T>
> {};
template<typename T>
concept pointer_like = pointer_like_dereferenceable<T> && pointer_like_arrow_dereferenceable<T>;


// dereference<T>
//   pointer-like型の参照先の値を得る。
//   operator*による間接参照を行った後の型を返すものとする。
//   shared_ptr<T>等はTではなくT&を返すので、Tそのものを得たい場合は別途remove_referenceで囲む必要がある。
template<typename T, bool = pointer_like<T>>
struct dereference {};
template<typename T>
struct dereference<T,true> {
    using type = decltype(*std::declval<T>());
};
template<typename T>
struct dereference<T,false> {
    using type = T;
};
template<typename T> using dereference_t = typename dereference<T>::type;

//
// get_true_value_type<T>
//   ポインタや参照型に対し、その参照先の値の型(CV修飾なし)を得る。
//   再帰処理は行わないので入れ子になった型(ダブルポインタ等)は最外の参照のみ外される。
template<typename T>
struct get_true_value_type {
    using type = std::remove_cv_t<std::remove_reference_t<dereference_t<T>>>;
};
template<typename T> using get_true_value_type_t = typename get_true_value_type<T>::type;

ASRC_NAMESPACE_END(traits)
ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)

