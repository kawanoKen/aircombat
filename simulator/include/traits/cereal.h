// Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
//
// cerealの利用に関するtraits
//
#pragma once
#include <cereal/details/traits.hpp>
#include <cereal/details/helpers.hpp>
#include <concepts>
#include "../util/macros/common_macros.h"
#include "pointer_like.h"

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)
ASRC_NAMESPACE_BEGIN(traits)

template<typename T>
using is_minimal_type = cereal::traits::is_minimal_type<T>;
template<typename T>
concept cereal_minimal_type = is_minimal_type<T>::value;

template<class Archive, class KeyType=std::string>
concept archive_has_contains = std::derived_from<Archive,cereal::detail::InputArchiveBase> &&
requires (Archive& ar, KeyType&& key) {
    { ar.contains(std::forward<KeyType>(key)) } -> std::same_as<bool>;
};

//
// cereal用のtraitsをconcept化したものを定義する。
// cereal::traits内の従来のtraitと使い分ける際に名前空間を修正する手間を省くため、
// 定義したconceptに対応する従来型traitはasrc::core::traitsにも定義する。

//
// Archiveの種類に関するもの
//
template <class ArchiveT, class CerealArchiveT>
using is_same_archive = cereal::traits::is_same_archive<ArchiveT,CerealArchiveT>;
template <class ArchiveT, class CerealArchiveT>
concept same_archive_as = is_same_archive<ArchiveT,CerealArchiveT>::value;

template <class A>
using is_text_archive = cereal::traits::is_text_archive<A>;
template <class A>
concept text_archive = is_text_archive<A>::value;
template <class A>
concept non_text_archive = !text_archive<A>;

template <class A>
struct is_output_archive : std::is_base_of<cereal::detail::OutputArchiveBase,A> {};
template <class A>
concept output_archive = is_output_archive<A>::value;

template <class A>
struct is_input_archive : std::is_base_of<cereal::detail::InputArchiveBase,A> {};
template <class A>
concept input_archive = is_input_archive<A>::value;

template <class A>
struct is_cereal_archive : std::disjunction<
    is_output_archive<A>,
    is_input_archive<A>
> {};
template <class A>
concept cereal_archive = output_archive<A> || input_archive<A>;

//
// NameValuePairかどうかの判定に関するもの
//

//中身の型を問わずNameValuePair<T>を判定する。
template<typename T>
struct is_cereal_nvp_impl : std::false_type {};
template<typename T>
struct is_cereal_nvp_impl<cereal::NameValuePair<T>> : std::true_type {};
template<typename T>
struct is_cereal_nvp {
    private:
    using TT=std::remove_cv_t<std::remove_reference_t<T>>;
    public:
    static constexpr bool value=is_cereal_nvp_impl<TT>::value;
};
template<typename T>
concept cereal_nvp = is_cereal_nvp<T>::value;

//中身を指定してNameValuePair<U>を判定する。CV修飾と参照修飾を無視する。
template<typename T, typename U>
struct is_cereal_nvp_of_impl : std::false_type {};
template<typename T, typename U>
struct is_cereal_nvp_of_impl<cereal::NameValuePair<T>,U> : std::is_same<U,T> {};
template<typename T, typename U>
struct is_cereal_nvp_of_impl<cereal::NameValuePair<T&>,U> : std::is_same<U,T> {};
template<typename T, typename U>
struct is_cereal_nvp_of_impl<cereal::NameValuePair<T&&>,U> : std::is_same<U,T> {};
template<typename T, typename U>
struct is_cereal_nvp_of {
    private:
    using TT=std::remove_cv_t<std::remove_reference_t<T>>;
    using UU=std::remove_reference_t<U>;
    public:
    static constexpr bool value=is_cereal_nvp_of_impl<TT,UU>::value;
};
template<typename T, typename U>
concept cereal_nvp_of = cereal_nvp<T> && is_cereal_nvp_of<T,U>::value;

//NameValuePair<T>のうちTがUの派生クラスであるものを判定する。CV修飾と参照修飾を無視する。
template<typename T, typename U>
struct is_cereal_nvp_of_derived_from_impl : std::false_type {};
template<typename T, typename U>
struct is_cereal_nvp_of_derived_from_impl<cereal::NameValuePair<T>,U> : std::is_base_of<U,T> {};
template<typename T, typename U>
struct is_cereal_nvp_of_derived_from_impl<cereal::NameValuePair<T&>,U> : std::is_base_of<U,T> {};
template<typename T, typename U>
struct is_cereal_nvp_of_derived_from_impl<cereal::NameValuePair<T&&>,U> : std::is_base_of<U,T> {};
template<typename T, typename U>
struct is_cereal_nvp_of_derived_from {
    private:
    using TT=std::remove_cv_t<std::remove_reference_t<T>>;
    using UU=std::remove_reference_t<U>;
    public:
    static constexpr bool value=is_cereal_nvp_of_derived_from_impl<TT,UU>::value;
};
template<typename T, typename U>
concept cereal_nvp_of_derived_from = cereal_nvp<T> && is_cereal_nvp_of_derived_from<T,U>::value;

//
// シリアライズ可否に関するもの
//
template<typename T, class Archive>
using is_output_serializable = cereal::traits::is_output_serializable<T,Archive>;
template<typename T, class Archive>
concept output_serializable_by = is_output_serializable<T,Archive>::value;

template<typename T, class Archive>
using is_input_serializable = cereal::traits::is_input_serializable<T,Archive>;
template<typename T, class Archive>
concept input_serializable_by = is_input_serializable<T,Archive>::value;

template<typename T, class Archive>
struct is_two_way_serializable_o : std::conjunction<
    is_output_archive<Archive>,
    is_output_serializable<std::remove_const_t<T>,Archive>,
    is_input_serializable<std::remove_const_t<T>,typename cereal::traits::detail::get_input_from_output<Archive>::type>
> {};
template<typename T, class Archive>
struct is_two_way_serializable_i : std::conjunction<
    is_input_archive<Archive>,
    is_input_serializable<std::remove_const_t<T>,Archive>,
    is_output_serializable<std::remove_const_t<T>,typename cereal::traits::detail::get_output_from_input<Archive>::type>
> {};
template<typename T, class Archive>
struct is_two_way_serializable : std::disjunction<
    is_two_way_serializable_o<T,Archive>,
    is_two_way_serializable_i<T,Archive>
> {};
template<typename T, class Archive>
concept two_way_serializable_by_o = is_two_way_serializable_o<T,Archive>::value;
template<typename T, class Archive>
concept two_way_serializable_by_i = is_two_way_serializable_i<T,Archive>::value;
template<typename T, class Archive>
concept two_way_serializable_by = two_way_serializable_by_o<T,Archive> || two_way_serializable_by_i<T,Archive>;

//
// save_as_reference
//
template<typename T, class Archive>
concept has_static_member_save_as_reference = requires (T& t, Archive& ar) {
    get_true_value_type_t<T>::save_as_reference(ar, std::as_const(t));
};
template<typename T, class Archive>
concept has_non_member_save_as_reference = requires (T& t, Archive& ar) {
    save_as_reference(ar, std::as_const(t));
};
template<typename T, class Archive>
concept has_save_as_reference = (
    has_static_member_save_as_reference<T,Archive>
    || has_non_member_save_as_reference<T,Archive>
);

//
// load_as_reference
//
template<typename T, class Archive>
concept has_static_member_load_as_reference = requires (T& t, Archive& ar) {
    get_true_value_type_t<T>::load_as_reference(ar, t);
};
template<typename T, class Archive>
concept has_non_member_load_as_reference = requires (T& t, Archive& ar) {
    load_as_reference(ar, t);
};
template<typename T, class Archive>
concept has_load_as_reference = (
    has_static_member_load_as_reference<T,Archive>
    || has_non_member_load_as_reference<T,Archive>
);

//
// serialize
//
template<typename T, class Archive>
concept has_member_serialize_by_dot = requires (T& t, Archive& ar) {
    t.serialize(ar);
};
template<typename T, class Archive>
concept has_member_serialize_by_arrow = requires (T& t, Archive& ar) {
    t->serialize(ar);
};
template<typename T, class Archive>
concept has_member_serialize = has_member_serialize_by_dot<T,Archive> || has_member_serialize_by_arrow<T,Archive>;
template<typename T, class Archive>
concept has_static_member_serialize = requires (T& t, Archive& ar) {
    get_true_value_type_t<T>::serialize(ar,t);
};
template<typename T, class Archive>
concept has_non_member_serialize = requires (T& t, Archive& ar) {
    serialize(ar,t);
};
template<typename T, class Archive>
concept has_serialize = (
    has_member_serialize<T,Archive>
    || has_static_member_serialize<T,Archive>
    || has_non_member_serialize<T,Archive>
);

//
// versioned serialize
//
template<typename T, class Archive>
concept has_member_versioned_serialize_by_dot = requires (T& t, Archive& ar) {
    t.serialize(ar,0);
};
template<typename T, class Archive>
concept has_member_versioned_serialize_by_arrow = requires (T& t, Archive& ar) {
    t->serialize(ar,0);
};
template<typename T, class Archive>
concept has_member_versioned_serialize = has_member_versioned_serialize_by_dot<T,Archive> || has_member_versioned_serialize_by_arrow<T,Archive>;
template<typename T, class Archive>
concept has_static_member_versioned_serialize = requires (T& t, Archive& ar) {
    get_true_value_type_t<T>::serialize(ar,t,0);
};
template<typename T, class Archive>
concept has_non_member_versioned_serialize = requires (T& t, Archive& ar) {
    serialize(ar,t,0);
};
template<typename T, class Archive>
concept has_versioned_serialize = (
    has_member_versioned_serialize<T,Archive>
    || has_static_member_versioned_serialize<T,Archive>
    || has_non_member_versioned_serialize<T,Archive>
);

//
// save
//
#define ASRC_CEREAL_MAKE_VERSIONED_TEST ,0
#define ASRC_MAKE_HAS_SAVE_TEST(name,prefix,suffix,versioned) \
template<typename T, class Archive> \
concept has_member_##prefix##name##suffix##_by_dot = std::derived_from<Archive,cereal::detail::OutputArchiveBase> && \
requires (T& t, Archive& ar) { \
    std::as_const(t). name (ar versioned); \
}; \
template<typename T, class Archive> \
concept has_member_##prefix##name##suffix##_by_arrow = std::derived_from<Archive,cereal::detail::OutputArchiveBase> && \
requires (T& t, Archive& ar) { \
    std::as_const(t)-> name (ar versioned); \
}; \
template<typename T, class Archive> \
concept has_member_##prefix##name##suffix = has_member_##prefix##name##suffix##_by_dot<T,Archive> || has_member_##prefix##name##suffix##_by_arrow<T,Archive>; \
template<typename T, class Archive> \
concept has_static_member_##prefix##name##suffix = std::derived_from<Archive,cereal::detail::OutputArchiveBase> && \
requires (T& t, Archive& ar) { \
    get_true_value_type_t<T>:: name (ar,std::as_const(t) versioned); \
}; \
template<typename T, class Archive> \
concept has_non_member_##prefix##name##suffix = std::derived_from<Archive,cereal::detail::OutputArchiveBase> && \
requires (T& t, Archive& ar) { \
    name (ar,std::as_const(t) versioned); \
}; \
template<typename T, class Archive> \
concept has_##prefix##name##suffix = ( \
    has_member_##prefix##name##suffix<T,Archive> \
    || has_static_member_##prefix##name##suffix<T,Archive> \
    || has_non_member_##prefix##name##suffix<T,Archive> \
); \

//
// load
//
#define ASRC_MAKE_HAS_LOAD_TEST(name,prefix,suffix,versioned) \
template<typename T, class Archive> \
concept has_member_##prefix##name##suffix##_by_dot = std::derived_from<Archive,cereal::detail::InputArchiveBase> && \
requires (T& t, Archive& ar) { \
    t. name (ar versioned); \
}; \
template<typename T, class Archive> \
concept has_member_##prefix##name##suffix##_by_arrow = std::derived_from<Archive,cereal::detail::InputArchiveBase> && \
requires (T& t, Archive& ar) { \
    t-> name (ar versioned); \
}; \
template<typename T, class Archive> \
concept has_member_##prefix##name##suffix = has_member_##prefix##name##suffix##_by_dot<T,Archive> || has_member_##prefix##name##suffix##_by_arrow<T,Archive>; \
template<typename T, class Archive> \
concept has_static_member_##prefix##name##suffix = std::derived_from<Archive,cereal::detail::InputArchiveBase> && \
requires (T& t, Archive& ar) { \
    get_true_value_type_t<T>:: name (ar,t versioned); \
}; \
template<typename T, class Archive> \
concept has_non_member_##prefix##name##suffix = std::derived_from<Archive,cereal::detail::InputArchiveBase> && \
requires (T& t, Archive& ar) { \
    name (ar,t versioned); \
}; \
template<typename T, class Archive> \
concept has_##prefix##name##suffix = ( \
    has_member_##prefix##name##suffix<T,Archive> \
    || has_static_member_##prefix##name##suffix<T,Archive> \
    || has_non_member_##prefix##name##suffix<T,Archive> \
); \

ASRC_MAKE_HAS_SAVE_TEST(save,,,)
ASRC_MAKE_HAS_SAVE_TEST(save,versioned_,,ASRC_CEREAL_MAKE_VERSIONED_TEST)

ASRC_MAKE_HAS_SAVE_TEST(load,,,)
ASRC_MAKE_HAS_SAVE_TEST(load,versioned_,,ASRC_CEREAL_MAKE_VERSIONED_TEST)


//
// save_minimal
//
#define ASRC_MAKE_HAS_SAVE_MINIMAL_TEST(name,prefix,suffix,versioned) \
template<typename T, class Archive> \
concept has_member_##prefix##name##suffix##_by_dot = std::derived_from<Archive,cereal::detail::OutputArchiveBase> && \
requires (T& t, Archive& ar) { \
    { std::as_const(t). name (ar versioned) } -> cereal_minimal_type; \
}; \
template<typename T, class Archive> \
concept has_member_##prefix##name##suffix##_by_arrow = std::derived_from<Archive,cereal::detail::OutputArchiveBase> && \
requires (T& t, Archive& ar) { \
    { std::as_const(t)-> name (ar versioned) } -> cereal_minimal_type; \
}; \
template<typename T, class Archive> \
concept has_member_##prefix##name##suffix = has_member_##prefix##name##suffix##_by_dot<T,Archive> || has_member_##prefix##name##suffix##_by_arrow<T,Archive>; \
template<typename T, class Archive> \
concept has_static_member_##prefix##name##suffix = std::derived_from<Archive,cereal::detail::OutputArchiveBase> && \
requires (T& t, Archive& ar) { \
    { get_true_value_type_t<T>:: name (ar,std::as_const(t) versioned) } -> cereal_minimal_type; \
}; \
template<typename T, class Archive> \
concept has_non_member_##prefix##name##suffix = std::derived_from<Archive,cereal::detail::OutputArchiveBase> && \
requires (T& t, Archive& ar) { \
    { name (ar,std::as_const(t) versioned) } -> cereal_minimal_type; \
}; \
template<typename T, class Archive> \
concept has_##prefix##name##suffix = ( \
    has_member_##prefix##name##suffix<T,Archive> \
    || has_static_member_##prefix##name##suffix<T,Archive> \
    || has_non_member_##prefix##name##suffix<T,Archive> \
); \

//
// versioned load_minimal
//
#define ASRC_MAKE_HAS_LOAD_MINIMAL_TEST(name,saver_name,prefix,suffix,versioned) \
template<typename T, class Archive, class AOut=typename cereal::traits::detail::get_output_from_input<Archive>::type> \
concept has_member_##prefix##name##suffix##_by_dot = std::derived_from<Archive,cereal::detail::InputArchiveBase> && \
requires (T& t, Archive& ar, AOut& aout) { \
    t. name (ar,std::as_const(t). saver_name (aout versioned) versioned); \
}; \
template<typename T, class Archive, class AOut=typename cereal::traits::detail::get_output_from_input<Archive>::type> \
concept has_member_##prefix##name##suffix##_by_arrow = std::derived_from<Archive,cereal::detail::InputArchiveBase> && \
requires (T& t, Archive& ar, AOut& aout) { \
    t-> name (ar,std::as_const(t)-> saver_name (aout versioned) versioned); \
}; \
template<typename T, class Archive, class AOut=typename cereal::traits::detail::get_output_from_input<Archive>::type> \
concept has_member_##prefix##name##suffix = has_member_##prefix##name##suffix##_by_dot<T,Archive> || has_member_##prefix##name##suffix##_by_arrow<T,Archive>; \
template<typename T, class Archive, class AOut=typename cereal::traits::detail::get_output_from_input<Archive>::type> \
concept has_static_member_##prefix##name##suffix = std::derived_from<Archive,cereal::detail::InputArchiveBase> && \
requires (T& t, Archive& ar, AOut& aout) { \
    get_true_value_type_t<T>:: name (ar,t,get_true_value_type_t<T>:: saver_name (aout,std::as_const(t) versioned) versioned); \
}; \
template<typename T, class Archive, class AOut=typename cereal::traits::detail::get_output_from_input<Archive>::type> \
concept has_non_member_##prefix##name##suffix = std::derived_from<Archive,cereal::detail::InputArchiveBase> && \
requires (T& t, Archive& ar, AOut& aout) { \
    name(ar,t, saver_name (aout,std::as_const(t) versioned) versioned); \
}; \
template<typename T, class Archive> \
concept has_##prefix##name##suffix = ( \
    has_member_##prefix##name##suffix<T,Archive> \
    || has_static_member_##prefix##name##suffix<T,Archive> \
    || has_non_member_##prefix##name##suffix<T,Archive> \
); \

ASRC_MAKE_HAS_SAVE_MINIMAL_TEST(save_minimal,,,)
ASRC_MAKE_HAS_SAVE_MINIMAL_TEST(save_minimal,versioned_,,ASRC_CEREAL_MAKE_VERSIONED_TEST)
ASRC_MAKE_HAS_LOAD_MINIMAL_TEST(load_minimal,save_minimal,,,)
ASRC_MAKE_HAS_LOAD_MINIMAL_TEST(load_minimal,save_minimal,versioned_,,ASRC_CEREAL_MAKE_VERSIONED_TEST)

#undef ASRC_ASRC_CEREAL_MAKE_VERSIONED_TEST
#undef ASRC_MAKE_HAS_SAVE_TEST
#undef ASRC_MAKE_HAS_LOAD_TEST

ASRC_NAMESPACE_END(traits)
ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)

