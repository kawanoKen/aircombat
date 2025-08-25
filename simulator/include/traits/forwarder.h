// Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
//
// ある型Tから暗黙に変換可能な型のうち、一部の型への暗黙的な変換を禁止するためのラッパークラス
//
#pragma once
#include "../util/macros/common_macros.h"
#include "condition_for.h"

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)
ASRC_NAMESPACE_BEGIN(traits)

//
// ある型Tから暗黙に変換可能な型のうち、一部の型への暗黙的な変換を禁止するためのラッパーの基底クラス。
//
struct ForwarderBase {};

template<template<class> class Forwarder, class T,
    std::enable_if_t<
        std::is_base_of_v<ForwarderBase,Forwarder<T>>
    ,std::nullptr_t> = nullptr
>
struct make_forwarder {
    using type = Forwarder<std::remove_cv_t<std::remove_reference_t<T>>>;
};
template<template<class> class Forwarder, class T>
using make_forwarder_t = typename make_forwarder<Forwarder,T>::type;

//
// ある型Tから「CV修飾子又は参照修飾子の異なるT」以外への暗黙的な変換を禁止するためのラッパー。
// 大抵の型と変換できてしまう型(nl::json等)を経由して想定外のテンプレート実体化が成立してしまうのを抑制するためのSFINAEやconcept制約に用いる。
//
template<class T>
struct NoConvertForwarder : ForwarderBase {
    using RawType = std::remove_cv_t<std::remove_reference_t<T>>;
    using StoredType = std::conditional_t<std::is_reference_v<T>,T,std::decay_t<T>>;

    StoredType value;

    explicit NoConvertForwarder(T value_) : value(value_){}
    
    operator StoredType& () & {return value;}
    operator const StoredType& () const & {return value;}
    operator StoredType&& () && {return value;}
    operator const StoredType&& () const && {return value;}

    template<class Dest=StoredType, std::enable_if_t<!std::is_reference_v<Dest>, std::nullptr_t> = nullptr>
    Dest* operator& (){return std::addressof(value);}
    template<class Dest=StoredType, std::enable_if_t<!std::is_reference_v<Dest>, std::nullptr_t> = nullptr>
    const Dest* operator& () const {return std::addressof(value);}
};

//
// ある型Tから暗黙に変換可能な型のうち、BlackListで列挙した型から派生するもの(ポインタ型を含む)への暗黙的な変換を禁止するためのラッパー。
// 大抵の型と変換できてしまう型(nl::json等)を経由して想定外のテンプレート実体化が成立してしまうのを抑制するためのSFINAEやconcept制約に用いる。
//
struct BlackListedForwarderBase : ForwarderBase {};
template<class T, class ... BlackList>
struct BlackListedForwarder : BlackListedForwarderBase {
    using RawType = std::remove_cv_t<std::remove_reference_t<T>>;
    using StoredType = std::conditional_t<std::is_reference_v<T>,T,std::decay_t<T>>;
    template<class U, class Dest>
    struct Checker : std::negation<std::is_base_of<get_true_value_type_t<U>,get_true_value_type_t<Dest>>> {};
    template<class Dest>
    using Condition = conjunction_for<Checker, pack<BlackList ...>, Dest>;

    StoredType value;

    explicit BlackListedForwarder(T value_) : value(value_){}
    
    operator StoredType& () & {return value;}
    operator const StoredType& () const & {return value;}
    operator StoredType&& () && {return value;}
    operator const StoredType&& () const && {return value;}

    template<class Dest=StoredType, std::enable_if_t<!std::is_reference_v<Dest>, std::nullptr_t> = nullptr>
    Dest* operator& (){return std::addressof(value);}
    template<class Dest=StoredType, std::enable_if_t<!std::is_reference_v<Dest>, std::nullptr_t> = nullptr>
    const Dest* operator& () const {return std::addressof(value);}

    template <class Dest, std::enable_if_t<
        std::conjunction_v<
            std::negation<std::is_same<StoredType, std::conditional_t<std::is_reference_v<Dest>,Dest,std::decay_t<Dest>>>>,
            std::conjunction<
                Condition<Dest>,
                std::is_convertible<StoredType, Dest>
            >
        >
    , std::nullptr_t> = nullptr>
    operator Dest () {return value;}
    template <class Dest, std::enable_if_t<
        std::conjunction_v<
            std::negation<std::is_same<StoredType, std::conditional_t<std::is_reference_v<Dest>,Dest,std::decay_t<Dest>>>>,
            std::negation<std::conjunction<
                Condition<Dest>,
                std::is_convertible<StoredType, Dest>
            >>
        >
    , std::nullptr_t> = nullptr>
    operator Dest () = delete;
};

//
// ある型Tから暗黙に変換可能な型のうち、WhiteListで列挙した型から派生するもの(ポインタ型を含む)以外への暗黙的な変換を禁止するためのラッパー。
// 大抵の型と変換できてしまう型(nl::json等)を経由して想定外のテンプレート実体化が成立してしまうのを抑制するためのSFINAEやconcept制約に用いる。
// ただし、実際にこの型のインスタンスを生成してキャストする機能は設けていないので、あくまでメタ関数への引数として使用すること。
//
struct WhiteListedForwarderBase : ForwarderBase {};
template<class T, class ... WhiteList>
struct WhiteListedForwarder : WhiteListedForwarderBase {
    using RawType = std::remove_cv_t<std::remove_reference_t<T>>;
    using StoredType = std::conditional_t<std::is_reference_v<T>,T,std::decay_t<T>>;
    template<class U, class Dest>
    struct Checker : std::negation<std::is_base_of<get_true_value_type_t<U>,get_true_value_type_t<Dest>>> {};
    template<class Dest>
    using Condition = disjunction_for<Checker, pack<WhiteList ...>, Dest>;
    
    StoredType value;

    explicit WhiteListedForwarder(T value_) : value(value_){}

    operator StoredType& () & {return value;}
    operator const StoredType& () const & {return value;}
    operator StoredType&& () && {return value;}
    operator const StoredType&& () const && {return value;}

    template<class Dest=StoredType, std::enable_if_t<!std::is_reference_v<Dest>, std::nullptr_t> = nullptr>
    Dest* operator& (){return std::addressof(value);}
    template<class Dest=StoredType, std::enable_if_t<!std::is_reference_v<Dest>, std::nullptr_t> = nullptr>
    const Dest* operator& () const {return std::addressof(value);}

    template <class Dest, std::enable_if_t<
        std::conjunction_v<
            std::negation<std::is_same<StoredType, std::conditional_t<std::is_reference_v<Dest>,Dest,std::decay_t<Dest>>>>,
            std::conjunction<
                Condition<Dest>,
                std::is_convertible<StoredType, Dest>
            >
        >
    , std::nullptr_t> = nullptr>
    operator Dest () {return value;}
    template <class Dest, std::enable_if_t<
        std::conjunction_v<
            std::negation<std::is_same<StoredType, std::conditional_t<std::is_reference_v<Dest>,Dest,std::decay_t<Dest>>>>,
            std::negation<std::conjunction<
                Condition<Dest>,
                std::is_convertible<StoredType, Dest>
            >>
        >
    , std::nullptr_t> = nullptr>
    operator Dest () = delete;
};

ASRC_NAMESPACE_END(traits)
ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)

