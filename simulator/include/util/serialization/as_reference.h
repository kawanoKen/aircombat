// Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
//
// shared_ptrとweak_ptrを中身の値を含めず単なる参照としてシリアライズする機能を提供する。
// cerealのsmart pointerに関する機能をC++20のconceptを利用して上書きすることで実現している。
//
// あるクラスTに対して、shared_ptr<T>を引数に取る2つの関数save_as_referenceとload_as_referenceが定義されている場合、
// これらがcerealのシリアライズ関数として使用されるようになる。
// save_as_referenceとload_as_referenceは、cerealの本来のsave/loadと同様にメンバ関数又は非メンバ関数として実装してもよいし、
// staticメンバ関数として実装してもよい。
//
// クラスT側で何らかの形でインスタンスが管理されていなければならず、
// save_as_referenceとload_as_referenceはインスタンスを特定可能な最低限の情報(ID等)のみを読み書きする必要がある。
//
#pragma once
#include <cereal/cereal.hpp>
#include "../../traits/pointer_like.h"
#include "../../traits/cereal.h"

namespace cereal{
    template <class Archive, class T>
    requires (
        ::asrc::core::traits::has_save_as_reference<std::shared_ptr<T>,Archive>
    )
    inline void CEREAL_SAVE_FUNCTION_NAME( Archive & ar, std::shared_ptr<T> const & ptr )
    {
        if constexpr (::asrc::core::traits::has_static_member_save_as_reference<std::shared_ptr<T>,Archive>) {
            ::asrc::core::traits::get_true_value_type_t<std::shared_ptr<T>>::save_as_reference(ar,ptr);
        }else{ // if constexpr (::asrc::core::traits::has_non_member_save_as_reference<std::shared_ptr<T>,Archive>) {
            save_as_reference(ar,ptr);
        }
    }
    template <class Archive, class T>
    requires (
        ::asrc::core::traits::has_load_as_reference<std::shared_ptr<T>,Archive>
    )
    inline void CEREAL_LOAD_FUNCTION_NAME( Archive & ar, std::shared_ptr<T> & ptr )
    {
        if constexpr (::asrc::core::traits::has_static_member_load_as_reference<std::shared_ptr<T>,Archive>) {
            ::asrc::core::traits::get_true_value_type_t<std::shared_ptr<T>>::load_as_reference(ar,ptr);
        }else{ // if constexpr (::asrc::core::traits::has_non_member_load_as_reference<std::shared_ptr<T>,Archive>) {
            load_as_reference(ar,ptr);
        }
    }
    template <class Archive, class T>
    requires (
        ::asrc::core::traits::has_save_as_reference<std::shared_ptr<T>,Archive>
    )
    inline void CEREAL_SAVE_FUNCTION_NAME( Archive & ar, std::weak_ptr<T> const & ptr )
    {
        if(!ptr.expired()){
            std::shared_ptr<T> sptr=ptr.lock();
            CEREAL_SAVE_FUNCTION_NAME(ar,sptr);
        }else{
            std::shared_ptr<T> sptr=nullptr;
            CEREAL_SAVE_FUNCTION_NAME(ar,sptr);
        }
    }
    template <class Archive, class T>
    requires (
        ::asrc::core::traits::has_load_as_reference<std::shared_ptr<T>,Archive>
    )
    inline void CEREAL_LOAD_FUNCTION_NAME( Archive & ar, std::weak_ptr<T> & ptr )
    {
        std::shared_ptr<T> sptr;
        CEREAL_LOAD_FUNCTION_NAME(ar,sptr);
        ptr=sptr;
    }
}
