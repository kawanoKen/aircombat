// Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
//
// nlohmann's jsonとcerealの連接に関すること
//
#pragma once
#include <nlohmann/json.hpp>
#include <cereal/cereal.hpp>
#include "NLJSONArchive.h"
#include "enum.h"
#include "../../traits/cereal.h"
#include "../../traits/json.h"

namespace cereal{
    //
    // nlohmann::basic_json::value_t (enum)は文字列としてシリアライズ
    //
    template<::asrc::core::traits::output_archive Archive,
             template<typename, typename, typename...> class ObjectType,
             template<typename, typename...> class ArrayType,
             class StringType, class BooleanType, class NumberIntegerType,
             class NumberUnsignedType, class NumberFloatType,
             template<typename> class AllocatorType,
             template<typename, typename = void> class JSONSerializer,
             class BinaryType>
    std::string save_minimal(
        const Archive & archive,
        const typename nlohmann::basic_json<ObjectType, ArrayType, StringType, BooleanType,
            NumberIntegerType, NumberUnsignedType, NumberFloatType,
            AllocatorType, JSONSerializer, BinaryType
        >::value_t& m
    ){
        return ::asrc::core::util::enumToStr(m);
    }
    template<::asrc::core::traits::input_archive Archive,
             template<typename, typename, typename...> class ObjectType,
             template<typename, typename...> class ArrayType,
             class StringType, class BooleanType, class NumberIntegerType,
             class NumberUnsignedType, class NumberFloatType,
             template<typename> class AllocatorType,
             template<typename, typename = void> class JSONSerializer,
             class BinaryType>
    void load_minimal(
        const Archive & archive,
        typename nlohmann::basic_json<ObjectType, ArrayType, StringType, BooleanType,
            NumberIntegerType, NumberUnsignedType, NumberFloatType,
            AllocatorType, JSONSerializer, BinaryType
        >::value_t& m,
        const std::string& value
    ){
        using value_t=typename nlohmann::basic_json<ObjectType, ArrayType, StringType, BooleanType,
            NumberIntegerType, NumberUnsignedType, NumberFloatType,
            AllocatorType, JSONSerializer, BinaryType
        >::value_t;
        m=::asrc::core::util::strToEnum<value_t>(value);
    }

    //
    // nlohmann::ordered_mapはvectorをmapのように扱うものであり、emplace_hintを持たないため
    // cereal/types/concepts/pair_associative_conteiner.hppを使えない。個別に特殊化する必要がある。
    //
    template <::asrc::core::traits::output_archive Archive,
              class Key, class T, class IgnoredLess, class Allocator>
    requires (!::asrc::core::traits::same_archive_as<Archive, ::asrc::core::util::NLJSONOutputArchive>)
    inline void CEREAL_SAVE_FUNCTION_NAME( Archive & ar, nl::ordered_map<Key,T,IgnoredLess,Allocator> const & map )
    {
        ar( cereal::make_size_tag( static_cast<cereal::size_type>(map.size()) ) );

        for( const auto & i : map ){
            ar( cereal::make_map_item(i.first, i.second) );
        }
    }

    template <::asrc::core::traits::input_archive Archive,
              class Key, class T, class IgnoredLess, class Allocator>
    requires (!::asrc::core::traits::same_archive_as<Archive, ::asrc::core::util::NLJSONInputArchive>)
    inline void CEREAL_LOAD_FUNCTION_NAME( Archive & ar, nl::ordered_map<Key,T,IgnoredLess,Allocator> & map )
    {
        cereal::size_type size;
        ar( cereal::make_size_tag( size ) );

        map.clear();

        for( size_t i = 0; i < size; ++i )
        {
            Key key;
            T value;

            ar( cereal::make_map_item(key, value) );
            map.emplace_back( std::move( key ), std::move( value ) );
        }
    }

    //
    // nlohmann::basic_jsonをNLJSONOutputArchive,NLJSONInputArchive以外のArchiveでシリアライズする場合は値の種類(value_t)の情報を付加する
    //
    template<::asrc::core::traits::output_archive Archive, ::asrc::core::traits::basic_json T>
    requires (!::asrc::core::traits::same_archive_as<Archive, ::asrc::core::util::NLJSONOutputArchive>)
    inline void CEREAL_SAVE_FUNCTION_NAME(Archive & archive, const T& j){
        archive(cereal::make_nvp<Archive>("type",j.type()));
        if (j.is_null())
        {
            // skip
        }
        else if (j.is_boolean())
        {
            archive(cereal::make_nvp<Archive>("value",j.template get<typename T::boolean_t>()));
        }
        else if (j.is_number_unsigned())
        {
            archive(cereal::make_nvp<Archive>("value",j.template get<typename T::number_unsigned_t>()));
        }
        else if (j.is_number_integer())
        {
            archive(cereal::make_nvp<Archive>("value",j.template get<typename T::number_integer_t>()));
        }
        else if (j.is_number_float())
        {
            archive(cereal::make_nvp<Archive>("value",j.template get<typename T::number_float_t>()));
        }
        else if (j.is_string())
        {
            archive(cereal::make_nvp<Archive>("value",j.template get_ref<const typename T::string_t&>()));
        }
        else if (j.is_array())
        {
            archive(cereal::make_nvp<Archive>("value",j.template get_ref<const typename T::array_t&>()));
        }
        else if (j.is_object())
        {
            archive(cereal::make_nvp<Archive>("value",j.template get_ref<const typename T::object_t&>()));
        }
        else if (j.is_binary())
        {
            archive(cereal::make_nvp<Archive>("value",j.template get_ref<const typename T::binary_t&>()));
        }
        else
        {
            throw cereal::Exception("Invalid nlohmann::basic_json type to serialize.");
        }
    }
    
    template<::asrc::core::traits::input_archive Archive, ::asrc::core::traits::basic_json T>
    requires (!::asrc::core::traits::same_archive_as<Archive, ::asrc::core::util::NLJSONInputArchive>)
    inline void CEREAL_LOAD_FUNCTION_NAME(Archive & archive, T& j){
        typename T::value_t j_type;
        archive(cereal::make_nvp<Archive>("type",j_type));
        if (j_type == T::value_t::null)
        {
            j=T();
        }
        else if (j_type == T::value_t::boolean)
        {
            typename T::boolean_t v;
            archive(cereal::make_nvp<Archive>("value",v));
            j=v;
        }
        else if (j_type == T::value_t::number_unsigned)
        {
            typename T::number_unsigned_t v;
            archive(cereal::make_nvp<Archive>("value",v));
            j=v;
        }
        else if (j_type == T::value_t::number_integer)
        {
            typename T::number_integer_t v;
            archive(cereal::make_nvp<Archive>("value",v));
            j=v;
        }
        else if (j_type == T::value_t::number_float)
        {
            typename T::number_float_t v;
            archive(cereal::make_nvp<Archive>("value",v));
            j=v;
        }
        else if (j_type == T::value_t::string)
        {
            typename T::string_t v;
            archive(cereal::make_nvp<Archive>("value",v));
            j=v;
        }
        else if (j_type == T::value_t::array)
        {
            typename T::array_t v;
            archive(cereal::make_nvp<Archive>("value",v));
            j=v;
        }
        else if (j_type == T::value_t::object)
        {
            typename T::object_t v;
            archive(cereal::make_nvp<Archive>("value",v));
            j=v;
        }
        else if (j_type == T::value_t::binary)
        {
            typename T::binary_t v;
            archive(cereal::make_nvp<Archive>("value",v));
            j=v;
        }
        else
        {
            throw cereal::Exception("Invalid nlohmann::basic_json type to serialize.");
        }
    }

    template <class Archive,
        template<typename, typename, typename...> class ObjectType,
        template<typename, typename...> class ArrayType,
        class StringType, class BooleanType, class NumberIntegerType,
        class NumberUnsignedType, class NumberFloatType,
        template<typename> class AllocatorType,
        template<typename, typename = void> class JSONSerializer,
        class BinaryType
    >
    struct specialize<Archive, 
        nlohmann::basic_json<ObjectType, ArrayType, StringType, BooleanType, 
            NumberIntegerType, NumberUnsignedType, NumberFloatType, 
            AllocatorType, JSONSerializer, BinaryType
        >,
        cereal::specialization::non_member_load_save
    > {};
}


NLOHMANN_JSON_NAMESPACE_BEGIN
    //
    // nlohmann's jsonの標準の変換機能を使用してシリアライズを行う型(*1)以外の型のうち、
    // NLJSONOutputArchive,NLJSONInputArchiveで読み書きできる型について、
    // これらのArchiveを介してnlohmann::basic_jsonとの相互変換を行うものとする。
    //
    // *1 数値型、文字列型、basic_json、nlohmann::detail::value_t及びそれらを要素とする以下のSTLコンテナ(*2)(コンテナのコンテナ及びコンテナの派生クラスを含む)が除外される。
    // *2 set,multiset,unordered_set,unordered_multiset,
    //    array,deque,forward_list,list,vector,valarray,
    //    map,multimap,unordered_map,unordered_multimap,
    //    pair,tuple
    //
    template<typename T>
    requires(
        std::conjunction_v< //use std::conjunction for short-circuit evaluation
            std::negation<::asrc::core::traits::is_direct_serializable_with_json<T>>,
            ::asrc::core::traits::is_output_serializable<std::remove_const_t<T>,::asrc::core::util::NLJSONOutputArchive>,
            ::asrc::core::traits::is_input_serializable<std::remove_const_t<T>,::asrc::core::util::NLJSONInputArchive>
        >
    )
    struct adl_serializer<T>
    {
        template<typename BasicJsonType, typename TargetType=T>
        static void to_json(BasicJsonType& j, TargetType&& m) {
            if constexpr (std::same_as<std::remove_reference_t<BasicJsonType>,ordered_json>){
                ::asrc::core::util::NLJSONOutputArchive ar(j,true);
                ar(std::forward<TargetType>(m));
            }else{
                ordered_json oj;
                ::asrc::core::util::NLJSONOutputArchive ar(oj,true);
                ar(std::forward<TargetType>(m));
                j=std::move(oj);
            }
        }
        template<typename BasicJsonType, typename TargetType=T>
        static void from_json(BasicJsonType&& j, TargetType& m) {
            using Unref=std::remove_reference_t<BasicJsonType>;
            if constexpr (std::same_as<Unref,ordered_json>){
                ::asrc::core::util::NLJSONInputArchive ar(std::forward<BasicJsonType>(j),true);
                ar(std::forward<TargetType>(m));
            }else{
                ordered_json oj=std::forward<BasicJsonType>(j);
                ::asrc::core::util::NLJSONInputArchive ar(oj,true);
                ar(std::forward<TargetType>(m));
            }
        }
    };
NLOHMANN_JSON_NAMESPACE_END
