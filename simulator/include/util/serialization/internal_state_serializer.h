// Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
//
// インスタンスを新規生成せず内部状態のみのシリアライゼーションを行うためのメンバ関数として
// void serializeInternalState(AvailableArchiveTypes&,bool)が実装されている場合、
// 通常のsave,load,serializeの代わりにそれが呼び出されるようにするためのラッパークラス。
//
// archive(value)の代わりにarchive(makeInternalStateSerializer(value,full))のように用いる。
// 直接 value.serializeInternalState(archive,full) を呼び出した場合とはjson archive内の階層が異なる。
// 前者の場合は次のjsonノードに移動して入出力が行われるのでNameValuePairとの併用が可能である。
// 後者の場合は現在のjsonノードに続けて入出力が行われるので階層を深くしたくない場合にはそちらを用いることになる。
// 基本的には前者(makeInternalStateSerializerを用いてラップ)を推奨する。
//
// valueとしてコンテナを与えた場合は、その要素型(key_type,value_type,mapped_type)について再帰的にserializeInternalStateの呼び出しを試みる。
// このとき、serializeInternalStateを持たない型であれば通常のarchive.operator()が使用される。
// bool型引数fullの値はコンテナの全要素について適用されるため、
// 要素ごとに切り替えたい場合はInternalStateSerializerを要素型とする新コンテナを別途作成し、
// 要素ごとにfullの値を設定した状態で、改めてその新コンテナに対してmakeInternalStateSerializer(value,full)を呼び出す必要がある。
//
#pragma once
#include <stack>
#include <memory>
#include <variant>
#include <pybind11/pybind11.h>
#include <nlohmann/json.hpp>
#include <cereal/cereal.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/external/base64.hpp>
#include <boost/iostreams/stream.hpp>
#include "../../util/macros/common_macros.h"
#include "../../traits/traits.h"
#include "NLJSONArchive.h"
#include "pyobject.h"
namespace py=pybind11;
namespace nl=nlohmann;

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)

ASRC_NAMESPACE_BEGIN(traits)
template<class T>
concept has_serialize_internal_state_by_dot = requires(T&& t, AvailableArchiveTypes& ar, bool b) {
    t.serializeInternalState(ar,b);
};
template<class T>
concept has_serialize_internal_state_by_arrow = requires(T&& t, AvailableArchiveTypes& ar, bool b) {
    t->serializeInternalState(ar,b);
};
template<class T>
concept has_serialize_internal_state = has_serialize_internal_state_by_dot<T> || has_serialize_internal_state_by_arrow<T>;
ASRC_NAMESPACE_END(traits)

ASRC_INLINE_NAMESPACE_BEGIN(util)

struct InternalStateSerializerTag {};
template<class T>
struct InternalStateSerializer;

template<class T>
requires (
    traits::has_serialize_internal_state<T>
    ||traits::stl_set<T>
    || traits::stl_multiset<T>
    || traits::stl_unordered_set<T>
    || traits::stl_unordered_multiset<T>
    || traits::stl_array<T>
    || traits::stl_deque<T>
    || traits::stl_forward_list<T>
    || traits::stl_list<T>
    || traits::stl_vector<T>
    || traits::stl_valarray<T>
    || traits::stl_map<T>
    || traits::stl_multimap<T>
    || traits::stl_unordered_map<T>
    || traits::stl_unordered_multimap<T>
    || traits::stl_pair<T>
    || traits::stl_tuple<T>
)
inline InternalStateSerializer<T> makeInternalStateSerializer(T&& value,bool full){
    return InternalStateSerializer<T>(std::forward<T>(value),full);
}
template<class T>
requires (!(
    traits::has_serialize_internal_state<T>
    ||traits::stl_set<T>
    || traits::stl_multiset<T>
    || traits::stl_unordered_set<T>
    || traits::stl_unordered_multiset<T>
    || traits::stl_array<T>
    || traits::stl_deque<T>
    || traits::stl_forward_list<T>
    || traits::stl_list<T>
    || traits::stl_vector<T>
    || traits::stl_valarray<T>
    || traits::stl_map<T>
    || traits::stl_multimap<T>
    || traits::stl_unordered_map<T>
    || traits::stl_unordered_multimap<T>
    || traits::stl_pair<T>
    || traits::stl_tuple<T>
    || traits::pyobject<T>
))
inline T&& makeInternalStateSerializer(T&& value,bool full){
    return std::forward<T>(value);
}

template<traits::pyobject T>
inline InternalStateSerializer<T> makeInternalStateSerializer(T&& value,bool full){
    return InternalStateSerializer<T>(std::forward<T>(value),full);
}

template<traits::has_serialize_internal_state T>
struct InternalStateSerializer<T>: InternalStateSerializerTag{
    private:
    using StoredType = std::conditional_t<std::is_lvalue_reference_v<T>,T,std::decay_t<T>>;
    StoredType value;
    bool isFull;
    public:
    InternalStateSerializer(T&& value_,bool full):value(std::forward<T>(value_)),isFull(full){}
    template<class Archive>
    void serialize(Archive& archive){
        AvailableArchiveTypes wrapped(std::ref(archive));
        if constexpr(traits::pointer_like<T>){
            value->serializeInternalState(wrapped,isFull);
        }else{
            value.serializeInternalState(wrapped,isFull);
        }
    }
};

template<class T>
requires (
    traits::stl_set<T>
    || traits::stl_multiset<T>
    || traits::stl_unordered_set<T>
    || traits::stl_unordered_multiset<T>
)
struct InternalStateSerializer<T>: InternalStateSerializerTag {
    private:
    using StoredType = std::conditional_t<std::is_lvalue_reference_v<T>,T,std::decay_t<T>>;
    using key_type = typename traits::get_true_value_type_t<T>::key_type;
    StoredType value;
    bool isFull;
    public:
    InternalStateSerializer(T&& value_,bool full):value(std::forward<T>(value_)),isFull(full){}
    template<class Archive>
    void serialize(Archive& archive){
        cereal::size_type size=value.size();
        archive( cereal::make_size_tag( size ) );
        for(auto && v : value){
            archive(makeInternalStateSerializer(v,isFull));
        }
    }
};
template<class T>
requires (
    !traits::ordered_map<T>
    && (
        traits::stl_array<T>
        || traits::stl_deque<T>
        || traits::stl_forward_list<T>
        || traits::stl_list<T>
        || traits::stl_vector<T>
        || traits::stl_valarray<T>
    )
)
struct InternalStateSerializer<T>: InternalStateSerializerTag {
    private:
    using StoredType = std::conditional_t<std::is_lvalue_reference_v<T>,T,std::decay_t<T>>;
    using value_type = typename traits::get_true_value_type_t<T>::value_type;
    StoredType value;
    bool isFull;
    public:
    InternalStateSerializer(T&& value_,bool full):value(std::forward<T>(value_)),isFull(full){}
    template<class Archive>
    void serialize(Archive& archive){
        cereal::size_type size=value.size();
        archive( cereal::make_size_tag( size ) );
        for(auto && v : value){
            archive(makeInternalStateSerializer(v,isFull));
        }
    }
};
template<class T>
requires (
    traits::ordered_map<T>
    || traits::stl_map<T>
    || traits::stl_multimap<T>
    || traits::stl_unordered_map<T>
    || traits::stl_unordered_multimap<T>
)
struct InternalStateSerializer<T>: InternalStateSerializerTag {
    private:
    using StoredType = std::conditional_t<std::is_lvalue_reference_v<T>,T,std::decay_t<T>>;
    using key_type = typename traits::get_true_value_type_t<T>::key_type;
    using mapped_type = typename traits::get_true_value_type_t<T>::mapped_type;
    StoredType value;
    bool isFull;
    public:
    InternalStateSerializer(T&& value_,bool full):value(std::forward<T>(value_)),isFull(full){}
    template<class Archive>
    void serialize(Archive& archive){
        if constexpr (
            traits::string_like<key_type>
            && (
                traits::same_archive_as<Archive,::asrc::core::util::NLJSONOutputArchive>
                || traits::same_archive_as<Archive,::asrc::core::util::NLJSONInputArchive>
            )
        ){
            for( auto & [k,v] : value ){
                std::string key_str(k);
                archive(cereal::make_nvp<Archive>(key_str.c_str(),makeInternalStateSerializer(v,isFull)));
            }
        }else{
            cereal::size_type size=value.size();
            archive( cereal::make_size_tag( size ) );
            for( auto & [k,v] : value ){
                if constexpr (traits::output_archive<Archive>){
                    archive(
                        cereal::make_map_item(
                            k,
                            makeInternalStateSerializer(v,isFull)
                        )
                    );
                }else{
                    key_type key;
                    archive(
                        cereal::make_map_item(
                            key,
                            makeInternalStateSerializer(v,isFull)
                        )
                    );
                    assert(k==key);
                }
            }
        }
    }
};
template<traits::stl_pair T>
struct InternalStateSerializer<T>: InternalStateSerializerTag {
    private:
    using StoredType = std::conditional_t<std::is_lvalue_reference_v<T>,T,std::decay_t<T>>;
    using first_type = typename traits::get_true_value_type_t<T>::first_type;
    using second_type = typename traits::get_true_value_type_t<T>::second_type;
    StoredType value;
    bool isFull;
    public:
    InternalStateSerializer(T&& value_,bool full):value(std::forward<T>(value_)),isFull(full){}
    template<class Archive>
    void serialize(Archive& archive){
        if constexpr (
            traits::same_archive_as<Archive,::asrc::core::util::NLJSONOutputArchive>
            || traits::same_archive_as<Archive,::asrc::core::util::NLJSONInputArchive>
        ){
            cereal::size_type size=2;
            archive(cereal::make_size_tag( size ));
            archive(makeInternalStateSerializer(value.first,isFull));
            archive(makeInternalStateSerializer(value.second,isFull));
        }else{
            archive(CEREAL_NVP_("first",  makeInternalStateSerializer(value.first,isFull)));
            archive(CEREAL_NVP_("second",  makeInternalStateSerializer(value.second,isFull)));
        }
    }
};

template<traits::stl_tuple T>
struct InternalStateSerializer<T>: InternalStateSerializerTag {
    private:
    using StoredType = std::conditional_t<std::is_lvalue_reference_v<T>,T,std::decay_t<T>>;
    static constexpr size_t tuple_size = std::tuple_size_v<traits::get_true_value_type_t<T>>;
    StoredType value;
    bool isFull;
    public:
    InternalStateSerializer(T&& value_,bool full):value(std::forward<T>(value_)),isFull(full){}
    template<class Archive>
    void serialize(Archive& archive){
        if constexpr (
            traits::same_archive_as<Archive,::asrc::core::util::NLJSONOutputArchive>
            || traits::same_archive_as<Archive,::asrc::core::util::NLJSONInputArchive>
        ){
            cereal::size_type size=tuple_size;
            archive( cereal::make_size_tag( size ) );
        }
        sub<tuple_size>::template apply<Archive>(archive,this);
    }
    private:
    template<size_t Height>
    struct sub {
        template<class Archive>
        void apply(Archive& archive,InternalStateSerializer<T>* parent){
            if constexpr (Height>0){
                apply<Height-1>(archive);
                if constexpr (
                    traits::same_archive_as<Archive,::asrc::core::util::NLJSONOutputArchive>
                    || traits::same_archive_as<Archive,::asrc::core::util::NLJSONInputArchive>
                ){
                    archive(makeInternalStateSerializer(std::get<Height-1>(parent->value),parent->isFull));
                }else{
                    archive(CEREAL_NVP_(
                        cereal::tuple_detail::tuple_element_name<Height-1>::c_str(),
                        makeInternalStateSerializer(std::get<Height-1>(parent->value),parent->isFull)
                    ));
                }
            }
        }
    };
};

//
// AvailableArchiveTypesとboolの組を引数に取る任意の関数オブジェクトをArchiveのoperator()で呼び出せるようにするためのラッパー
//
struct PYBIND11_EXPORT VariantArchiveInternalStateSaverWrapper{
    std::function<void(AvailableOutputArchiveTypes&, bool)> saver;
    bool isFull;
    VariantArchiveInternalStateSaverWrapper(const std::function<void(AvailableOutputArchiveTypes&, bool)>& f, bool full):saver(f),isFull(full){}
    template<class Archive>
    void save(Archive & archive) const{
        AvailableOutputArchiveTypes wrapped(std::ref(archive));
        saver(wrapped,isFull);
    }
};
struct PYBIND11_EXPORT VariantArchiveInternalStateLoaderWrapper{
    std::function<void(AvailableInputArchiveTypes&, bool)> loader;
    bool isFull;
    VariantArchiveInternalStateLoaderWrapper(const std::function<void(AvailableInputArchiveTypes&,bool)>& f, bool full):loader(f),isFull(full){}
    template<class Archive>
    void load(Archive & archive){
        AvailableInputArchiveTypes wrapped(std::ref(archive));
        loader(wrapped,isFull);
    }
};
struct PYBIND11_EXPORT VariantArchiveInternalStateSerializerWrapper{
    std::function<void(AvailableArchiveTypes&, bool)> serializer;
    bool isFull;
    VariantArchiveInternalStateSerializerWrapper(const std::function<void(AvailableArchiveTypes&, bool)>& f, bool full):serializer(f),isFull(full){}
    template<class Archive>
    void serialize(Archive & archive){
        AvailableArchiveTypes wrapped(std::ref(archive));
        serializer(wrapped,isFull);
    }
};

template<traits::pyobject T>
struct InternalStateSerializer<T>: InternalStateSerializerTag{
    private:
    using StoredType = std::conditional_t<std::is_lvalue_reference_v<T>,T,std::decay_t<T>>;
    StoredType value;
    bool isFull;
    public:
    InternalStateSerializer(T&& value_,bool full):value(std::forward<T>(value_)),isFull(full){}
    template<class Archive>
    void serialize(Archive& archive){
        if(pybind11::hasattr(value,"serializeInternalState")){
            VariantArchiveInternalStateSerializerWrapper wrapped([this](AvailableArchiveTypes& ar, bool full){
                value.attr("serializeInternalState")(ar,full);
            },isFull);
            archive(wrapped);
        }else{
            serialize_with_type_info(archive,value);
        }
    }
};
template<traits::pyobject T>
inline void prologue(util::NLJSONOutputArchive & archive, const InternalStateSerializer<T>& obj){}
template<traits::pyobject T>
inline void epilogue(util::NLJSONOutputArchive & archive, const InternalStateSerializer<T>& obj){}
template<traits::pyobject T>
inline void prologue(util::NLJSONInputArchive & archive, const InternalStateSerializer<T>& obj){}
template<traits::pyobject T>
inline void epilogue(util::NLJSONInputArchive & archive, const InternalStateSerializer<T>& obj){}
template<traits::pyobject T>
inline void prologue(cereal::JSONOutputArchive & archive, const InternalStateSerializer<T>& obj){}
template<traits::pyobject T>
inline void epilogue(cereal::JSONOutputArchive & archive, const InternalStateSerializer<T>& obj){}
template<traits::pyobject T>
inline void prologue(cereal::JSONInputArchive & archive, const InternalStateSerializer<T>& obj){}
template<traits::pyobject T>
inline void epilogue(cereal::JSONInputArchive & archive, const InternalStateSerializer<T>& obj){}
template<traits::pyobject T>
inline void prologue(cereal::XMLOutputArchive & archive, const InternalStateSerializer<T>& obj){}
template<traits::pyobject T>
inline void epilogue(cereal::XMLOutputArchive & archive, const InternalStateSerializer<T>& obj){}
template<traits::pyobject T>
inline void prologue(cereal::XMLInputArchive & archive, const InternalStateSerializer<T>& obj){}
template<traits::pyobject T>
inline void epilogue(cereal::XMLInputArchive & archive, const InternalStateSerializer<T>& obj){}

ASRC_NAMESPACE_END(util)
ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
