// Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
//
// Pythonオブジェクトをcerealでシリアライズする機能と、それを用いてnlohmann::basic_jsonを相互変換する機能
//
// pybind11::object等への代入は参照先を変える形となるため、基本的には既存のPythonオブジェクトに対するin-placeなloadができない。
// in-placeなloadに相当する処理として、以下の2種類を提供する。Python側からは(2)の利用を推奨する。なお、C++側で保持するpybind11::object型変数に対してはin-placeなloadによる参照先の変更で事足りることが多いと思われる。
//   (1)pybind11::objectインスタンスを返す関数(load_python_object)
//   (2)setattrを用いて他のオブジェクトの属性としてloadする関数(load_attr)
// また、load時には型情報を与えないと復元できないため、型情報の与え方として以下の2種類を提供する。(1)の利用を推奨する。
//   (1)saveの代わりにsave_with_type_infoを呼ぶことで型情報と一緒に保存しておき、loadの代わりにload_with_type_infoで同時に読み込む。
//   (2)save_type_info_ofを呼んで型情報を別途保存しておき、loadの代わりにload_with_external_type_infoでその型情報を引数に与えて読み込む。
//
// 上記の推奨する方式を踏まえて、Python側からは読み・書き共通のインターフェースとして以下の関数を用いることを推奨する。
//  (1) serialize_attr_with_type_info(archive: AvailableArchiveTypes ,obj: Any, *args: str)
//    objが持つargsで与えた属性について型情報とともに読み書きする。
//  (2) serialize_by_func(archive: AvailableArchiveTypes, func: Callable[[AvailableArchiveTypes],None])
//    serialize関数としてfunc(archive)を持つような仮想オブジェクトとして読み書きする。
//  (3) serialize_by_func(archive: AvailableArchiveTypes, attr_name: str, func: Callable[[AvailableArchiveTypes],None])
//    serialize関数としてfunc(archive)を持つような仮想オブジェクトを、attr_nameという名称を持つNameValuePairとして読み書きする。
//  (4) serialize_internal_state_by_func(archive: AvailableArchiveTypes, bool full, func: Callable[[AvailableArchiveTypes, bool],None])
//    serializeInternalState関数としてfunc(archive,full)を持つような仮想オブジェクトとして読み書きする。
//  (5) serialize_internal_state_by_func(archive: AvailableArchiveTypes, bool full, attr_name: str, func: Callable[[AvailableArchiveTypes, bool],None])
//    serializeInternalState関数としてfunc(archive,full)を持つような仮想オブジェクトを、attr_nameという名称を持つNameValuePairとして読み書きする。
//  (6) save_with_type_info(archive: AvailableOutputArchiveTypes, obj: Any)
//    objを型情報とともに書き込む。名前を指定しないのでarchiveによりデフォルトの名前が付与される。
//  (7) save_with_type_info(archive: AvailableOutputArchiveTypes, name: str, obj: Any)
//    nameという名前でcereal::NameValuePairを生成し、objを型情報とともに書き込む。
//  (8) load_with_type_info(archive: AvailableInputArchiveTypes) -> Any
//    型情報とともにobjectを読み込んで返す。名前を指定しないのでarchiveによりデフォルトの名前が付与される。
//  (9) load_with_type_info(archive: AvailableInputArchiveTypes name: str) -> Any
//    nameという名前でcereal::NameValuePairを生成し、型情報とともにobjectを読み込んで返す。
//
#pragma once
#include "../../util/macros/common_macros.h"
#include "pyobject_util.h"
#include "numpy.h"
#include "nljson.h"
#include "../../traits/traits.h"
#include <pybind11/pybind11.h>
#include <nlohmann/json.hpp>
#include <cereal/cereal.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/archives/xml.hpp>
#include <cereal/external/base64.hpp>
#include <cereal/types/map.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/utility.hpp>

// Pythonオブジェクトに対するprologue/epilogueのオーバーロード
// JSON,XMLのArchiveは「その他全て」枠でSFINAEによる実装がすでに行われているが、
// Pythonオブジェクトはその中身の型に応じてシリアライズ処理を分岐させたいので、
// Pythonオブジェクトとしてのprologue/epilogueを無効化する必要がある。
// 既存のArchiveの実装を変更せずに実現するためには、
// Archiveとデータ型の両方を特殊化した関数オーバーロードを用意する必要があると思われる。
// TODO よりスマートな方法で実現できる方法が見つかったら修正
#define ASRC_INTERNAL_CEREAL_PROLOGUE_EPILOGUE_DISABLER_FOR_PYTHON_OBJECT(Archive) \
inline void prologue(Archive & archive, const ::pybind11::handle& obj){} \
inline void epilogue(Archive & archive, const ::pybind11::handle& obj){} \
inline void prologue(Archive & archive, const ::pybind11::detail::kwargs_proxy& obj){} \
inline void epilogue(Archive & archive, const ::pybind11::detail::kwargs_proxy& obj){} \
inline void prologue(Archive & archive, const ::pybind11::detail::args_proxy& obj){} \
inline void epilogue(Archive & archive, const ::pybind11::detail::args_proxy& obj){} \
inline void prologue(Archive & archive, const ::pybind11::object& obj){} \
inline void epilogue(Archive & archive, const ::pybind11::object& obj){} \
inline void prologue(Archive & archive, const ::pybind11::iterator& obj){} \
inline void epilogue(Archive & archive, const ::pybind11::iterator& obj){} \
inline void prologue(Archive & archive, const ::pybind11::type& obj){} \
inline void epilogue(Archive & archive, const ::pybind11::type& obj){} \
inline void prologue(Archive & archive, const ::pybind11::iterable& obj){} \
inline void epilogue(Archive & archive, const ::pybind11::iterable& obj){} \
inline void prologue(Archive & archive, const ::pybind11::str& obj){} \
inline void epilogue(Archive & archive, const ::pybind11::str& obj){} \
inline void prologue(Archive & archive, const ::pybind11::bytes& obj){} \
inline void epilogue(Archive & archive, const ::pybind11::bytes& obj){} \
inline void prologue(Archive & archive, const ::pybind11::bytearray& obj){} \
inline void epilogue(Archive & archive, const ::pybind11::bytearray& obj){} \
inline void prologue(Archive & archive, const ::pybind11::none& obj){} \
inline void epilogue(Archive & archive, const ::pybind11::none& obj){} \
inline void prologue(Archive & archive, const ::pybind11::ellipsis& obj){} \
inline void epilogue(Archive & archive, const ::pybind11::ellipsis& obj){} \
inline void prologue(Archive & archive, const ::pybind11::bool_& obj){} \
inline void epilogue(Archive & archive, const ::pybind11::bool_& obj){} \
inline void prologue(Archive & archive, const ::pybind11::int_& obj){} \
inline void epilogue(Archive & archive, const ::pybind11::int_& obj){} \
inline void prologue(Archive & archive, const ::pybind11::float_& obj){} \
inline void epilogue(Archive & archive, const ::pybind11::float_& obj){} \
inline void prologue(Archive & archive, const ::pybind11::weakref& obj){} \
inline void epilogue(Archive & archive, const ::pybind11::weakref& obj){} \
inline void prologue(Archive & archive, const ::pybind11::slice& obj){} \
inline void epilogue(Archive & archive, const ::pybind11::slice& obj){} \
inline void prologue(Archive & archive, const ::pybind11::capsule& obj){} \
inline void epilogue(Archive & archive, const ::pybind11::capsule& obj){} \
inline void prologue(Archive & archive, const ::pybind11::tuple& obj){} \
inline void epilogue(Archive & archive, const ::pybind11::tuple& obj){} \
inline void prologue(Archive & archive, const ::pybind11::dict& obj){} \
inline void epilogue(Archive & archive, const ::pybind11::dict& obj){} \
inline void prologue(Archive & archive, const ::pybind11::sequence& obj){} \
inline void epilogue(Archive & archive, const ::pybind11::sequence& obj){} \
inline void prologue(Archive & archive, const ::pybind11::list& obj){} \
inline void epilogue(Archive & archive, const ::pybind11::list& obj){} \
inline void prologue(Archive & archive, const ::pybind11::args& obj){} \
inline void epilogue(Archive & archive, const ::pybind11::args& obj){} \
inline void prologue(Archive & archive, const ::pybind11::kwargs& obj){} \
inline void epilogue(Archive & archive, const ::pybind11::kwargs& obj){} \
inline void prologue(Archive & archive, const ::pybind11::anyset& obj){} \
inline void epilogue(Archive & archive, const ::pybind11::anyset& obj){} \
inline void prologue(Archive & archive, const ::pybind11::set& obj){} \
inline void epilogue(Archive & archive, const ::pybind11::set& obj){} \
inline void prologue(Archive & archive, const ::pybind11::frozenset& obj){} \
inline void epilogue(Archive & archive, const ::pybind11::frozenset& obj){} \
inline void prologue(Archive & archive, const ::pybind11::function& obj){} \
inline void epilogue(Archive & archive, const ::pybind11::function& obj){} \
inline void prologue(Archive & archive, const ::pybind11::staticmethod& obj){} \
inline void epilogue(Archive & archive, const ::pybind11::staticmethod& obj){} \
inline void prologue(Archive & archive, const ::pybind11::buffer& obj){} \
inline void epilogue(Archive & archive, const ::pybind11::buffer& obj){} \
inline void prologue(Archive & archive, const ::pybind11::memoryview& obj){} \
inline void epilogue(Archive & archive, const ::pybind11::memoryview& obj){} \
inline void prologue(Archive & archive, const ::pybind11::array& obj){} \
inline void epilogue(Archive & archive, const ::pybind11::array& obj){} \
template <typename T, int ExtraFlags> \
inline void prologue(Archive & archive, const ::pybind11::array_t<T,ExtraFlags>& obj){} \
template <typename T, int ExtraFlags> \
inline void epilogue(Archive & archive, const ::pybind11::array_t<T,ExtraFlags>& obj){} \
template<class Policy> \
inline void prologue(Archive & archive, const ::pybind11::detail::accessor<Policy>& obj){} \
template<class Policy> \
inline void epilogue(Archive & archive, const ::pybind11::detail::accessor<Policy>& obj){} \
template<::asrc::core::traits::pyobject Object, class Hint> \
inline void prologue(Archive & archive, const ::asrc::core::util::PyObjectWrapper<Object,Hint>& obj){} \
template<::asrc::core::traits::pyobject Object, class Hint> \
inline void epilogue(Archive & archive, const ::asrc::core::util::PyObjectWrapper<Object,Hint>& obj){} \

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)
ASRC_INLINE_NAMESPACE_BEGIN(util)
// forward declaration
template<traits::pyobject Object, class Hint=PyTypeInfo>
requires (std::same_as<traits::get_true_value_type_t<Hint>,PyTypeInfo>)
struct PyObjectWrapper;

template<traits::pyobject Object>
inline PyObjectWrapper<Object> makePyObjectWrapper(Object&& obj);
template<traits::pyobject Object, class Hint=PyTypeInfo>
requires (std::same_as<traits::get_true_value_type_t<Hint>,PyTypeInfo>)
inline PyObjectWrapper<Object,Hint> makePyObjectWrapper(Object&& obj,Hint&& hint);

// NLJSON archive用のprologue/epilogue
ASRC_INTERNAL_CEREAL_PROLOGUE_EPILOGUE_DISABLER_FOR_PYTHON_OBJECT(NLJSONOutputArchive)
ASRC_INTERNAL_CEREAL_PROLOGUE_EPILOGUE_DISABLER_FOR_PYTHON_OBJECT(NLJSONInputArchive)

ASRC_NAMESPACE_END(util)
ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)


namespace cereal{

// cereal標準のJSON,XML archive用のprologue/epilogue
ASRC_INTERNAL_CEREAL_PROLOGUE_EPILOGUE_DISABLER_FOR_PYTHON_OBJECT(JSONOutputArchive)
ASRC_INTERNAL_CEREAL_PROLOGUE_EPILOGUE_DISABLER_FOR_PYTHON_OBJECT(JSONInputArchive)
ASRC_INTERNAL_CEREAL_PROLOGUE_EPILOGUE_DISABLER_FOR_PYTHON_OBJECT(XMLOutputArchive)
ASRC_INTERNAL_CEREAL_PROLOGUE_EPILOGUE_DISABLER_FOR_PYTHON_OBJECT(XMLInputArchive)

template<class Archive, ::asrc::core::traits::pyobject Object>
inline void CEREAL_SERIALIZE_FUNCTION_NAME(Archive& archive, Object& obj){
    archive(::asrc::core::util::makePyObjectWrapper(obj));
}

template <class Archive, ::asrc::core::traits::pyobject Object>
struct specialize<Archive, Object, cereal::specialization::non_member_serialize> {};

}


ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)
ASRC_INLINE_NAMESPACE_BEGIN(util)

template<traits::pyobject Object, class Hint=PyTypeInfo>
requires (std::same_as<traits::get_true_value_type_t<Hint>,PyTypeInfo>)
struct PYBIND11_EXPORT Set{
    using StoredObjectType = std::conditional_t<std::is_lvalue_reference_v<Object>,Object,std::decay_t<Object>>;
    using StoredHintType = std::conditional_t<std::is_lvalue_reference_v<Hint>,Hint,std::decay_t<Hint>>;
    StoredObjectType obj;
    bool withHint;
    StoredHintType hint;
    Set(Object&& obj_):obj(std::forward<Object>(obj_)),withHint(false),hint(){}
    Set(Object&& obj_,Hint&& hint_):obj(std::forward<Object>(obj_)),withHint(true),hint(std::forward<Hint>(hint_)){}
    template<class Archive>
    void save(Archive& archive) const; //define later
    template<class Archive>
    inline void load(Archive& archive); //define later
};
template<traits::pyobject Object>
inline Set<Object> makeSet(Object&& obj_){
    return Set<Object>(std::forward<Object>(obj_));
}
template<traits::pyobject Object, class Hint=PyTypeInfo>
requires (std::same_as<traits::get_true_value_type_t<Hint>,PyTypeInfo>)
inline Set<Object,Hint> makeSet(Object&& obj_,Hint&& hint_){
    return Set<Object,Hint>(std::forward<Object>(obj_),std::forward<Hint>(hint_));
}

template<traits::pyobject Object, class Hint=PyTypeInfo>
requires (std::same_as<traits::get_true_value_type_t<Hint>,PyTypeInfo>)
struct PYBIND11_EXPORT Sequence{
    using StoredObjectType = std::conditional_t<std::is_lvalue_reference_v<Object>,Object,std::decay_t<Object>>;
    using StoredHintType = std::conditional_t<std::is_lvalue_reference_v<Hint>,Hint,std::decay_t<Hint>>;
    StoredObjectType obj;
    bool withHint;
    StoredHintType hint;
    Sequence(Object&& obj_):obj(std::forward<Object>(obj_)),withHint(false),hint(){}
    Sequence(Object&& obj_,Hint&& hint_):obj(std::forward<Object>(obj_)),withHint(true),hint(std::forward<Hint>(hint_)){}
    template<class Archive>
    inline void save(Archive& archive) const; //define later
    template<class Archive>
    inline void load(Archive& archive); //define later
};
template<traits::pyobject Object>
inline Sequence<Object> makeSequence(Object&& obj_){
    return Sequence<Object>(std::forward<Object>(obj_));
}
template<traits::pyobject Object, class Hint=PyTypeInfo>
requires (std::same_as<traits::get_true_value_type_t<Hint>,PyTypeInfo>)
inline Sequence<Object,Hint> makeSequence(Object&& obj_,Hint&& hint_){
    return Sequence<Object,Hint>(std::forward<Object>(obj_),std::forward<Hint>(hint_));
}

template<traits::pyobject Key,traits::pyobject Value,
    class KeyHint=PyTypeInfo, class ValueHint=PyTypeInfo
>
struct PYBIND11_EXPORT MappingItem{
    using StoredKeyType = std::conditional_t<std::is_lvalue_reference_v<Key>,Key,std::decay_t<Key>>;
    using StoredValueType = std::conditional_t<std::is_lvalue_reference_v<Value>,Value,std::decay_t<Value>>;
    using StoredKeyHintType = std::conditional_t<std::is_lvalue_reference_v<KeyHint>,KeyHint,std::decay_t<KeyHint>>;
    using StoredValueHintType = std::conditional_t<std::is_lvalue_reference_v<ValueHint>,ValueHint,std::decay_t<ValueHint>>;
    StoredKeyType key;
    StoredValueType value;
    bool withHint;
    StoredKeyHintType keyHint;
    StoredValueHintType valueHint;
    MappingItem(Key&& key_, Value&& value_):key(std::forward<Key>(key_)),value(std::forward<Value>(value_)),
                                            withHint(false),keyHint(),valueHint() {}
    MappingItem(Key&& key_, Value&& value_, KeyHint&& keyHint_, ValueHint&& valueHint_):key(std::forward<Key>(key_)),value(std::forward<Value>(value_)),
                                            withHint(true),keyHint(std::forward<KeyHint>(keyHint_)),valueHint(std::forward<ValueHint>(valueHint_)) {}
    template<class Archive>
    inline void save(Archive& archive) const; //define later
    template<class Archive>
    inline void load(Archive& archive); //define later
};
template<traits::pyobject Key,traits::pyobject Value>
inline MappingItem<Key,Value> makeMappingItem(Key&& key_,Value&& value_){
    return MappingItem<Key,Value>(std::forward<Key>(key_),std::forward<Value>(value_));
}
template<traits::pyobject Key,traits::pyobject Value,
    class KeyHint=PyTypeInfo,class ValueHint=PyTypeInfo
>
inline MappingItem<Key,Value,KeyHint,ValueHint> makeMappingItem(Key&& key_,Value&& value_,KeyHint&& keyHint_,ValueHint&& valueHint_){
    return MappingItem<Key,Value,KeyHint,ValueHint>(std::forward<Key>(key_),std::forward<Value>(value_),std::forward<KeyHint>(keyHint_),std::forward<ValueHint>(valueHint_));
}

template<traits::pyobject Object, class Hint=PyTypeInfo>
requires (std::same_as<traits::get_true_value_type_t<Hint>,PyTypeInfo>)
struct PYBIND11_EXPORT Mapping{
    using StoredObjectType = std::conditional_t<std::is_lvalue_reference_v<Object>,Object,std::decay_t<Object>>;
    using StoredHintType = std::conditional_t<std::is_lvalue_reference_v<Hint>,Hint,std::decay_t<Hint>>;
    StoredObjectType obj;
    bool withHint;
    StoredHintType hint;
    Mapping(Object&& obj_):obj(std::forward<Object>(obj_)),withHint(false),hint(){}
    Mapping(Object&& obj_,Hint&& hint_):obj(std::forward<Object>(obj_)),withHint(true),hint(std::forward<Hint>(hint_)){}
    template<class Archive>
    inline void save(Archive& archive) const; //define later
    template<class Archive>
    inline void load(Archive& archive); //define later
};
template<traits::pyobject Object>
inline Mapping<Object> makeMapping(Object&& obj_){
    return Mapping<Object>(std::forward<Object>(obj_));
}
template<traits::pyobject Object, class Hint=PyTypeInfo>
requires (std::same_as<traits::get_true_value_type_t<Hint>,PyTypeInfo>)
inline Mapping<Object,Hint> makeMapping(Object&& obj_,Hint&& hint_){
    return Mapping<Object,Hint>(std::forward<Object>(obj_),std::forward<Hint>(hint_));
}

template<traits::pyobject Object, class Hint/*=PyTypeInfo*/>
requires (std::same_as<traits::get_true_value_type_t<Hint>,PyTypeInfo>)
struct PYBIND11_EXPORT PyObjectWrapper{
    using StoredObjectType = std::conditional_t<std::is_lvalue_reference_v<Object>,Object,std::decay_t<Object>>;
    using StoredHintType = std::conditional_t<std::is_lvalue_reference_v<Hint>,Hint,std::decay_t<Hint>>;
    StoredObjectType obj;
    bool withHint;
    StoredHintType hint;
    explicit PyObjectWrapper(Object&& obj_):obj(std::forward<Object>(obj_)),withHint(false),hint(){}
    explicit PyObjectWrapper(Object&& obj_,Hint&& hint_):obj(std::forward<Object>(obj_)),withHint(true),hint(std::forward<Hint>(hint_)){}
    template<class Archive>
    inline void serialize(Archive& archive){
        // withHint(with type info)であって、C++型でなく、基本型でもないならC++側からはシリアライゼーション方法を確定できないためpickleに委ねる。
        if(withHint && hint.kind!=PyObjectTypeKind::cpp_type && !hint.is_basic_type){
            serializeByPickle(archive);
            return;
        }
        // C++側の型で判別可能ならconstexprで個々の型に応じたserializeを呼ぶ
        if constexpr (std::same_as<traits::get_true_value_type_t<Object>,::pybind11::none>){
            serializeNone(archive);
        }else if constexpr (std::same_as<traits::get_true_value_type_t<Object>,::pybind11::type>){
            serializeByPickle(archive);
        }else if constexpr (std::same_as<traits::get_true_value_type_t<Object>,::pybind11::bool_>){
            serializeBool(archive);
        }else if constexpr (std::same_as<traits::get_true_value_type_t<Object>,::pybind11::int_>){
            serializeInt(archive);
        }else if constexpr (std::same_as<traits::get_true_value_type_t<Object>,::pybind11::float_>){
            serializeFloat(archive);
        }else if constexpr (std::same_as<traits::get_true_value_type_t<Object>,::pybind11::bytearray>){
            serializeBytearray(archive);
        }else if constexpr(std::same_as<traits::get_true_value_type_t<Object>,::pybind11::bytes>){
            serializeBytes(archive);
        }else if constexpr (std::same_as<traits::get_true_value_type_t<Object>,::pybind11::str>){
            serializeString(archive);
        }else if constexpr (std::derived_from<traits::get_true_value_type_t<Object>,::pybind11::array>){
            serializeNumpy(archive);
        }else if constexpr (
            std::derived_from<traits::get_true_value_type_t<Object>,::pybind11::anyset>
        ){
            serializeSet(archive);
        }else if constexpr (
            std::derived_from<traits::get_true_value_type_t<Object>,::pybind11::sequence>
            || std::derived_from<traits::get_true_value_type_t<Object>,::pybind11::list>
            || std::derived_from<traits::get_true_value_type_t<Object>,::pybind11::tuple>
        ){
            serializeSequence(archive);
        }else if constexpr (
            std::derived_from<traits::get_true_value_type_t<Object>,::pybind11::dict>
        ){
            serializeMapping(archive);
        }else{
            // ::pybind11::objectのように中身の種類が不定ならインスタンス情報から通常の分岐でserializeを呼び分ける。
            PyObjectTypeKind kind;
            bool is_basic_type;
            if constexpr (traits::output_archive<Archive>){
                if(withHint){
                    kind=hint.kind;
                    is_basic_type=hint.is_basic_type;
                }else{
                    // save時は中身がNoneの場合にkindをnoneで上書きする
                    if(obj.ptr()==nullptr || obj.is_none()){
                        kind=PyObjectTypeKind::none;
                    }else{
                        kind=get_type_kind_of_object(obj);
                    }
                    // hintなしのsaveは基本型として行う
                    is_basic_type=true;
                }
            }else{
                // load時はhintの反映を行う
                if(withHint){
                    kind=hint.kind;
                    is_basic_type=hint.is_basic_type;
                }else{
                    // hintなしのloadは基本型として行い、格納先の変数の現在の値と同じkindを仮定する。
                    // objがNoneの場合kindがnoneになるので、新規オブジェクトとしてloadするような場合にはNLJSON以外のArchiveを使用できない。
                    kind=get_type_kind_of_object(obj);
                    is_basic_type=true;
                    if constexpr(traits::same_archive_as<Archive,::asrc::core::util::NLJSONInputArchive>){
                        // NLJSONArchiveについてはkindを推定可能なので新規オブジェクトとしてもload可能
                        if(kind==PyObjectTypeKind::none){
                            const nlohmann::ordered_json& j=archive.getNextNode();
                            auto j_type=j.type();
                            if (j_type == nlohmann::ordered_json::value_t::null){
                                kind=PyObjectTypeKind::none;
                            }else if (j_type == nlohmann::ordered_json::value_t::boolean){
                                kind=PyObjectTypeKind::boolean;
                            }else if (j_type == nlohmann::ordered_json::value_t::number_unsigned){
                                kind=PyObjectTypeKind::integer;
                            }else if (j_type == nlohmann::ordered_json::value_t::number_integer){
                                kind=PyObjectTypeKind::integer;
                            }else if (j_type == nlohmann::ordered_json::value_t::number_float){
                                kind=PyObjectTypeKind::float_;
                            }else if (j_type == nlohmann::ordered_json::value_t::binary){
                                kind=PyObjectTypeKind::bytes;
                            }else if (j_type == nlohmann::ordered_json::value_t::string){
                                kind=PyObjectTypeKind::string;
                            }else if (j_type == nlohmann::ordered_json::value_t::array){
                                kind=PyObjectTypeKind::mutable_sequence;
                            }else if (j_type == nlohmann::ordered_json::value_t::object){
                                kind=PyObjectTypeKind::mutable_mapping;
                            }
                        }
                    }
                }
            }
            if(kind!=PyObjectTypeKind::cpp_type && !is_basic_type){
                serializeByPickle(archive);
            }else if(kind==PyObjectTypeKind::none){
                serializeNone(archive);
            }else if(
                kind==PyObjectTypeKind::nl_json
                || kind==PyObjectTypeKind::nl_ordered_json
            ){
                serializeJson(archive,kind);
            }else if(kind==PyObjectTypeKind::cpp_type){
                serializeCppType(archive);
            }else if(kind==PyObjectTypeKind::boolean){
                serializeBool(archive);
            }else if(kind==PyObjectTypeKind::integer){
                serializeInt(archive);
            }else if(kind==PyObjectTypeKind::float_){
                serializeFloat(archive);
            }else if(kind==PyObjectTypeKind::bytearray){
                serializeBytearray(archive);
            }else if(kind==PyObjectTypeKind::bytes){
                serializeBytes(archive);
            }else if(kind==PyObjectTypeKind::string){
                serializeString(archive);
            }else if(
                kind==PyObjectTypeKind::mutable_set
                || kind==PyObjectTypeKind::set
            ){
                serializeSet(archive);
            }else if(
                kind==PyObjectTypeKind::mutable_sequence
                || kind==PyObjectTypeKind::sequence
            ){
                serializeSequence(archive);
            }else if(
                kind==PyObjectTypeKind::mutable_mapping
                || kind==PyObjectTypeKind::mapping
            ){
                serializeMapping(archive);
            }else if(kind==PyObjectTypeKind::numpy_array){
                serializeNumpy(archive);
            }else{
                serializeByPickle(archive);
            }
        }
    }
    // 中身の種類に応じたserialize処理
    template<class Archive>
    inline void serializeNone(Archive& archive){
        if constexpr (traits::output_archive<Archive>){
            // save
            if constexpr (
                traits::same_archive_as<Archive,::asrc::core::util::NLJSONOutputArchive>
                || traits::same_archive_as<Archive,::cereal::JSONOutputArchive>
            ){
                archive(nullptr);
            }
        }else{
            // load
            if constexpr (
                traits::same_archive_as<Archive,::asrc::core::util::NLJSONInputArchive>
                || traits::same_archive_as<Archive,::cereal::JSONInputArchive>
            ){
                archive(nullptr);
            }
            obj=::pybind11::none();
        }
    }
    template<class Archive>
    inline void serializeJson(Archive& archive,PyObjectTypeKind kind){
        if constexpr (traits::output_archive<Archive>){
            // save
            if(kind==PyObjectTypeKind::nl_json){
                archive(obj.template cast<std::reference_wrapper<const nlohmann::json>>().get());
            }else{ //kind==PyObjectTypeKind::nl_ordered_json
                archive(obj.template cast<std::reference_wrapper<const nlohmann::ordered_json>>().get());
            }
        }else{
            // load
            if(kind==PyObjectTypeKind::nl_json){
                obj=::pybind11::cast(nlohmann::json());
                archive(obj.template cast<std::reference_wrapper<nlohmann::json>>().get());
            }else{ //kind==PyObjectTypeKind::nl_ordered_json
                obj=::pybind11::cast(nlohmann::ordered_json());
                archive(obj.template cast<std::reference_wrapper<nlohmann::ordered_json>>().get());
            }
        }
    }
    template<class Archive>
    inline void serializeCppType(Archive& archive){
        if constexpr (traits::output_archive<Archive>){
            // save
            obj.attr("save")(::pybind11::cast(archive,::pybind11::return_value_policy::reference));
        }else{
            // load
            // C++からPython側へ公開したクラスのうち、C++側でのシリアライズを明示したもの
            // ヒントなしでは復元できない(NLJSONInputArchiveの場合は代わりに基本型としてloadする)
            if(withHint){
                obj=hint.self.attr("static_load")(::pybind11::cast(archive,::pybind11::return_value_policy::reference));
            }else{
                throw cereal::Exception("Non-primitive C++ type cannot be deserialized without type hint.");
            }
        }
    }
    template<class Archive>
    inline void serializeBool(Archive& archive){
        if constexpr (traits::output_archive<Archive>){
            // save
            archive(obj.template cast<nlohmann::ordered_json::boolean_t>());
        }else{
            // load
            nlohmann::ordered_json::boolean_t value;
            archive(value);
            obj=::pybind11::bool_(value);
        }
    }
    template<class Archive>
    inline void serializeInt(Archive& archive){
        if constexpr (traits::output_archive<Archive>){
            // save
            // TODO long long intより大きな整数の処理
            auto value=obj.template cast<nlohmann::ordered_json::number_integer_t>();
            if(!::pybind11::int_(value).equal(obj)){
                throw cereal::Exception("Integer out of range for serialization. Currently it is limited within the range of 64bit signed integer.");
            }
            archive(value);
        }else{
            // load
            nlohmann::ordered_json::number_integer_t value;
            archive(value);
            obj=::pybind11::int_(value);
        }
    }
    template<class Archive>
    inline void serializeFloat(Archive& archive){
        if constexpr (traits::output_archive<Archive>){
            // save
            archive(obj.template cast<nlohmann::ordered_json::number_float_t>());
        }else{
            // load
            nlohmann::ordered_json::number_float_t value;
            archive(value);
            obj=::pybind11::float_(value);
        }
    }
    template<class Archive>
    inline void serializeBytearray(Archive& archive){
        if constexpr (traits::output_archive<Archive>){
            // save
            auto value=obj.template cast<nlohmann::ordered_json::string_t>();
            if constexpr (traits::text_archive<Archive>){
                // text-based archiveならbase64を挟む
                value=cereal::base64::encode(
                    reinterpret_cast<const unsigned char *>(value.c_str()),
                    value.size()
                );
            }
            archive(value);
        }else{
            // load
            nlohmann::ordered_json::string_t value;
            archive(value);
            if constexpr (traits::text_archive<Archive>){
                value=cereal::base64::decode(value);
            }
            obj=::pybind11::bytearray(std::move(value));
        }
    }
    template<class Archive>
    inline void serializeBytes(Archive& archive){
        if constexpr (traits::output_archive<Archive>){
            // save
            auto value=obj.template cast<nlohmann::ordered_json::string_t>();
            if constexpr (traits::text_archive<Archive>){
                // text-based archiveならbase64を挟む
                value=cereal::base64::encode(
                    reinterpret_cast<const unsigned char *>(value.c_str()),
                    value.size()
                );
            }
            archive(value);
        }else{
            // load
            nlohmann::ordered_json::string_t value;
            archive(value);
            if constexpr (traits::text_archive<Archive>){
                value=cereal::base64::decode(value);
            }
            obj=::pybind11::bytes(std::move(value));
        }
    }
    template<class Archive>
    inline void serializeString(Archive& archive){
        if constexpr (traits::output_archive<Archive>){
            // save
            archive(obj.template cast<nlohmann::ordered_json::string_t>());
        }else{
            // load
            nlohmann::ordered_json::string_t value;
            archive(value);
            obj=::pybind11::str(std::move(value));
        }
    }
    template<class Archive>
    inline void serializeSet(Archive& archive){
        if(withHint){
            archive(makeSet(obj,hint));
        }else{
            archive(makeSet(obj));
        }
    }
    template<class Archive>
    inline void serializeSequence(Archive& archive){
        if(withHint){
            archive(makeSequence(obj,hint));
        }else{
            archive(makeSequence(obj));
        }
    }
    template<class Archive>
    inline void serializeMapping(Archive& archive){
        if(withHint){
            archive(makeMapping(obj,hint));
        }else{
            archive(makeMapping(obj));
        }
    }
    template<class Archive>
    inline void serializeNumpy(Archive& archive){
        //Numpy Array
        if constexpr (traits::output_archive<Archive>){
            // save
            auto arr=::pybind11::array::ensure(obj);
            archive(numpy_array_wrapper(arr));
        }else{
            // load
            if(::pybind11::isinstance<::pybind11::array>(obj)){
                auto arr=::pybind11::array::ensure(obj);
                if(withHint){
                    archive(numpy_array_wrapper(arr,hint.dtype,hint.is_col_major));
                }else{
                    archive(numpy_array_wrapper(arr));
                }
                obj=arr;
            }else{
                ::pybind11::array arr;
                if(withHint){
                    archive(numpy_array_wrapper(arr,hint.dtype,hint.is_col_major));
                }else{
                    archive(numpy_array_wrapper(arr));
                }
                obj=arr;
            }
        }
    }
    template<class Archive>
    inline void serializeByPickle(Archive& archive){
        // PyObjectTypeKind::type or PyObjectTypeKind::other
        // pickleを使用する
        auto cloudpickle=::pybind11::module::import("cloudpickle");
        if constexpr (traits::output_archive<Archive>){
            // save
            auto value=cloudpickle.attr("dumps")(obj).template cast<nlohmann::ordered_json::string_t>();
            if constexpr (traits::text_archive<Archive>){
                // text-based archiveならbase64を挟む
                value=cereal::base64::encode(
                    reinterpret_cast<const unsigned char *>(value.c_str()),
                    value.size()
                );
            }
            archive(value);
        }else{
            // load
            std::string value;
            archive(value);

            if constexpr (traits::text_archive<Archive>){
                // text-based archiveならbase64を挟む
                obj=cloudpickle.attr("loads")(::pybind11::bytes(cereal::base64::decode(value)));
            }else{
                obj=cloudpickle.attr("loads")(::pybind11::bytes(value));
            }
        }
    }
};

template<traits::pyobject Object>
inline PyObjectWrapper<Object> makePyObjectWrapper(Object&& obj){
    return PyObjectWrapper<Object>(std::forward<Object>(obj));
}
template<traits::pyobject Object, class Hint/*=PyTypeInfo*/>
requires (std::same_as<traits::get_true_value_type_t<Hint>,PyTypeInfo>)
inline PyObjectWrapper<Object,Hint> makePyObjectWrapper(Object&& obj,Hint&& hint){
    return PyObjectWrapper<Object,Hint>(std::forward<Object>(obj),std::forward<Hint>(hint));
}

template<traits::pyobject Object, class Hint>
requires (std::same_as<traits::get_true_value_type_t<Hint>,PyTypeInfo>)
template<class Archive>
void Set<Object,Hint>::save(Archive& archive) const{
    archive( ::cereal::make_size_tag( static_cast<::cereal::size_type>(::pybind11::len(obj) ) ));
    ::cereal::size_type i=0;
    for(auto&& value : obj){
        if(withHint){
            archive(makePyObjectWrapper(value,hint.sequence[i]));
        }else{
            archive(makePyObjectWrapper(value));
        }
        ++i;
    }
}
template<traits::pyobject Object, class Hint>
requires (std::same_as<traits::get_true_value_type_t<Hint>,PyTypeInfo>)
template<class Archive>
void Set<Object,Hint>::load(Archive& archive){
    ::cereal::size_type size;
    archive( ::cereal::make_size_tag( size ));
    ::pybind11::set s;
    if(withHint){
        assert(
            (
                hint.kind==PyObjectTypeKind::mutable_set
                || hint.kind==PyObjectTypeKind::set
            )
            && hint.sequence.size()==size
        );
    }
    for(::cereal::size_type i=0;i<size;++i){
        ::pybind11::object value;
        if(withHint){
            archive(makePyObjectWrapper(value,hint.sequence[i]));
        }else{
            archive(makePyObjectWrapper(value));
        }
        s.add(value);
    }
    if(withHint && hint.kind==PyObjectTypeKind::set){
        obj=::pybind11::frozenset(s);
    }else{
        obj=s;
    }
}
template<traits::pyobject Object, class Hint>
requires (std::same_as<traits::get_true_value_type_t<Hint>,PyTypeInfo>)
template<class Archive>
void Sequence<Object,Hint>::save(Archive& archive) const{
    archive( ::cereal::make_size_tag( static_cast<::cereal::size_type>(::pybind11::len(obj) ) ));
    ::cereal::size_type i=0;
    for(auto&& value : obj){
        if(withHint){
            archive(makePyObjectWrapper(value,hint.sequence[i]));
        }else{
            archive(makePyObjectWrapper(value));
        }
        ++i;
    }
}
template<traits::pyobject Object, class Hint>
requires (std::same_as<traits::get_true_value_type_t<Hint>,PyTypeInfo>)
template<class Archive>
void Sequence<Object,Hint>::load(Archive& archive){
    ::cereal::size_type size;
    archive( ::cereal::make_size_tag( size ));
    ::pybind11::list l(size);
    if(withHint){
        assert(
            (
                hint.kind==PyObjectTypeKind::mutable_sequence
                || hint.kind==PyObjectTypeKind::sequence
            )
            && hint.sequence.size()==size
        );
    }
    for(::cereal::size_type i=0;i<size;++i){
        ::pybind11::object value;
        if(withHint){
            archive(makePyObjectWrapper(value,hint.sequence[i]));
        }else{
            archive(makePyObjectWrapper(value));
        }
        l[i]=value;
    }
    if(withHint && hint.kind==PyObjectTypeKind::sequence){
        obj=py::tuple(l);
    }else{
        obj=l;
    }
}
template<traits::pyobject Key,traits::pyobject Value,class KeyHint, class ValueHint>
template<class Archive>
void MappingItem<Key,Value,KeyHint,ValueHint>::save(Archive& archive) const{
    if constexpr (traits::same_archive_as<Archive,::asrc::core::util::NLJSONOutputArchive>){
        archive( ::cereal::make_size_tag( static_cast<::cereal::size_type>(2) ));
        if(withHint){
            archive(
                makePyObjectWrapper(key,keyHint),
                makePyObjectWrapper(value,valueHint)
            );
        }else{
            archive(
                makePyObjectWrapper(key),
                makePyObjectWrapper(value)
            );
        }
    }else{
        if(withHint){
            archive(
                ::cereal::make_nvp<Archive>("key",makePyObjectWrapper(key,keyHint)),
                ::cereal::make_nvp<Archive>("value",makePyObjectWrapper(value,valueHint))
            );
        }else{
            archive(
                ::cereal::make_nvp<Archive>("key",makePyObjectWrapper(key)),
                ::cereal::make_nvp<Archive>("value",makePyObjectWrapper(value))
            );
        }
    }
}
template<traits::pyobject Key,traits::pyobject Value,class KeyHint, class ValueHint>
template<class Archive>
void MappingItem<Key,Value,KeyHint,ValueHint>::load(Archive& archive){
    if constexpr (traits::same_archive_as<Archive,::asrc::core::util::NLJSONInputArchive>){
        ::cereal::size_type size;
        archive( ::cereal::make_size_tag( size ));
        assert(size==2);
        if(withHint){
            archive(
                makePyObjectWrapper(key,keyHint),
                makePyObjectWrapper(value,valueHint)
            );
        }else{
            archive(
                makePyObjectWrapper(key),
                makePyObjectWrapper(value)
            );
        }
    }else{
        if(withHint){
            archive(
                ::cereal::make_nvp<Archive>("key",makePyObjectWrapper(key,keyHint)),
                ::cereal::make_nvp<Archive>("value",makePyObjectWrapper(value,valueHint))
            );
        }else{
            archive(
                ::cereal::make_nvp<Archive>("key",makePyObjectWrapper(key)),
                ::cereal::make_nvp<Archive>("value",makePyObjectWrapper(value))
            );
        }
    }
}
template<traits::pyobject Object, class Hint>
requires (std::same_as<traits::get_true_value_type_t<Hint>,PyTypeInfo>)
template<class Archive>
void Mapping<Object,Hint>::save(Archive& archive) const{
    if constexpr (traits::same_archive_as<Archive,::asrc::core::util::NLJSONOutputArchive>){
        // keyが全てstring compatibleならjson objectとして書き込む
        // keyが一つでもstring compatibleでないならkey-value pairのjson arrayとして書き込む
        auto asdict=::pybind11::reinterpret_borrow<::pybind11::dict>(obj);
        if(is_all_keys_string_compatible(asdict)){
            ::cereal::size_type i=0;
            for(auto&& [key,value] : asdict){
                auto key_str=key.template cast<std::string>();
                if(withHint){
                    archive(cereal::make_nvp<Archive>(key_str.c_str(),makePyObjectWrapper(value,hint.mapping_str_key_only[i].second)));
                }else{
                    archive(cereal::make_nvp<Archive>(key_str.c_str(),makePyObjectWrapper(value)));
                }
                ++i;
            }

            return;
        }
    }
    archive( ::cereal::make_size_tag( static_cast<::cereal::size_type>(::pybind11::len(obj) ) ));
    ::cereal::size_type i=0;
    auto strTypeInfo=PyTypeInfo(::pybind11::str());
    for(auto&& [key,value] : ::pybind11::reinterpret_borrow<::pybind11::dict>(obj)){
        if(withHint){
            if(hint.str_key_only){
                archive(makeMappingItem(key,value,strTypeInfo,hint.mapping_str_key_only[i].second));
            }else{
                archive(makeMappingItem(key,value,hint.mapping_generic[i].first,hint.mapping_generic[i].second));
            }
        }else{
            archive(makeMappingItem(key,value));
        }
        ++i;
    }
}
template<traits::pyobject Object, class Hint>
requires (std::same_as<traits::get_true_value_type_t<Hint>,PyTypeInfo>)
template<class Archive>
void Mapping<Object,Hint>::load(Archive& archive){
    ::cereal::size_type size;
    if constexpr (traits::same_archive_as<Archive,::asrc::core::util::NLJSONInputArchive>){
        const nlohmann::ordered_json& j=archive.getCurrentNode();
        size=j.size();
    }else{
        archive( ::cereal::make_size_tag( size ));
    }
    if(withHint){
        assert(
            (
                hint.kind==PyObjectTypeKind::mutable_mapping
                || hint.kind==PyObjectTypeKind::mapping
            )
            && (
                hint.str_key_only ? 
                hint.mapping_str_key_only.size()==size :
                hint.mapping_generic.size()==size
            )
        );
    }
    obj=::pybind11::dict();
    if(withHint){
        if(hint.str_key_only){
            for(auto&& [h_key,h_value] : hint.mapping_str_key_only){
                ::pybind11::object key,value;
                if constexpr (traits::same_archive_as<Archive,::asrc::core::util::NLJSONInputArchive>){
                    // object
                    archive(::cereal::make_nvp(
                        h_key.c_str(),
                        makePyObjectWrapper(value,h_value)
                    ));
                    key=::pybind11::str(h_key);
                }else{
                    auto strTypeInfo=PyTypeInfo(::pybind11::str());
                    archive(makeMappingItem(key,value,strTypeInfo,h_value));
                }
                obj[key]=value;
            }
        }else{
            std::size_t i=0;
            for(auto&& [h_key,h_value] : hint.mapping_generic){
                ::pybind11::object key,value;
                if constexpr (traits::same_archive_as<Archive,::asrc::core::util::NLJSONInputArchive>){
                    // array
                    const nlohmann::ordered_json& j=archive.getCurrentNode();
                    assert(j.at(i).is_array() && j.at(i).size()==2);
                }
                archive(makeMappingItem(key,value,h_key,h_value));
                obj[key]=value;
                ++i;
            }
        }
    }else{
        if constexpr (traits::same_archive_as<Archive,::asrc::core::util::NLJSONInputArchive>){
            // object
            const nlohmann::ordered_json& j=archive.getCurrentNode();
            for(auto&& [j_key, j_value] : j.items()){
                ::pybind11::object key,value;
                archive(::cereal::make_nvp(
                    j_key.c_str(),
                    makePyObjectWrapper(value)
                ));
                obj[::pybind11::str(j_key)]=value;
            }
        }else{
            for(std::size_t i=0;i<size;++i){
                ::pybind11::object key,value;
                archive(makeMappingItem(key,value));
                obj[key]=value;
            }
        }
    }
}

template<traits::pyobject Object>
struct PYBIND11_EXPORT PyTypeInfoAndObjectWrapper{
    using StoredObjectType = std::conditional_t<std::is_lvalue_reference_v<Object>,Object,std::decay_t<Object>>;
    StoredObjectType obj;
    PyTypeInfoAndObjectWrapper(Object&& obj_):obj(std::forward<Object>(obj_)){}
    template<class Archive>
    inline void serialize(Archive & archive){
        if constexpr (traits::output_archive<Archive>){
            // save
            PyTypeInfo info(obj);
            archive(cereal::make_nvp("type",info));
            archive(cereal::make_nvp("value",makePyObjectWrapper(obj,info)));
        }else{
            // load
            PyTypeInfo info;
            archive(cereal::make_nvp("type",info));
            archive(cereal::make_nvp("value",makePyObjectWrapper(obj,info)));
        }
    }
};
template<traits::pyobject Object>
inline PyTypeInfoAndObjectWrapper<Object> makePyTypeInfoAndObjectWrapper(Object&& obj){
    return {std::forward<Object>(obj)};
}

template<traits::output_archive Archive, class T>
inline void save_type_info_of(Archive& archive, T&& obj){
    if constexpr (traits::cereal_nvp<T>){
        if constexpr (traits::cereal_nvp_of_derived_from<T, ::pybind11::detail::pyobject_tag>){
            archive(cereal::make_nvp(std::forward<T>(obj).name,PyTypeInfo(std::forward<T>(obj).value)));
        }else{
            archive(cereal::make_nvp(std::forward<T>(obj).name,PyTypeInfo(::pybind11::cast(std::forward<T>(obj).value),::pybind11::return_value_policy::reference)));
        }
    }else{
        if constexpr (traits::pyobject<T>){
            archive(PyTypeInfo(std::forward<T>(obj)));
        }else{
            archive(PyTypeInfo(::pybind11::cast(std::forward<T>(obj),::pybind11::return_value_policy::reference)));
        }
    }
}

template<traits::cereal_archive Archive, class T>
inline void serialize_with_type_info(Archive& archive, T&& obj){
    if constexpr (traits::pyobject<T>){
        archive(makePyTypeInfoAndObjectWrapper(std::forward<T>(obj)));
    }else if constexpr (traits::cereal_nvp_of_derived_from<T, ::pybind11::detail::pyobject_tag>){
        archive(cereal::make_nvp(std::forward<T>(obj).name,makePyTypeInfoAndObjectWrapper(std::forward<T>(obj).value)));
    }else{
        archive(std::forward<T>(obj));
    }
}
template<traits::output_archive Archive, class T>
inline void save_with_type_info(Archive& archive, T&& obj){
    serialize_with_type_info(archive,std::forward<T>(obj));
}
template<traits::input_archive Archive, class T>
inline void load_with_type_info(Archive& archive, T&& obj){
    serialize_with_type_info(archive,std::forward<T>(obj));
}
template<traits::cereal_archive Archive, class T, class ... Others>
inline void serialize_with_type_info(Archive& archive, T&& obj, Others&& ... others){
    serialize_with_type_info(archive,std::forward<T>(obj));
    serialize_with_type_info(archive,std::forward<Others>(others)...);
}
template<traits::output_archive Archive, class T, class ... Others>
inline void save_with_type_info(Archive& archive, T&& obj, Others&& ... others){
    save_with_type_info(archive,std::forward<T>(obj));
    save_with_type_info(archive,std::forward<Others>(others)...);
}
template<traits::input_archive Archive, class T, class ... Others>
inline void load_with_type_info(Archive& archive, T&& obj, Others&& ... others){
    load_with_type_info(archive,std::forward<T>(obj));
    load_with_type_info(archive,std::forward<Others>(others)...);
}

template<traits::input_archive Archive, traits::cereal_nvp_of_derived_from<::pybind11::detail::pyobject_tag> NVP>
inline void load_with_external_type_info(Archive& archive, NVP&& obj, const PyTypeInfo& hint){
    archive(cereal::make_nvp(obj.name,makePyObjectWrapper(obj.value,hint)));
}
template<traits::input_archive Archive, traits::pyobject Object>
inline void load_with_external_type_info(Archive& archive, Object& obj, const PyTypeInfo& hint){
    archive(makePyObjectWrapper(obj,hint));
}

template<traits::input_archive Archive, traits::pyobject Object=::pybind11::object>
inline Object load_python_object(Archive& archive){
    // クラスヒントなしで新たなpybind11::objectとしてload
    Object ret;
    archive(ret);
    return std::move(ret);
}
template<traits::input_archive Archive, traits::pyobject Object=::pybind11::object>
inline Object load_python_object(Archive& archive, const std::string& name){
    // クラスヒントなしで新たなpybind11::objectとしてload
    Object ret;
    archive(cereal::make_nvp(name.c_str(),ret));
    return std::move(ret);
}
template<traits::input_archive Archive, traits::pyobject Object=::pybind11::object>
inline Object load_python_object_with_type_info(Archive& archive){
    // Archive中のクラスヒントとともに新たなpybind11::objectとしてload
    Object ret;
    load_with_type_info(archive,ret);
    return std::move(ret);
}
template<traits::input_archive Archive, traits::pyobject Object=::pybind11::object>
inline Object load_python_object_with_type_info(Archive& archive, const std::string& name){
    // Archive中のクラスヒントとともに新たなpybind11::objectとしてload
    Object ret;
    load_with_type_info(archive,cereal::make_nvp(name.c_str(),ret));
    return std::move(ret);
}
template<traits::input_archive Archive, traits::pyobject Object=::pybind11::object>
inline Object load_python_object_with_external_type_info(Archive& archive, const PyTypeInfo& hint){
    // 外部からのクラスヒントありで新たなpybind11::objectとしてload
    Object ret;
    load_with_external_type_info(archive,ret,hint);
    return std::move(ret);
}
template<traits::input_archive Archive, traits::pyobject Object=::pybind11::object>
inline Object load_python_object_with_external_type_info(Archive& archive, const PyTypeInfo& hint, const std::string& name){
    // 外部からのクラスヒントありで新たなpybind11::objectとしてload
    Object ret;
    load_with_external_type_info(archive,cereal::make_nvp(name.c_str(),ret),hint);
    return std::move(ret);
}

template<traits::output_archive Archive>
inline void save_attr(Archive& archive, ::pybind11::handle& obj, const std::string& attr_name){
    if(!obj.is_none()){
        archive(cereal::make_nvp(attr_name.c_str(),::pybind11::getattr(obj,attr_name.c_str())));
    }
}
template<traits::input_archive Archive>
inline void load_attr(Archive& archive, ::pybind11::handle& obj, const std::string& attr_name){
    if(!obj.is_none()){
        ::pybind11::object attr;
        archive(cereal::make_nvp(attr_name.c_str(),attr));
        ::pybind11::setattr(obj,attr_name.c_str(),attr);
    }
}
template<traits::cereal_archive Archive>
inline void serialize_attr(Archive& archive, ::pybind11::handle& obj, const std::string& attr_name){
    if constexpr (traits::output_archive<Archive>){
        save_attr(archive,obj,attr_name);
    }else{
        load_attr(archive,obj,attr_name);
    }
}
template<traits::output_archive Archive, class ... Others>
requires (std::conjunction_v<traits::is_string_like<Others>...>)
inline void save_attr(Archive& archive, ::pybind11::handle& obj, const std::string& attr_name, Others&& ... others){
    save_attr(archive,obj,attr_name);
    save_attr(archive,obj,std::forward<Others>(others)...);
}
template<traits::output_archive Archive, class ... Others>
requires (std::conjunction_v<traits::is_string_like<Others>...>)
inline void load_attr(Archive& archive, ::pybind11::handle& obj, const std::string& attr_name, Others&& ... others){
    load_attr(archive,obj,attr_name);
    load_attr(archive,obj,std::forward<Others>(others)...);
}
template<traits::output_archive Archive, class ... Others>
requires (std::conjunction_v<traits::is_string_like<Others>...>)
inline void serialize_attr(Archive& archive, ::pybind11::handle& obj, const std::string& attr_name, Others&& ... others){
    serialize_attr(archive,obj,attr_name);
    serialize_attr(archive,obj,std::forward<Others>(others)...);
}

template<traits::output_archive Archive>
inline void save_attr_with_type_info(Archive& archive, ::pybind11::handle& obj, const std::string& attr_name){
    if(!obj.is_none()){
        save_with_type_info(archive,cereal::make_nvp(attr_name.c_str(),::pybind11::getattr(obj,attr_name.c_str())));
    }
}
template<traits::input_archive Archive>
inline void load_attr_with_type_info(Archive& archive, ::pybind11::handle& obj, const std::string& attr_name){
    if(!obj.is_none()){
        ::pybind11::object attr;
        load_with_type_info(archive,cereal::make_nvp(attr_name.c_str(),attr));
        ::pybind11::setattr(obj,attr_name.c_str(),attr);
    }
}
template<traits::cereal_archive Archive>
inline void serialize_attr_with_type_info(Archive& archive, ::pybind11::handle& obj, const std::string& attr_name){
    if constexpr (traits::output_archive<Archive>){
        save_attr_with_type_info(archive,obj,attr_name);
    }else{
        load_attr_with_type_info(archive,obj,attr_name);
    }
}
template<traits::output_archive Archive, class ... Others>
requires (std::conjunction_v<traits::is_string_like<Others>...>)
inline void save_attr_with_type_info(Archive& archive, ::pybind11::handle& obj, const std::string& attr_name, Others&& ... others){
    save_attr_with_type_info(archive,obj,attr_name);
    save_attr_with_type_info(archive,obj,std::forward<Others>(others)...);
}
template<traits::output_archive Archive, class ... Others>
requires (std::conjunction_v<traits::is_string_like<Others>...>)
inline void load_attr_with_type_info(Archive& archive, ::pybind11::handle& obj, const std::string& attr_name, Others&& ... others){
    load_attr_with_type_info(archive,obj,attr_name);
    load_attr_with_type_info(archive,obj,std::forward<Others>(others)...);
}
template<traits::output_archive Archive, class ... Others>
requires (std::conjunction_v<traits::is_string_like<Others>...>)
inline void serialize_attr_with_type_info(Archive& archive, ::pybind11::handle& obj, const std::string& attr_name, Others&& ... others){
    serialize_attr_with_type_info(archive,obj,attr_name);
    serialize_attr_with_type_info(archive,obj,std::forward<Others>(others)...);
}

template<traits::input_archive Archive>
inline void load_attr_with_external_type_info(Archive& archive, ::pybind11::handle& obj, const std::string& attr_name, const PyTypeInfo& hint){
    if(!obj.is_none()){
        ::pybind11::object attr;
        load_with_external_type_info(archive,cereal::make_nvp(attr_name.c_str(),attr),hint);
        ::pybind11::setattr(obj,attr_name.c_str(),attr);
    }
}

void exportPyObjectSerialization(::pybind11::module &m);

ASRC_NAMESPACE_END(util)

ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)

#undef ASRC_INTERNAL_CEREAL_PROLOGUE_EPILOGUE_DISABLER_FOR_PYTHON_OBJECT
