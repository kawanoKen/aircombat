// Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
//
// pybind11::objectをcerealでシリアライズする機能に関するユーティリティを分離したもの
//
#pragma once
#include "../../util/macros/common_macros.h"
#include "../../traits/cereal.h"
#include "../../traits/pybind.h"
#include "nljson.h"
#include "numpy.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <nlohmann/json.hpp>
#include <type_traits>
#include <cereal/cereal.hpp>
#include <cereal/types/vector.hpp>

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)
ASRC_INLINE_NAMESPACE_BEGIN(util)

enum class PyObjectTypeKind : std::uint8_t {
    none,
    type,
    boolean,
    integer,
    float_,
    bytearray,
    bytes,
    string,
    set,
    mutable_set,
    sequence,
    mutable_sequence,
    mapping,
    mutable_mapping,
    numpy_array,
    nl_json,
    nl_ordered_json,
    cpp_type,
    other
};

template<traits::pyobject Object>
PyObjectTypeKind PYBIND11_EXPORT get_type_kind_of_object(const Object& obj); // forward declaration
PyObjectTypeKind PYBIND11_EXPORT get_type_kind_of_type(const pybind11::type& cls);

template<traits::pyobject Object>
bool PYBIND11_EXPORT is_basic_type_object(const Object& obj); // forward declaration
bool PYBIND11_EXPORT is_basic_type_object(const pybind11::type& cls);

bool PYBIND11_EXPORT is_primitive_type(const PyObjectTypeKind& kind);
bool PYBIND11_EXPORT is_primitive_type(const pybind11::type& cls);
bool PYBIND11_EXPORT is_primitive_object(const pybind11::handle& obj);

void exportPyObjectSerializationUtil(pybind11::module &m);

bool PYBIND11_EXPORT is_all_keys_string_compatible(const pybind11::dict& d);

struct PYBIND11_EXPORT PyTypeInfo{
    PyObjectTypeKind kind;
    bool is_basic_type;
    pybind11::type self;
    bool str_key_only;
    std::vector<PyTypeInfo> sequence; // setもこれを使う
    std::vector<std::pair<std::string,PyTypeInfo>> mapping_str_key_only;
    std::vector<std::pair<PyTypeInfo,PyTypeInfo>> mapping_generic;
    asrc::core::util::DTypes dtype;
    bool is_col_major;
    PyTypeInfo();
    template<traits::pyobject Object>
    PyTypeInfo(const Object& obj)
    :kind(get_type_kind_of_object(obj)),
    is_basic_type(is_basic_type_object(obj)),
    self(pybind11::type::of(obj))
    {
        if constexpr (
            traits::sequence_like_pyobject<Object>
            || traits::set_like_pyobject<Object>
        ){
            for(auto&& v : obj){
                sequence.emplace_back(PyTypeInfo(v));
            }
            assert(pybind11::len(obj)==(sequence.size()));
        }else if constexpr (traits::mapping_like_pyobject<Object>){
                auto asdict=pybind11::reinterpret_borrow<pybind11::dict>(obj);
                str_key_only=is_all_keys_string_compatible(asdict);
                if(str_key_only){
                    for(auto&& [k,v] : asdict){
                        mapping_str_key_only.emplace_back(k.template cast<std::string>(),PyTypeInfo(v));
                    }
                }else{
                    for(auto&& [k,v] : asdict){
                        mapping_generic.emplace_back(PyTypeInfo(k),PyTypeInfo(v));
                    }
                }
                assert(pybind11::len(asdict)==(mapping_str_key_only.size()+mapping_generic.size()));
        }else if constexpr (std::derived_from<Object,pybind11::array>){
            dtype=asrc::core::util::get_dtype_enum(obj);
            is_col_major=asrc::core::util::is_col_major(obj);
        }else{
            if(
                kind==PyObjectTypeKind::set
                || kind==PyObjectTypeKind::mutable_set
                || kind==PyObjectTypeKind::sequence
                || kind==PyObjectTypeKind::mutable_sequence
            ){
                for(auto&& v : obj){
                    sequence.emplace_back(PyTypeInfo(v));
                }
                assert(pybind11::len(obj)==(sequence.size()));
            }
            if(
                kind==PyObjectTypeKind::mapping
                || kind==PyObjectTypeKind::mutable_mapping
            ){
                auto asdict=pybind11::reinterpret_borrow<pybind11::dict>(obj);
                str_key_only=is_all_keys_string_compatible(asdict);
                if(str_key_only){
                    for(auto&& [k,v] : asdict){
                        mapping_str_key_only.emplace_back(k.template cast<std::string>(),PyTypeInfo(v));
                    }
                }else{
                    for(auto&& [k,v] : asdict){
                        mapping_generic.emplace_back(PyTypeInfo(k),PyTypeInfo(v));
                    }
                }
                assert(pybind11::len(asdict)==(mapping_str_key_only.size()+mapping_generic.size()));
            }
            if(kind==PyObjectTypeKind::numpy_array){
                auto arr=pybind11::reinterpret_borrow<pybind11::array>(obj);
                dtype=asrc::core::util::get_dtype_enum(arr);
                is_col_major=asrc::core::util::is_col_major(arr);
            }
        }
    }
    template<class Archive>
    inline void serialize(Archive& archive){
        archive(cereal::make_nvp("kind",kind));
        archive(cereal::make_nvp("is_basic_type",is_basic_type));
        if(!is_basic_type){
            if constexpr (traits::output_archive<Archive>){
                auto cloudpickle=pybind11::module::import("cloudpickle");
                auto type_string=cloudpickle.attr("dumps")(self).cast<std::string>();

                std::uint32_t nameid;
                if constexpr (traits::text_archive<Archive>){
                    type_string=cereal::base64::encode(
                        reinterpret_cast<const unsigned char *>(type_string.c_str()),
                        type_string.size()
                    );
                }
                // output archiveではpolymorphic_idがconst char*をキーとしたmapで管理されているため、
                // registerPolymorphicTypeに一時変数のtype_string.c_str()を渡すと上手く動かない(同じ文字列でもアドレスが異なるため。)
                // この対策として、PyTypeInfoのstaticメンバとしてtype_stringを保持しておくことでポインタを固定する。
                auto it=PyTypeInfo::typeStringCache.find(type_string);
                if(it==PyTypeInfo::typeStringCache.end()){
                    it=PyTypeInfo::typeStringCache.insert(type_string).first;
                }
                nameid = archive.registerPolymorphicType((*it).c_str());
                archive( CEREAL_NVP_("polymorphic_id", nameid) );
                if( nameid & cereal::detail::msb_32bit ){
                    archive( CEREAL_NVP_("polymorphic_name", type_string) );
                }
            }else{
                std::uint32_t nameid;
                archive( CEREAL_NVP_("polymorphic_id", nameid) );
                std::string loaded_string;
                if( nameid & cereal::detail::msb_32bit ){
                    archive( CEREAL_NVP_("polymorphic_name", loaded_string) );
                    archive.registerPolymorphicName(nameid, loaded_string);
                }else{
                    loaded_string = archive.getPolymorphicName(nameid);
                }
                auto cloudpickle=pybind11::module::import("cloudpickle");
                if constexpr (traits::text_archive<Archive>){
                    self=cloudpickle.attr("loads")(pybind11::bytes(cereal::base64::decode(loaded_string)));
                }else{
                    self=cloudpickle.attr("loads")(pybind11::bytes(loaded_string));
                }
            }
        }else{
            if constexpr (traits::input_archive<Archive>){
                switch(kind){
                    case PyObjectTypeKind::none :
                        self=pybind11::type::of(pybind11::none());
                        break;
                    case PyObjectTypeKind::type :
                        self=pybind11::type::of(pybind11::handle((PyObject*)(&PyBool_Type)));
                        break;
                    case PyObjectTypeKind::boolean :
                        self=pybind11::type::of(pybind11::bool_());
                        break;
                    case PyObjectTypeKind::integer :
                        self=pybind11::type::of(pybind11::int_());
                        break;
                    case PyObjectTypeKind::float_ :
                        self=pybind11::type::of(pybind11::float_());
                        break;
                    case PyObjectTypeKind::bytearray :
                        self=pybind11::type::of(pybind11::bytearray());
                        break;
                    case PyObjectTypeKind::bytes :
                        self=pybind11::type::of(pybind11::bytes());
                        break;
                    case PyObjectTypeKind::string :
                        self=pybind11::type::of(pybind11::str());
                        break;
                    case PyObjectTypeKind::set :
                        self=pybind11::type::of(pybind11::frozenset(pybind11::set()));
                        break;
                    case PyObjectTypeKind::mutable_set :
                        self=pybind11::type::of(pybind11::set());
                        break;
                    case PyObjectTypeKind::sequence :
                        self=pybind11::type::of(pybind11::tuple());
                        break;
                    case PyObjectTypeKind::mutable_sequence :
                        self=pybind11::type::of(pybind11::list());
                        break;
                    case PyObjectTypeKind::mutable_mapping :
                        self=pybind11::type::of(pybind11::dict());
                        break;
                    case PyObjectTypeKind::numpy_array :
                        self=pybind11::type::of(pybind11::array());
                        break;
                    default :
                        break;
                }
            }
        }
        if(
            kind==PyObjectTypeKind::set
            || kind==PyObjectTypeKind::mutable_set
            || kind==PyObjectTypeKind::sequence
            || kind==PyObjectTypeKind::mutable_sequence
        ){
            archive(cereal::make_nvp("sequence",sequence));
        }else{
            if constexpr (traits::input_archive<Archive>){
                sequence.clear();
            }
        }
        if(
            kind==PyObjectTypeKind::mapping
            || kind==PyObjectTypeKind::mutable_mapping
        ){
            archive(cereal::make_nvp("str_key_only",str_key_only));
            if(str_key_only){
                archive(cereal::make_nvp("mapping_str_key_only",mapping_str_key_only));
            }else{
                archive(cereal::make_nvp("mapping_generic",mapping_generic));
            }
        }else{
            if constexpr (traits::input_archive<Archive>){
                mapping_str_key_only.clear();
                mapping_generic.clear();
            }
        }
        if(kind==PyObjectTypeKind::numpy_array){
            archive(cereal::make_nvp("dtype",dtype));
            archive(cereal::make_nvp("is_col_major",is_col_major));
        }
    }
    static bool isBoolean(const pybind11::type& cls);
    static bool isIntegral(const pybind11::type& cls);
    static bool isReal(const pybind11::type& cls);
    static bool isSet(const pybind11::type& cls);
    static bool isSequence(const pybind11::type& cls);
    static bool isMapping(const pybind11::type& cls);
    static bool isMutableSet(const pybind11::type& cls);
    static bool isMutableSequence(const pybind11::type& cls);
    static bool isMutableMapping(const pybind11::type& cls);
    private:
    static PyObject* numpy_bool_;
    static PyObject* realABC;
    static PyObject* integralABC;
    static PyObject* setABC;
    static PyObject* sequenceABC;
    static PyObject* mappingABC;
    static PyObject* mutableSetABC;
    static PyObject* mutableSequenceABC;
    static PyObject* mutableMappingABC;
    static std::unordered_set<std::string> typeStringCache;
};

template<traits::pyobject Object>
bool PYBIND11_EXPORT is_basic_type_object(const Object& obj){
    if(obj.ptr()==nullptr || obj.is_none()){
        return true;
    }
    if constexpr (std::derived_from<Object,pybind11::none>){
        return true;
    }else if constexpr (std::derived_from<Object,pybind11::type>){
        return true;
    }else if constexpr (std::derived_from<Object,pybind11::bool_>){
        return pybind11::type::of(obj).ptr() == (PyObject*)(&PyBool_Type);
    }else if constexpr (std::derived_from<Object,pybind11::int_>){
        return pybind11::type::of(obj).ptr() == (PyObject*)(&PyLong_Type);
    }else if constexpr (std::derived_from<Object,pybind11::float_>){
        return pybind11::type::of(obj).ptr() == (PyObject*)(&PyFloat_Type);
    }else if constexpr (std::derived_from<Object,pybind11::bytearray>){
        return pybind11::type::of(obj).ptr() == (PyObject*)(&PyByteArray_Type);
    }else if constexpr (std::derived_from<Object,pybind11::bytes>){
        return pybind11::type::of(obj).ptr() == (PyObject*)(&PyBytes_Type);
    }else if constexpr (std::derived_from<Object,pybind11::str>){
        return pybind11::type::of(obj).ptr() == (PyObject*)(&PyUnicode_Type);
    }else if constexpr (std::derived_from<Object,pybind11::array>){
        return pybind11::type::of(obj).ptr() == (PyObject*)(pybind11::detail::npy_api::get().PyArray_Type_);
    }else if constexpr (std::derived_from<Object,pybind11::set>){
        return pybind11::type::of(obj).ptr() == (PyObject*)(&PySet_Type);
    }else if constexpr (std::derived_from<Object,pybind11::frozenset>){
        return pybind11::type::of(obj).ptr() == (PyObject*)(&PyFrozenSet_Type);
    }else if constexpr (std::derived_from<Object,pybind11::list>){
        return pybind11::type::of(obj).ptr() == (PyObject*)(&PyList_Type);
    }else if constexpr (std::derived_from<Object,pybind11::tuple>){
        return pybind11::type::of(obj).ptr() == (PyObject*)(&PyTuple_Type);
    }else if constexpr (std::derived_from<Object,pybind11::sequence>){
        return pybind11::type::of(obj).ptr() == (PyObject*)(&PyList_Type) || pybind11::type::of(obj).ptr() == (PyObject*)(&PyTuple_Type);
    }else if constexpr (std::derived_from<Object,pybind11::dict>){
        return pybind11::type::of(obj).ptr() == (PyObject*)(&PyDict_Type);
    }else{
        // それ以外はisinstanceで判別
        if(PyType_Check(obj.ptr())){
            return true;
        }else if(pybind11::isinstance<nlohmann::json>(obj)){
            return pybind11::type::of(obj).is(pybind11::type::of<nlohmann::json>());
        }else if(pybind11::isinstance<nlohmann::ordered_json>(obj)){
            return pybind11::type::of(obj).is(pybind11::type::of<nlohmann::ordered_json>());
        }else if(
            pybind11::hasattr(obj,"_allow_cereal_serialization_in_cpp")
            && obj.attr("_allow_cereal_serialization_in_cpp").template cast<bool>()
        ){
            return false;
        }else if(pybind11::isinstance<pybind11::bool_>(obj)){
            return pybind11::type::of(obj).ptr() == (PyObject*)(&PyBool_Type);
        }else if(pybind11::isinstance<pybind11::int_>(obj)){
            return pybind11::type::of(obj).ptr() == (PyObject*)(&PyLong_Type);
        }else if(pybind11::isinstance<pybind11::float_>(obj)){
            return pybind11::type::of(obj).ptr() == (PyObject*)(&PyFloat_Type);
        }else if(pybind11::isinstance<pybind11::bytearray>(obj)){
            return pybind11::type::of(obj).ptr() == (PyObject*)(&PyByteArray_Type);
        }else if(pybind11::isinstance<pybind11::bytes>(obj)){
            return pybind11::type::of(obj).ptr() == (PyObject*)(&PyBytes_Type);
        }else if(pybind11::isinstance<pybind11::str>(obj)){
            return pybind11::type::of(obj).ptr() == (PyObject*)(&PyUnicode_Type);
        }else if(pybind11::isinstance<pybind11::array>(obj)){
            return pybind11::type::of(obj).ptr() == (PyObject*)(pybind11::detail::npy_api::get().PyArray_Type_);
        }else if(pybind11::isinstance<pybind11::set>(obj)){
            return pybind11::type::of(obj).ptr() == (PyObject*)(&PySet_Type);
        }else if(pybind11::isinstance<pybind11::frozenset>(obj)){
            return pybind11::type::of(obj).ptr() == (PyObject*)(&PyFrozenSet_Type);
        }else if(pybind11::isinstance<pybind11::list>(obj)){
            return pybind11::type::of(obj).ptr() == (PyObject*)(&PyList_Type);
        }else if(pybind11::isinstance<pybind11::tuple>(obj)){
            return pybind11::type::of(obj).ptr() == (PyObject*)(&PyTuple_Type);
        }else if(pybind11::isinstance<pybind11::dict>(obj)){
            return pybind11::type::of(obj).ptr() == (PyObject*)(&PyDict_Type);
        }else{
            return false;
        }
    }
}
template<traits::pyobject Object>
PyObjectTypeKind PYBIND11_EXPORT get_type_kind_of_object(const Object& obj){
    // まずはC++側の型で判別可能なら中身がnoneでも型情報から判定
    if constexpr (std::derived_from<Object,pybind11::none>){
        return PyObjectTypeKind::none;
    }else if constexpr (std::derived_from<Object,pybind11::type>){
        return PyObjectTypeKind::type;
    }else if constexpr (std::derived_from<Object,pybind11::bool_>){
        return PyObjectTypeKind::boolean;
    }else if constexpr (std::derived_from<Object,pybind11::int_>){
        return PyObjectTypeKind::integer;
    }else if constexpr (std::derived_from<Object,pybind11::float_>){
        return PyObjectTypeKind::float_;
    }else if constexpr (std::derived_from<Object,pybind11::bytearray>){
        return PyObjectTypeKind::bytearray;
    }else if constexpr (std::derived_from<Object,pybind11::bytes>){
        return PyObjectTypeKind::bytes;
    }else if constexpr (std::derived_from<Object,pybind11::str>){
        return PyObjectTypeKind::string;
    }else if constexpr (std::derived_from<Object,pybind11::array>){
        return PyObjectTypeKind::numpy_array;
    }else if constexpr (std::derived_from<Object,pybind11::set>){
        return PyObjectTypeKind::mutable_set;
    }else if constexpr (std::derived_from<Object,pybind11::frozenset>){
        return PyObjectTypeKind::set;
    }else if constexpr (std::derived_from<Object,pybind11::list>){
        return PyObjectTypeKind::mutable_sequence;
    }else if constexpr (std::derived_from<Object,pybind11::tuple>){
        return PyObjectTypeKind::sequence;
    }else if constexpr (std::derived_from<Object,pybind11::sequence>){
        if(pybind11::type::of(obj).ptr() == (PyObject*)(&PyTuple_Type)){
            return PyObjectTypeKind::sequence;
        }else{
            return PyObjectTypeKind::mutable_sequence;
        }
    }else if constexpr (std::derived_from<Object,pybind11::dict>){
        return PyObjectTypeKind::mutable_mapping;
    }else{
        // それ以外はisinstanceで判別
        if(obj.ptr()==nullptr || obj.is_none()){
            return PyObjectTypeKind::none;
        }else if(PyType_Check(obj.ptr())){
            return PyObjectTypeKind::type;
        }else if(pybind11::isinstance<nlohmann::json>(obj)){
                return PyObjectTypeKind::nl_json;
        }else if(pybind11::isinstance<nlohmann::ordered_json>(obj)){
                return PyObjectTypeKind::nl_ordered_json;
        }else if(
            pybind11::hasattr(obj,"_allow_cereal_serialization_in_cpp")
            && obj.attr("_allow_cereal_serialization_in_cpp").template cast<bool>()
        ){
            return PyObjectTypeKind::cpp_type;
        }else if(PyTypeInfo::isBoolean(pybind11::type::of(obj))){
            return PyObjectTypeKind::boolean;
        }else if(PyTypeInfo::isIntegral(pybind11::type::of(obj))){
            return PyObjectTypeKind::integer;
        }else if(PyTypeInfo::isReal(pybind11::type::of(obj))){
            return PyObjectTypeKind::float_;
        }else if(pybind11::isinstance<pybind11::bytearray>(obj)){
            return PyObjectTypeKind::bytearray;
        }else if(pybind11::isinstance<pybind11::bytes>(obj)){
            return PyObjectTypeKind::bytes;
        }else if(pybind11::isinstance<pybind11::str>(obj)){
            return PyObjectTypeKind::string;
        }else if(pybind11::isinstance<pybind11::array>(obj)){
            return PyObjectTypeKind::numpy_array;
        }else if(PyTypeInfo::isMutableSet(pybind11::type::of(obj))){
            return PyObjectTypeKind::mutable_set;
        }else if(PyTypeInfo::isSet(pybind11::type::of(obj))){
            return PyObjectTypeKind::set;
        }else if(PyTypeInfo::isMutableSequence(pybind11::type::of(obj))){
            return PyObjectTypeKind::mutable_sequence;
        }else if(PyTypeInfo::isSequence(pybind11::type::of(obj))){
            return PyObjectTypeKind::sequence;
        }else if(PyTypeInfo::isMutableMapping(pybind11::type::of(obj))){
            return PyObjectTypeKind::mutable_mapping;
        }else{
            return PyObjectTypeKind::other;
        }
    }
}

ASRC_NAMESPACE_END(util)

ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)

//
// PyTypeInfoはメンバ変数に自分自身と同じ型の変数を持つため、nlohmann::adl_serializerを個別に特殊化する必要がある。
// (特殊化しない場合、./nljson.hに定義された共通のadl_serializerとNLJSONArchiveとの間でconstraintsの循環参照が発生してしまう)
//
NLOHMANN_JSON_NAMESPACE_BEGIN
    template<>
    struct adl_serializer<::asrc::core::util::PyTypeInfo>
    {
        template<typename BasicJsonType, typename TargetType=::asrc::core::util::PyTypeInfo>
        static void to_json(BasicJsonType& j, TargetType&& info) {
            if constexpr (std::same_as<BasicJsonType,ordered_json>){
                ::asrc::core::util::NLJSONOutputArchive ar(j,true);
                ar(std::forward<TargetType>(info));
            }else{
                ordered_json oj;
                ::asrc::core::util::NLJSONOutputArchive ar(oj,true);
                ar(std::forward<TargetType>(info));
                j=std::move(oj);
            }
        }
        template<typename BasicJsonType, typename TargetType=::asrc::core::util::PyTypeInfo>
        static void from_json(BasicJsonType&& j, TargetType& info) {
            if constexpr (std::same_as<BasicJsonType,ordered_json>){
                ::asrc::core::util::NLJSONInputArchive ar(std::forward<BasicJsonType>(j),true);
                ar(std::forward<TargetType>(info));
            }else{
                ordered_json oj(std::forward<BasicJsonType>(j));
                ::asrc::core::util::NLJSONInputArchive ar(oj,true);
                ar(std::forward<TargetType>(info));
            }
        }
    };
NLOHMANN_JSON_NAMESPACE_END
