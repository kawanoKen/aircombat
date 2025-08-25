// Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#include "util/serialization/pyobject_util.h"
#include <pybind11/numpy.h>
#include <cereal/cereal.hpp>
#include <nlohmann/json.hpp>
#include "Utility.h"

namespace py=pybind11;
namespace nl=nlohmann;

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)
ASRC_INLINE_NAMESPACE_BEGIN(util)

bool is_basic_type_object(const py::type& cls){
    return (
        cls.ptr() == (PyObject*)(&PyType_Type)
        || cls.is(pybind11::type::of<nlohmann::json>())
        || cls.is(pybind11::type::of<nlohmann::ordered_json>())
        || (
            pybind11::hasattr(cls,"_allow_cereal_serialization_in_cpp")
            && cls.attr("_allow_cereal_serialization_in_cpp").cast<bool>()
        )
        || cls.ptr() == (PyObject*)(&PyBool_Type)
        || cls.ptr() == (PyObject*)(&PyLong_Type)
        || cls.ptr() == (PyObject*)(&PyFloat_Type)
        || cls.ptr() == (PyObject*)(&PyByteArray_Type)
        || cls.ptr() == (PyObject*)(&PyBytes_Type)
        || cls.ptr() == (PyObject*)(&PyUnicode_Type)
        || cls.ptr() == (PyObject*)(pybind11::detail::npy_api::get().PyArray_Type_)
        || cls.ptr() == (PyObject*)(&PySet_Type)
        || cls.ptr() == (PyObject*)(&PyFrozenSet_Type)
        || cls.ptr() == (PyObject*)(&PyList_Type)
        || cls.ptr() == (PyObject*)(&PyTuple_Type)
        || cls.ptr() == (PyObject*)(&PyDict_Type)
    );
}

PyObjectTypeKind get_type_kind_of_type(const py::type& cls){
    if(PyObject_IsSubclass(cls.ptr(), py::type::of<nl::json>().ptr())){
        return PyObjectTypeKind::nl_json;
    }else if(PyObject_IsSubclass(cls.ptr(), py::type::of<nl::ordered_json>().ptr())){
        return PyObjectTypeKind::nl_ordered_json;
    }else if(
        py::hasattr(cls,"_allow_cereal_serialization_in_cpp")
        && cls.attr("_allow_cereal_serialization_in_cpp").cast<bool>()
    ){
        return PyObjectTypeKind::cpp_type;
    }else if(cls.ptr() == py::type::of(py::none()).ptr()){
        return PyObjectTypeKind::none;
    }else if(PyTypeInfo::isBoolean(cls)){
        return PyObjectTypeKind::boolean;
    }else if(PyTypeInfo::isIntegral(cls)){
        return PyObjectTypeKind::integer;
    }else if(PyTypeInfo::isReal(cls)){
        return PyObjectTypeKind::float_;
    }else if(PyObject_IsSubclass(cls.ptr(), (PyObject*)(&PyByteArray_Type))){
        return PyObjectTypeKind::bytearray;
    }else if(PyObject_IsSubclass(cls.ptr(), (PyObject*)(&PyBytes_Type))){
        return PyObjectTypeKind::bytes;
    }else if(PyObject_IsSubclass(cls.ptr(), (PyObject*)(&PyUnicode_Type))){
        return PyObjectTypeKind::string;
    }else if(PyTypeInfo::isMutableSet(cls)){
        return PyObjectTypeKind::mutable_set;
    }else if(PyTypeInfo::isSet(cls)){
        return PyObjectTypeKind::set;
    }else if(PyTypeInfo::isMutableSequence(cls)){
        return PyObjectTypeKind::mutable_sequence;
    }else if(PyTypeInfo::isSequence(cls)){
        return PyObjectTypeKind::sequence;
    }else if(PyTypeInfo::isMutableMapping(cls)){
        return PyObjectTypeKind::mutable_mapping;
    }else if(PyTypeInfo::isMapping(cls)){
        return PyObjectTypeKind::mapping;
    }else if(PyObject_IsSubclass(cls.ptr(), (PyObject*)(py::detail::npy_api::get().PyArray_Type_))){
        return PyObjectTypeKind::numpy_array;
    }else{
        return PyObjectTypeKind::other;
    }
}

bool is_all_keys_string_compatible(const py::dict& d){
    for(auto&& [key,value] : d){
        try{
            auto tmp_str=key.cast<std::string>();
        }catch(...){
            return false;
        }
    }
    return true;
}
bool is_primitive_type(const PyObjectTypeKind& kind){
    return (
        kind==PyObjectTypeKind::none
        || kind==PyObjectTypeKind::type
        || kind==PyObjectTypeKind::boolean
        || kind==PyObjectTypeKind::integer
        || kind==PyObjectTypeKind::float_
        || kind==PyObjectTypeKind::string
        || kind==PyObjectTypeKind::bytearray
        || kind==PyObjectTypeKind::bytes
    );
}
bool is_primitive_type(const py::type& cls){
    return is_primitive_type(get_type_kind_of_type(cls));
}
bool is_primitive_object(const py::handle& obj){
    return is_primitive_type(get_type_kind_of_object(obj));
}

PyObject* PyTypeInfo::numpy_bool_=nullptr;
PyObject* PyTypeInfo::realABC=nullptr;
PyObject* PyTypeInfo::integralABC=nullptr;
PyObject* PyTypeInfo::setABC=nullptr;
PyObject* PyTypeInfo::sequenceABC=nullptr;
PyObject* PyTypeInfo::mappingABC=nullptr;
PyObject* PyTypeInfo::mutableSetABC=nullptr;
PyObject* PyTypeInfo::mutableSequenceABC=nullptr;
PyObject* PyTypeInfo::mutableMappingABC=nullptr;
std::unordered_set<std::string> PyTypeInfo::typeStringCache;

PyTypeInfo::PyTypeInfo()
:kind(PyObjectTypeKind::none),
 self(py::type::of(py::none()))
{
}

bool PyTypeInfo::isBoolean(const py::type& cls){
    if(!numpy_bool_){
        numpy_bool_=py::dtype::of<bool>().attr("type").ptr();
    }
    return (
        PyObject_IsSubclass(cls.ptr(), (PyObject*)(&PyBool_Type))
        || PyObject_IsSubclass(cls.ptr(), numpy_bool_)
    );
}
bool PyTypeInfo::isReal(const py::type& cls){
    if(!realABC){
        auto numbers=py::module::import("numbers");
        realABC=numbers.attr("Real").ptr();
    }
    return PyObject_IsSubclass(cls.ptr(), realABC);
}
bool PyTypeInfo::isIntegral(const py::type& cls){
    if(!integralABC){
        auto numbers=py::module::import("numbers");
        integralABC=numbers.attr("Integral").ptr();
    }
    return PyObject_IsSubclass(cls.ptr(), integralABC);
}
bool PyTypeInfo::isSet(const py::type& cls){
    if(!setABC){
        auto collections=py::module::import("collections");
        setABC=collections.attr("abc").attr("Set").ptr();
    }
    return PyObject_IsSubclass(cls.ptr(), setABC);
}
bool PyTypeInfo::isSequence(const py::type& cls){
    if(!sequenceABC){
        auto collections=py::module::import("collections");
        sequenceABC=collections.attr("abc").attr("Sequence").ptr();
    }
    return PyObject_IsSubclass(cls.ptr(), sequenceABC);
}
bool PyTypeInfo::isMapping(const py::type& cls){
    if(!mappingABC){
        auto collections=py::module::import("collections");
        mappingABC=collections.attr("abc").attr("Mapping").ptr();
    }
    return PyObject_IsSubclass(cls.ptr(), mappingABC);
}
bool PyTypeInfo::isMutableSet(const py::type& cls){
    if(!mutableSetABC){
        auto collections=py::module::import("collections");
        mutableSetABC=collections.attr("abc").attr("MutableSet").ptr();
    }
    return PyObject_IsSubclass(cls.ptr(), mutableSetABC);
}
bool PyTypeInfo::isMutableSequence(const py::type& cls){
    if(!mutableSequenceABC){
        auto collections=py::module::import("collections");
        mutableSequenceABC=collections.attr("abc").attr("MutableSequence").ptr();
    }
    return PyObject_IsSubclass(cls.ptr(), mutableSequenceABC);
}
bool PyTypeInfo::isMutableMapping(const py::type& cls){
    if(!mutableMappingABC){
        auto collections=py::module::import("collections");
        mutableMappingABC=collections.attr("abc").attr("MutableMapping").ptr();
    }
    return PyObject_IsSubclass(cls.ptr(), mutableMappingABC);
}

void exportPyObjectSerializationUtil(py::module &m){
    using namespace py::literals;

    expose_enum_value_helper(
        expose_enum_class<PyObjectTypeKind>(m,"PyObjectTypeKind")
        ,"none"
        ,"type"
        ,"boolean"
        ,"integer"
        ,"float_"
        ,"bytearray"
        ,"bytes"
        ,"string"
        ,"set"
        ,"mutable_set"
        ,"sequence"
        ,"mutable_sequence"
        ,"mapping"
        ,"mutable_mapping"
        ,"numpy_array"
        ,"nl_json"
        ,"nl_ordered_json"
        ,"cpp_type"
        ,"other"
    );
    

    py::class_<PyTypeInfo>(m,"PyTypeInfo")
    .def(py::init())
    .def(py::init<const py::handle&>())
    DEF_READWRITE(PyTypeInfo,str_key_only)
    DEF_STATIC_FUNC(PyTypeInfo,isSet)
    DEF_STATIC_FUNC(PyTypeInfo,isSequence)
    DEF_STATIC_FUNC(PyTypeInfo,isMapping)
    DEF_STATIC_FUNC(PyTypeInfo,isMutableSet)
    DEF_STATIC_FUNC(PyTypeInfo,isMutableSequence)
    DEF_STATIC_FUNC(PyTypeInfo,isMutableMapping)
    ;

    m.def("get_type_kind_of_object",&get_type_kind_of_object<py::handle>);
    m.def("get_type_kind_of_type",&get_type_kind_of_type);

    m.def("is_primitive_type",py::overload_cast<const PyObjectTypeKind&>(&is_primitive_type));
    m.def("is_primitive_type",py::overload_cast<const py::type&>(&is_primitive_type));
    m.def("is_primitive_object",&is_primitive_object);

    m.def("is_all_keys_string_compatible",&is_all_keys_string_compatible);

}

ASRC_NAMESPACE_END(util)
ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
