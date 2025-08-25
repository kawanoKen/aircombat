// Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#include "util/serialization/pyobject.h"
#include "Utility.h"
#include <pybind11/functional.h>

namespace py=pybind11;
namespace nl=nlohmann;

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)
ASRC_INLINE_NAMESPACE_BEGIN(util)

void exportPyObjectSerialization(py::module &m){
    using namespace pybind11::literals;

    m
    .def("save_type_info_of",[](AvailableOutputArchiveTypes& archive, const py::object& obj){
        ASRC_CALL_ARCHIVE_FUNC(save_type_info_of, archive, obj);
    })
    .def("save_without_type_info_to_json",[](const py::object& obj){
        nl::ordered_json ret;
        {
            NLJSONOutputArchive archive(ret,true);
            archive(obj);
        }
        return std::move(ret);
    })
    .def("save_without_type_info_to_binary",[](const py::object& obj){
        std::stringstream oss;
        {
            cereal::PortableBinaryOutputArchive archive(oss);
            archive(obj);
        }
        return py::bytes(oss.str());
    })
    .def("save_without_type_info",[](AvailableOutputArchiveTypes& archive, const py::object& obj){
        call_operator(archive, obj);
    })
    .def("save_without_type_info",[](AvailableOutputArchiveTypes& archive, const std::string& name, const py::object& obj){
        call_operator(archive, cereal::make_nvp(name.c_str(),obj));
    })
    .def("save_with_type_info_to_json",[](const py::object& obj){
        nl::ordered_json ret;
        {
            NLJSONOutputArchive archive(ret,true);
            save_with_type_info(archive,obj);
        }
        return std::move(ret);
    })
    .def("save_with_type_info_to_binary",[](const py::object& obj){
        std::stringstream oss;
        {
            cereal::PortableBinaryOutputArchive archive(oss);
            save_with_type_info(archive,obj);
        }
        return py::bytes(oss.str());
    })
    .def("save_with_type_info",[](AvailableOutputArchiveTypes& archive, const py::object& obj){
        ASRC_CALL_ARCHIVE_FUNC(save_with_type_info, archive, obj);
    })
    .def("save_with_type_info",[](AvailableOutputArchiveTypes& archive, const std::string& name, const py::object& obj){
        ASRC_CALL_ARCHIVE_FUNC(save_with_type_info, archive, cereal::make_nvp(name.c_str(),obj));
    })
    .def("load_without_type_info_from_json",[](const nl::ordered_json& j){
        py::object ret;
        {
            NLJSONInputArchive archive(j,true);
            archive(ret);
        }
        return std::move(ret);
    })
    .def("load_without_type_info_from_binary",[](const py::bytes& str){
        py::object ret;
        {
            std::istringstream iss(str.cast<std::string>());
            cereal::PortableBinaryInputArchive archive(iss);
            archive(ret);
        }
        return std::move(ret);
    })
    .def("load_without_type_info",[](AvailableInputArchiveTypes& archive){
        return ASRC_CALL_ARCHIVE_FUNC(load_python_object, archive);
    })
    .def("load_without_type_info",[](AvailableInputArchiveTypes& archive, const std::string& name){
        return ASRC_CALL_ARCHIVE_FUNC(load_python_object, archive, name);
    })
    .def("load_with_type_info_from_json",[](const nl::ordered_json& j){
        py::object ret;
        {
            NLJSONInputArchive archive(j,true);
            load_with_type_info(archive,ret);
        }
        return std::move(ret);
    })
    .def("load_with_type_info_from_binary",[](const py::bytes& str){
        py::object ret;
        {
            std::istringstream iss(str.cast<std::string>());
            cereal::PortableBinaryInputArchive archive(iss);
            load_with_type_info(archive,ret);
        }
        return std::move(ret);
    })
    .def("load_with_type_info",[](AvailableInputArchiveTypes& archive){
        return ASRC_CALL_ARCHIVE_FUNC(load_python_object_with_type_info, archive);
    })
    .def("load_with_type_info",[](AvailableInputArchiveTypes& archive, const std::string& name){
        return ASRC_CALL_ARCHIVE_FUNC(load_python_object_with_type_info, archive, name);
    })
    .def("load_with_external_type_info_from_json",[](const nl::ordered_json& j, const PyTypeInfo& hint){
        py::object ret;
        {
            NLJSONInputArchive archive(j,true);
            load_with_external_type_info(archive,ret, hint);
        }
        return std::move(ret);
    })
    .def("load_with_external_type_info_from_binary",[](const py::bytes& str, const PyTypeInfo& hint){
        py::object ret;
        {
            std::istringstream iss(str.cast<std::string>());
            cereal::PortableBinaryInputArchive archive(iss);
            load_with_external_type_info(archive,ret, hint);
        }
        return std::move(ret);
    })
    .def("load_with_external_type_info",[](AvailableInputArchiveTypes& archive, const PyTypeInfo& hint){
        return ASRC_CALL_ARCHIVE_FUNC(load_python_object_with_external_type_info, archive, hint);
    })
    .def("load_with_external_type_info",[](AvailableInputArchiveTypes& archive, const PyTypeInfo& hint, const std::string& name){
        return ASRC_CALL_ARCHIVE_FUNC(load_python_object_with_external_type_info, archive, hint, name);
    })
    .def("save_attr",[](AvailableOutputArchiveTypes& archive, py::handle& obj, const py::args& attr_names){
        for(auto&& elem : attr_names){
            auto attr_name =elem.cast<std::string>();
            ASRC_CALL_ARCHIVE_FUNC(save_attr, archive, obj, attr_name);
        }
    })
    .def("load_attr",[](AvailableInputArchiveTypes& archive, py::handle& obj, const py::args& attr_names){
        for(auto&& elem : attr_names){
            auto attr_name =elem.cast<std::string>();
            ASRC_CALL_ARCHIVE_FUNC(load_attr, archive, obj, attr_name);
        }
    })
    .def("serialize_attr",[](AvailableArchiveTypes& archive, py::handle& obj, const py::args& attr_names){
        for(auto&& elem : attr_names){
            auto attr_name =elem.cast<std::string>();
            ASRC_CALL_ARCHIVE_FUNC(serialize_attr, archive, obj, attr_name);
        }
    })
    .def("save_attr_with_type_info",[](AvailableOutputArchiveTypes& archive, py::handle& obj, const py::args& attr_names){
        for(auto&& elem : attr_names){
            auto attr_name =elem.cast<std::string>();
            ASRC_CALL_ARCHIVE_FUNC(save_attr_with_type_info, archive, obj, attr_name);
        }
    })
    .def("load_attr_with_type_info",[](AvailableInputArchiveTypes& archive, py::handle& obj, const py::args& attr_names){
        for(auto&& elem : attr_names){
            auto attr_name =elem.cast<std::string>();
            ASRC_CALL_ARCHIVE_FUNC(load_attr_with_type_info, archive, obj, attr_name);
        }
    })
    .def("serialize_attr_with_type_info",[](AvailableArchiveTypes& archive, py::handle& obj, const py::args& attr_names){
        for(auto&& elem : attr_names){
            auto attr_name =elem.cast<std::string>();
            ASRC_CALL_ARCHIVE_FUNC(serialize_attr_with_type_info, archive, obj, attr_name);
        }
    })
    .def("load_attr_with_external_type_info",[](AvailableInputArchiveTypes& archive, py::handle& obj, const std::string& attr_name, const PyTypeInfo& hint){
        ASRC_CALL_ARCHIVE_FUNC(load_attr_with_external_type_info, archive, obj, attr_name, hint);
    })
    .def("load_attr_with_external_type_info",[](AvailableInputArchiveTypes& archive, py::handle& obj, const py::tuple& attr_names, const py::tuple& hints){
        assert(attr_names.size()==hints.size());
        for(std::size_t i=0;i<attr_names.size();++i){
            auto attr_name = attr_names[i].cast<std::string>();
            auto hint = hints[i].cast<PyTypeInfo>();
            ASRC_CALL_ARCHIVE_FUNC(load_attr_with_external_type_info, archive, obj, attr_name, hint);
        }
    })
    .def("save_internal_state_of_attr",[](AvailableOutputArchiveTypes& archive, bool full, py::handle& obj, const py::args& attr_names){
        for(auto&& elem : attr_names){
            auto attr_name =elem.cast<std::string>();
            auto wrapped=makeInternalStateSerializer(pybind11::getattr(obj,attr_name.c_str()),full);
            call_operator(archive,cereal::make_nvp(attr_name.c_str(),wrapped));
        }
    })
    .def("load_internal_state_of_attr",[](AvailableInputArchiveTypes& archive, bool full, py::handle& obj, const py::args& attr_names){
        for(auto&& elem : attr_names){
            auto attr_name =elem.cast<std::string>();
            auto wrapped=makeInternalStateSerializer(pybind11::getattr(obj,attr_name.c_str()),full);
            call_operator(archive,cereal::make_nvp(attr_name.c_str(),wrapped));
        }
    })
    .def("serialize_internal_state_of_attr",[](AvailableArchiveTypes& archive, bool full, py::handle& obj, const py::args& attr_names){
        for(auto&& elem : attr_names){
            auto attr_name =elem.cast<std::string>();
            auto wrapped=makeInternalStateSerializer(pybind11::getattr(obj,attr_name.c_str()),full);
            call_operator(archive,cereal::make_nvp(attr_name.c_str(),wrapped));
        }
    })
    .def("save_by_func",[](AvailableOutputArchiveTypes& archive,
        const std::function<void(AvailableOutputArchiveTypes&)>& saver
    ){
        VariantArchiveSaverWrapper wrapped(saver);
        call_operator(archive,wrapped);
    })
    .def("save_by_func",[](AvailableOutputArchiveTypes& archive, const std::string& attr_name,
        const std::function<void(AvailableOutputArchiveTypes&)>& saver
    ){
        VariantArchiveSaverWrapper wrapped(saver);
        call_operator(archive,cereal::make_nvp(attr_name.c_str(),wrapped));
    })
    .def("load_by_func",[](AvailableInputArchiveTypes& archive,
        const std::function<void(AvailableInputArchiveTypes&)>& loader
    ){
        VariantArchiveLoaderWrapper wrapped(loader);
        call_operator(archive,wrapped);
    })
    .def("load_by_func",[](AvailableInputArchiveTypes& archive, const std::string& attr_name,
        const std::function<void(AvailableInputArchiveTypes&)>& loader
    ){
        VariantArchiveLoaderWrapper wrapped(loader);
        call_operator(archive,cereal::make_nvp(attr_name.c_str(),wrapped));
    })
    .def("serialize_by_func",[](AvailableArchiveTypes& archive,
        const std::function<void(AvailableArchiveTypes&)>& serializer
    ){
        VariantArchiveSerializerWrapper wrapped(serializer);
        call_operator(archive,wrapped);
    })
    .def("serialize_by_func",[](AvailableArchiveTypes& archive, const std::string& attr_name,
        const std::function<void(AvailableArchiveTypes&)>& serializer
    ){
        VariantArchiveSerializerWrapper wrapped(serializer);
        call_operator(archive,cereal::make_nvp(attr_name.c_str(),wrapped));
    })
    .def("save_internal_state_by_func",[](AvailableOutputArchiveTypes& archive, bool full,
        const std::function<void(AvailableOutputArchiveTypes&, bool)>& saver
    ){
        VariantArchiveInternalStateSaverWrapper wrapped(saver,full);
        call_operator(archive,wrapped);
    })
    .def("save_internal_state_by_func",[](AvailableOutputArchiveTypes& archive, bool full, const std::string& attr_name,
        const std::function<void(AvailableOutputArchiveTypes&, bool)>& saver
    ){
        VariantArchiveInternalStateSaverWrapper wrapped(saver,full);
        call_operator(archive,cereal::make_nvp(attr_name.c_str(),wrapped));
    })
    .def("load_internal_state_by_func",[](AvailableInputArchiveTypes& archive, bool full,
        const std::function<void(AvailableInputArchiveTypes&, bool)>& loader
    ){
        VariantArchiveInternalStateLoaderWrapper wrapped(loader,full);
        call_operator(archive,wrapped);
    })
    .def("load_internal_state_by_func",[](AvailableInputArchiveTypes& archive, bool full, const std::string& attr_name,
        const std::function<void(AvailableInputArchiveTypes&, bool)>& loader
    ){
        VariantArchiveInternalStateLoaderWrapper wrapped(loader,full);
        call_operator(archive,cereal::make_nvp(attr_name.c_str(),wrapped));
    })
    .def("serialize_internal_state_by_func",[](AvailableArchiveTypes& archive, bool full,
        const std::function<void(AvailableArchiveTypes&, bool)>& serializer
    ){
        VariantArchiveInternalStateSerializerWrapper wrapped(serializer,full);
        call_operator(archive,wrapped);
    })
    .def("serialize_internal_state_by_func",[](AvailableArchiveTypes& archive, bool full, const std::string& attr_name,
        const std::function<void(AvailableArchiveTypes&, bool)>& serializer
    ){
        VariantArchiveInternalStateSerializerWrapper wrapped(serializer,full);
        call_operator(archive,cereal::make_nvp(attr_name.c_str(),wrapped));
    })
    ;
}

ASRC_NAMESPACE_END(util)
ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
