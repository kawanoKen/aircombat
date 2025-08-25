// Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#include "util/serialization/Serializer.h"
#include "util/serialization/pyobject.h"
#include <pybind11/stl.h>

namespace py=pybind11;

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)

ASRC_INLINE_NAMESPACE_BEGIN(util)

void exportSerializer(py::module &m)
{
    using namespace pybind11::literals;

    py::class_<cereal::detail::OutputArchiveBase>(m,"OutputArchiveBase")
    ;
    py::class_<cereal::detail::InputArchiveBase>(m,"InputArchiveBase")
    ;
    py::class_<NLJSONOutputArchive,cereal::detail::OutputArchiveBase>(m,"NLJSONOutputArchive")
    ;
    py::class_<NLJSONInputArchive,cereal::detail::InputArchiveBase>(m,"NLJSONInputArchive")
    ;
    py::class_<cereal::PortableBinaryOutputArchive,cereal::detail::OutputArchiveBase>(m,"PortableBinaryOutputArchive")
    ;
    py::class_<cereal::PortableBinaryInputArchive,cereal::detail::InputArchiveBase>(m,"PortableBinaryIntputArchive")
    ;

    m
    .def("isValidArchive",[](AvailableArchiveTypes& archive){
        return isValidArchive(archive);
    })
    .def("isOutputArchive",[](AvailableArchiveTypes& archive){
        return isOutputArchive(archive);
    })
    .def("isInputArchive",[](AvailableArchiveTypes& archive){
        return isInputArchive(archive);
    })
    ;

}

ASRC_NAMESPACE_END(util)

ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
