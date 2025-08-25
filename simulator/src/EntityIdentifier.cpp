// Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#include "EntityIdentifier.h"
#include "Utility.h"
#include "Entity.h"

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)
using namespace util;

EntityIdentifier::EntityIdentifier(){
    worker_index = EntityManager::getWorkerIndex();
    vector_index = -1;
    episode_index = -1;
    entity_index = -1;
}
EntityIdentifier::EntityIdentifier(const std::int32_t& w, const std::int32_t& v, const std::int32_t& ep, const std::int32_t& en){
    worker_index = w;
    vector_index = v>=0 ? v : -1;
    episode_index = ep>=0 ? ep : -1;
    entity_index = en>=0 ? en : -1;
}
EntityIdentifier::EntityIdentifier(const nl::json& j){
    load_from_json(j);
}
std::string EntityIdentifier::toString() const{
    std::ostringstream ss;
    ss << *this;
    return ss.str();
}
std::int32_t EntityIdentifier::getWorkerIndex(){
    return EntityManager::getWorkerIndex();
}

void exportEntityIdentifier(py::module &m)
{
    bind_stl_container<std::set<EntityIdentifier>>(m,py::module_local(false));
    bind_stl_container<std::vector<EntityIdentifier>>(m,py::module_local(false));
    bind_stl_container<std::pair<EntityIdentifier,bool>>(m,py::module_local(false));
    bind_stl_container<std::vector<std::pair<EntityIdentifier,bool>>>(m,py::module_local(false));
    bind_stl_container<std::map<std::string,EntityIdentifier>>(m,py::module_local(false));
    bind_stl_container<std::map<EntityIdentifier,boost::uuids::uuid>>(m,py::module_local(false));
    bind_stl_container<std::map<boost::uuids::uuid,EntityIdentifier>>(m,py::module_local(false));

    expose_common_class<EntityIdentifier>(m,"EntityIdentifier")
    .def(py_init<>())
    .def(py_init<const std::int32_t&,const std::int32_t&,const std::int32_t&,const std::int32_t&>())
    .def(py_init<const nl::json&>())
    DEF_READWRITE(EntityIdentifier,worker_index)
    DEF_READWRITE(EntityIdentifier,vector_index)
    DEF_READWRITE(EntityIdentifier,episode_index)
    DEF_READWRITE(EntityIdentifier,entity_index)
    .def(py::self == py::self)
    .def(py::self != py::self)
    .def(py::self < py::self)
    .def(py::self > py::self)
    .def(py::self <= py::self)
    .def(py::self >= py::self)
    .def("__str__",[](const EntityIdentifier& ei){return ei.toString();})
    .def("__repr__",[](const EntityIdentifier& ei){
        std::ostringstream s;
        s<<"EntityIdentifier("<<ei.worker_index<<","<<ei.vector_index<<","<<ei.episode_index<<","<<ei.entity_index<<")";
        return s.str();
    })
    .def("__hash__",[](const EntityIdentifier& ei){
        return std::hash<EntityIdentifier>{}(ei);
    })
    ;
}

ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
