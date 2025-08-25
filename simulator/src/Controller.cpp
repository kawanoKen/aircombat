// Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#include "Controller.h"
#include <pybind11/stl.h>
#include "Utility.h"
#include "Factory.h"
#include "SimulationManager.h"
#include "Asset.h"
#include "Agent.h"

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)
using namespace util;

const std::string Controller::baseName="Controller"; // baseNameを上書き

void Controller::initialize(){
    BaseType::initialize();
    auto fullName=getFullName();
    if(fullName.find("/")>=0){
        team=fullName.substr(0,fullName.find("/"));
        group=fullName.substr(0,fullName.rfind("/"));
        name=fullName.substr(fullName.rfind("/")+1);
    }else{
        team=group=name=fullName;
    }
    if(instanceConfig.contains("parent")){
        parent=instanceConfig.at("parent");
        removeWhenKilled=parent.lock()->removeWhenKilled;
    }
    if(parent.expired()){
        throw std::runtime_error("Controller must have a parent. fullName="+getFullName());
    }
}

void Controller::serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full){
    BaseType::serializeInternalState(archive,full);

    if(full){
        if(asrc::core::util::isInputArchive(archive)){
            auto fullName=getFullName();
            if(fullName.find("/")>=0){
                team=fullName.substr(0,fullName.find("/"));
                group=fullName.substr(0,fullName.rfind("/"));
                name=fullName.substr(fullName.rfind("/")+1);
            }else{
                team=group=name=fullName;
            }
        }

        ASRC_SERIALIZE_NVP(archive
            ,parent
        )
    }
}

bool Controller::isAlive() const{
    return parent.lock()->isAlive();
}
std::string Controller::getTeam() const{
    return team;
}
std::string Controller::getGroup() const{
    return group;
}
std::string Controller::getName() const{
    return name;
}
void Controller::perceive(bool inReset){
}
void Controller::control(){
}
void Controller::behave(){
}
void Controller::kill(){
    observables=nl::json::object();
    commands=nl::json::object();
}
bool Controller::generateCommunicationBuffer(const std::string& name_,const nl::json& participants_,const nl::json& inviteOnRequest_){
    return manager->generateCommunicationBuffer(name_,participants_,inviteOnRequest_);
}
void Controller::setManagerAccessor(const std::shared_ptr<EntityManagerAccessor>& ema){
    BaseType::setManagerAccessor(ema);
    manager=std::dynamic_pointer_cast<SimulationManagerAccessorForPhysicalAsset>(ema);
    assert(manager);
}

void exportController(py::module &m, const std::shared_ptr<asrc::core::FactoryHelper>& factoryHelper)
{
    using namespace pybind11::literals;

    FACTORY_ADD_BASE(Controller,{"Controller","PhysicalAsset","Callback","!AnyOthers"},{"!Agent","AnyOthers"})

    expose_entity_subclass<Controller>(m,"Controller")
    DEF_READWRITE(Controller,name)
    DEF_READWRITE(Controller,parent)
    ;
}

ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
