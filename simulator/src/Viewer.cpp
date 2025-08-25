// Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#include "Viewer.h"
#include "Utility.h"
#include "SimulationManager.h"
#include "Asset.h"
#include "Agent.h"

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)
using namespace util;

const std::string Viewer::baseName="Viewer"; // baseNameを上書き

void Viewer::initialize(){
    BaseType::initialize();
    isValid=false;
}
Viewer::~Viewer(){
    close();
}
void Viewer::serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full){
    BaseType::serializeInternalState(archive,full);

    ASRC_SERIALIZE_NVP(archive
        ,isValid
    );
}
void Viewer::validate(){
    BaseType::validate();
    isValid=true;
}
void Viewer::display(){
}
void Viewer::close(){
}
void Viewer::onEpisodeBegin(){
    if(!isValid){
        validate();
    }
}
void Viewer::onInnerStepBegin(){
}
void Viewer::onInnerStepEnd(){
    if(isValid){
        display();
    }
}
std::shared_ptr<GUIDataFrameCreator> Viewer::getDataFrameCreator(){
    return std::make_shared<GUIDataFrameCreator>();
}
void exportViewer(py::module &m, const std::shared_ptr<asrc::core::FactoryHelper>& factoryHelper)
{
    using namespace pybind11::literals;

    FACTORY_ADD_BASE(Viewer,{"Viewer","!AnyOthers"},{"Callback","!AnyOthers"})

    expose_entity_subclass<Viewer>(m,"Viewer")
    DEF_FUNC(Viewer,display)
    DEF_FUNC(Viewer,close)
    DEF_FUNC(Viewer,getDataFrameCreator)
    DEF_READWRITE(Viewer,isValid)
    ;
    FACTORY_ADD_CLASS(Viewer,Viewer) //Register to Factory for "None" viewer.
}

ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
