// Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#include "Asset.h"
#include "Utility.h"
#include "SimulationManager.h"
#include "Agent.h"
#include "CommunicationBuffer.h"
ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)
using namespace util;

Asset::Asset(const nl::json& modelConfig_,const nl::json& instanceConfig_)
:Entity(modelConfig_,instanceConfig_),observables(nl::json::object()),commands(nl::json::object()){
}
void Asset::initialize(){
    BaseType::initialize();
    removeWhenKilled=false;
    std::shared_ptr<SimulationManagerAccessorBase> manager=std::dynamic_pointer_cast<SimulationManagerAccessorBase>(this->Entity::manager);
    if(modelConfig.contains("interval")){
        nl::json intervals=modelConfig.at("interval");
        std::string intervalUnit="tick";
        if(intervals.contains("unit")){
            intervalUnit=intervals.at("unit");
        }
        if(intervalUnit=="time"){
            double dt=manager->getBaseTimeStep();
            interval[SimPhase::PERCEIVE]=std::max<std::uint64_t>(1,std::lround(getValueFromJsonKRD(intervals,"perceive",randomGen,dt)/dt));//reset at t=0 is not regarded as the "first time."
            interval[SimPhase::CONTROL]=std::max<std::uint64_t>(1,std::lround(getValueFromJsonKRD(intervals,"control",randomGen,dt)/dt));
            interval[SimPhase::BEHAVE]=std::max<std::uint64_t>(1,std::lround(getValueFromJsonKRD(intervals,"behave",randomGen,dt)/dt));
        }else if(intervalUnit=="tick"){
            interval[SimPhase::PERCEIVE]=getValueFromJsonKRD(intervals,"perceive",randomGen,1);//reset at t=0 is not regarded as the "first time."
            interval[SimPhase::CONTROL]=getValueFromJsonKRD(intervals,"control",randomGen,1);
            interval[SimPhase::BEHAVE]=getValueFromJsonKRD(intervals,"behave",randomGen,1);
        }else{
            throw std::runtime_error("Only \"tick\" or \"time\" can be designated as the unit of interval."+intervalUnit);
        }
    }else{
        interval[SimPhase::PERCEIVE]=1;
        interval[SimPhase::CONTROL]=1;
        interval[SimPhase::BEHAVE]=1;
    }
    interval[SimPhase::VALIDATE]=std::numeric_limits<std::uint64_t>::max();
    if(modelConfig.contains("firstTick")){
        nl::json firstTicks=modelConfig.at("firstTick");
        std::string firstTickUnit="tick";
        if(firstTicks.contains("unit")){
            firstTickUnit=firstTicks.at("unit");
        }
        if(firstTickUnit=="time"){
            double dt=manager->getBaseTimeStep();
            firstTick[SimPhase::PERCEIVE]=std::lround(getValueFromJsonKRD<double>(firstTicks,"perceive",randomGen,interval[SimPhase::PERCEIVE]*dt)/dt);//reset at t=0 is not regarded as the "first time."
            firstTick[SimPhase::CONTROL]=std::lround(getValueFromJsonKRD(firstTicks,"control",randomGen,0.0)/dt);
            firstTick[SimPhase::BEHAVE]=std::lround(getValueFromJsonKRD(firstTicks,"behave",randomGen,0.0)/dt);
        }else if(firstTickUnit=="tick"){
            firstTick[SimPhase::PERCEIVE]=getValueFromJsonKRD(firstTicks,"perceive",randomGen,interval[SimPhase::PERCEIVE]);//reset at t=0 is not regarded as the "first time."
            firstTick[SimPhase::CONTROL]=getValueFromJsonKRD(firstTicks,"control",randomGen,0);
            firstTick[SimPhase::BEHAVE]=getValueFromJsonKRD(firstTicks,"behave",randomGen,0);
        }else{
            throw std::runtime_error("Only \"tick\" or \"time\" can be designated as the unit of firstTick."+firstTickUnit);
        }
    }else{
        firstTick[SimPhase::PERCEIVE]=interval[SimPhase::PERCEIVE];//reset at t=0 is not regarded as the "first time."
        firstTick[SimPhase::CONTROL]=0;
        firstTick[SimPhase::BEHAVE]=0;
    }
    firstTick[SimPhase::VALIDATE]=0;//meaningless but defined for same interface
}

void Asset::serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full){
    BaseType::serializeInternalState(archive,full);

    ASRC_SERIALIZE_NVP(archive
        ,observables
        ,commands
    )
}

void Asset::addCommunicationBuffer(const std::shared_ptr<CommunicationBuffer>& buffer){
    communicationBuffers[buffer->getName()]=buffer;
}
void Asset::setDependency(){
    //Inform children parent's sequential process order.
    //Call manager->addDependency(predecessor,successor) to add dependency.
}
void Asset::perceive(bool inReset){
}
void Asset::control(){
}
void Asset::behave(){
}
void Asset::kill(){
}
std::shared_ptr<EntityAccessor> Asset::getAccessorImpl(){
    return std::make_shared<AssetAccessor>(getShared<Asset>(this->shared_from_this()));
}

nl::json AssetAccessor::dummy=nl::json();
AssetAccessor::AssetAccessor(const std::shared_ptr<Asset>& a)
:EntityAccessor(a)
,asset(a)
,observables(a ? a->observables : dummy)
{
}
bool AssetAccessor::isAlive() const{
    return !asset.expired() && asset.lock()->isAlive();
}
std::string AssetAccessor::getTeam() const{
    return asset.lock()->getTeam();
}
std::string AssetAccessor::getGroup() const{
    return asset.lock()->getGroup();
}
std::string AssetAccessor::getName() const{
    return asset.lock()->getName();
}

void exportAsset(py::module &m, const std::shared_ptr<asrc::core::FactoryHelper>& factoryHelper)
{
    using namespace pybind11::literals;
    expose_entity_subclass<Asset>(m,"Asset")
    DEF_FUNC(Asset,isAlive)
    DEF_FUNC(Asset,getTeam)
    DEF_FUNC(Asset,getGroup)
    DEF_FUNC(Asset,getName)
    DEF_FUNC(Asset,setDependency)
    DEF_FUNC(Asset,perceive)
    DEF_FUNC(Asset,control)
    DEF_FUNC(Asset,behave)
    DEF_FUNC(Asset,kill)
    DEF_READWRITE(Asset,communicationBuffers)
    DEF_READWRITE(Asset,observables)
    DEF_READWRITE(Asset,commands)
    ;

    expose_common_class<AssetAccessor>(m,"AssetAccessor")
    .def(py_init<const std::shared_ptr<Asset>&>())
    DEF_FUNC(AssetAccessor,isAlive)
    DEF_FUNC(AssetAccessor,getTeam)
    DEF_FUNC(AssetAccessor,getGroup)
    DEF_FUNC(AssetAccessor,getName)
    .def_property_readonly("observables",[](const AssetAccessor& v){return v.observables;})
    ;
}

ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
