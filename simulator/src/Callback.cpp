// Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#include "Callback.h"
#include "Utility.h"
#include "SimulationManager.h"

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)
using namespace util;

const std::string Callback::baseName="Callback"; // baseNameを上書き

void Callback::initialize(){
    BaseType::initialize();
    auto fullName=getFullName();
    if(fullName.find("/")>=0){
        group=fullName.substr(0,fullName.find("/"));
        name=fullName.substr(fullName.find("/")+1);
    }else{
        group="";
        name=fullName;
    }
    try{
        acceptReconfigure=modelConfig.at("acceptReconfigure");
    }catch(...){
        acceptReconfigure=false;
    }
    firstTick[SimPhase::ON_EPISODE_BEGIN]=0;//meaningless but defined for same interface
    firstTick[SimPhase::ON_VALIDATION_END]=0;//meaningless but defined for same interface
    firstTick[SimPhase::ON_STEP_BEGIN]=0;//meaningless but defined for same interface
    nl::json firstTicks=nl::json::object();
    if(instanceConfig.contains("firstTick")){
        firstTicks=instanceConfig.at("firstTick");
    }else if(modelConfig.contains("firstTick")){
        firstTicks=modelConfig.at("firstTick");
    }
    std::string firstTickUnit="tick";
    if(firstTicks.contains("unit")){
        firstTickUnit=firstTicks.at("unit");
    }
    if(firstTickUnit=="time"){
        double dt=manager->getBaseTimeStep();
        firstTick[SimPhase::ON_INNERSTEP_BEGIN]=std::lround(getValueFromJsonKRD(firstTicks,"value",randomGen,0.0)/dt);
    }else if(firstTickUnit=="tick"){
        firstTick[SimPhase::ON_INNERSTEP_BEGIN]=getValueFromJsonKRD(firstTicks,"value",randomGen,0);
    }else{
        throw std::runtime_error("Only \"tick\" or \"time\" can be designated as the unit of firstTick."+firstTickUnit);
    }
    firstTick[SimPhase::ON_STEP_END]=0;//meaningless but defined for same interface
    firstTick[SimPhase::ON_EPISODE_END]=0;//meaningless but defined for same interface
    firstTick[SimPhase::ON_GET_OBSERVATION_SPACE]=0;//meaningless but defined for same interface
    firstTick[SimPhase::ON_GET_ACTION_SPACE]=0;//meaningless but defined for same interface
    firstTick[SimPhase::ON_MAKE_OBS]=0;//meaningless but defined for same interface
    firstTick[SimPhase::ON_DEPLOY_ACTION]=0;//meaningless but defined for same interface
    interval[SimPhase::ON_EPISODE_BEGIN]=std::numeric_limits<std::uint64_t>::max();//meaningless but defined for same interface
    interval[SimPhase::ON_VALIDATION_END]=std::numeric_limits<std::uint64_t>::max();//meaningless but defined for same interface
    interval[SimPhase::ON_STEP_BEGIN]=0;//meaningless but defined for same interface
    nl::json intervals=nl::json::object();
    if(instanceConfig.contains("interval")){
        intervals=instanceConfig.at("interval");
    }else if(modelConfig.contains("interval")){
        intervals=modelConfig.at("interval");
    }
    std::string intervalUnit="tick";
    if(intervals.contains("unit")){
        intervalUnit=intervals.at("unit");
    }
    if(intervalUnit=="time"){
        double dt=manager->getBaseTimeStep();
        interval[SimPhase::ON_INNERSTEP_BEGIN]=std::max<std::uint64_t>(1,std::lround(getValueFromJsonKRD(intervals,"value",randomGen,dt)/dt));
    }else if(intervalUnit=="tick"){
        interval[SimPhase::ON_INNERSTEP_BEGIN]=getValueFromJsonKRD(intervals,"value",randomGen,1);
    }else{
        throw std::runtime_error("Only \"tick\" or \"time\" can be designated as the unit of interval."+intervalUnit);
    }
    firstTick[SimPhase::ON_INNERSTEP_END]=firstTick[SimPhase::ON_INNERSTEP_BEGIN]+interval[SimPhase::ON_INNERSTEP_BEGIN];
    interval[SimPhase::ON_INNERSTEP_END]=interval[SimPhase::ON_INNERSTEP_BEGIN];
    interval[SimPhase::ON_STEP_END]=0;//meaningless but defined for same interface
    interval[SimPhase::ON_EPISODE_END]=std::numeric_limits<std::uint64_t>::max();//meaningless but defined for same interface
    interval[SimPhase::ON_GET_OBSERVATION_SPACE]=0;//meaningless but defined for same interface
    interval[SimPhase::ON_GET_ACTION_SPACE]=0;//meaningless but defined for same interface
    interval[SimPhase::ON_MAKE_OBS]=0;//meaningless but defined for same interface
    interval[SimPhase::ON_DEPLOY_ACTION]=0;//meaningless but defined for same interface
}

void Callback::serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full){
    BaseType::serializeInternalState(archive,full);

    if(full){
        if(asrc::core::util::isInputArchive(archive)){
            auto fullName=getFullName();
            if(fullName.find("/")>=0){
                group=fullName.substr(0,fullName.find("/"));
                name=fullName.substr(fullName.find("/")+1);
            }else{
                group="";
                name=fullName;
            }
            try{
                acceptReconfigure=modelConfig.at("acceptReconfigure");
            }catch(...){
                acceptReconfigure=false;
            }
        }
    }
}

std::string Callback::getGroup() const{
    return group;
}
std::string Callback::getName() const{
    return name;
}
void Callback::setDependency(){
    //Inform children parent's sequential process order.
    //Call manager->addDependency(predecessor,successor) to add dependency.
}
void Callback::onGetObservationSpace(){
}
void Callback::onGetActionSpace(){
}
void Callback::onMakeObs(){
}
void Callback::onDeployAction(){
}
void Callback::onEpisodeBegin(){
}
void Callback::onValidationEnd(){
}
void Callback::onStepBegin(){
}
void Callback::onInnerStepBegin(){
}
void Callback::onInnerStepEnd(){
}
void Callback::onStepEnd(){
}
void Callback::onEpisodeEnd(){
}
void Callback::setManagerAccessor(const std::shared_ptr<EntityManagerAccessor>& ema){
    BaseType::setManagerAccessor(ema);
    manager=std::dynamic_pointer_cast<SimulationManagerAccessorForCallback>(ema);
    assert(manager);
}

void exportCallback(py::module &m, const std::shared_ptr<asrc::core::FactoryHelper>& factoryHelper)
{
    using namespace pybind11::literals;

    FACTORY_ADD_BASE(Callback,{"Callback","!AnyOthers"},{"Callback","!AnyOthers"})

    expose_entity_subclass<Callback>(m,"Callback")
    DEF_FUNC(Callback,getGroup)
    DEF_FUNC(Callback,getName)
    DEF_FUNC(Callback,setDependency)
    DEF_FUNC(Callback,onGetObservationSpace)
    DEF_FUNC(Callback,onGetActionSpace)
    DEF_FUNC(Callback,onMakeObs)
    DEF_FUNC(Callback,onDeployAction)
    DEF_FUNC(Callback,onEpisodeBegin)
    DEF_FUNC(Callback,onValidationEnd)
    DEF_FUNC(Callback,onStepBegin)
    DEF_FUNC(Callback,onInnerStepBegin)
    DEF_FUNC(Callback,onInnerStepEnd)
    DEF_FUNC(Callback,onStepEnd)
    DEF_FUNC(Callback,onEpisodeEnd)
    DEF_READWRITE(Callback,group)
    DEF_READWRITE(Callback,name)
    DEF_READWRITE(Callback,acceptReconfigure)
    ;
    //FACTORY_ADD_CLASS(Callback,Callback) //Do not register to Factory because Callback is abstract class.
}

ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
