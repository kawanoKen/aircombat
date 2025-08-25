// Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#include "Ruler.h"
#include "Utility.h"
#include "SimulationManager.h"
#include "Asset.h"
#include "Agent.h"

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)
using namespace util;

const std::string Ruler::baseName="Ruler"; // baseNameを上書き

void Ruler::initialize(){
    BaseType::initialize();
    maxTime=getValueFromJsonKRD(modelConfig,"maxTime",randomGen,1200);
    winner="";
    dones["__all__"]=false;
    endReason=EndReason::NOTYET;
    if(instanceConfig.contains("crs")){
        localCRS=createOrGetEntity<CoordinateReferenceSystem>(instanceConfig.at("crs"));
    }else if(modelConfig.contains("crs")){
        localCRS=createOrGetEntity<CoordinateReferenceSystem>(modelConfig.at("crs"));
    }else{
        localCRS=manager->getRootCRS();
    }
    observables={
        {"maxTime",maxTime},
        {"endReason",endReason},
        {"localCRS",localCRS}
    };
}
void Ruler::serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full){
    BaseType::serializeInternalState(archive,full);

    if(full){
        ASRC_SERIALIZE_NVP(archive
            ,localCRS
            ,maxTime
            ,teams
        )
    }

    ASRC_SERIALIZE_NVP(archive
        ,observables
        ,dones
        ,score
        ,stepScore
        ,winner
    )

    if(asrc::core::util::isOutputArchive(archive)){
        call_operator(archive,cereal::make_nvp("endReason",getEndReason()));
    }else{
        std::string loadedEndReason;
        call_operator(archive,cereal::make_nvp("endReason",loadedEndReason));
        setEndReason(loadedEndReason);
    }
}

void Ruler::onEpisodeBegin(){
    teams.clear();
    score.clear();
    stepScore.clear();
    nl::json j_teams;
    try{
        j_teams=modelConfig.at("teams");
    }catch(...){
        j_teams="All";
    }
    if(j_teams.is_null()){
        j_teams="All";
    }
    if(j_teams.is_string()){
        std::string team=j_teams;
        if(team=="All"){
            for(auto &&t:manager->getTeams()){
                teams.push_back(t);
                score[t]=0.0;
                stepScore[t]=0.0;
            }
        }else{
            teams.push_back(team);
            score[team]=0.0;
            stepScore[team]=0.0;
        }
    }else if(j_teams.is_array()){
        teams=j_teams.get<std::vector<std::string>>();
        for(auto &&t:teams){
            score[t]=0.0;
            stepScore[t]=0.0;
        }
    }else{
        std::cout<<"modelConfig['teams']="<<j_teams<<std::endl;
        throw std::runtime_error("Invalid designation of 'teams' for Ruler.");
    }
    dones.clear();
    for(auto &&e:manager->getAgents()){
        auto a=getShared(e);
        dones[a->getName()]=!a->isAlive();
    }
    dones["__all__"]=false;
    endReason=EndReason::NOTYET;
    observables["endReason"]=endReason;
}
void Ruler::onStepBegin(){
    for(auto &&t:teams){
        stepScore[t]=0;
    }
}
void Ruler::onInnerStepBegin(){
}
void Ruler::onInnerStepEnd(){
}
void Ruler::onStepEnd(){
    for(auto &&t:teams){
        score[t]+=stepScore[t];
    }
    checkDone();
}
void Ruler::onEpisodeEnd(){
}
void Ruler::checkDone(){
    dones.clear();
    for(auto &&e:manager->getAgents()){
        auto a=getShared(e);
        dones[a->getName()]=!a->isAlive();
    }
    if(manager->getElapsedTime()>=maxTime){
        int index=0;
        double maxScore=0;
        for(auto &&t:teams){
            if(index==0 || maxScore<score[t]){
                maxScore=score[t];
                winner=t;
            }
            index++;
        }
        for(auto &e:dones){
            e.second=true;
        }
        dones["__all__"]=true;
        endReason=EndReason::TIMEUP;
        return;
    }
    if(dones.size()==0){
        dones["__all__"]=false;
        return;
    }
    dones["__all__"]=true;
    for(auto &&e:dones){
        if(!e.second){
            dones["__all__"]=false;
        }
    }
    if(dones["__all__"]){
        endReason=EndReason::NO_ONE_ALIVE;
    }
    observables["endReason"]=endReason;
}
std::string Ruler::getEndReason() const{
    return util::enumToStr<EndReason>(endReason);
}
void Ruler::setEndReason(const std::string& newReason){
    endReason=util::strToEnum<EndReason>(newReason);
}
double Ruler::getStepScore(const std::string &key){
    if(stepScore.count(key)>0){
        return stepScore[key];
    }else{
        return 0.0;
    }
}
double Ruler::getScore(const std::string &key){
    if(score.count(key)>0){
        return score[key];
    }else{
        return 0.0;
    }
}
double Ruler::getStepScore(const std::shared_ptr<Agent>& key){
    return getStepScore(key->getTeam());
}
double Ruler::getScore(const std::shared_ptr<Agent>& key){
    return getScore(key->getTeam());
}
std::shared_ptr<CoordinateReferenceSystem> Ruler::getLocalCRS() const{
    return localCRS;
}
std::shared_ptr<EntityAccessor> Ruler::getAccessorImpl(){
    return std::make_shared<RulerAccessor>(getShared<Ruler>(this->shared_from_this()));
}

nl::json RulerAccessor::dummy=nl::json();
RulerAccessor::RulerAccessor(const std::shared_ptr<Ruler>& r)
:EntityAccessor(r)
,ruler(r)
,observables(r ? r->observables : dummy)
{
}
double RulerAccessor::getStepScore(const std::string &key){
    return ruler.lock()->getStepScore(key);
}
double RulerAccessor::getScore(const std::string &key){
    return ruler.lock()->getScore(key);
}
double RulerAccessor::getStepScore(const std::shared_ptr<Agent>& key){
    return ruler.lock()->getStepScore(key);
}
double RulerAccessor::getScore(const std::shared_ptr<Agent>& key){
    return ruler.lock()->getScore(key);
}
std::shared_ptr<CoordinateReferenceSystem> RulerAccessor::getLocalCRS() const{
    return ruler.lock()->getLocalCRS();
}

void exportRuler(py::module &m, const std::shared_ptr<asrc::core::FactoryHelper>& factoryHelper)
{
    using namespace pybind11::literals;

    FACTORY_ADD_BASE(Ruler,{"!Callback","Ruler","!AnyOthers"},{"Callback","!AnyOthers"})

    auto cls=expose_entity_subclass<Ruler>(m,"Ruler");
    cls
    DEF_FUNC(Ruler,checkDone)
    DEF_FUNC(Ruler,getEndReason)
    DEF_FUNC(Ruler,setEndReason)
    .def("getStepScore",py::overload_cast<const std::string&>(&Ruler::getStepScore))
    .def("getStepScore",py::overload_cast<const std::shared_ptr<Agent>&>(&Ruler::getStepScore))
    .def("getScore",py::overload_cast<const std::string&>(&Ruler::getScore))
    .def("getScore",py::overload_cast<const std::shared_ptr<Agent>&>(&Ruler::getScore))
    DEF_FUNC(Ruler,getLocalCRS)
    DEF_READWRITE(Ruler,dones)
    DEF_READWRITE(Ruler,maxTime)
    DEF_READWRITE(Ruler,teams)
    DEF_READWRITE(Ruler,winner)
    DEF_READWRITE(Ruler,score)
    DEF_READWRITE(Ruler,stepScore)
    DEF_READWRITE(Ruler,observables)
    DEF_READWRITE(Ruler,endReason)
    ;
    FACTORY_ADD_CLASS(Ruler,Ruler)

    expose_enum_value_helper(
        expose_enum_class<Ruler::EndReason>(cls,"EndReason")
        ,"NOTYET"
        ,"TIMEUP"
        ,"NO_ONE_ALIVE"
    );

    expose_common_class<RulerAccessor>(m,"RulerAccessor")
    .def(py_init<const std::shared_ptr<Ruler>&>())
    .def("getStepScore",py::overload_cast<const std::string&>(&RulerAccessor::getStepScore))
    .def("getStepScore",py::overload_cast<const std::shared_ptr<Agent>&>(&RulerAccessor::getStepScore))
    .def("getScore",py::overload_cast<const std::string&>(&RulerAccessor::getScore))
    .def("getScore",py::overload_cast<const std::shared_ptr<Agent>&>(&RulerAccessor::getScore))
    DEF_FUNC(RulerAccessor,getLocalCRS)
    .def_property_readonly("observables",[](const RulerAccessor& v){return v.observables;})
    ;
}

ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
