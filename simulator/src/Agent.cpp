// Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#include "Agent.h"
#include <algorithm>
#include <pybind11/stl.h>
#include "Utility.h"
#include "Factory.h"
#include "SimulationManager.h"
#include "PhysicalAsset.h"
#include "Ruler.h"

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)
using namespace util;

const std::string Agent::baseName="Agent"; // baseNameを上書き

void Agent::initialize(){
    BaseType::initialize();
    name=instanceConfig.at("name");
    type=instanceConfig.at("type");
    model=instanceConfig.at("model");
    policy=instanceConfig.at("policy");
    instanceConfig.at("parents").get_to(parents);
    commands=nl::json::object();
    observables=nl::json::object();
    for(auto &&e:parents){
        commands[e.second->getFullName()]=nl::json::object();
        observables[e.second->getFullName()]={{"decision",nl::json::object()}};
    }
    if(modelConfig.contains("interval")){
        nl::json intervals=modelConfig.at("interval");
        std::string intervalUnit="tick";
        if(intervals.contains("unit")){
            intervalUnit=intervals.at("unit");
        }
        if(intervalUnit=="time"){
            double dt=manager->getBaseTimeStep();
            interval[SimPhase::AGENT_STEP]=std::max<std::uint64_t>(1,std::lround(getValueFromJsonKRD(intervals,"step",randomGen,manager->getDefaultAgentStepInterval()*dt)/dt));
        }else if(intervalUnit=="tick"){
            interval[SimPhase::AGENT_STEP]=getValueFromJsonKRD(intervals,"step",randomGen,manager->getDefaultAgentStepInterval());
        }else{
            throw std::runtime_error("Only \"tick\" or \"time\" can be designated as the unit of interval."+intervalUnit);
        }
    }else{
        interval[SimPhase::AGENT_STEP]=manager->getDefaultAgentStepInterval();
    }
    firstTick[SimPhase::AGENT_STEP]=0;
    dontExpose=getValueFromJsonKD(modelConfig,"dontExpose",false);
    if(dontExpose){
        firstTick[SimPhase::AGENT_STEP]=std::numeric_limits<std::uint64_t>::max();
        interval[SimPhase::AGENT_STEP]=std::numeric_limits<std::uint64_t>::max();
    }
}

void Agent::serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full){
    BaseType::serializeInternalState(archive,full);

    if(full){
        if(asrc::core::util::isInputArchive(archive)){
            name=instanceConfig.at("name");
            type=instanceConfig.at("type");
            model=instanceConfig.at("model");
            policy=instanceConfig.at("policy");
            instanceConfig.at("parents").get_to(parents);
        }
    }
    ASRC_SERIALIZE_NVP(archive
        ,localCRS
    )

}

bool Agent::isAlive() const{
    bool ret=false;
    for(auto&& e:parents){
        ret=ret || e.second->isAlive();
    }
    return ret;
}
std::string Agent::getTeam() const{
    return (*parents.begin()).second->getTeam();
}
std::string Agent::getGroup() const{
    return (*parents.begin()).second->getGroup();
}
std::string Agent::getName() const{
    return name;
}
std::uint64_t Agent::getStepCount() const{
    return round(manager->getTickCount()/getStepInterval());
}
std::uint64_t Agent::getStepInterval() const{
    return getInterval(SimPhase::AGENT_STEP);
}
std::string Agent::repr() const{
    if(type=="Learning"){
        return name+":Learning("+model+")";
    }else if(type=="Clone"){
        return name+":Clone("+model+")";
    }else if(type=="External"){
        return name+":"+model+"["+policy+"]";
    }else{
        return name+":"+model;
    }
}
void Agent::setDependency(){
    //In the parent PhysicalAsset's setDependency, agent's dependency can be set as follows:
    //  manager->addDependency(SimPhase::PERCEIVE,shared_from_this(),agent.lock()); // "this" precedes agent at perceive
    //  manager->addDependency(SimPhase::CONTROL,agent.lock(),shared_from_this()); // agent precedes "this" at control
}
py::object Agent::observation_space(){
    py::module_ spaces=py::module_::import("gymnasium.spaces");
    return spaces.attr("Discrete")(1);
}
py::object Agent::makeObs(){
    return py::cast((int)0);
}
py::object Agent::action_space(){
    py::module_ spaces=py::module_::import("gymnasium.spaces");
    return spaces.attr("Discrete")(1);
}
void Agent::deploy(py::object action){
    for(auto &&e:parents){
        observables[e.second->getFullName()]={{"decision",nl::json::object()}};
    }
}
void Agent::perceive(bool inReset){
}
void Agent::control(){
    for(auto &&e:parents){
        commands[e.second->getFullName()]=nl::json::object();
    }
}
void Agent::behave(){
}
py::object Agent::convertActionFromAnother(const nl::json& decision,const nl::json& command){
    return py::cast((int)0);
}
void Agent::controlWithAnotherAgent(const nl::json& decision,const nl::json& command){
    //ExpertWrapperでimitatorとして使用する際、controlの内容をexpertの出力を用いて上書きしたい場合に用いる
    control();
}
std::shared_ptr<CoordinateReferenceSystem> Agent::getLocalCRS(){
    if(!localCRS){
        if(instanceConfig.contains("crs")){
            localCRS=createOrGetEntity<CoordinateReferenceSystem>(instanceConfig.at("crs"));
        }else if(modelConfig.contains("crs")){
            localCRS=createOrGetEntity<CoordinateReferenceSystem>(modelConfig.at("crs"));
        }else{
            localCRS=manager->getRuler().lock()->getLocalCRS();
        }
    }
    return localCRS;
}
void Agent::setManagerAccessor(const std::shared_ptr<EntityManagerAccessor>& ema){
    BaseType::setManagerAccessor(ema);
    manager=std::dynamic_pointer_cast<SimulationManagerAccessorForAgent>(ema);
    assert(manager);
}

void ExpertWrapper::initialize(){
    BaseType::initialize();
    whichOutput=modelConfig.at("whichOutput").get<std::string>();
    imitatorModelName=instanceConfig.at("imitatorModelName").get<std::string>();
    expertModelName=instanceConfig.at("expertModelName").get<std::string>();
    expertPolicyName=instanceConfig.at("expertPolicyName").get<std::string>();
    trajectoryIdentifier=instanceConfig.at("identifier").get<std::string>();
    whichExpose=modelConfig.at("whichExpose").get<std::string>();
}
void ExpertWrapper::serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full){
    BaseType::serializeInternalState(archive,full);

    if(full){
        if(asrc::core::util::isInputArchive(archive)){
            whichOutput=modelConfig.at("whichOutput").get<std::string>();
            imitatorModelName=instanceConfig.at("imitatorModelName").get<std::string>();
            expertModelName=instanceConfig.at("expertModelName").get<std::string>();
            expertPolicyName=instanceConfig.at("expertPolicyName").get<std::string>();
            trajectoryIdentifier=instanceConfig.at("identifier").get<std::string>();
            whichExpose=modelConfig.at("whichExpose").get<std::string>();
        }

        ASRC_SERIALIZE_NVP(archive
            ,imitator
            ,expert
            ,isInternal
        )
    }
    ASRC_SERIALIZE_NVP(archive
        ,hasImitatorDeployed
        ,expertObs
        ,imitatorObs
        ,imitatorAction
    )
}

void ExpertWrapper::makeChildren(){
    BaseType::makeChildren();
    nl::json sub={
        {"name",name+"/Expert"},
        {"seed",randomGen()},
        {"parents",parents},
        {"type","Child"},
        {"model",expertModelName},
        {"policy",expertPolicyName},
        {"entityFullName",name+"/Expert:"+model+":"+policy}
    };
    expert=createUnmanagedEntity<Agent>(
        isEpisodic(),
        "Agent",
        expertModelName,
        sub
    );
    sub["name"]=name+"/Imitator";
    sub["seed"]=randomGen();
    sub["model"]=imitatorModelName;
    sub["policy"]="Expert";
    sub["entityFullName"]=name+"/Imitator"+model+":"+policy;
    imitator=createUnmanagedEntity<Agent>(
        isEpisodic(),
        "Agent",
        imitatorModelName,
        sub
    );
    expertObs=py::none();
    imitatorObs=py::none();
    imitatorAction=py::none();
    isInternal=false;
    hasImitatorDeployed=false;
    observables=expert->observables;
    assert(expert->getStepInterval()==imitator->getStepInterval());
}
void ExpertWrapper::validate(){
    BaseType::validate();
    if(whichOutput=="Expert"){
        imitator->validate();
        expert->validate();
    }else{
        expert->validate();
        imitator->validate();
    }
    observables=expert->observables;
}

std::uint64_t ExpertWrapper::getFirstTick(const SimPhase& phase) const{
    return std::min(expert->getFirstTick(phase),imitator->getFirstTick(phase));
}
std::uint64_t ExpertWrapper::getInterval(const SimPhase& phase) const{
    return std::gcd(expert->getInterval(phase),imitator->getInterval(phase));
}
std::uint64_t ExpertWrapper::getNextTick(const SimPhase& phase,const std::uint64_t now){
    return std::min(expert->getNextTick(phase,now),imitator->getNextTick(phase,now));
}
std::uint64_t ExpertWrapper::getStepCount() const{
    return expert->getStepCount();
}
std::uint64_t ExpertWrapper::getStepInterval() const{
    return expert->getStepInterval();
}
std::string ExpertWrapper::repr() const{
    std::string ret=name+":"+type+"("+imitatorModelName+"<-"+expertModelName;
    if(expertPolicyName!="" && expertPolicyName!="Internal"){
        ret+="["+expertPolicyName+"]";
    }
    return ret;
}
py::object ExpertWrapper::observation_space(){
    py::module_ spaces=py::module_::import("gymnasium.spaces");
    return spaces.attr("Tuple")(py::make_tuple(imitator->observation_space(), expert->observation_space()));
}
py::object ExpertWrapper::makeObs(){
    expertObs=expert->makeObs();
    imitatorObs=imitator->makeObs();
    return py::make_tuple(imitatorObs,expertObs);
}
py::object ExpertWrapper::action_space(){
    py::module_ spaces=py::module_::import("gymnasium.spaces");
    return spaces.attr("Tuple")(py::make_tuple(imitator->action_space(), expert->action_space()));
}
py::object ExpertWrapper::expert_observation_space(){
    return expert->observation_space();
}
py::object ExpertWrapper::expert_action_space(){
    return expert->action_space();
}
py::object ExpertWrapper::imitator_observation_space(){
    return imitator->observation_space();
}
py::object ExpertWrapper::imitator_action_space(){
    return imitator->action_space();
}
void ExpertWrapper::deploy(py::object action){
    if(expertPolicyName=="Internal"){
        expert->deploy(py::none());
    }else{
        expert->deploy(action);
    }
    observables=expert->observables;
    hasImitatorDeployed=false;
}
void ExpertWrapper::perceive(bool inReset){
    if(inReset){
        imitator->perceive(inReset);
        expert->perceive(inReset);
    }else{
        std::uint64_t now=manager->getTickCount();
        if(imitator->isTickToRunPhaseFunc(SimPhase::PERCEIVE,now)){
            imitator->perceive(inReset);
        }
        if(expert->isTickToRunPhaseFunc(SimPhase::PERCEIVE,now)){
            expert->perceive(inReset);
        }
    }
    observables=expert->observables;
    if(whichOutput=="Expert"){
        commands=expert->commands;
    }else{
        commands=imitator->commands;
    }
}
void ExpertWrapper::control(){
    std::uint64_t now=manager->getTickCount();
    if(expert->isTickToRunPhaseFunc(SimPhase::CONTROL,now)){
        expert->control();
    }
    observables=expert->observables;
    nl::json decisions=nl::json::object();
    for(auto&&e:expert->observables.items()){
        decisions[e.key()]=e.value().at("decision");
    }
    if(imitator->isTickToRunPhaseFunc(SimPhase::CONTROL,now)){
        if(!hasImitatorDeployed){
            py::gil_scoped_acquire acquire;
            imitatorAction=imitator->convertActionFromAnother(decisions,expert->commands);
            imitator->deploy(imitatorAction);
            hasImitatorDeployed=true;
        }
        imitator->controlWithAnotherAgent(decisions,expert->commands);
    }
    if(whichOutput=="Expert"){
        commands=expert->commands;
    }else{
        commands=imitator->commands;
    }
}
void ExpertWrapper::behave(){
    std::uint64_t now=manager->getTickCount();
    if(imitator->isTickToRunPhaseFunc(SimPhase::BEHAVE,now)){
        imitator->behave();
    }
    if(expert->isTickToRunPhaseFunc(SimPhase::BEHAVE,now)){
        expert->behave();
    }
    observables=expert->observables;
    if(whichOutput=="Expert"){
        commands=expert->commands;
    }else{
        commands=imitator->commands;
    }
}

void MultiPortCombiner::serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full){
    BaseType::serializeInternalState(archive,full);

    if(full){
        ASRC_SERIALIZE_NVP(archive
            ,ports
            ,children
        )
    }
}

void MultiPortCombiner::makeChildren(){
    BaseType::makeChildren();
    std::map<std::string,nl::json> childrenConfigs=modelConfig.at("children");
    std::map<std::string,nl::json> buffer;
    std::string childName,childModel,childPort;
    for(auto &&e:childrenConfigs){
        std::string key=e.first;
        nl::json value=e.second;
        try{
            childName=value.at("name").get<std::string>();
        }catch(...){
            childName="child_"+key;
        }
        childModel=value.at("model").get<std::string>();
        try{
            childPort=value.at("port").get<std::string>();
        }catch(...){
            childPort="0";
        }
        if(parents.count(key)>0){
            //instanciate only when parents for the child actually exist.
            if(buffer.count(childName)>0){
                buffer[childName]["ports"][childPort]=key;
                buffer[childName]["parents"][childPort]=parents[key];
            }else{
                buffer[childName]={
                    {"ports",{{childPort,key}}},
                    {"model",childModel},
                    {"parents",{{childPort,parents[key]}}}
                };
            }
            ports[key]={
                {"childName",childName},
                {"childPort",childPort}
            };
        }
    }
    nl::json sub;
    for(auto &&e:buffer){
        std::string key=e.first;
        nl::json value=e.second;
        sub={
            {"name",name+"/"+key},
            {"seed",randomGen()},
            {"parents",buffer[key]["parents"]},
            {"type","Child"},
            {"model",buffer[key]["model"]},
            {"policy",policy},
            {"entityFullName",name+"/"+key+":"+model+":"+policy}
        };
        children[key]=createUnmanagedEntity<Agent>(
            isEpisodic(),
            "Agent",
            buffer[key]["model"],
            sub
        );
    }
    std::uint64_t stepInterval=children.begin()->second->getStepInterval();
    for(auto &&e:children){
        assert(e.second->getStepInterval()==stepInterval);
    }
}
void MultiPortCombiner::validate(){
    BaseType::validate();
    for(auto &&e:children){
        e.second->validate();
    }
}

std::uint64_t MultiPortCombiner::getFirstTick(const SimPhase& phase) const{
    std::uint64_t ret=children.begin()->second->getFirstTick(phase);
    for(auto &&e:children){
        ret=std::min(ret,e.second->getFirstTick(phase));
    }
    return ret;
}
std::uint64_t MultiPortCombiner::getInterval(const SimPhase& phase) const{
    std::uint64_t ret=children.begin()->second->getInterval(phase);
    for(auto &&e:children){
        ret=std::gcd(ret,e.second->getInterval(phase));
    }
    return ret;
}
std::uint64_t MultiPortCombiner::getNextTick(const SimPhase& phase,const std::uint64_t now){
    std::uint64_t ret=children.begin()->second->getNextTick(phase,now);
    for(auto &&e:children){
        ret=std::min(ret,e.second->getNextTick(phase,now));
    }
    return ret;
}
std::uint64_t MultiPortCombiner::getStepCount() const{
    return children.begin()->second->getStepCount();
}
std::uint64_t MultiPortCombiner::getStepInterval() const{
    return children.begin()->second->getStepInterval();
}
py::object MultiPortCombiner::observation_space(){
    py::module_ spaces=py::module_::import("gymnasium.spaces");
    return spaces.attr("Discrete")(1);
}
py::object MultiPortCombiner::makeObs(){
    return py::cast((int)0);
}
py::object MultiPortCombiner::action_space(){
    py::module_ spaces=py::module_::import("gymnasium.spaces");
    return spaces.attr("Discrete")(1);
}
std::map<std::string,py::object> MultiPortCombiner::actionSplitter(py::object action){
    std::map<std::string,py::object> tmp;
    for(auto &&e:children){
        tmp[e.first]=py::cast((int)0);
    }
    return tmp;
}
void MultiPortCombiner::deploy(py::object action){
    std::map<std::string,py::object> splitted=actionSplitter(action);
    std::shared_ptr<Agent> child;
    for(auto &&e:children){
        if(e.second->isAlive()){
            e.second->deploy(splitted[e.first]);
        }
    }
    for(auto &&e:ports){
        std::string key=e.first;
        child=children[e.second["childName"]];
        std::string parentName=parents[key]->getFullName();
        commands[parentName]=child->commands[parentName];
        observables[parentName]=child->observables[parentName];
    }
}
void MultiPortCombiner::perceive(bool inReset){
    std::uint64_t now=manager->getTickCount();
    std::shared_ptr<Agent> child;
    for(auto &&e:children){
        if(e.second->isAlive()){
            if(inReset || e.second->isTickToRunPhaseFunc(SimPhase::PERCEIVE,now)){
                e.second->perceive(inReset);
            }
        }
    }
    for(auto &&e:ports){
        std::string key=e.first;
        child=children[e.second["childName"]];
        std::string parentName=parents[key]->getFullName();
        commands[parentName]=child->commands[parentName];
        observables[parentName]=child->observables[parentName];
    }
}
void MultiPortCombiner::control(){
    std::uint64_t now=manager->getTickCount();
    std::shared_ptr<Agent> child;
    for(auto &&e:children){
        if(e.second->isAlive()){
            if(e.second->isTickToRunPhaseFunc(SimPhase::CONTROL,now)){
                e.second->control();
            }
        }
    }
    for(auto &&e:ports){
        std::string key=e.first;
        child=children[e.second["childName"]];
        std::string parentName=parents[key]->getFullName();
        commands[parentName]=child->commands[parentName];
        observables[parentName]=child->observables[parentName];
    }
}
void MultiPortCombiner::behave(){
    std::uint64_t now=manager->getTickCount();
    std::shared_ptr<Agent> child;
    for(auto &&e:children){
        if(e.second->isAlive()){
            if(e.second->isTickToRunPhaseFunc(SimPhase::BEHAVE,now)){
                e.second->behave();
            }
        }
    }
    for(auto &&e:ports){
        std::string key=e.first;
        child=children[e.second["childName"]];
        std::string parentName=parents[key]->getFullName();
        commands[parentName]=child->commands[parentName];
        observables[parentName]=child->observables[parentName];
    }
}

void SimpleMultiPortCombiner::serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full){
    BaseType::serializeInternalState(archive,full);

    ASRC_SERIALIZE_NVP(archive
        ,lastChildrenObservations
    )
}

py::object SimpleMultiPortCombiner::observation_space(){
    py::module_ spaces=py::module_::import("gymnasium.spaces");
    py::dict space;
    for(auto &&e:children){
        space[e.first.c_str()]=e.second->observation_space();
    }
    return spaces.attr("Dict")(space);
}
py::object SimpleMultiPortCombiner::makeObs(){
    py::dict observation;
    py::dict children_is_alive;
    for(auto &&e:children){
        if(e.second->isAlive()){
            auto obs=e.second->makeObs();
            lastChildrenObservations[e.first]=obs;
            observation[e.first.c_str()]=obs;
        }else{
            observation[e.first.c_str()]=lastChildrenObservations[e.first];
        }
        children_is_alive[e.first.c_str()]=e.second->isAlive();
    }
    observation["isAlive"]=children_is_alive;
    return observation;
}
py::object SimpleMultiPortCombiner::action_space(){
    py::module_ spaces=py::module_::import("gymnasium.spaces");
    py::dict space;
    for(auto &&e:children){
        space[e.first.c_str()]=e.second->action_space();
    }
    return spaces.attr("Dict")(space);
}
std::map<std::string,py::object> SimpleMultiPortCombiner::actionSplitter(py::object action){
    std::map<std::string,py::object> tmp;
    py::dict asdict=py::cast<py::dict>(action);
    for(auto &&e:children){
        tmp[e.first]=asdict[e.first.c_str()];
    }
    return tmp;
}
py::object SimpleMultiPortCombiner::convertActionFromAnother(const nl::json& decision,const nl::json& command){
    py::dict converted;
    for(auto &&e:children){
        if(e.second->isAlive()){
            nl::json subD=nl::json::object();
            nl::json subC=nl::json::object();
            for(auto &&p:e.second->parents){
                std::string pName=p.second->getFullName();
                subD[pName]=decision.at(pName);
                subC[pName]=command.at(pName);
            }
            converted[e.first.c_str()]=e.second->convertActionFromAnother(subD,subC);
        }else{
            converted[e.first.c_str()]=py::none();
        }
    }
    return converted;
}
void SimpleMultiPortCombiner::controlWithAnotherAgent(const nl::json& decision,const nl::json& command){
    for(auto &&e:children){
        if(e.second->isAlive()){
            nl::json subD=nl::json::object();
            nl::json subC=nl::json::object();
            for(auto &&p:e.second->parents){
                std::string pName=p.second->getFullName();
                subD[pName]=decision.at(pName);
                subC[pName]=command.at(pName);
            }
            e.second->controlWithAnotherAgent(subD,subC);
        }
    }
    std::shared_ptr<Agent> child;
    for(auto &&e:ports){
        std::string key=e.first;
        child=children[e.second["childName"]];
        std::string parentName=parents[key]->getFullName();
        commands[parentName]=child->commands[parentName];
        observables[parentName]=child->observables[parentName];
    }
}

void SingleAssetAgent::initialize(){
    BaseType::initialize();
    port=(*parents.begin()).first;
    parent=(*parents.begin()).second;
    parents.clear();
    parents[port]=parent;
}

void SingleAssetAgent::serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full){
    BaseType::serializeInternalState(archive,full);

    if(full){
        if(asrc::core::util::isInputArchive(archive)){
            port=(*parents.begin()).first;
            parent=(*parents.begin()).second;
            parents.clear();
            parents[port]=parent;
        }
    }
}

void exportAgent(py::module &m, const std::shared_ptr<asrc::core::FactoryHelper>& factoryHelper)
{
    using namespace pybind11::literals;

    FACTORY_ADD_BASE(Agent,{"Agent","!AnyOther"},{"!Agent","AnyOthers"})

    expose_entity_subclass<Agent>(m,"Agent")
    DEF_FUNC(Agent,getStepCount)
    DEF_FUNC(Agent,getStepInterval)
    .def("__repr__",&Agent::repr)
    DEF_FUNC(Agent,observation_space)
    DEF_FUNC(Agent,makeObs)
    DEF_FUNC(Agent,action_space)
    DEF_FUNC(Agent,deploy)
    DEF_FUNC(Agent,convertActionFromAnother)
    DEF_FUNC(Agent,controlWithAnotherAgent)
    DEF_FUNC(Agent,getLocalCRS)
    DEF_READWRITE(Agent,name)
    DEF_READWRITE(Agent,parents)
    DEF_READWRITE(Agent,dontExpose)
    ;
    //FACTORY_ADD_CLASS(Agent,Agent) //Do not register to Factory because Agent is abstract class.

    expose_entity_subclass<ExpertWrapper>(m,"ExpertWrapper")
    DEF_FUNC(ExpertWrapper,expert_observation_space)
    DEF_FUNC(ExpertWrapper,expert_action_space)
    DEF_FUNC(ExpertWrapper,imitator_observation_space)
    DEF_FUNC(ExpertWrapper,imitator_action_space)
    DEF_READWRITE(ExpertWrapper,whichOutput)
    DEF_READWRITE(ExpertWrapper,expert)
    DEF_READWRITE(ExpertWrapper,imitator)
    DEF_READWRITE(ExpertWrapper,imitatorModelName)
    DEF_READWRITE(ExpertWrapper,expertModelName)
    DEF_READWRITE(ExpertWrapper,expertPolicyName)
    DEF_READWRITE(ExpertWrapper,trajectoryIdentifier)
    DEF_READWRITE(ExpertWrapper,expertObs)
    DEF_READWRITE(ExpertWrapper,imitatorObs)
    DEF_READWRITE(ExpertWrapper,imitatorAction)
    DEF_READWRITE(ExpertWrapper,isInternal)
    DEF_READWRITE(ExpertWrapper,hasImitatorDeployed)
    ;
    FACTORY_ADD_CLASS(Agent,ExpertWrapper)

    expose_entity_subclass<MultiPortCombiner>(m,"MultiPortCombiner")
    DEF_FUNC(MultiPortCombiner,actionSplitter)
    DEF_READWRITE(MultiPortCombiner,ports)
    DEF_READWRITE(MultiPortCombiner,children)
    ;
    FACTORY_ADD_CLASS(Agent,MultiPortCombiner)

    expose_entity_subclass<SimpleMultiPortCombiner>(m,"SimpleMultiPortCombiner")
    ;
    FACTORY_ADD_CLASS(Agent,SimpleMultiPortCombiner)

    expose_entity_subclass<SingleAssetAgent>(m,"SingleAssetAgent")
    DEF_READWRITE(SingleAssetAgent,parent)
    DEF_READWRITE(SingleAssetAgent,port)
    ;
    FACTORY_ADD_CLASS(Agent,SingleAssetAgent)
}

ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
