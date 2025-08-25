// Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#include "SupervisedTaskMixer.h"
#include <ASRCAISim1/SimulationManager.h>
#include <ASRCAISim1/Agent.h>
#include <regex>

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)
using namespace util;

void UnsupervisedTaskMixer::initialize(){
    BaseType::initialize();
    taskName=getValueFromJsonKRD<std::string>(modelConfig,"taskName",randomGen,"Unsupervised");
    targetAgentModels=getValueFromJsonKRD<std::map<std::string,std::vector<std::string>>>(
        modelConfig,"targetAgentModels",randomGen,{});//Dict[team,List[agentModelName]]
}
void UnsupervisedTaskMixer::serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full){
    BaseType::serializeInternalState(archive,full);

    if(full){
        ASRC_SERIALIZE_NVP(archive
            ,taskName
            ,targetAgentModels
        )
    }
}
py::object UnsupervisedTaskMixer::overrideObservationSpaceImpl(const std::shared_ptr<Agent>& agent,py::object oldSpace){
    py::module_ spaces=py::module_::import("gymnasium.spaces");
    py::object dictSpace=spaces.attr("Dict");
    py::dict newSpace;
    if(py::isinstance(oldSpace,dictSpace)){
        for(auto&& item:py::cast<py::dict>(oldSpace.attr("spaces"))){
            newSpace[py::cast<std::string>(item.first).c_str()]=item.second;
        }
    }else{
        newSpace["original"]=oldSpace;
    }
    newSpace[(taskName+"_input").c_str()]=calcTaskInputSpace(agent);
    return dictSpace(newSpace);
}
py::object UnsupervisedTaskMixer::overrideObservationImpl(const std::shared_ptr<Agent>& agent,py::object oldObs){
    py::dict newObs;
    if(py::isinstance<py::dict>(oldObs)){
        newObs=oldObs;
    }else{
        newObs["original"]=oldObs;
    }
    newObs[(taskName+"_input").c_str()]=calcTaskInput(agent);
    return newObs;
}
bool UnsupervisedTaskMixer::isTarget(const std::shared_ptr<Agent>& agent){
    std::string team=agent->getTeam();
    auto found=targetAgentModels.find(team);
    if(found==targetAgentModels.end()){
        return false;
    }
    for(auto&& query:found->second){
        std::regex re=std::regex(query);
        if(std::regex_match(agent->getFactoryModelName(),re)){
            return true;
        }
    }
    return false;
}

void SupervisedTaskMixer::initialize(){
    BaseType::initialize();
    taskName=getValueFromJsonKRD<std::string>(modelConfig,"taskName",randomGen,"Supervised");
}
void SupervisedTaskMixer::serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full){
    BaseType::serializeInternalState(archive,full);

    if(full){
        ASRC_SERIALIZE_NVP(archive
            ,truths
            ,truth_spaces
        )
    }
}
void SupervisedTaskMixer::onEpisodeBegin(){
    truths.clear();
    truth_spaces.clear();
    std::map<std::string,py::object> truth;
    py::dict& observation=manager->observation();//makeObsは既に呼ばれた後である
    for(auto&& e:observation){
        std::string key=py::cast<std::string>(e.first);
        auto agent=getShared(manager->getAgent(key));
        if(isTarget(agent)){
            std::pair<bool,py::object> ret=calcTaskTruth(agent);
            if(ret.first){//即時計算出来た場合
                truth[agent->getFullName()]=ret.second;
            }else{//即時計算出来ない場合は一旦Noneを入れておき、派生クラスにおいて適切な場所で計算して格納するものとする
                truth[agent->getFullName()]=py::none();
            }
            truth_spaces[agent->getFullName()]=calcTaskTruthSpace(agent);
        }
    }
    truths.push_back(truth);
}
void SupervisedTaskMixer::onStepEnd(){
    std::map<std::string,py::object> truth;
    py::dict& observation=manager->observation();//makeObsは既に呼ばれた後である
    for(auto&& e:observation){
        std::string key=py::cast<std::string>(e.first);
        auto agent=getShared(manager->getAgent(key));
        if(isTarget(agent)){
            std::pair<bool,py::object> ret=calcTaskTruth(agent);
            if(ret.first){//即時計算出来た場合
                truth[agent->getFullName()]=ret.second;
            }else{//即時計算出来ない場合は一旦Noneを入れておき、派生クラスにおいて適切な場所で計算して格納するものとする
                truth[agent->getFullName()]=py::none();
            }
            if(truth_spaces.find(agent->getFullName())==truth_spaces.end()){
                truth_spaces[agent->getFullName()]=calcTaskTruthSpace(agent);
            }
        }
    }
    truths.push_back(truth);
}

void exportSupervisedTaskMixer(py::module &m, const std::shared_ptr<asrc::core::FactoryHelper>& factoryHelper)
{
    using namespace pybind11::literals;
    expose_entity_subclass<UnsupervisedTaskMixer>(m,"UnsupervisedTaskMixer")
    DEF_FUNC(UnsupervisedTaskMixer,calcTaskInputSpace)
    DEF_FUNC(UnsupervisedTaskMixer,calcTaskInput)
    DEF_FUNC(UnsupervisedTaskMixer,isTarget)
    DEF_READWRITE(UnsupervisedTaskMixer,targetAgentModels)
    ;

    expose_entity_subclass<SupervisedTaskMixer>(m,"SupervisedTaskMixer")
    DEF_FUNC(SupervisedTaskMixer,calcTaskTruthSpace)
    DEF_FUNC(SupervisedTaskMixer,calcTaskTruth)
    DEF_READWRITE(SupervisedTaskMixer,truths)
    DEF_READWRITE(SupervisedTaskMixer,truth_spaces)
    ;
}

ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
