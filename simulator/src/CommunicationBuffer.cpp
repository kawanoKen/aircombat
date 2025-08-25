// Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#include "CommunicationBuffer.h"
#include "Utility.h"
#include "Units.h"
#include "PhysicalAsset.h"
#include "Controller.h"
#include "Agent.h"
#include "SimulationManager.h"

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)
ASRC_INLINE_NAMESPACE_BEGIN(util)

CommunicationBuffer::CommunicationBuffer(const std::shared_ptr<SimulationManager>& manager_,const std::string& name_,const nl::json& participants_,const nl::json& inviteOnRequest_)
:updatedTimes(),manager(manager_),name(name_),buffer(nl::json::object()),
j_participants(participants_),j_inviteOnRequest(inviteOnRequest_){
    if(j_inviteOnRequest.is_string()){
        inviteOnRequest.emplace_back(j_inviteOnRequest);
    }else if(j_inviteOnRequest.is_array()){
        for(auto&& e:j_inviteOnRequest){
            inviteOnRequest.emplace_back(e);
        }
    }
    validated=false;
    hasBeenSerialized=false;
}
CommunicationBuffer::~CommunicationBuffer(){
}

void CommunicationBuffer::serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full){
    ASRC_SERIALIZE_NVP(archive
        ,validated
        ,participants
        ,inviteOnRequest
        ,invitationRequested
        ,updatedTimes
        ,buffer
    )

    if(asrc::core::util::isOutputArchive(archive)){
        hasBeenSerialized=true;
    }else{
        for(auto&& e:participants){
            e.second.lock()->addCommunicationBuffer(this->shared_from_this());
        }
    }
}

std::string CommunicationBuffer::getName() const{
    return name;
}
void CommunicationBuffer::send(const nl::json& data,CommunicationBuffer::UpdateType updateType){
    std::lock_guard<std::shared_mutex> lock(mtx);
    assert(data.is_object());
    if(updateType==UpdateType::REPLACE){
        buffer=data;
        for(auto&& e:data.items()){
            updatedTimes[e.key()]=manager.lock()->getTime();
        }
    }else if(updateType==UpdateType::MERGE){
        buffer.merge_patch(data);
        for(auto&& e:data.items()){
            updatedTimes[e.key()]=manager.lock()->getTime();
        }
    }else{
        throw std::runtime_error("Unknown UpdateType");
    }
}
std::pair<Time,nl::json> CommunicationBuffer::receive(const std::string& key) const{
    std::shared_lock<std::shared_mutex> lock(mtx);
    if(updatedTimes.count(key)>0){
        if(buffer.contains(key)){
            return std::make_pair(updatedTimes.at(key),buffer.at(key));
        }else{
            return std::make_pair(updatedTimes.at(key),nl::json());
        }
    }else{
        return std::make_pair(Time(),nl::json());
    }
}
bool CommunicationBuffer::requestInvitation(const std::shared_ptr<Asset>& asset){
    bool ret=false;
    for(auto&& query:inviteOnRequest){
        std::regex re;
        bool typeMatch;
        if(query.find("PhysicalAsset:")==0){
            re=std::regex(query.substr(14));
            typeMatch=util::isinstance<PhysicalAsset>(asset);
        }else if(query.find("Controller:")==0){
            re=std::regex(query.substr(11));
            typeMatch=util::isinstance<Controller>(asset);
        }else if(query.find("Agent:")==0){
            re=std::regex(query.substr(6));
            typeMatch=util::isinstance<Agent>(asset);
        }else{
            throw std::runtime_error("Invalid query for CommunicationBuffer's inviteOnRequest. query="+query);
        }
        if(typeMatch && std::regex_match(asset->getFullName(),re)){
            invitationRequested.emplace_back(asset);
            ret=true;
            break;
        }
    }
    if(ret && validated){
        //初期化後の追加の場合
        participants[asset->getFullName()]=asset;
        asset->addCommunicationBuffer(this->shared_from_this());
    }
    return ret;
}
void CommunicationBuffer::validate(){
    participants.clear();
    if(j_participants.is_string()){
        validateSub(j_participants);
    }else if(j_participants.is_array()){
        for(auto&& e:j_participants){
            validateSub(e);
        }
    }
    for(auto&& e:invitationRequested){
        participants[e.lock()->getFullName()]=e;
    }
    for(auto&& e:participants){
        e.second.lock()->addCommunicationBuffer(this->shared_from_this());
    }
    validated=true;
}
void CommunicationBuffer::validateSub(const std::string& query){
    if(query.find("PhysicalAsset:")==0){
        std::regex re(query.substr(14));
        for(auto&& e:manager.lock()->getAssets([&re](const std::shared_ptr<const PhysicalAsset>& asset)->bool {
            return std::regex_match(asset->getFullName(),re);
        })){
            participants[e.lock()->getFullName()]=e;
        }
    }else if(query.find("Controller:")==0){
        std::regex re(query.substr(11));
        for(auto&& e:manager.lock()->getControllers([&re](const std::shared_ptr<const Controller>& cont)->bool {
            return std::regex_match(cont->getFullName(),re);
        })){
            participants[e.lock()->getFullName()]=e;
        }
    }else if(query.find("Agent:")==0){
        std::regex re(query.substr(6));
        for(auto&& e:manager.lock()->getAgents([&re](const std::shared_ptr<const Agent>& agent)->bool {
            return std::regex_match(agent->getFullName(),re);
        })){
            participants[e.lock()->getFullName()]=e;
        }
    }else{
        throw std::runtime_error("Invalid query for CommunicationBuffer's participants. name="+name+", query="+query);
    }
}
std::shared_ptr<CommunicationBuffer> CommunicationBuffer::create(const std::shared_ptr<SimulationManager>& manager_,const std::string& name_,const nl::json& participants_,const nl::json& inviteOnRequest_){
    return std::make_shared<CommunicationBuffer>(manager_,name_,participants_,inviteOnRequest_);
}

void exportCommunicationBuffer(py::module &m)
{
    using namespace pybind11::literals;

    bind_stl_container<std::map<std::string,std::weak_ptr<CommunicationBuffer>>>(m,py::module_local(false));

    auto buf=expose_common_class<CommunicationBuffer>(m,"CommunicationBuffer")
    DEF_FUNC(CommunicationBuffer,send)
    DEF_FUNC(CommunicationBuffer,receive)
    ;
    expose_enum_value_helper(
        expose_enum_class<CommunicationBuffer::UpdateType>(buf,"UpdateType")
        ,"MERGE"
        ,"REPLACE"
    );
}

ASRC_NAMESPACE_END(util)
ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
