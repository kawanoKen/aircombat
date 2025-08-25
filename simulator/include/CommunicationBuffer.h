/**
 * Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
 * @file
 * 
 * @brief Asset 間通信の簡易的な模擬として、一つの json object(dict)を共有バッファとして参加中の Asset 間で読み書きを行うためのクラス。
 */
#pragma once
#include "Common.h"
#include <memory>
#include <regex>
#include <vector>
#include <map>
#include <shared_mutex>
#include <pybind11/pybind11.h>
#include <nlohmann/json.hpp>
#include "Utility.h"
#include "TimeSystem.h"
namespace py=pybind11;
namespace nl=nlohmann;

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)

class SimulationManager;
class Asset;

ASRC_INLINE_NAMESPACE_BEGIN(util)

/**
 * @class CommunicationBuffer
 * @brief Asset 間通信の簡易的な模擬として、一つの json object(dict)を共有バッファとして参加中の Asset 間で読み書きを行うためのクラス。
 */
ASRC_DECLARE_BASE_REF_CLASS(CommunicationBuffer)
    private:
    friend class core::SimulationManager;
    mutable std::shared_mutex mtx;
    std::string name;
    nl::json j_participants,j_inviteOnRequest;
    bool validated;
    std::map<std::string,std::weak_ptr<Asset>> participants;
    std::vector<std::string> inviteOnRequest;
    std::vector<std::weak_ptr<Asset>> invitationRequested;
    std::map<std::string,Time> updatedTimes;
    nl::json buffer;
    std::weak_ptr<SimulationManager> manager;
    mutable bool hasBeenSerialized;
    //constructors & destructor
    bool requestInvitation(const std::shared_ptr<Asset>& asset);
    void validate();
    void validateSub(const std::string& query);
    public:
    enum UpdateType{
        REPLACE=0,
        MERGE=1
    };
    //functions
    CommunicationBuffer(const std::shared_ptr<SimulationManager>& manager_,const std::string& name_,const nl::json& participants_,const nl::json& inviteOnRequest_);
    virtual ~CommunicationBuffer();
    virtual void serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full);
    std::string getName() const;
    void send(const nl::json& data,UpdateType updateType);
    std::pair<Time,nl::json> receive(const std::string& key) const;
    static std::shared_ptr<CommunicationBuffer> create(const std::shared_ptr<SimulationManager>& manager_,const std::string& name_,const nl::json& participants_,const nl::json& inviteOnRequest_);
};
DEFINE_SERIALIZE_ENUM_AS_STRING(CommunicationBuffer::UpdateType)
ASRC_DECLARE_BASE_REF_TRAMPOLINE(CommunicationBuffer)
};

void exportCommunicationBuffer(py::module &m);

ASRC_NAMESPACE_END(util)
ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)

ASRC_PYBIND11_MAKE_OPAQUE(std::map<std::string,std::weak_ptr<::asrc::core::util::CommunicationBuffer>>);
