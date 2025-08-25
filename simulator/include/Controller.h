/**
 * Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
 * @file
 * 
 * @brief Asset のうち物理的実体を持たないものを表すクラス
 */
#pragma once
#include <map>
#include <pybind11/pybind11.h>
#include "Asset.h"
#include "Utility.h"
#include "SimulationManager.h"

namespace py=pybind11;
namespace nl=nlohmann;

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)

class Agent;

/**
 * @class Controller
 * @brief Asset のうち物理的実体を持つものを表すクラス
 * 
 * @details 必ず一つの PhysicalAsset に従属してその「子」として生成される
 */
ASRC_DECLARE_DERIVED_CLASS_WITH_TRAMPOLINE(Controller,Asset)
    friend class SimulationManager;
    public:
    static const std::string baseName; // baseNameを上書き
    std::shared_ptr<SimulationManagerAccessorForPhysicalAsset> manager; //!< SimulationManager の Accessor
    std::string team,group,name;
    std::weak_ptr<Asset> parent; //!< 親となる PhysicalAsset
    //constructors & destructor
    using BaseType::BaseType;
    //functions
    virtual void initialize() override;
    virtual void serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full) override;
    virtual bool isAlive() const override;
    virtual std::string getTeam() const override;
    virtual std::string getGroup() const override;
    virtual std::string getName() const override;
    virtual void perceive(bool inReset) override;
    virtual void control() override;
    virtual void behave() override;
    virtual void kill() override;
    /**
     * @brief CommunicationBufferを生成する。
     * 
     * @param [in] name_ CommunicationBuffer を識別する名前
     * @param [in] participants_ 無条件で参加する Asset のリスト
     * @param [in] inviteOnRequest_ Asset 側から要求があった場合のみ参加させる Asset のリスト
     * 
     * @sa [CommunicationBuffer による Asset 間通信の表現](\ref page_simulation_CommunicationBuffer)
     */
    bool generateCommunicationBuffer(const std::string& name_,const nl::json& participants_,const nl::json& inviteOnRequest_);
    protected:
    virtual void setManagerAccessor(const std::shared_ptr<EntityManagerAccessor>& ema) override;
};

ASRC_DECLARE_DERIVED_TRAMPOLINE(Controller)
    virtual bool isAlive() const override{
        PYBIND11_OVERRIDE(bool,Base,isAlive);
    }
    virtual std::string getTeam() const override{
        PYBIND11_OVERRIDE(std::string,Base,getTeam);
    }
    virtual std::string getGroup() const override{
        PYBIND11_OVERRIDE(std::string,Base,getGroup);
    }
    virtual std::string getName() const override{
        PYBIND11_OVERRIDE(std::string,Base,getName);
    }
};

void exportController(py::module &m, const std::shared_ptr<asrc::core::FactoryHelper>& factoryHelper);

ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
