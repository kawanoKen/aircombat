// Copyright (c) 2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#pragma once
#include <ASRCAISim1/Common.h>
#include <ASRCAISim1/Agent.h>
#include <R7ContestModels/R7ContestMassPointFighter.h>
#include "Common.h"

ASRC_PLUGIN_NAMESPACE_BEGIN

// 名前空間全体をusingすることも可能ではあるが推奨はしない。
// using directive is possible, but not recommended unless this module will not be inherited by another module.
//using namespace asrc::core; 

ASRC_DECLARE_DERIVED_CLASS_WITHOUT_TRAMPOLINE(your_class,asrc::core::SingleAssetAgent)
    // asrc::core::をなるべく省略したい場合は、クラス内で一度だけalias宣言する方が安全である。
    // Alias declration inside the class is safer way to prevent writing "asrc::core::" everywhere.
    using Track3D=asrc::core::Track3D;
    using MotionState=asrc::core::MotionState;
    using AltitudeKeeper=asrc::core::util::AltitudeKeeper;
    using Time=asrc::core::Time;
    public:
    //constructors & destructor
    using BaseType::BaseType;
    //functions
    //以下を必要に応じてオーバーライドすること。
    virtual void initialize() override;
    virtual void serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full) override;
    virtual void validate() override;
    virtual py::object observation_space() override;
    virtual py::object makeObs() override;
    virtual py::object action_space() override;
    virtual void deploy(py::object action) override;
    virtual void control() override;
    virtual py::object convertActionFromAnother(const nl::json& decision,const nl::json& command) override;
    virtual void controlWithAnotherAgent(const nl::json& decision,const nl::json& command) override;
};

void export_your_class(py::module &m, const std::shared_ptr<asrc::core::FactoryHelper>& factoryHelper);

ASRC_PLUGIN_NAMESPACE_END

