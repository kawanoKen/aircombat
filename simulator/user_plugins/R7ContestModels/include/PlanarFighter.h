/**
 * Copyright (c) 2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
 * @file
 * 
 * @brief R7年度コンテストのユース部門用の2次元平面上に運動を拘束した戦闘機クラス
 */
#pragma once
#include <deque>
#include <vector>
#include <functional>
#include <pybind11/pybind11.h>
#include <Eigen/Core>
#include <ASRCAISim1/MathUtility.h>
#include <ASRCAISim1/MassPointFighter.h>
#include <ASRCAISim1/Controller.h>
#include "Common.h"
#include "R7ContestMassPointFighter.h"
namespace py=pybind11;
namespace nl=nlohmann;

ASRC_PLUGIN_NAMESPACE_BEGIN

/**
 * @class PlanarFighter
 * @brief R7年度コンテストのユース部門用の2次元平面上に運動を拘束した戦闘機クラス
 */
ASRC_DECLARE_DERIVED_CLASS_WITHOUT_TRAMPOLINE(PlanarFighter,R7ContestMassPointFighter)
    //運動を2次元平面上に拘束した戦闘機モデル。
    public:
    //constructors & destructor
    using BaseType::BaseType;
    //functions
    virtual void calcMotion(double dt) override;
};

ASRC_DECLARE_DERIVED_CLASS_WITHOUT_TRAMPOLINE(PlanarFlightController,R7ContestMassPointFlightController)
    public:
    //constructors & destructor
    using BaseType::BaseType;
    //functions
    virtual nl::json getDefaultCommand() override;
    virtual nl::json calc(const nl::json &cmd) override;
    virtual nl::json calcDirect(const nl::json &cmd) override;
    virtual nl::json calcFromDirAndVel(const nl::json &cmd) override;
};

void exportPlanarFighter(py::module &m, const std::shared_ptr<asrc::core::FactoryHelper>& factoryHelper);

ASRC_PLUGIN_NAMESPACE_END
