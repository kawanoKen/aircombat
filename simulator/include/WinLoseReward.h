/**
 * Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
 * @file
 * 
 * @brief 得点差や途中経過によらず、終了時の勝ち負けのみによる報酬を与えるクラス
 */
#pragma once
#include <iostream>
#include <vector>
#include <map>
#include <pybind11/pybind11.h>
#include <Eigen/Core>
#include "MathUtility.h"
#include "Utility.h"
#include "Reward.h"
#include "Fighter.h"
#include "Missile.h"
namespace py=pybind11;
namespace nl=nlohmann;

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)

/**
 * @class WinLoseReward
 * @brief 得点差や途中経過によらず、終了時の勝ち負けのみによる報酬を与えるクラス
 */
ASRC_DECLARE_DERIVED_CLASS_WITHOUT_TRAMPOLINE(WinLoseReward,TeamReward)
    public:
    //parameters
    double win; //!< 勝利時の報酬
    double lose; //!< 敗北時の報酬
    double draw; //!< 引き分け時の報酬
    //internal variables
    std::string westSider,eastSider;
    //constructors & destructor
    using BaseType::BaseType;
    //functions
    virtual void initialize() override;
    virtual void serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full) override;
    virtual void onEpisodeBegin() override;
    virtual void onStepEnd() override;
};
void exportWinLoseReward(py::module &m, const std::shared_ptr<asrc::core::FactoryHelper>& factoryHelper);

ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
