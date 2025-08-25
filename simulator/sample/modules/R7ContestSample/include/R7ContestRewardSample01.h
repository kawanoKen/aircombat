// Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#pragma once
#include <iostream>
#include <vector>
#include <map>
#include <pybind11/pybind11.h>
#include <Eigen/Core>
#include <ASRCAISim1/MathUtility.h>
#include <ASRCAISim1/Utility.h>
#include <ASRCAISim1/Reward.h>
#include <ASRCAISim1/Fighter.h>
#include <ASRCAISim1/Missile.h>
#include "Common.h"
namespace py=pybind11;
namespace nl=nlohmann;

ASRC_PLUGIN_NAMESPACE_BEGIN

// 名前空間全体をusingすることも可能ではあるが推奨はしない。
// using directive is possible, but not recommended unless this module will not be inherited by another module.
//using namespace asrc::core; 

ASRC_DECLARE_DERIVED_CLASS_WITHOUT_TRAMPOLINE(R7ContestRewardSample01,asrc::core::TeamReward)
    /*いくつかの観点に基づいた報酬の実装例。
    (1)Bite(誘導弾シーカで目標を捕捉)への加点
    (2)誘導弾目標のメモリトラック落ちへの減点
    (3)敵探知への加点(生存中の敵の何%を探知できているか)
    (4)過剰な機動への減点
    (5)前進・後退への更なる加減点
    (6)保持している力学的エネルギー(回転を除く)の多寡による加減点
    */

	// asrc::core::をなるべく省略したい場合は、クラス内で一度だけalias宣言する方が安全である。
    // Alias declration inside the class is safer way to prevent writing "asrc::core::" everywhere.
    using Fighter=asrc::core::Fighter;
    using Missile=asrc::core::Missile;
    public:
    //parameters
    double pBite,pMemT,pDetect,pVel,pOmega,pLine,pEnergy;
    bool pLineAsPeak;
    //internal variables
    std::string westSider,eastSider;
    double dLine;
    std::map<std::string,double> leadRange;
    std::map<std::string,double> leadRangePrev;
    std::map<std::string,Eigen::Vector2d> forwardAx;
    std::map<std::string,std::size_t> numMissiles;
    std::map<std::string,Eigen::VectorX<bool>> biteFlag,memoryTrackFlag;
    std::map<std::string,std::vector<std::weak_ptr<Fighter>>> friends,enemies;
    std::map<std::string,std::vector<std::weak_ptr<Missile>>> friendMsls;
    std::map<std::string,double> totalEnergy;
    //constructors & destructor
    using BaseType::BaseType;
    //functions
    virtual void initialize() override;
    virtual void serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full) override;
    virtual void onEpisodeBegin() override;
    virtual void onInnerStepEnd() override;
};
void exportR7ContestRewardSample01(py::module& m, const std::shared_ptr<asrc::core::FactoryHelper>& factoryHelper);

ASRC_PLUGIN_NAMESPACE_END
