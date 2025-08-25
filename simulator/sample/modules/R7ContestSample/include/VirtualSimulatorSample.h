// Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
// 仮想シミュレータを生成するサンプル

#pragma once
#include <ASRCAISim1/Agent.h>
#include <ASRCAISim1/MotionState.h>
#include <ASRCAISim1/Track.h>
#include <ASRCAISim1/Missile.h>
#include <ASRCAISim1/SimulationManager.h>
#include <BasicAgentUtility/util/MissileRangeUtility.h>
#include "Common.h"

ASRC_PLUGIN_NAMESPACE_BEGIN

// 名前空間全体をusingすることも可能ではあるが推奨はしない。
// using directive is possible, but not recommended unless this module will not be inherited by another module.
//using namespace asrc::core; 

//彼機からの仮想誘導弾を生成するサンプル
ASRC_DECLARE_BASE_REF_CLASS(VirtualSimulatorSample)
	public:
	int maxNumPerParent; // 各parentあたりの最大仮想誘導弾数
	int launchInterval; // 仮想誘導弾の生成間隔 (agent step数単位)
	double kShoot; // 仮想誘導弾の射撃条件(彼から我へのRNormがこの値以下のとき射撃)

	std::shared_ptr<asrc::core::SimulationManager> virtualSimulator; // 仮想シミュレータインスタンス

    std::map<std::string,std::vector<std::shared_ptr<asrc::core::Missile>>> virtualMissiles; // 各parentを目標として飛翔中の仮想誘導弾

	std::map<std::string,double> minimumDistances; // 各parentの、最近傍の仮想誘導弾との距離

	// コンストラクタ。ここでは仮想シミュレータを起動しない
	VirtualSimulatorSample(const nl::json& config);

	// agentの持つ情報に基づいて仮想シミュレータを起動する。agentのvalidate()で呼び出すことを想定。
	void createVirtualSimulator(const std::shared_ptr<asrc::core::Agent>& agent);

	// agentの持つ情報に基づいて仮想シミュレータの時間を進め、情報を取得する。仮想敵の情報をtracksで与える。
	// agentのmakeObs()又はdeploy(action)で毎回呼び出すことを想定。
	void step(const std::shared_ptr<asrc::core::Agent>& agent, const std::vector<asrc::core::Track3D>& tracks);

	// 仮想誘導弾の射撃条件を指定する。この例では、motionからtrackへのRNormがkShoot以下の場合に射撃する。
	virtual bool launchCondition(const std::shared_ptr<asrc::core::PhysicalAssetAccessor>& parent,const asrc::core::MotionState& motion, const asrc::core::Track3D& track);
};

ASRC_DECLARE_BASE_REF_TRAMPOLINE(VirtualSimulatorSample)
    virtual bool launchCondition(const std::shared_ptr<asrc::core::PhysicalAssetAccessor>& parent,const asrc::core::MotionState& motion, const asrc::core::Track3D& track) override{
        PYBIND11_OVERRIDE(bool,Base,launchCondition,parent,motion,track);
    }
};

void exportVirtualSimulatorSample(py::module &m);

ASRC_PLUGIN_NAMESPACE_END

ASRC_PYBIND11_MAKE_OPAQUE(std::map<std::string,std::vector<std::shared_ptr<asrc::core::Missile>>>);
