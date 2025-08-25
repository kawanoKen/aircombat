// Copyright (c) 2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#pragma once
#include <ASRCAISim1/Agent.h>
#include <ASRCAISim1/FlightControllerUtility.h>
#include "Common.h"

ASRC_PLUGIN_NAMESPACE_BEGIN

ASRC_DECLARE_DERIVED_CLASS_WITHOUT_TRAMPOLINE(CapsuleCoursePatrolAgent,asrc::core::SingleAssetAgent)
	/*
		空間上の2点p1とp2を囲むカプセル状の軌道を指定速度で周回するAgentモデル。
		自身に飛来する誘導弾を検知した場合はそれに背を向けて回避行動を行い、回避完了後は元の軌道に復帰を試みる。
	*/
	using AltitudeKeeper=asrc::core::util::AltitudeKeeper;
	public:
	//modelConfigで設定するもの
	Eigen::Vector3d p1,p2;
	double velocity; //周回速度
	double radius; //軌跡の半径
	double deltaAZLimit; //旋回コマンドの現在方位とのAZ差の上限
	double minPitchForNormal,maxPitchForNormal; //通常時のピッチ角制限
	double minPitchForEvasion,maxPitchForEvasion,minAltForEvasion; //誘導弾回避時のピッチ角制限と下限高度
	//内部変数
	AltitudeKeeper altitudeKeeper;
	double course_length;

	public:
    //constructors & destructor
    using BaseType::BaseType;
    //functions
    virtual void initialize() override;
    virtual void serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full) override;
	virtual void validate() override;
	virtual void control() override;

	//周回軌道上の位置を0〜1で正規化して表現する。
	//p1-p2を結んだ直線が周回経路と交わる2点のうちp1に近い点を0とし、
	//進行方向に沿った移動距離に比例した値とし、終点を1とする。
	std::pair<Eigen::Vector3d,double> getCoursePoint(const double& phase) const;
	double getCoursePhase(const Eigen::Vector3d& pos) const;

	bool chkMWS();
};

void exportCapsuleCoursePatrolAgent(py::module &m, const std::shared_ptr<asrc::core::FactoryHelper>& factoryHelper);

ASRC_PLUGIN_NAMESPACE_END
