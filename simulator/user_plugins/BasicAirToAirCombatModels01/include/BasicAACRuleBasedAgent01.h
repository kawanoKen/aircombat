// Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#pragma once
#include "Common.h"
#include <ASRCAISim1/Track.h>
#include <ASRCAISim1/Agent.h>
#include <ASRCAISim1/Sensor.h>
#include <ASRCAISim1/FlightControllerUtility.h>

ASRC_PLUGIN_NAMESPACE_BEGIN

ASRC_DECLARE_DERIVED_CLASS_WITH_TRAMPOLINE(TrackInfo,asrc::core::Track3D)
    using Time=asrc::core::Time;
    enum class SensingState{
        INSIDE,
        LIMIT,
        OUTSIDE
    };
    enum class UpdateState{
        TRACK,
        MEMORY,
        LOST
    };
    int idx;
    double distance,myRHead,myRTail,hisRHead,hisRTail;
    SensingState inOurSensor,inMySensor;
    std::size_t numTracker,numTrackerLimit;
    std::vector<std::string> trackers,limitTrackers,nonLimitTrackers;
    UpdateState state;
    Time memoryStartTime;
    public:
    //constructors & destructor
    TrackInfo();
    TrackInfo(const Track3D& original_,int idx_=-1);
    TrackInfo(const TrackInfo& other);
    TrackInfo(const nl::json& j_);
    virtual ~TrackInfo();
    //functions
    TrackInfo copy() const;
    virtual void update(const TrackInfo& other,bool isFull=false);
    virtual void polymorphic_assign(const PtrBaseType& other) override;
    virtual void serialize_impl(asrc::core::util::AvailableArchiveTypes& archive) override;
};
DEFINE_SERIALIZE_ENUM_AS_STRING(TrackInfo::SensingState)
DEFINE_SERIALIZE_ENUM_AS_STRING(TrackInfo::UpdateState)

ASRC_DECLARE_DERIVED_TRAMPOLINE(TrackInfo)
    virtual void update(const TrackInfo& other,bool isFull) override{
        PYBIND11_OVERRIDE(void,Base,update,other,isFull);
    }
};

ASRC_DECLARE_DERIVED_CLASS_WITHOUT_TRAMPOLINE(BasicAACRuleBasedAgent01,asrc::core::SingleAssetAgent)
    using AltitudeKeeper=asrc::core::AltitudeKeeper;
	using Time=asrc::core::Time;
    public:
    enum class TransitionCause{
        FUEL_SHORTAGE,//1
        NO_ENEMY,//2
        TRY_BREAK,//3
        INTERCEPT_LEAKER,//3
        TRACKING_BY_ALLY,//4
        TRACKING_BY_NOBODY,//5
        TRACKING_BY_MYSELF,//6
        ABNORMAL,//0
        NONE//0
    };
    enum class State{
        ADVANCE,
        APPROACH_TARGET,
        KEEP_SENSING,
        RTB,
        WITHDRAW,
        EVADE,
        NONE
    };
    //失探後の航跡の保持に関するパラメータ
	double tMaxMemory;
    //探知状況の分類に関するパラメータ
	double sensorInRangeLimit,sensorInCoverageLimit;
    bool considerStateForTrackingCondition;
	//(c1)射撃条件に関するパラメータ
	double kShoot;
    int nMslSimul;
	//(c2)離脱条件に関するパラメータ
	double kBreak;
    //(c3)離脱終了条件に関するパラメ−タ
    double tWithdraw;
    //(s1)通常時の行動選択に関するパラメータ
    //advance,approach,keepSensing
	double pAdvanceAlly,pApproachAlly,pKeepSensingAlly,pApproachMyself,pKeepSensingMyself;
    double reconsiderationCycle;
    //(s1-1)前進に関するパラメータ
    double dPrioritizedAdvance,thetaPrioritizedAdvance;
    //(s1-3)横行に関するパラメータ
    double thetaKeepSensing;
    //(s1-4)後退に関するパラメータ
    double RTBEnterMargin,RTBExitMargin,RTBAltitude,RTBVelocity,RTBPitchLimit;
    //(s3)回避に関するパラメータ
	double thetaEvasion,hEvasion;
    //(o1)副目標の追尾に関するパラメータ
    double thetaModForSensing;
    //(o2)高度維持に関するパラメータ
	double thetaNormal,hNormal;
    double thetaWithdraw,hWithdraw;
    //(o3)場外の防止に関するパラメータ
	double dOutLimit,dOutLimitTurnAxis,dOutLimitKeepSensing,dOutLimitThreshold,dOutLimitStrength;
	//目標選択に関するパラメータ
	double dPrioritizedAimLeaker,thetaPrioritizedAimLeaker;
    //速度維持のためのパラメータ
    double minimumV,minimumRecoveryV,minimumRecoveryDstV,nominalThrottle;
    double cmdDeltaAzLimit;
    //内部変数
    State state;
    std::vector<TrackInfo> track,additionalTargets,allEnemies,withdrawnTrigger;
    TrackInfo target;
    bool isAssignedAsInterceptor;
    bool launchFlag,velRecovery;;
    TrackInfo launchedTarget;
    TrackInfo fireTgt;
    Time withdrawnTime;
    bool waitingTransition;
    double transitionDelay;
    Time transitionTriggeredTime;
    State transitionBefore,transitionAfter;
    TransitionCause transitionCause;
    Time lastNormalTransitionTime;
    bool isAfterDeploy;
    Eigen::Vector3d forward,rightSide;
    double dOut,dLine;
    Eigen::Vector3d pos,vel;
    Eigen::Vector3d dstDir;
    AltitudeKeeper altitudeKeeper;
    public:
    //constructors & destructor
    using BaseType::BaseType;
    //functions
    virtual void initialize() override;
    virtual void serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full) override;
    virtual void validate() override;
    virtual void deploy(py::object action) override;
    virtual void perceive(bool inReset) override;
    virtual void control() override;
    //情報の更新
    virtual void updateMyInfo();
    virtual void updateTracks(bool isFull);
    virtual void updateTrackInfo(TrackInfo& ti,bool isFull);
    virtual void updateTargetStatus(bool isFull);
    virtual void updateTargetStatusSub(TrackInfo& tgt,bool isFull);

    virtual bool chkFuelSufficiency();//余剰燃料のチェック
    //(c1)射撃条件
    virtual bool chkLaunchable();
    virtual std::pair<bool,std::pair<TrackInfo,double>> chkConditionForShoot(TrackInfo& tgt);
    //(c2)離脱条件
    virtual std::vector<bool> chkBreak();
    //(s3)回避条件
    virtual bool chkMWS();
    virtual bool chkTransitionCount();
    virtual void reserveTransition(const State& dst,double delay);
    virtual void completeTransition();
    virtual void cancelTransition();
    virtual void immidiateTransition(const State& dst);
    virtual void selectTarget();
    virtual void decideShoot(TrackInfo& tgt);
    virtual bool chooseTransitionNormal();
    virtual bool deploySub();
    virtual void makeCommand();
    void overrideFlightControl();
    protected:
    virtual double calcRHead(const Eigen::Vector3d &rs,const Eigen::Vector3d &vs,const Eigen::Vector3d &rt,const Eigen::Vector3d &vt);
    virtual double calcRTail(const Eigen::Vector3d &rs,const Eigen::Vector3d &vs,const Eigen::Vector3d &rt,const Eigen::Vector3d &vt);
};
DEFINE_SERIALIZE_ENUM_AS_STRING(BasicAACRuleBasedAgent01::TransitionCause)
DEFINE_SERIALIZE_ENUM_AS_STRING(BasicAACRuleBasedAgent01::State)

void exportBasicAACRuleBasedAgent01(py::module &m, const std::shared_ptr<asrc::core::FactoryHelper>& factoryHelper);

ASRC_PLUGIN_NAMESPACE_END

ASRC_PYBIND11_MAKE_OPAQUE(std::vector<ASRC_PLUGIN_NAMESPACE::TrackInfo>);
ASRC_PYBIND11_MAKE_OPAQUE(std::vector<std::vector<ASRC_PLUGIN_NAMESPACE::TrackInfo>>);
