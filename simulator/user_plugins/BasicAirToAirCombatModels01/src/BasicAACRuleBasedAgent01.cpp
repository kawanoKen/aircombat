// Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#include "BasicAACRuleBasedAgent01.h"
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <ASRCAISim1/SimulationManager.h>
#include <ASRCAISim1/Fighter.h>
#include <ASRCAISim1/Missile.h>
#include <ASRCAISim1/Ruler.h>
#include <ASRCAISim1/Units.h>
#include <ASRCAISim1/StaticCollisionAvoider2D.h>

ASRC_PLUGIN_NAMESPACE_BEGIN

using namespace asrc::core;
using namespace util;

TrackInfo::TrackInfo():Track3D(){
    idx=-1;
    distance=myRHead=myRTail=hisRHead=hisRTail=0.0;
    inOurSensor=SensingState::OUTSIDE;inMySensor=SensingState::OUTSIDE;
    numTracker=numTrackerLimit=0;
    trackers.clear();
    limitTrackers.clear();
    nonLimitTrackers.clear();
    state=UpdateState::LOST;
    memoryStartTime=Time();
}
TrackInfo::TrackInfo(const Track3D& original_,int idx_)
:Track3D(original_){
    idx=idx_;
    distance=myRHead=myRTail=hisRHead=hisRTail=0.0;
    inOurSensor=SensingState::INSIDE;inMySensor=SensingState::INSIDE;
    numTracker=numTrackerLimit=0;
    trackers.clear();
    limitTrackers.clear();
    nonLimitTrackers.clear();
    state=UpdateState::TRACK;
    memoryStartTime=Time();
}
TrackInfo::TrackInfo(const nl::json& j_)
:TrackInfo(j_.get<TrackInfo>()){
}
TrackInfo::TrackInfo(const TrackInfo &other):Track3D(other){
    update(other,true);
}
TrackInfo::~TrackInfo(){}
TrackInfo TrackInfo::copy() const{
    return TrackInfo(*this);
}
void TrackInfo::update(const TrackInfo& other,bool isFull){
    Track3D::update(other);
	distance=other.distance;
	if(isFull){
        idx=other.idx;
		myRHead=other.myRHead;
		myRTail=other.myRTail;
		hisRHead=other.hisRHead;
		hisRTail=other.hisRTail;
		inOurSensor=other.inOurSensor;
		inMySensor=other.inMySensor;
		numTracker=other.numTracker;
		numTrackerLimit=other.numTrackerLimit;
        trackers.clear();
        for(auto & e:other.trackers){
            trackers.push_back(e);
        }
        limitTrackers.clear();
        for(auto & e:other.limitTrackers){
            limitTrackers.push_back(e);
        }
        nonLimitTrackers.clear();
        for(auto & e:other.nonLimitTrackers){
            nonLimitTrackers.push_back(e);
        }
		state=other.state;
		memoryStartTime=other.memoryStartTime;
    }
}
void TrackInfo::polymorphic_assign(const TrackInfo::PtrBaseType& other){
    BaseType::polymorphic_assign(other);
    try{
        auto casted=dynamic_cast<const TrackInfo&>(other);
	    distance=casted.distance;
        idx=casted.idx;
		myRHead=casted.myRHead;
		myRTail=casted.myRTail;
		hisRHead=casted.hisRHead;
		hisRTail=casted.hisRTail;
		inOurSensor=casted.inOurSensor;
		inMySensor=casted.inMySensor;
		numTracker=casted.numTracker;
		numTrackerLimit=casted.numTrackerLimit;
        trackers.clear();
        for(auto & e:casted.trackers){
            trackers.push_back(e);
        }
        limitTrackers.clear();
        for(auto & e:casted.limitTrackers){
            limitTrackers.push_back(e);
        }
        nonLimitTrackers.clear();
        for(auto & e:casted.nonLimitTrackers){
            nonLimitTrackers.push_back(e);
        }
		state=casted.state;
		memoryStartTime=casted.memoryStartTime;
    }catch(std::bad_cast){
        //
    }
}
void TrackInfo::serialize_impl(asrc::core::util::AvailableArchiveTypes& archive){
    BaseType::serialize_impl(archive);

    ASRC_SERIALIZE_NVP(archive
        ,idx
        ,distance
        ,myRHead
        ,myRTail
        ,hisRHead
        ,hisRTail
        ,inOurSensor
        ,inMySensor
        ,numTracker
        ,numTrackerLimit
        ,trackers
        ,limitTrackers
        ,nonLimitTrackers
        ,state
        ,memoryStartTime
    )
}

void BasicAACRuleBasedAgent01::initialize(){
    BaseType::initialize();
    //失探後の航跡の保持に関するパラメータ
    tMaxMemory=getValueFromJsonKRD(modelConfig,"tMaxMemory",randomGen,10.0);
    //探知状況の分類に関するパラメータ
    sensorInRangeLimit=getValueFromJsonKRD(modelConfig,"sensorInRangeLimit",randomGen,0.0);
    sensorInCoverageLimit=deg2rad(getValueFromJsonKRD(modelConfig,"sensorInCoverageLimit",randomGen,0.0));
    considerStateForTrackingCondition=getValueFromJsonKRD(modelConfig,"considerStateForTrackingCondition",randomGen,false);
    //(c1)射撃条件に関するパラメータ
    kShoot=getValueFromJsonKRD(modelConfig,"kShoot",randomGen,0.25);
    nMslSimul=getValueFromJsonKRD(modelConfig,"nMslSimul",randomGen,2);
    //(c2)離脱条件に関するパラメータ
    kBreak=getValueFromJsonKRD(modelConfig,"kBreak",randomGen,0.15);
    //(c3)離脱終了条件に関するパラメータ
    tWithdraw=getValueFromJsonKRD(modelConfig,"tWithdraw",randomGen,30.0);
    //(s1)通常時の行動選択に関するパラメータ
    pAdvanceAlly=getValueFromJsonKRD(modelConfig,"pAdvanceAlly",randomGen,0.0);
    pApproachAlly=getValueFromJsonKRD(modelConfig,"pApproachAlly",randomGen,60.0);
    pKeepSensingAlly=getValueFromJsonKRD(modelConfig,"pKeepSensingAlly",randomGen,40.0);
    pApproachMyself=getValueFromJsonKRD(modelConfig,"pApproachMyself",randomGen,60.0);
    pKeepSensingMyself=getValueFromJsonKRD(modelConfig,"pKeepSensingMyself",randomGen,40.0);
    reconsiderationCycle=getValueFromJsonKRD(modelConfig,"reconsiderationCycle",randomGen,10.0);
    //(s1-1)前進に関するパラメータ
    dPrioritizedAdvance=getValueFromJsonKRD(modelConfig,"dPrioritizedAdvance",randomGen,-1.0);
    thetaPrioritizedAdvance=deg2rad(getValueFromJsonKRD(modelConfig,"thetaPrioritizedAdvance",randomGen,180.0));
    //(s1-3)横行に関するパラメータ
    thetaKeepSensing=deg2rad(getValueFromJsonKRD(modelConfig,"thetaKeepSensing",randomGen,10.0));
    //(s1-4)後退に関するパラメータ
    RTBEnterMargin=getValueFromJsonKRD(modelConfig,"RTBEnterMargin",randomGen,0.0);
    RTBExitMargin=getValueFromJsonKRD(modelConfig,"RTBExitMargin",randomGen,0.0);
    RTBAltitude=getValueFromJsonKRD(modelConfig,"RTBAltitude",randomGen,9000.0);
    RTBVelocity=getValueFromJsonKRD(modelConfig,"RTBVelocity",randomGen,250.0);
    RTBPitchLimit=deg2rad(getValueFromJsonKRD(modelConfig,"RTBPitchLimit",randomGen,10.0));
    //(s3)回避に関するパラメータ
    thetaEvasion=deg2rad(getValueFromJsonKRD(modelConfig,"thetaEvasion",randomGen,-45.0));
    hEvasion=getValueFromJsonKRD(modelConfig,"hEvasion",randomGen,4000.0);
    //(o1)副目標の追尾に関するパラメータ
    thetaModForSensing=deg2rad(getValueFromJsonKRD(modelConfig,"thetaModForSensing",randomGen,0.0));
    //(o2)高度維持に関するパラメータ
    thetaNormal=deg2rad(getValueFromJsonKRD(modelConfig,"thetaNormal",randomGen,10.0));
    hNormal=getValueFromJsonKRD(modelConfig,"hNormal",randomGen,12000.0);
    thetaWithdraw=deg2rad(getValueFromJsonKRD(modelConfig,"thetaWithdraw",randomGen,15.0));
    hWithdraw=getValueFromJsonKRD(modelConfig,"hWithdraw",randomGen,8000.0);
    nl::json sub={
        {"pGain",-3e-1},
        {"dGain",-1e-1},
        {"minPitch",-thetaNormal},
        {"maxPitch",thetaNormal}
    };
	altitudeKeeper=AltitudeKeeper(sub);
    //(o3)場外の防止に関するパラメータ
    dOutLimit=getValueFromJsonKRD(modelConfig,"dOutLimit",randomGen,7500.0);
    dOutLimitTurnAxis=getValueFromJsonKRD(modelConfig,"dOutLimitTurnAxis",randomGen,15000.0);
    dOutLimitKeepSensing=getValueFromJsonKRD(modelConfig,"dOutLimitKeepSensing",randomGen,10000.0);
    dOutLimitThreshold=getValueFromJsonKRD(modelConfig,"dOutLimitThreshold",randomGen,10000.0);
    dOutLimitStrength=getValueFromJsonKRD(modelConfig,"dOutLimitStrength",randomGen,1e-3);
    //目標選択に関するパラメータ(突破優先に関するもの)
    dPrioritizedAimLeaker=getValueFromJsonKRD(modelConfig,"dPrioritizedAimLeaker",randomGen,-1.0);
    thetaPrioritizedAimLeaker=deg2rad(getValueFromJsonKRD(modelConfig,"thetaPrioritizedAimLeaker",randomGen,180.0));
    //最低速度維持のためのパラメータ
    minimumV=getValueFromJsonKRD(modelConfig,"minimumV",randomGen,200.0);
    minimumRecoveryV=getValueFromJsonKRD(modelConfig,"minimumRecoveryV",randomGen,220.0);
    minimumRecoveryDstV=getValueFromJsonKRD(modelConfig,"minimumRecoveryDstV",randomGen,250.0);
    nominalThrottle=getValueFromJsonKRD(modelConfig,"nominalThrottle",randomGen,1.0);
    cmdDeltaAzLimit=deg2rad(getValueFromJsonKRD(modelConfig,"cmdDeltaAzLimit",randomGen,45.0));
    state=State::ADVANCE;
    track.clear();
    target=TrackInfo();
    additionalTargets.clear();
    isAssignedAsInterceptor=false;
    launchFlag=false;
    launchedTarget=TrackInfo();
    withdrawnTime=Time();
    waitingTransition=false;
    transitionDelay=0.0;
    transitionTriggeredTime=Time();
    transitionBefore=State::NONE;
    transitionAfter=State::NONE;
    transitionCause=TransitionCause::NONE;
    lastNormalTransitionTime=manager->getEpoch()-reconsiderationCycle;
    velRecovery=false;
    isAfterDeploy=false;
}
void BasicAACRuleBasedAgent01::serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full){
    BaseType::serializeInternalState(archive,full);

    if(full){
        ASRC_SERIALIZE_NVP(archive
            ,tMaxMemory
            ,sensorInRangeLimit,sensorInCoverageLimit
            ,considerStateForTrackingCondition
            ,kShoot
            ,nMslSimul
            ,kBreak
            ,tWithdraw
            ,pAdvanceAlly,pApproachAlly,pKeepSensingAlly,pApproachMyself,pKeepSensingMyself
            ,reconsiderationCycle
            ,dPrioritizedAdvance,thetaPrioritizedAdvance
            ,thetaKeepSensing
            ,RTBEnterMargin,RTBExitMargin,RTBAltitude,RTBVelocity,RTBPitchLimit
            ,thetaEvasion,hEvasion
            ,thetaModForSensing
            ,thetaNormal,hNormal
            ,thetaWithdraw,hWithdraw
            ,dOutLimit,dOutLimitTurnAxis,dOutLimitKeepSensing,dOutLimitThreshold,dOutLimitStrength
            ,dPrioritizedAimLeaker,thetaPrioritizedAimLeaker
            ,minimumV,minimumRecoveryV,minimumRecoveryDstV,nominalThrottle
            ,cmdDeltaAzLimit
            ,forward,rightSide
            ,dOut,dLine
        )
    }

    ASRC_SERIALIZE_NVP(archive
        ,state
        ,track,additionalTargets,allEnemies,withdrawnTrigger
        ,target
        ,isAssignedAsInterceptor
        ,launchFlag,velRecovery
        ,launchedTarget
        ,fireTgt
        ,withdrawnTime
        ,waitingTransition
        ,transitionDelay
        ,transitionTriggeredTime
        ,transitionBefore,transitionAfter
        ,transitionCause
        ,lastNormalTransitionTime
        ,isAfterDeploy
        ,pos,vel
        ,dstDir
        ,altitudeKeeper
    )
}

void BasicAACRuleBasedAgent01::validate(){
    BaseType::validate();
    auto rulerObs=manager->getRuler().lock()->observables;
    std::string eastSider=rulerObs.at("eastSider");
    if(getTeam()==eastSider){
        forward<<0,-1,0;
        rightSide<<1,0,0;
    }else{
        forward<<0,1,0;
        rightSide<<-1,0,0;
    }
    dOut=rulerObs.at("dOut");
    dLine=rulerObs.at("dLine");
    std::dynamic_pointer_cast<FighterAccessor>(parent)->setFlightControllerMode("fromDirAndVel");
}
void BasicAACRuleBasedAgent01::perceive(bool inReset){
    observables.merge_patch({{parent->getFullName(),{
        {"state",state},
        {"target",target},
        {"launchedTarget",launchedTarget}
    }}});
}
void BasicAACRuleBasedAgent01::updateMyInfo(){
    MotionState myMotion(parent->observables.at("motion"));
    pos=myMotion.pos(getLocalCRS());
    vel=myMotion.vel(getLocalCRS());
}
void BasicAACRuleBasedAgent01::updateTracks(bool isFull){
    track.clear();
    int idx=0;
    for(auto&& t:parent->observables.at("/sensor/track"_json_pointer)){
        track.push_back(TrackInfo(t.get<Track3D>(),idx));
        updateTrackInfo(track[idx],isFull);
        idx++;
    }
}
void BasicAACRuleBasedAgent01::updateTrackInfo(TrackInfo& ti,bool isFull){
	/*TrackInfoの付帯情報の更新を行う。
	* 射程情報: 双方の距離,RmaxHead,Rmax1,Rmax2
	* 追尾状況: 自機が追尾中か否か,誰かが追尾中か否か,追尾機数(IN,LIMIT,OUT) ただし、stateがEVADEまたはWITHDRAWの機体はINでもLIMIT扱いとする。
	* isFullにTrueを入れたときはdeploy用に射程計算等を含む全てを計算し、Falseを入れたときは位置と速度だけ更新する。
	*/
    Eigen::Vector3d tpos=ti.pos(getLocalCRS());
    Eigen::Vector3d tvel=ti.vel(getLocalCRS());
    Eigen::Vector3d dpos=pos-tpos;
    double L=dpos.norm();
    ti.distance=L;
    if(isFull && ti.state!=TrackInfo::UpdateState::LOST){
        ti.myRHead=calcRHead(pos,vel,tpos,tvel);
        ti.myRTail=calcRTail(pos,vel,tpos,tvel);
        ti.hisRHead=calcRHead(tpos,tvel,pos,vel);
        ti.hisRTail=calcRTail(tpos,tvel,pos,vel);
        Eigen::Vector3d tEx=tvel.normalized();
        if(ti.state==TrackInfo::UpdateState::TRACK){
            ti.trackers.clear();
            for(auto& src:parent->observables.at("/sensor/trackSource"_json_pointer)[ti.idx]){
                if(std::find(ti.trackers.begin(),ti.trackers.end(),src.get<std::string>())==ti.trackers.end()){
                    ti.trackers.push_back(src);
                }
            }
            ti.numTracker=ti.trackers.size();
            ti.nonLimitTrackers.clear();
            ti.limitTrackers.clear();
            ti.numTrackerLimit=0;
            ti.inOurSensor=TrackInfo::SensingState::OUTSIDE;
            ti.inMySensor=TrackInfo::SensingState::OUTSIDE;
            for(auto& e:ti.trackers){
                const nl::json& fObs=e==parent->getFullName() ?
                    parent->observables :
                    parent->observables.at("/shared/fighter"_json_pointer).at(e);
                const nl::json& aObs=parent->observables.at("/shared/agent"_json_pointer).at(e);
                State aState=State::NONE;
                try{
                    aObs.at("state").get_to(aState);
                }catch(...){
                }
                Eigen::Vector3d tposS=MotionState(fObs.at("motion")).absAtoB(tpos,getLocalCRS());
                double L=tposS.norm();
                double angle=acos(std::max(-1.0,std::min(1.0,tposS(0)/L)));
                double fLref,fAngle;
                if(fObs.contains("/spec/sensor/radar"_json_pointer)){
                    fLref=fObs.at("/spec/sensor/radar/Lref"_json_pointer);
                    fAngle=fObs.at("/spec/sensor/radar/thetaFOR"_json_pointer);
                }else{
                    fLref=0;
                    fAngle=0;
                }
                if(
                    L<=fLref-sensorInRangeLimit &&
                    angle<=fAngle-sensorInCoverageLimit &&
                    (!considerStateForTrackingCondition || (
                    aState!=State::EVADE &&
                    aState!=State::WITHDRAW &&
                    aState!=State::NONE
                    ))
                ){
                    ti.inOurSensor=TrackInfo::SensingState::INSIDE;
                    ti.nonLimitTrackers.push_back(e);
                    if(e==parent->getFullName()){
                        ti.inMySensor=TrackInfo::SensingState::INSIDE;
                    }
                }else if(L<=fLref && angle<=fAngle){
                    ti.limitTrackers.push_back(e);
                    ti.numTrackerLimit++;
                    if(ti.inOurSensor!=TrackInfo::SensingState::INSIDE){
                        ti.inOurSensor=TrackInfo::SensingState::LIMIT;
                    }
                    if(e==parent->getFullName()){
                        if(ti.inMySensor!=TrackInfo::SensingState::INSIDE){
                            ti.inMySensor=TrackInfo::SensingState::LIMIT;
                        }
                    }
                }
            }
        }else{
            ti.trackers.clear();
            ti.nonLimitTrackers.clear();
            ti.limitTrackers.clear();
            ti.numTracker=0;
            ti.numTrackerLimit=0;
            ti.inOurSensor=TrackInfo::SensingState::OUTSIDE;
            ti.inMySensor=TrackInfo::SensingState::OUTSIDE;
        }
    }
}
void BasicAACRuleBasedAgent01::updateTargetStatus(bool isFull){
	/*主目標と副目標を更新する。副目標は、前回の各機の主目標と、まだ誘導が必要な全ての誘導弾目標とし、副目標内でのtruthの重複はないようにする。
	「誘導が必要」とは、誘導弾自身のセンサで捉えていない状態(mode!=SELF)を指すものとする。
	なお、副目標には自機の主目標も含む(敢えて除外していない)
	isFullにTrueを入れたときは副目標の再抽出も行い全て更新するが、Falseを入れたときはその時点で抽出されているものの位置と速度だけ更新する。
	*/
	//主目標の更新
    updateTargetStatusSub(target,isFull);
    updateTargetStatusSub(launchedTarget,isFull);
    //副目標の確定(isFullの時のみ)
    if(isFull){
        std::vector<bool> flag;//いらないものを削除するためのフラグ
        for(auto& t:additionalTargets){
            flag.push_back(false);
        }
        for(auto&& fr:parent->observables.at("/shared/fighter"_json_pointer).items()){
            if(fr.value().at("isAlive")){
                const nl::json& ag=parent->observables.at("/shared/agent"_json_pointer).at(fr.key());
                TrackInfo tgt;
                //主目標
                if(ag.contains("target")){
                    tgt=TrackInfo(ag.at("target"),-1);
                    if(!tgt.is_none()){
                        bool ret=true;
                        int idx=0;
                        for(auto& aTgt:additionalTargets){
                            if(aTgt.isSame(tgt)){
                                ret=false;
                                flag[idx]=true;
                                break;
                            }
                            idx++;
                        }
                        if(ret){
                            additionalTargets.push_back(tgt);
                            flag.push_back(true);
                        }
                    }
                }
                //直近の射撃目標
                if(ag.contains("launchedTarget")){
                    tgt=TrackInfo(ag.at("launchedTarget"),-1);
                    if(!tgt.is_none()){
                        int nextMsl=fr.value().at("/weapon/nextMsl"_json_pointer);
                        if(nextMsl>0){
                            const nl::json& msl=fr.value().at("/weapon/missiles"_json_pointer)[nextMsl-1];
                            if(msl.at("isAlive").get<bool>() && msl.at("mode").get<Missile::Mode>()!=Missile::Mode::SELF){//まだ誘導が必要
                                bool ret=true;
                                int idx=0;
                                for(auto& aTgt:additionalTargets){
                                    if(aTgt.isSame(tgt)){
                                        ret=false;
                                        flag[idx]=true;
                                        break;
                                    }
                                    idx++;
                                }
                                if(ret){
                                    additionalTargets.push_back(tgt);
                                    flag.push_back(true);
                                }
                            }
                        }
                    }
                }
            }
        }
        //まだ誘導が必要な誘導弾の目標は直近のでなくても残す
        for(auto&& msl:parent->observables.at("/weapon/missiles"_json_pointer)){
            if(msl.at("isAlive").get<bool>() && msl.at("hasLaunched").get<bool>() && msl.at("mode").get<Missile::Mode>()!=Missile::Mode::SELF){
                int idx=0;
                for(auto& aTgt:additionalTargets){
                    if(msl.at("target").get<Track3D>().isSame(aTgt)){
                        flag[idx]=true;
                            idx++;
                    }
                }
            }
        }
        for(auto&& fr:parent->observables.at("/shared/fighter"_json_pointer).items()){
            if(fr.key()!=parent->getFullName()){
                for(auto&& msl:fr.value().at("/weapon/missiles"_json_pointer)){
                    if(msl.at("isAlive").get<bool>() && msl.at("hasLaunched").get<bool>() && msl.at("mode").get<Missile::Mode>()!=Missile::Mode::SELF){
                        int idx=0;
                        for(auto& aTgt:additionalTargets){
                            if(msl.at("target").get<Track3D>().isSame(aTgt)){
                                flag[idx]=true;
                                    idx++;
                            }
                        }
                    }
                }
            }
        }
        auto ait=additionalTargets.begin();
        for(auto fit=flag.begin();fit!=flag.end();){
            if((*fit) && (*ait).state!=TrackInfo::UpdateState::LOST){
                ++fit;
                ++ait;
            }else{
                fit=flag.erase(fit);
                ait=additionalTargets.erase(ait);
            }
        }
    }
    //副目標の更新
    for(auto& aTgt:additionalTargets){
        updateTargetStatusSub(aTgt,isFull);
    }
    //行動判断の考慮対象(航跡＋メモリトラック中の主or副目標)の抽出
    allEnemies.clear();
    for(auto& t:track){
        allEnemies.push_back(t);
    }
    for(auto& t:additionalTargets){
        if(t.state==TrackInfo::UpdateState::MEMORY){
            allEnemies.push_back(t);
        }
    }
}
void BasicAACRuleBasedAgent01::updateTargetStatusSub(TrackInfo& tgt,bool isFull){
    /*指定した目標情報を更新する。
    */
    if(!tgt.is_none()){
        bool updated=false;
        for(auto& t:track){//航跡情報から更新対象を検索
            if(tgt.isSame(t)){
                tgt.update(t,isFull);
                updated=true;
                break;
            }
        }
        if(!updated){//見つからなかった場合、メモリトラック
            if(tgt.state==TrackInfo::UpdateState::TRACK){
                tgt.memoryStartTime=manager->getTime();
                tgt.state=TrackInfo::UpdateState::MEMORY;
            }else if(tgt.state==TrackInfo::UpdateState::MEMORY){
                if(manager->getTime()>=tMaxMemory+tgt.memoryStartTime){
                    tgt.state=TrackInfo::UpdateState::LOST;
                }
            }
        }
        if(tgt.state!=TrackInfo::UpdateState::TRACK){//外挿
            tgt.extrapolateTo(manager->getTime());
            updateTrackInfo(tgt,isFull);
        }
    }
}
bool BasicAACRuleBasedAgent01::chkFuelSufficiency(){
    //余剰燃料のチェック
    std::shared_ptr<FighterAccessor> f=std::dynamic_pointer_cast<FighterAccessor>(parent);
    double maxReachableRange=f->getMaxReachableRange();
    auto rulerObs=manager->getRuler().lock()->observables;
    double dLine=rulerObs.at("dLine");
    double distanceFromLine=forward.block<2,1>(0,0,2,1).dot(pos.block<2,1>(0,0,2,1))+dLine;
    double distanceFromBase=rulerObs.at("distanceFromBase").at(getTeam());
    double fuelMargin=rulerObs.at("fuelMargin");
    if(state==State::RTB){
        return maxReachableRange/(1+fuelMargin)-(distanceFromLine+distanceFromBase) >= RTBExitMargin;
    }else{
        return maxReachableRange/(1+fuelMargin)-(distanceFromLine+distanceFromBase) >= RTBEnterMargin;
    }
}
bool BasicAACRuleBasedAgent01::chkMWS(){
    if(parent->observables.contains("/sensor/mws"_json_pointer)){
        return parent->observables.at("/sensor/mws/track"_json_pointer).get<std::vector<Track2D>>().size()>0;
    }else{
        return false;
    }
}
bool BasicAACRuleBasedAgent01::chkLaunchable(){
    //自機が射撃可能な状態かどうかを返す。
    return getShared<FighterAccessor>(parent)->isLaunchable();
}
std::pair<bool,std::pair<TrackInfo,double>> BasicAACRuleBasedAgent01::chkConditionForShoot(TrackInfo& tgt){
	/*(c1)射撃条件の判定
    1. 射程による判定・・・その航跡に対してRtailを0、Rheadを1としたときに現在の距離がkShoot以下であること
    2. 探知状況による判定・・・その航跡を自身か味方の少なくとも１機が余裕をもって捉えていること
    3. 射撃状況による判定・・・その航跡に対して自身が発射した飛翔中の誘導弾がnMslSimul発未満であること
	*/
    //条件0：機体側の制限も確認しておく
    bool ret0=getShared<FighterAccessor>(parent)->isLaunchableAt(tgt);
    //条件1：射程による判定
    bool ret1=!tgt.is_none();
    double r;
    if(ret1){
        r=tgt.distance-tgt.myRTail;
        double delta=tgt.myRHead-tgt.myRTail;
        if(delta==0){
            if(r<0){
                r=-std::numeric_limits<double>::infinity();
            }else if(r>0){
                r=std::numeric_limits<double>::infinity();
            }else{
                r=0;
            }
        }else{
            r/=delta;
        }
        ret1=r<=kShoot;
    }else{
        r=0;
    }
    //条件2：探知状況による判定
    bool ret2=false;
    for(auto& f:tgt.nonLimitTrackers){
        State aState=State::NONE;
        try{
            parent->observables.at("/shared/agent"_json_pointer).at(f).at("state").get_to(aState);
        }catch(...){
        }
        if(!considerStateForTrackingCondition || (aState!=State::EVADE && aState!=State::WITHDRAW&& aState!=State::NONE)){
            ret2=true;
            break;
        }
    }
    //条件3:射撃状況による判定
    bool ret3=true;
    int count=0;
    for(auto&& msl:parent->observables.at("/weapon/missiles"_json_pointer)){
        if(msl.at("isAlive").get<bool>() && msl.at("hasLaunched").get<bool>()){
            if(msl.at("target").get<Track3D>().isSame(tgt)){
                count++;
                if(count>=nMslSimul){
                    ret3=false;
                    break;
                }
            }
        }
    }
    return std::make_pair(ret0 && ret1 && ret2 && ret3,std::make_pair(tgt,r));
}
std::vector<bool> BasicAACRuleBasedAgent01::chkBreak(){
    //(c2)離脱条件の判定
    std::vector<bool> ret;
    for(auto& t:allEnemies){
        ret.push_back((t.distance-t.hisRTail)<=(t.hisRHead-t.hisRTail)*kBreak);
    }
    return ret;
}
bool BasicAACRuleBasedAgent01::chkTransitionCount(){
    return waitingTransition && manager->getTime()-transitionTriggeredTime>=transitionDelay;
}
void BasicAACRuleBasedAgent01::reserveTransition(const State& dst,double delay){
    transitionTriggeredTime=manager->getTime();
    transitionBefore=state;
    transitionAfter=dst;
    transitionDelay=delay;
    waitingTransition=true;
}
void BasicAACRuleBasedAgent01::completeTransition(){
    bool isAbnormal=(
        (transitionAfter!=State::ADVANCE && transitionAfter!=State::APPROACH_TARGET && transitionAfter!=State::KEEP_SENSING && transitionAfter!=State::RTB)
    );
    if(isAbnormal){
        transitionCause=TransitionCause::ABNORMAL;
    }else{
        lastNormalTransitionTime=manager->getTime();
    }
    waitingTransition=false;
    state=transitionAfter;
}
void BasicAACRuleBasedAgent01::cancelTransition(){
    waitingTransition=false;
}
void BasicAACRuleBasedAgent01::immidiateTransition(const State& dst){
    bool isAbnormal=(
        (dst!=State::ADVANCE && dst!=State::APPROACH_TARGET && dst!=State::KEEP_SENSING && dst!=State::RTB)
    );
    if(isAbnormal){
        transitionCause=TransitionCause::ABNORMAL;
        //std::cout<<"abnormal, dst="<<dst<<std::endl;
    }else{
        lastNormalTransitionTime=manager->getTime();
    }
    waitingTransition=false;
    state=dst;
}
void BasicAACRuleBasedAgent01::selectTarget(){
	/*主目標の選択を行う。候補は全航跡に加え、メモリトラック中の副目標(主目標も含む)とする
	1. 自陣に十分近いか、敵陣側扇範囲内に捉えている味方がいないのいずれを満たす敵がいれば最も近い味方を割り振る。
	パラメータはdPrioritizedAimLeaker,thetaPrioritizedAimLeakerの二つ
	2. 距離の近い順に割り振る。
	*/
    //まず、lostしてしまったものを削除
    if(!target.is_none() && target.state==TrackInfo::UpdateState::LOST){
        target=TrackInfo();
    }
    if(!launchedTarget.is_none() && launchedTarget.state==TrackInfo::UpdateState::LOST){
        launchedTarget=TrackInfo();
    }
    std::vector<Eigen::Vector3d> fpos,fvel;
    std::vector<bool> assigned;
    assigned.push_back(false);
    fpos.push_back(pos);
    fvel.push_back(vel);
    for(auto&& f:parent->observables.at("/shared/fighter"_json_pointer).items()){
        if(f.value().at("isAlive").get<bool>()){
            assigned.push_back(false);
            MotionState mo(f.value().at("motion"));
            fpos.push_back(mo.pos(getLocalCRS()));
            fvel.push_back(mo.vel(getLocalCRS()));
        }
    }
    std::size_t numF=fpos.size();
    std::vector<TrackInfo*> candidates;
    std::vector<Eigen::Vector3d> tpos;
    for(auto& t:allEnemies){
        if(t.state!=TrackInfo::UpdateState::LOST){
            candidates.push_back(&t);
            tpos.push_back(t.pos(getLocalCRS()));
        }
    }
    isAssignedAsInterceptor=false;
    if(candidates.size()==0){//何も見えていない場合
        target=TrackInfo();
        return;
    }
	//(1)突破阻止(複数該当した場合は、とりあえずラインに近い敵から順に、最も近い未割当の味方を割り当てる)
    std::vector<std::pair<double,int>> leakers,nearestF;
    Eigen::MatrixXd dist=Eigen::MatrixXd::Zero(fpos.size(),tpos.size());
    int i=0;int j=0;
    for(auto& tp:tpos){
        double distFromLine=dLine-tp.dot(-forward);
        bool isLeaker=distFromLine<dPrioritizedAimLeaker*dLine;
        double angle=M_PI;
        i=0;
        for(auto& fp:fpos){
            dist(i,j)=(fp-tp).norm();
            angle=std::min(angle,
                acos(std::max(-1.0,std::min(1.0,
                    (tp-fp).block<2,1>(0,0,2,1).normalized().dot(forward.block<2,1>(0,0,2,1))
                )))
            );
            i++;
        }
        if(angle>thetaPrioritizedAimLeaker){
            isLeaker=true;
        }
        if(isLeaker){
            leakers.push_back(std::make_pair(distFromLine,j));
        }
        j++;
    }
    std::sort(leakers.begin(),leakers.end(),
        [](const std::pair<double,int> &lhs,const std::pair<double,int> &rhs){
        return lhs.first<rhs.first;
    });
    int numAssigned=0;
    //typedef std::pair<double,int> viType;
    for(auto& li:leakers){
        nearestF.clear();
        for(int i=0;i<fpos.size();++i){
            nearestF.push_back(std::make_pair(dist(i,li.second),i));
        }
        std::sort(nearestF.begin(),nearestF.end(),
            [](const std::pair<double,int> &lhs,const std::pair<double,int> &rhs){
            return lhs.first<rhs.first;
        });
        for(auto& nf:nearestF){
            if(!assigned[nf.second]){
                numAssigned++;
                if(nf.second==0){
                    target=*candidates[li.second];
                    isAssignedAsInterceptor=true;
                }
                break;
            }
        }
        if(numAssigned==fpos.size()){
            break;
        }
    }
    //(2)最も近い敵(重複可)
    std::vector<TrackInfo*> nearestE;
    for(int i=0;i<fpos.size();++i){
        int idx;
        Eigen::VectorXd tmp=dist.block(i,0,1,tpos.size()).transpose();
        tmp.minCoeff(&idx);
        nearestE.push_back(candidates[idx]);
    }
    target=*nearestE[0];
    return;
}
void BasicAACRuleBasedAgent01::decideShoot(TrackInfo& tgt){
    launchFlag=true;
    launchedTarget=tgt.copy();
}
bool BasicAACRuleBasedAgent01::chooseTransitionNormal(){
	/*通常時の状態遷移(前進、接近、横行、後退)の選択を行う。状況に応じて確率で決定
    1. 帰還できなくなりそうなとき、確定で後退
	2. 見えている敵がいない場合、確定で前進
    3. 敵陣に十分近いか、敵陣側扇範囲内に敵がいないのいずれかであればライン突破を優先し前進、
    　主目標が突破阻止を目的として割り当てられているとき、突破阻止を優先し接近を選択する。
    　両方満たす場合、自身が先に突破しそうであれば突破を優先し前進を選択、相手が先に突破しそうであれば突破阻止を優先し接近を選択
	4. 主目標を味方が余裕をもって捉えていれば前進、接近、横行から確率で選択
	5. 主目標を味方が余裕をもって捉えておらず、自身で余裕をもって捉えていなければ確定で接近
	6. 主目標を味方が余裕をもって捉えておらず、自身で余裕をもって捉えていれば横行または接近から確率で選択
	Returns:
		bool: Trueのとき、もう一度回す。
	*/
    const nl::json& fObs=parent->observables;
    bool reconsider=manager->getTime()-lastNormalTransitionTime>=reconsiderationCycle;
    //1. 帰還できなくなりそう
    if(!chkFuelSufficiency()){
        if(transitionCause!=TransitionCause::FUEL_SHORTAGE || reconsider){
            transitionCause=TransitionCause::FUEL_SHORTAGE;
            immidiateTransition(State::RTB);
        }
        return false;
    }
    //2. 見えている相手がいない
    if(allEnemies.size()==0){
        if(transitionCause!=TransitionCause::NO_ENEMY || reconsider){
            transitionCause=TransitionCause::NO_ENEMY;
            immidiateTransition(State::ADVANCE);
        }
        return false;
    }
    //3. ライン突破を優先 or 突破阻止を優先
    double dist=dLine-pos.dot(forward);
    double angle=M_PI;
    bool tryBreak=false;
    for(auto& t:allEnemies){
        Eigen::Vector3d tpos=t.pos(getLocalCRS());
        angle=std::min(angle,acos(std::max(-1.0,std::min(1.0,(tpos-pos).block<2,1>(0,0,2,1).normalized().dot(forward.block<2,1>(0,0,2,1))))));
    }
    tryBreak=dist<dPrioritizedAdvance*dLine || angle>thetaPrioritizedAdvance;
    if(isAssignedAsInterceptor){
        if(tryBreak){
            double eDist=dLine-target.pos(getLocalCRS()).dot(-forward);
            if(eDist<dist){
                //相手の方が先に突破する可能性が高い場合、突破阻止を優先
                if(transitionCause!=TransitionCause::INTERCEPT_LEAKER || reconsider){
                    transitionCause=TransitionCause::INTERCEPT_LEAKER;
                    immidiateTransition(State::APPROACH_TARGET);
                }
                return false;
            }
        }
    }
    if(tryBreak){
        //自分の方が先に突破する可能性が高い場合、突破を優先
        if(transitionCause!=TransitionCause::TRY_BREAK || reconsider){
            transitionCause=TransitionCause::TRY_BREAK;
            immidiateTransition(State::ADVANCE);
        }
        return false;
    }
    //4〜6
    if(target.nonLimitTrackers.size()>=2){
        //4. 味方が余裕をもって捉えている
        if(transitionCause!=TransitionCause::TRACKING_BY_ALLY || reconsider){
            std::vector<nl::json> candidates={State::ADVANCE,State::APPROACH_TARGET,State::KEEP_SENSING};
            Eigen::VectorXd probs(3);
            probs<<pAdvanceAlly,pApproachAlly,pKeepSensingAlly;
            probs/=probs.sum();
            nl::json config={
                {"type","choice"},
                {"weights",probs},
                {"candidates",candidates}
            };
            auto nextState=getValueFromJsonR(config,randomGen).get<State>();
            transitionCause=TransitionCause::TRACKING_BY_ALLY;
            immidiateTransition(nextState);
        }
        return false;
    }else if(target.nonLimitTrackers.size()==1){
        if(target.nonLimitTrackers[0]!=parent->getFullName()){
            //4. 味方が余裕をもって捉えている
            if(transitionCause!=TransitionCause::TRACKING_BY_ALLY || reconsider){
                std::vector<nl::json> candidates={State::ADVANCE,State::APPROACH_TARGET,State::KEEP_SENSING};
                Eigen::VectorXd probs(3);
                probs<<pAdvanceAlly,pApproachAlly,pKeepSensingAlly;
                probs/=probs.sum();
                nl::json config={
                    {"type","choice"},
                    {"weights",probs},
                    {"candidates",candidates}
                };
                auto nextState=getValueFromJsonR(config,randomGen).get<State>();
                transitionCause=TransitionCause::TRACKING_BY_ALLY;
                immidiateTransition(nextState);
            }
            return false;
        }else{
            //5. 自分だけが余裕をもって捉えている
            if(transitionCause!=TransitionCause::TRACKING_BY_MYSELF || reconsider){
                std::vector<nl::json> candidates={State::APPROACH_TARGET,State::KEEP_SENSING};
                Eigen::VectorXd probs(2);
                probs<<pApproachMyself,pKeepSensingMyself;
                probs/=probs.sum();
                nl::json config={
                    {"type","choice"},
                    {"weights",probs},
                    {"candidates",candidates}
                };
                auto nextState=getValueFromJsonR(config,randomGen).get<State>();
                transitionCause=TransitionCause::TRACKING_BY_MYSELF;
                immidiateTransition(nextState);
            }
            return false;
        }
    }else{
        //6. 誰も余裕をもって捉えていない
        if(transitionCause!=TransitionCause::TRACKING_BY_NOBODY || reconsider){
            transitionCause=TransitionCause::TRACKING_BY_NOBODY;
            immidiateTransition(State::APPROACH_TARGET);
        }
        return false;
    }
}
void BasicAACRuleBasedAgent01::deploy(py::object action){
    launchFlag=false;
    updateMyInfo();
    updateTracks(true);
    updateTargetStatus(true);
    selectTarget();
    bool again=deploySub();
    while(again){
        again=deploySub();
    }
    isAfterDeploy=true;
}

bool BasicAACRuleBasedAgent01::deploySub(){
	/*
	遷移待機の終了判定⇛回避判定⇛離脱判定⇛射撃判定⇛通常遷移の順に判定
	特に遅延遷移の問題で、一度のdeployで二周以上回すこともある。
	Returns:
		bool: Trueのとき、もう一度回す。
	*/
	//遷移待ち中
	if(waitingTransition){
		if(chkTransitionCount()){//遷移待ち時間の経過
			//一旦遷移を完了させてもう一度回す。
			completeTransition();
			return true;
        }
    }
	//回避の判定(回避は最優先)
	if(chkMWS()){
		if(state!=State::EVADE){
			immidiateTransition(State::EVADE);
        }
		return false;
    }
	//離脱の判定(離脱する場合も射撃は行うためすぐにreturnはしない)
    bool needToWithdraw=false;
	bool isWithdrawing= state==State::WITHDRAW || (waitingTransition && transitionAfter==State::WITHDRAW);
	if(isWithdrawing){
		needToWithdraw=true;
        Time t=manager->getTime();
		if(t>=withdrawnTime+tWithdraw){
			needToWithdraw=false;
        }
    }
	std::vector<bool> bs=chkBreak();
    std::vector<bool> breakFlag;
    bool bAny=false;
    for(auto&& b:bs){
        breakFlag.push_back(b);
        bAny=bAny||b;
    }
    if(bAny){
		if(state!=State::WITHDRAW){
			withdrawnTrigger.clear();
            for(int i=0;i<allEnemies.size();++i){
                if(breakFlag[i]){
                    withdrawnTrigger.push_back(allEnemies[i].copy());
                }
            }
			immidiateTransition(State::WITHDRAW);
        }
		withdrawnTime=manager->getTime(); //時間経過による離脱終了の起点の時刻だけは更新
		needToWithdraw=true;
    }
	//射撃の判定
	if(chkLaunchable()){//自機が射撃可能状態
        std::vector<std::pair<bool,std::pair<TrackInfo,double>>> res;
        for(auto&& t:track){
            res.push_back(chkConditionForShoot(t));
        }
        std::sort(res.begin(),res.end(),
            [](const std::pair<bool,std::pair<TrackInfo,double>>& lhs,const std::pair<bool,std::pair<TrackInfo,double>>& rhs){
            if(lhs.first){
                if(rhs.first){
                    return lhs.second.second<rhs.second.second;
                }else{
                    return true;
                }
            }else{
                if(rhs.first){
                    return false;
                }else{
                    return lhs.second.second<rhs.second.second;
                }
            }
        });
        if(res.size()>0 && res[0].first){
            decideShoot(res[0].second.first);
        }
    }
	if(needToWithdraw){
		return false;
    }
	//離脱も回避もしなかった場合、通常時の行動選択を実施
	return chooseTransitionNormal();
}

void BasicAACRuleBasedAgent01::control(){
	/*stateに応じた制御出力を決定する。
	* stateに応じて進みたい方向dstDirと進みたい速さdstVを計算する。
    * dstDirに対しては、現時点の進行方向から導出される回転方向に回転するような角速度を発生させる。
	* dstVに対しては、現時点の速度から導出される加速度を発生させる。
	* 高度維持、マルチセンシングもこの関数内でdstDirに補正をかけることで行う。
	* 目標への接近はその瞬間に目標がいる方向へ最大速度とする。
	*/
    if(!isAfterDeploy){
		updateMyInfo();//自機情報の更新
		updateTracks(false);//航跡情報の更新
		updateTargetStatus(false);//目標情報の更新
    }
    isAfterDeploy=false;
	MotionState myMotion=MotionState(parent->observables.at("motion")).transformTo(getLocalCRS());
    Eigen::Vector3d ex=myMotion.dirBtoP(Eigen::Vector3d(1,0,0));
    Eigen::Vector3d dstDirH;
    //水平方向の決定
    if(state==State::ADVANCE){
		//前進
		dstDirH=forward;
    }else if(state==State::APPROACH_TARGET){
		//その瞬間に目標がいる方向
		Eigen::Vector3d tpos=target.pos(getLocalCRS());
		dstDirH=tpos-pos;
        dstDirH(2)=0;
        dstDirH.normalize();
    }else if(state==State::KEEP_SENSING){
		//横行ははじめから複数捉えることを考慮する
		//主目標方向を正面として、左右どちらに進むかを決める
		//(1)場外に出そうならば内側向き(dOutLimitKeepSensing)
		//(2)場外に出ないならば副目標を多数捉えられる向きを優先
		//(3)副目標が同数ならば前進優先
		Eigen::Vector3d tDir=target.pos(getLocalCRS())-pos;
		double az=atan2(tDir(1),tDir(0));
		Eigen::Vector3d right=Eigen::Vector3d(cos(az+M_PI/2.-thetaKeepSensing),sin(az+M_PI/2.-thetaKeepSensing),0.);
		Eigen::Vector3d left=Eigen::Vector3d(cos(az-M_PI/2.+thetaKeepSensing),sin(az-M_PI/2.+thetaKeepSensing),0.);
		if(abs(pos(0))>=dOut-dOutLimitKeepSensing){
			//(1)
            Eigen::Vector3d inner;
			if(pos(0)>0){
				inner<<-1.,0.,0.;
            }else{
				inner<<1.,0.,0.;
            }
			double r=right.dot(inner);
			double l=left.dot(inner);
			if(r>=l){
				dstDirH=right;
            }else{
				dstDirH=left;
            }
        }else{
			//(2)
			int leftCount=0;
			int rightCount=0;
            if(parent->observables.contains("/spec/sensor/radar"_json_pointer)){
                for(auto& t:additionalTargets){
                    Eigen::Vector3d tpos=t.pos(getLocalCRS());
                    tDir=(tpos-pos).array()*Eigen::Array3d(1,1,0);
                    double sAngle=parent->observables.at("/spec/sensor/radar/thetaFOR"_json_pointer);
                    if(tDir.dot(right)>=cos(sAngle)){
                        rightCount++;
                    }
                    if(tDir.dot(left)>=cos(sAngle)){
                        leftCount++;
                    }
                }
            }
			if(rightCount>leftCount){
				dstDirH=right;
            }else if(leftCount>rightCount){
				dstDirH=left;
            }else{
				//(3)
				double r=right.dot(forward);
				double l=left.dot(forward);
				if(r>=l){
				    dstDirH=right;
                }else{
					dstDirH=left;
                }
            }
        }
    }else if(state==State::RTB){
		//後退
		dstDirH=-forward;
    }else if(state==State::WITHDRAW){
		//基本的には離脱のきっかけとなった敵から最も離れる方位に向かう
        std::vector<Eigen::Vector3d> tpos;
        for(auto& t:allEnemies){
            for(auto& trig:withdrawnTrigger){
                if(t.isSame(trig)){
                    tpos.push_back(t.pos(getLocalCRS())-pos);
                    break;
                }
            }
        }
        if(tpos.size()==0){
			//対象の敵が見えなくなったら真後ろへ
			dstDirH=-forward;
        }else{
            std::vector<double> az;
            for(auto& p:tpos){
                az.push_back(atan2(p(1),p(0)));
            }
            std::sort(az.begin(),az.end());
            az.push_back(az[0]+2*M_PI);
            Eigen::VectorXd delta=Eigen::VectorXd::Zero(az.size()-1);
            for(int i=0;i<az.size()-1;++i){
                delta(i)=az[i+1]-az[i];
            }
            int idx=0;
            delta.minCoeff(&idx);
			double dstAz=(az[idx]+az[idx+1])/2.0;
			dstDirH=Eigen::Vector3d(cos(dstAz),sin(dstAz),0.0);
        }
    }else if(state==State::EVADE){
		//自身の正面に一番近いものに背を向ける。見えなくなっていたら現状維持
		if(chkMWS()){
            std::vector<std::pair<double,Eigen::Vector3d>> tmp;
            for(auto& m:parent->observables.at("/sensor/mws/track"_json_pointer)){
                Eigen::Vector3d dir=m.get<Track2D>().dir(getLocalCRS());
                tmp.push_back(std::make_pair(-dir.dot(myMotion.dirBtoP(Eigen::Vector3d(1,0,0))),-dir));
            }
            Eigen::Vector3d dr=(*std::min_element(tmp.begin(),tmp.end(),
                [](const std::pair<double,Eigen::Vector3d> &lhs,const std::pair<double,Eigen::Vector3d> &rhs){
                return lhs.first<rhs.first;
            })).second;
            dstDirH=dr;
            dstDirH(2)=0;
            dstDirH.normalize();
        }else{
			dstDirH=vel;
            dstDirH(2)=0;
            dstDirH.normalize();
        }
    }else{
		std::cout<<"Invalid state. state="<<magic_enum::enum_name(state)<<std::endl;
    }
	//副目標追尾のための方位補正(前進、接近)
	if(state==State::ADVANCE || state==State::APPROACH_TARGET){
		//前進or接近のときは、一定角度以内の補正で副目標が捉えられるならば、補正する。
		int leftCount=0;
		double leftDelta=0.0;
		int rightCount=0;
		double rightDelta=0.0;
		double az=atan2(dstDirH(1),dstDirH(0));
        if(parent->observables.contains("/spec/sensor/radar"_json_pointer)){
            for(auto& t:additionalTargets){
                Eigen::Vector3d tDir=(t.pos(getLocalCRS())-pos).array()*Eigen::Array3d(1,1,0);
                double delta=asin(std::max(-1.0,std::min(1.0,Eigen::Vector3d(dstDirH.array()*Eigen::Array3d(1,1,0)).cross(tDir)(2))));
                double sAngle=parent->observables.at("/spec/sensor/radar/thetaFOR"_json_pointer);
                double needAngle=(delta>=0) ? std::max(0.0,delta-sAngle) : std::max(0.0,delta+sAngle);
                if(abs(needAngle)<=thetaModForSensing){
                    if(needAngle>=0){
                            rightDelta=std::max(rightDelta,needAngle);
                            rightCount++;
                    }else{
                            leftDelta=std::max(leftDelta,-needAngle);
                            leftCount++;
                    }
                }
            }
        }
		if(leftCount>0 || rightCount>0){
			if(leftCount>rightCount){
				az-=leftDelta;
            }else if(leftCount<rightCount){
				az+=rightDelta;
            }else if(leftDelta>=rightDelta){
				az-=leftDelta;
            }else{
				az+=rightDelta;
            }
        }
		dstDirH<<cos(az),sin(az),0;
    }
    //場外防止のための方位補正
    StaticCollisionAvoider2D avoider;
	//北側
    nl::json c={
        {"p1",Eigen::Vector3d(+dOut,-5*dLine,0)},
        {"p2",Eigen::Vector3d(+dOut,+5*dLine,0)},
        {"infinite_p1",true},
        {"infinite_p2",true},
    	{"isOneSide",true},
        {"inner",Eigen::Vector2d(0.0,0.0)},
        {"limit",dOutLimit},
        {"threshold",dOutLimitThreshold},
		{"adjustStrength",dOutLimitStrength}
    };
    avoider.borders.push_back(std::make_shared<LinearSegment>(c));
	//南側
    c={
        {"p1",Eigen::Vector3d(-dOut,-5*dLine,0)},
        {"p2",Eigen::Vector3d(-dOut,+5*dLine,0)},
        {"infinite_p1",true},
        {"infinite_p2",true},
    	{"isOneSide",true},
        {"inner",Eigen::Vector2d(0.0,0.0)},
        {"limit",dOutLimit},
        {"threshold",dOutLimitThreshold},
		{"adjustStrength",dOutLimitStrength}
    };
    avoider.borders.push_back(std::make_shared<LinearSegment>(c));
    //東側
    c={
        {"p1",Eigen::Vector3d(-5*dOut,+dLine,0)},
        {"p2",Eigen::Vector3d(+5*dOut,+dLine,0)},
        {"infinite_p1",true},
        {"infinite_p2",true},
        {"isOneSide",true},
        {"inner",Eigen::Vector2d(0.0,0.0)},
        {"limit",dOutLimit},
        {"threshold",dOutLimitThreshold},
		{"adjustStrength",dOutLimitStrength}
    };
    avoider.borders.push_back(std::make_shared<LinearSegment>(c));
    //西側
    c={
        {"p1",Eigen::Vector3d(-5*dOut,-dLine,0)},
        {"p2",Eigen::Vector3d(+5*dOut,-dLine,0)},
        {"infinite_p1",true},
        {"infinite_p2",true},
        {"isOneSide",true},
        {"inner",Eigen::Vector2d(0.0,0.0)},
        {"limit",dOutLimit},
        {"threshold",dOutLimitThreshold},
		{"adjustStrength",dOutLimitStrength}
    };
    avoider.borders.push_back(std::make_shared<LinearSegment>(c));
    dstDirH=avoider(myMotion,dstDirH);

    //鉛直方向の決定
    double pitch=0.0;
    if(state==State::EVADE){
		//回避時のダイブ
        altitudeKeeper.minPitch=thetaEvasion;
        altitudeKeeper.maxPitch=thetaNormal;
        dstDir=altitudeKeeper(myMotion,dstDirH,hEvasion);
	}else{
		//回避時以外の高度維持
        double hN;
        if(state==State::RTB){
            altitudeKeeper.minPitch=-RTBPitchLimit;
            altitudeKeeper.maxPitch=RTBPitchLimit;
            hN=RTBAltitude;
        }else if(state==State::WITHDRAW){
            altitudeKeeper.minPitch=-thetaWithdraw;
            altitudeKeeper.maxPitch=thetaWithdraw;
            hN=hWithdraw;
        }else{
            altitudeKeeper.minPitch=-thetaNormal;
            altitudeKeeper.maxPitch=thetaNormal;
            hN=hNormal;
        }
        dstDir=altitudeKeeper(myMotion,dstDirH,hN);
    }
    if(launchedTarget.is_none() || launchedTarget.state==TrackInfo::UpdateState::LOST){
		fireTgt=TrackInfo();
    }else{
		fireTgt=TrackInfo(launchedTarget);
    }
    makeCommand();
}
void BasicAACRuleBasedAgent01::makeCommand(){
    double V=vel.norm();
    MotionState originalMyMotion(parent->observables.at("motion"));
    Eigen::Vector3d ex=originalMyMotion.dirBtoA(Eigen::Vector3d(1,0,0),Eigen::Vector3d(0,0,0),getLocalCRS());
    Eigen::Vector3d exh(ex(0),ex(1),0);
    exh.normalize();
    Eigen::Vector3d dstDirH=dstDir;
    dstDirH(2)=0;
    dstDirH.normalize();
    double pitch=asin(dstDir(2));
    Eigen::Vector3d cmdDstDir=dstDir;
    double cs=exh.dot(dstDirH);
    Eigen::Vector3d turnAxis=exh.cross(dstDirH);
    //回転中に場外に出そうな場合は回転軸を反転
    Eigen::Vector3d outside(0,0,0);
    if(abs(pos(0))>=dOut-dOutLimitTurnAxis){
        if(pos(0)>0){
            //北側
            outside(0)=1;
        }else{
            //南側
            outside(0)=-1;
        }
    }
    if(abs(pos(1))>=dLine-dOutLimitTurnAxis){
        if(pos(1)>0){
            //東側
            outside(1)=1;
        }else{
            //西側
            outside(1)=-1;
        }
    }
    if(outside(0)!=0 || outside(1)!=0){
        if(turnAxis.dot(exh.cross(outside))>=0 && turnAxis.dot(outside.cross(dstDirH))>=0){
            turnAxis=-turnAxis;
        }
    }
    double sn=turnAxis.norm();
    double theta=atan2(sn,cs);
    Eigen::Vector3d side=turnAxis.cross(exh).normalized();
    if(theta>cmdDeltaAzLimit){
        theta=cmdDeltaAzLimit;
        cmdDstDir=(exh*cos(theta)+side*sin(theta)).normalized();
        cmdDstDir<<cmdDstDir(0)*cos(pitch),cmdDstDir(1)*cos(pitch),sin(pitch);
    }
    cmdDstDir=originalMyMotion.dirAtoP(cmdDstDir,pos,getLocalCRS());
    commands[parent->getFullName()]={
        {"motion",{
            {"dstDir",cmdDstDir},
            {"dstThrottle",nominalThrottle}
        }},
        {"weapon",{
            {"launch",launchFlag},
            {"target",fireTgt}
        }}
    };
    double hN=(state==State::RTB ? RTBAltitude : hNormal);
    if(state==State::WITHDRAW){
        hN=hWithdraw;
    }
    observables[parent->getFullName()]["decision"]={
        {"Roll",nl::json::array({"Don't care"})},
        {"Horizontal",nl::json::array({"Az_NED",atan2(dstDir(1),dstDir(0))})},
        {"Vertical",(state==State::EVADE)?nl::json::array({"El",thetaEvasion}):nl::json::array({"Pos",-hN})},
        {"Throttle",nl::json::array({"Throttle",nominalThrottle})},
        {"Fire",nl::json::array({launchFlag,fireTgt})}
    };
    if(state==State::EVADE){
        //回避時はmaxAB
        commands[parent->getFullName()]["motion"].merge_patch(
            {
                {"dstThrottle",1.0}
            }
        );
        observables[parent->getFullName()]["decision"]["Throttle"]=nl::json::array({"Throttle",1.0});
    }else if(state==State::RTB){
        //後退時は速度指定
        commands[parent->getFullName()]["motion"].merge_patch(
            {
                {"dstThrottle",nullptr},
                {"dstV",RTBVelocity}
            }
        );
        observables[parent->getFullName()]["decision"]["Throttle"]=nl::json::array({"Vel",RTBVelocity});
    }else{
        if(V<minimumV){
            velRecovery=true;
        }
        if(V>=minimumRecoveryV){
            velRecovery=false;
        }
        if(velRecovery){
            commands[parent->getFullName()]["motion"].merge_patch(
                {
                    {"dstThrottle",nullptr},
                    {"dstV",minimumRecoveryDstV}
                }
            );
            observables[parent->getFullName()]["decision"]["Throttle"]=nl::json::array({"Vel",minimumRecoveryDstV});
        }

    }
}
double BasicAACRuleBasedAgent01::calcRHead(const Eigen::Vector3d &rs,const Eigen::Vector3d &vs,const Eigen::Vector3d &rt,const Eigen::Vector3d &vt){
    std::shared_ptr<FighterAccessor> f=std::dynamic_pointer_cast<FighterAccessor>(parent);
    return f->getRmax(rs,vs,rt,vt,M_PI,getLocalCRS());
}
double BasicAACRuleBasedAgent01::calcRTail(const Eigen::Vector3d &rs,const Eigen::Vector3d &vs,const Eigen::Vector3d &rt,const Eigen::Vector3d &vt){
    std::shared_ptr<FighterAccessor> f=std::dynamic_pointer_cast<FighterAccessor>(parent);
    return f->getRmax(rs,vs,rt,vt,0.0,getLocalCRS());
}

void exportBasicAACRuleBasedAgent01(py::module &m, const std::shared_ptr<asrc::core::FactoryHelper>& factoryHelper)
{
    TrackInfo::PtrBaseType::addDerivedCppClass<TrackInfo>();

    using namespace pybind11::literals;
    {
    auto cls=expose_common_class<TrackInfo>(m,"TrackInfo")
    .def(py_init<>())
    .def(py_init<const Track3D&,int>())
    .def(py_init<const nl::json&>())
    DEF_FUNC(TrackInfo,copy)
    DEF_FUNC(TrackInfo,update)
    DEF_READWRITE(TrackInfo,idx)
    DEF_READWRITE(TrackInfo,distance)
    DEF_READWRITE(TrackInfo,myRHead)
    DEF_READWRITE(TrackInfo,myRTail)
    DEF_READWRITE(TrackInfo,hisRHead)
    DEF_READWRITE(TrackInfo,hisRTail)
    DEF_READWRITE(TrackInfo,inOurSensor)
    DEF_READWRITE(TrackInfo,inMySensor)
    DEF_READWRITE(TrackInfo,numTracker)
    DEF_READWRITE(TrackInfo,numTrackerLimit)
    DEF_READWRITE(TrackInfo,trackers)
    DEF_READWRITE(TrackInfo,limitTrackers)
    DEF_READWRITE(TrackInfo,nonLimitTrackers)
    DEF_READWRITE(TrackInfo,state)
    DEF_READWRITE(TrackInfo,memoryStartTime)
    ;
    expose_enum_value_helper(
        expose_enum_class<TrackInfo::SensingState>(cls,"SensingState")
        ,"INSIDE"
        ,"LIMIT"
        ,"OUTSIDE"
    );
    expose_enum_value_helper(
        expose_enum_class<TrackInfo::UpdateState>(cls,"UpdateState")
        ,"TRACK"
        ,"MEMORY"
        ,"LOST"
    );
    }

    bind_stl_container<std::vector<TrackInfo>>(m,py::module_local(false));

    {
    auto cls=expose_entity_subclass<BasicAACRuleBasedAgent01>(m,"BasicAACRuleBasedAgent01");
    cls
    DEF_READWRITE(BasicAACRuleBasedAgent01,state)
    ;
    FACTORY_ADD_CLASS(Agent,BasicAACRuleBasedAgent01)

    expose_enum_value_helper(
        expose_enum_class<BasicAACRuleBasedAgent01::TransitionCause>(cls,"TransitionCause")
        ,"FUEL_SHORTAGE"
        ,"NO_ENEMY"
        ,"TRY_BREAK"
        ,"INTERCEPT_LEAKER"
        ,"TRACKING_BY_ALLY"
        ,"TRACKING_BY_NOBODY"
        ,"TRACKING_BY_MYSELF"
        ,"ABNORMAL"
        ,"NONE"
    );
    expose_enum_value_helper(
        expose_enum_class<BasicAACRuleBasedAgent01::State>(cls,"State")
        ,"ADVANCE"
        ,"APPROACH_TARGET"
        ,"KEEP_SENSING"
        ,"RTB"
        ,"WITHDRAW"
        ,"EVADE"
        ,"NONE"
    );
    }
}

ASRC_PLUGIN_NAMESPACE_END
