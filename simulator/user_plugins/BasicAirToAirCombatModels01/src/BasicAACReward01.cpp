// Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#include "BasicAACReward01.h"
#include <algorithm>
#include <ASRCAISim1/Utility.h>
#include <ASRCAISim1/SimulationManager.h>
#include <ASRCAISim1/Asset.h>
#include <ASRCAISim1/Fighter.h>
#include <ASRCAISim1/Agent.h>

ASRC_PLUGIN_NAMESPACE_BEGIN

using namespace asrc::core;
using namespace util;

void BasicAACReward01::initialize(){
    BaseType::initialize();
    debug=getValueFromJsonKRD(modelConfig,"debug",randomGen,false);
    rElim=getValueFromJsonKRD(modelConfig,"rElim",randomGen,0);
    rElimE=getValueFromJsonKRD(modelConfig,"rElimE",randomGen,0);
    rBreakRatio=getValueFromJsonKRD(modelConfig,"rBreakRatio",randomGen,0);
    rBreak=getValueFromJsonKRD(modelConfig,"rBreak",randomGen,0);
    rBreakE=getValueFromJsonKRD(modelConfig,"rBreakE",randomGen,0);
    adjustBreakEnd=getValueFromJsonKRD(modelConfig,"adjustBreakEnd",randomGen,true);
    rTimeup=getValueFromJsonKRD(modelConfig,"rTimeup",randomGen,0);
    rDisq=getValueFromJsonKRD(modelConfig,"rDisq",randomGen,0);
    rDisqE=getValueFromJsonKRD(modelConfig,"rDisqE",randomGen,0);
    rHitRatio=getValueFromJsonKRD(modelConfig,"rHitRatio",randomGen,0);
    rAdvRatio=getValueFromJsonKRD(modelConfig,"rAdvRatio",randomGen,0);
    acceptNegativeAdv=getValueFromJsonKRD(modelConfig,"acceptNegativeAdv",randomGen,true);
    rCrashRatio=getValueFromJsonKRD(modelConfig,"rCrashRatio",randomGen,0);
    rAliveRatio=getValueFromJsonKRD(modelConfig,"rAliveRatio",randomGen,0);
    rOutRatio=getValueFromJsonKRD(modelConfig,"rOutRatio",randomGen,0);
    adjustZerosum=getValueFromJsonKRD(modelConfig,"adjustZerosum",randomGen,true);
}
void BasicAACReward01::serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full){
    BaseType::serializeInternalState(archive,full);

    if(full){
        if(asrc::core::util::isInputArchive(archive)){
            ruler=getShared<BasicAACRuler01>(manager->getRuler());
            assert(ruler);
        }

        ASRC_SERIALIZE_NVP(archive
            ,debug
            ,rElim,rElimE,rBreakRatio,rBreak,rBreakE,rTimeup,rDisq,rDisqE,rHitRatio,rAdvRatio,rCrashRatio,rAliveRatio,rOutRatio
            ,rHit,rHitE,rCrash,rCrashE,rAlive,rAliveE
            ,rHitScale,rHitEScale,rCrashScale,rCrashEScale,rAliveScale,rAliveEScale
            ,rHitPerAircraft,rHitEPerAircraft,rCrashPerAircraft,rCrashEPerAircraft,rAlivePerAircraft,rAliveEPerAircraft
            ,adjustBreakEnd,acceptNegativeAdv,adjustZerosum
            ,opponentName
        )
    }
    ASRC_SERIALIZE_NVP(archive
        ,advPrev
        ,advOffset
        ,breakTime
        ,disqTime
        ,eliminatedTime
    )
}
void BasicAACReward01::debugPrint(const std::string& reason,const std::string& team,double value){
    if(debug){
        std::cout<<"["<<getFactoryModelName()<<","<<manager->getTickCount()<<"] "<<reason<<", "<<team<<", "<<value<<std::endl;
    }
}
void BasicAACReward01::validate(){
    rHitPerAircraft=getValueFromJsonKRD(modelConfig,"rHitPerAircraft",randomGen,ruler->pHitPerAircraft);
    rHitEPerAircraft=getValueFromJsonKRD(modelConfig,"rHitEPerAircraft",randomGen,ruler->pHitPerAircraft);
    rCrashPerAircraft=getValueFromJsonKRD(modelConfig,"rCrashPerAircraft",randomGen,ruler->pCrashPerAircraft);
    rCrashEPerAircraft=getValueFromJsonKRD(modelConfig,"rCrashEPerAircraft",randomGen,ruler->pCrashPerAircraft);
    rAlivePerAircraft=getValueFromJsonKRD(modelConfig,"rAlivePerAircraft",randomGen,ruler->pAlivePerAircraft);
    rAliveEPerAircraft=getValueFromJsonKRD(modelConfig,"rAliveEPerAircraft",randomGen,ruler->pAlivePerAircraft);
}
void BasicAACReward01::onEpisodeBegin(){
    ruler=getShared<BasicAACRuler01>(manager->getRuler());
    assert(ruler);
    this->TeamReward::onEpisodeBegin();
    assert(target.size()<=2);
    for(auto&& team:target){
        assert(team==ruler->westSider || team==ruler->eastSider);
    }
    opponentName.clear();
    opponentName[ruler->westSider]=ruler->eastSider;
    opponentName[ruler->eastSider]=ruler->westSider;
    advPrev.clear();
    advOffset.clear();
    eliminatedTime.clear();
    breakTime.clear();
    disqTime.clear();
    ruler->_setupPDownConfig(rHit,modelConfig,"rHit",0.0);
    ruler->_setupPDownConfig(rHitE,modelConfig,"rHitE",0.0);
    ruler->_setupPDownConfig(rCrash,modelConfig,"rCrash",0.0);
    ruler->_setupPDownConfig(rCrashE,modelConfig,"rCrashE",0.0);
    ruler->_setupPDownConfig(rAlive,modelConfig,"rAlive",0.0);
    ruler->_setupPDownConfig(rAliveE,modelConfig,"rAliveE",0.0);
    ruler->_setupPDownScale(rHitScale,rHit,rHitPerAircraft);
    ruler->_setupPDownScale(rHitEScale,rHitE,rHitEPerAircraft);
    ruler->_setupPDownScale(rCrashScale,rCrash,rCrashPerAircraft);
    ruler->_setupPDownScale(rCrashEScale,rCrashE,rCrashEPerAircraft);
    ruler->_setupPDownScale(rAliveScale,rAlive,rAlivePerAircraft);
    ruler->_setupPDownScale(rAliveEScale,rAliveE,rAliveEPerAircraft);
    for(auto& t:ruler->teams){
        eliminatedTime[t]=-1;
        breakTime[t]=-1;
        disqTime[t]=-1;
        advPrev[t]=(ruler->leadRange[t]-ruler->leadRange[opponentName[t]])/2;
        if(!acceptNegativeAdv){
            advPrev[t]=std::max(0.0,advPrev[t]);
        }else if(adjustZerosum){
            advPrev[t]/=2;
        }
        advOffset[t]=advPrev[t];
    }
}
void BasicAACReward01::onInnerStepEnd(){
    for(auto& team:target){
        //得点計算 1：撃墜による加点(1機あたりpHit点)
        for(auto&& c:ruler->hitCount[team]){
            debugPrint("1-1. Hit("+c.first+")",team,c.second*ruler->getPHit(team,c.first)*(1+rHitRatio));
    	    reward[team]+=c.second*ruler->getPHit(team,c.first)*(1+rHitRatio);
            debugPrint("1-2. Hit("+c.first+")",team,c.second*getRHit(team,c.first));
            reward[team]+=c.second*getRHit(team,c.first);
            debugPrint("1-3. Hit("+c.first+")",opponentName[team],c.second*getRHitE(opponentName[team],c.first));
            reward[opponentName[team]]+=c.second*getRHitE(opponentName[team],c.first);
        }
	    //得点計算 6-(a)：墜落による減点(1機あたりpCrash点)
        for(auto&& c:ruler->crashCount[team]){
            debugPrint("6-(a)-1. Crash("+c.first+")",team,-c.second*ruler->getPCrash(team,c.first)*(1+rCrashRatio));
    	    reward[team]-=c.second*ruler->getPCrash(team,c.first)*(1+rCrashRatio);
            debugPrint("6-(a)-2. Crash("+c.first+")",team,c.second*getRCrash(team,c.first));
            reward[team]+=c.second*getRCrash(team,c.first);
            debugPrint("6-(a)-3. Crash("+c.first+")",opponentName[team],c.second*getRCrashE(opponentName[team],c.first));
            reward[opponentName[team]]+=c.second*getRCrashE(opponentName[team],c.first);
        }
        //得点計算 6-(b)：場外に対する減点(1秒、1kmにつきpOut点)
        if(ruler->outDist[team]>0.0){
            debugPrint("6-(b). Out",team,-(ruler->outDist[team]/1000.)*ruler->pOut*interval[SimPhase::ON_INNERSTEP_END]*manager->getBaseTimeStep()*(1+rOutRatio));
        }
        reward[team]-=(ruler->outDist[team]/1000.)*ruler->pOut*interval[SimPhase::ON_INNERSTEP_END]*manager->getBaseTimeStep()*(1+rOutRatio);
    }
    for(auto& team:ruler->teams){
        //全滅に対する追加報酬
        if(ruler->eliminatedTime[team]>=0 && eliminatedTime[team]<0){
            if(reward.count(team)>0){
                debugPrint("7-1. Elim",team,rElimE);
                reward[team]+=rElimE;
            }
            if(reward.count(opponentName[team])>0){
                debugPrint("7-2. Elim",opponentName[team],rElim);
                reward[opponentName[team]]+=rElim;
            }
            eliminatedTime[team]=manager->getElapsedTime();
        }
        //突破に対する追加報酬
        if(ruler->breakTime[team]>=0 && breakTime[team]<0){
            if(reward.count(team)>0){
                debugPrint("2-1. Break",team,rBreak);
                reward[team]+=rBreak;
                //得点計算 2：突破による加点(pBreak点)
                debugPrint("2-2. Break",team,ruler->pBreak*(1+rBreakRatio));
    		    reward[team]+=ruler->pBreak*(1+rBreakRatio);
            }
            if(reward.count(opponentName[team])>0){
                debugPrint("2-3. Break",opponentName[team],rBreakE);
                reward[opponentName[team]]+=rBreakE;
            }
            breakTime[team]=manager->getElapsedTime();
        }
        //失格に対する追加報酬
        if(ruler->disqTime[team]>=0 && disqTime[team]<0){
            if(reward.count(team)>0){
                debugPrint("8-1. Disq",team,rDisq);
                reward[team]+=rDisq;
            }
            if(reward.count(opponentName[team])>0){
                debugPrint("8-1. Disq",opponentName[team],rDisqE);
                reward[opponentName[team]]+=rDisqE;
            }
            disqTime[team]=manager->getElapsedTime();
        }
    }
}
void BasicAACReward01::onStepEnd(){
    //得点計算 5：進出度合いに対する加点(1kmにつきpAdv点)
    double adv=(ruler->leadRange[ruler->westSider]-ruler->leadRange[ruler->eastSider])/2;
    if(!acceptNegativeAdv){
        adv=std::max(0.0,adv);
    }else if(adjustZerosum){
        adv/=2;
    }
    if(reward.count(ruler->westSider)>0){
        if(ruler->pAdv*(1+rAdvRatio)!=0.0){
            debugPrint("5. Adv",ruler->westSider,(adv-advPrev[ruler->westSider])/1000.*ruler->pAdv*(1+rAdvRatio));
        }
        reward[ruler->westSider]+=(adv-advPrev[ruler->westSider])/1000.*ruler->pAdv*(1+rAdvRatio);
    }
    advPrev[ruler->westSider]=adv;
    adv=(ruler->leadRange[ruler->eastSider]-ruler->leadRange[ruler->westSider])/2;
    if(!acceptNegativeAdv){
        adv=std::max(0.0,adv);
    }else if(adjustZerosum){
        adv/=2;
    }
    if(reward.count(ruler->eastSider)>0){
        if(ruler->pAdv*(1+rAdvRatio)!=0.0){
            debugPrint("5. Adv",ruler->eastSider,(adv-advPrev[ruler->eastSider])/1000.*ruler->pAdv*(1+rAdvRatio));
        }
        reward[ruler->eastSider]+=(adv-advPrev[ruler->eastSider])/1000.*ruler->pAdv*(1+rAdvRatio);
    }
    advPrev[ruler->eastSider]=adv;
    //終了時の帳尻合わせ
    if(ruler->endReason!=BasicAACRuler01::EndReason::NOTYET){
        bool considerAdvantage=false;
    	//終了条件(1)：全機撃墜or墜落
        if(ruler->endReason==BasicAACRuler01::EndReason::ELIMINATION){
            if(breakTime[ruler->westSider]<0 && breakTime[ruler->eastSider]<0){
                //両者未突破の場合、得点計算3及び5の考慮が必要
                if(eliminatedTime[ruler->westSider]>=0 && eliminatedTime[ruler->eastSider]>=0){
                    //両者全滅した場合、得点計算5に従い優勢度を考慮
                    considerAdvantage=true;
                }else{
                    //一方が生存していた場合
                    //得点計算 3：未突破の陣営に「そこから突破して更に帰還可能な」突破判定対象機が存在していれば+pBreak点
                    if(eliminatedTime[ruler->eastSider]<0){//東が生存
                        bool chk=false;
                        for(auto&& e:manager->getAssets([&](const std::shared_ptr<const Asset>& asset)->bool {
                            return asset->isAlive() && asset->getTeam()==ruler->eastSider && isinstance<Fighter>(asset) && ruler->isToBeConsideredForBreak(ruler->eastSider,asset->getFactoryModelName());
                        })){
                            if(ruler->isBreakableAndReturnableToBase(e.lock())){
                                chk=true;
                                break;
                            }
                        }
                        if(chk){
                            if(reward.count(ruler->eastSider)>0){
                                debugPrint("3-1. Break",ruler->eastSider,rBreak);
                                reward[ruler->eastSider]+=rBreak;
                                debugPrint("3-2. Break",ruler->eastSider,ruler->pBreak*(1+rBreakRatio));
    		                    reward[ruler->eastSider]+=ruler->pBreak*(1+rBreakRatio);
                            }
                            if(reward.count(ruler->westSider)>0){
                                debugPrint("3-3. Break",ruler->westSider,rBreakE);
                                reward[ruler->westSider]+=rBreakE;
                            }
                        }
                    }else{//西が生存
                        bool chk=false;
                        for(auto&& e:manager->getAssets([&](const std::shared_ptr<const Asset>& asset)->bool {
                            return asset->isAlive() && asset->getTeam()==ruler->westSider && isinstance<Fighter>(asset) && ruler->isToBeConsideredForBreak(ruler->westSider,asset->getFactoryModelName());
                        })){
                            if(ruler->isBreakableAndReturnableToBase(e.lock())){
                                chk=true;
                                break;
                            }
                        }
                        if(chk){
                            if(reward.count(ruler->westSider)>0){
                                debugPrint("3-1. Break",ruler->westSider,rBreak);
                                reward[ruler->westSider]+=rBreak;
                                debugPrint("3-2. Break",ruler->westSider,ruler->pBreak*(1+rBreakRatio));
    		                    reward[ruler->westSider]+=ruler->pBreak*(1+rBreakRatio);
                            }
                            if(reward.count(ruler->eastSider)>0){
                                debugPrint("3-3. Break",ruler->eastSider,rBreakE);
                                reward[ruler->eastSider]+=rBreakE;
                            }
                        }
                    }
                }
            }
        }
        //終了条件(2)：防衛ラインの突破
        else if(ruler->endReason==BasicAACRuler01::EndReason::BREAK){
            //追加の得点増減はなし
        }
        //終了条件(5)：ペナルティによる敗北
        else if(ruler->endReasonSub==BasicAACRuler01::EndReason::PENALTY){
            if(breakTime[ruler->westSider]<0 && breakTime[ruler->eastSider]<0){
                //両者未突破の場合、得点計算5に従い優勢度を計算
                considerAdvantage=true;
            }
        }
        //終了条件(3)：両者撤退による打ち切り
        else if(ruler->endReason==BasicAACRuler01::EndReason::WITHDRAWAL){
            //追加の得点増減はなし(優勢度も計算しない)
        }
        //終了条件(4)：時間切れ
        else if(manager->getElapsedTime()>=ruler->maxTime){
            if(breakTime[ruler->westSider]<0 && breakTime[ruler->eastSider]<0){
                //両者未突破の場合、得点計算5に従い優勢度を計算
                considerAdvantage=true;
            }
            for(auto&& team:target){
                debugPrint("9. TimeUp",team,rTimeup);
                reward[team]+=rTimeup;
            }
        }
        //得点計算 4：生存点(1機あたりpAlive点)
        //得点計算 6(a)：帰還不可による墜落ペナルティ(1機あたりpCrash点)
        for(auto& team:ruler->teams){
            for(auto&& e:manager->getAssets([&](const std::shared_ptr<const Asset>& asset)->bool {
                return asset->isAlive() && asset->getTeam()==team && isinstance<Fighter>(asset) && ruler->isToBeConsideredForElimination(team,asset->getFactoryModelName());
            })){
                auto asset=e.lock();
                if(ruler->isReturnableToBase(asset)){
                    //帰還可能なら生存点
                    if(reward.count(team)>0){
                        debugPrint("4-1. Alive("+asset->getFactoryModelName()+")",team,ruler->getPAlive(team,asset->getFactoryModelName())*(1+rAliveRatio));
    	                reward[team]+=ruler->getPAlive(team,asset->getFactoryModelName())*(1+rAliveRatio);
                        debugPrint("4-2. Alive("+asset->getFactoryModelName()+")",team,getRAlive(team,asset->getFactoryModelName()));
                        reward[team]+=getRAlive(team,asset->getFactoryModelName());
                    }
                    if(reward.count(opponentName[team])>0){
                        debugPrint("4-3. Alive("+asset->getFactoryModelName()+")",opponentName[team],getRAliveE(opponentName[team],asset->getFactoryModelName()));
                        reward[opponentName[team]]+=getRAliveE(opponentName[team],asset->getFactoryModelName());
                    }
                }else{
                    //帰還不可能なら墜落ペナルティ
                    if(reward.count(team)>0){
                        debugPrint("6-(a)-1. NoReturn("+asset->getFactoryModelName()+")",team,-ruler->getPCrash(team,asset->getFactoryModelName())*(1+rCrashRatio));
    	                reward[team]-=ruler->getPCrash(team,asset->getFactoryModelName())*(1+rCrashRatio);
                        debugPrint("6-(a)-2. NoReturn("+asset->getFactoryModelName()+")",team,getRCrash(team,asset->getFactoryModelName()));
                        reward[team]+=getRCrash(team,asset->getFactoryModelName());
                    }
                    if(reward.count(opponentName[team])>0){
                        debugPrint("6-(a)-3. NoReturn("+asset->getFactoryModelName()+")",opponentName[team],getRCrashE(opponentName[team],asset->getFactoryModelName()));
                        reward[opponentName[team]]+=getRCrashE(opponentName[team],asset->getFactoryModelName());
                    }
                }
            }
        }
        if(adjustBreakEnd && !considerAdvantage){
            //Rulerが進出度合いによる得点を使用しない場合に、即時報酬も無効化する場合
            if(reward.count(ruler->westSider)>0){
                debugPrint("5. Adv(cancel)",ruler->westSider,-(advPrev[ruler->westSider]-advOffset[ruler->westSider])/1000.*ruler->pAdv*(1+rAdvRatio));
                reward[ruler->westSider]-=(advPrev[ruler->westSider]-advOffset[ruler->westSider])/1000.*ruler->pAdv*(1+rAdvRatio);
            }
            if(reward.count(ruler->eastSider)>0){
                debugPrint("5. Adv(cancel)",ruler->eastSider,-(advPrev[ruler->eastSider]-advOffset[ruler->eastSider])/1000.*ruler->pAdv*(1+rAdvRatio));
                reward[ruler->eastSider]-=(advPrev[ruler->eastSider]-advOffset[ruler->eastSider])/1000.*ruler->pAdv*(1+rAdvRatio);
            }
        }
    }
    if(adjustZerosum){
        //ゼロサム化する場合
        std::map<std::string,double> buffer;
        for(auto&& t:target){
            buffer[t]=reward[t];
        }
        for(auto&& t:target){
            if(buffer[opponentName[t]]!=0.0){
                debugPrint("10. ZeroSum",t,-buffer[opponentName[t]]);
            }
            reward[t]-=buffer[opponentName[t]];
        }
    }
    this->TeamReward::onStepEnd();
    for(auto&& t:target){
        if(reward[t]!=0.0){
            debugPrint("reward",t,reward[t]);
            debugPrint("totalReward",t,totalReward[t]);
        }
    }
}
double BasicAACReward01::getRHit(const std::string& team,const std::string& modelName) const{
    return ruler->_getPDownImpl(rHitScale,rHit,team,modelName);
}
double BasicAACReward01::getRHitE(const std::string& team,const std::string& modelName) const{
    return ruler->_getPDownImpl(rHitEScale,rHitE,team,modelName);
}
double BasicAACReward01::getRCrash(const std::string& team,const std::string& modelName) const{
    return ruler->_getPDownImpl(rCrashScale,rCrash,team,modelName);
}
double BasicAACReward01::getRCrashE(const std::string& team,const std::string& modelName) const{
    return ruler->_getPDownImpl(rCrashEScale,rCrashE,team,modelName);
}
double BasicAACReward01::getRAlive(const std::string& team,const std::string& modelName) const{
    return ruler->_getPDownImpl(rAliveScale,rAlive,team,modelName);
}
double BasicAACReward01::getRAliveE(const std::string& team,const std::string& modelName) const{
    return ruler->_getPDownImpl(rAliveEScale,rAliveE,team,modelName);
}

void exportBasicAACReward01(py::module &m, const std::shared_ptr<asrc::core::FactoryHelper>& factoryHelper)
{
    using namespace pybind11::literals;
    expose_entity_subclass<BasicAACReward01>(m,"BasicAACReward01")
    DEF_FUNC(BasicAACReward01,getRHit)
    DEF_FUNC(BasicAACReward01,getRHitE)
    DEF_FUNC(BasicAACReward01,getRCrash)
    DEF_FUNC(BasicAACReward01,getRCrashE)
    DEF_FUNC(BasicAACReward01,getRAlive)
    DEF_FUNC(BasicAACReward01,getRAliveE)
    DEF_READWRITE(BasicAACReward01,rElim)
    DEF_READWRITE(BasicAACReward01,rElimE)
    DEF_READWRITE(BasicAACReward01,rBreakRatio)
    DEF_READWRITE(BasicAACReward01,rBreak)
    DEF_READWRITE(BasicAACReward01,rBreakE)
    DEF_READWRITE(BasicAACReward01,adjustBreakEnd)
    DEF_READWRITE(BasicAACReward01,rTimeup)
    DEF_READWRITE(BasicAACReward01,rDisq)
    DEF_READWRITE(BasicAACReward01,rDisqE)
    DEF_READWRITE(BasicAACReward01,rHitRatio)
    DEF_READWRITE(BasicAACReward01,rHit)
    DEF_READWRITE(BasicAACReward01,rHitE)
    DEF_READWRITE(BasicAACReward01,rAdvRatio)
    DEF_READWRITE(BasicAACReward01,acceptNegativeAdv)
    DEF_READWRITE(BasicAACReward01,rCrashRatio)
    DEF_READWRITE(BasicAACReward01,rCrash)
    DEF_READWRITE(BasicAACReward01,rCrashE)
    DEF_READWRITE(BasicAACReward01,rAliveRatio)
    DEF_READWRITE(BasicAACReward01,rAlive)
    DEF_READWRITE(BasicAACReward01,rAliveE)
    DEF_READWRITE(BasicAACReward01,rOutRatio)
    DEF_READWRITE(BasicAACReward01,adjustZerosum)
    DEF_READWRITE(BasicAACReward01,eliminatedTime)
    DEF_READWRITE(BasicAACReward01,breakTime)
    DEF_READWRITE(BasicAACReward01,disqTime)
    ;
    FACTORY_ADD_CLASS(Reward,BasicAACReward01)
}

ASRC_PLUGIN_NAMESPACE_END
