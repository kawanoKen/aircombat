// Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#include "R7ContestRewardSample01.h"
#include <algorithm>
#include <ASRCAISim1/Utility.h>
#include <ASRCAISim1/Units.h>
#include <ASRCAISim1/SimulationManager.h>
#include <ASRCAISim1/Asset.h>
#include <ASRCAISim1/Fighter.h>
#include <ASRCAISim1/Agent.h>
#include <ASRCAISim1/Ruler.h>

ASRC_PLUGIN_NAMESPACE_BEGIN

using namespace asrc::core;
using namespace util;

void R7ContestRewardSample01::initialize(){
    BaseType::initialize();
    pBite=getValueFromJsonKRD(modelConfig,"pBite",randomGen,+0.0);
    pMemT=getValueFromJsonKRD(modelConfig,"pMemT",randomGen,+0.0);
    pDetect=getValueFromJsonKRD(modelConfig,"pDetect",randomGen,+0.0);
    pVel=getValueFromJsonKRD(modelConfig,"pVel",randomGen,+0.0);
    pOmega=getValueFromJsonKRD(modelConfig,"pOmega",randomGen,+0.0);
    pLine=getValueFromJsonKRD(modelConfig,"pLine",randomGen,+0.0);
    pEnergy=getValueFromJsonKRD(modelConfig,"pEnergy",randomGen,+0.0);
    pLineAsPeak=getValueFromJsonKRD(modelConfig,"pLineAsPeak",randomGen,false);
}
void R7ContestRewardSample01::serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full){
    BaseType::serializeInternalState(archive,full);

    if(full){
        ASRC_SERIALIZE_NVP(archive
            ,pBite,pMemT,pDetect,pVel,pOmega,pLine,pEnergy
            ,pLineAsPeak
            ,westSider,eastSider
            ,dLine
            ,forwardAx
            ,numMissiles
            ,friends,enemies
            ,friendMsls
        )
    }
    ASRC_SERIALIZE_NVP(archive
        ,leadRange
        ,leadRangePrev
        ,biteFlag,memoryTrackFlag
        ,totalEnergy
    )
}
void R7ContestRewardSample01::onEpisodeBegin(){
    j_target="All";
    this->TeamReward::onEpisodeBegin();
    auto ruler_=getShared<Ruler,Ruler>(manager->getRuler());
    auto o=ruler_->observables;
    westSider=o.at("westSider");
    eastSider=o.at("eastSider");
    forwardAx=o.at("forwardAx").get<std::map<std::string,Eigen::Vector2d>>();
    dLine=o.at("dLine");
    friends.clear();
    enemies.clear();
    friendMsls.clear();
    numMissiles.clear();
    biteFlag.clear();
    memoryTrackFlag.clear();
    totalEnergy.clear();
    leadRangePrev.clear();
    leadRange.clear();
    auto crs=ruler_->getLocalCRS();
    for(auto&& team:target){
        friends[team].clear();
        totalEnergy[team]=0;
        leadRangePrev[team]=-dLine;
        for(auto&& e:manager->getAssets([&](const std::shared_ptr<const Asset>& asset)->bool {
            return asset->getTeam()==team && isinstance<Fighter>(asset);
        })){
            auto f=getShared<Fighter>(e);
            friends[team].push_back(f);
            totalEnergy[team]+=f->vel().squaredNorm()/2+gravity*f->getHeight();//TODO 慣性力モデルに置換。
            leadRangePrev[team]=std::max(leadRangePrev[team],forwardAx[team].dot(f->pos(crs).block<2,1>(0,0,2,1)));
        }
        leadRange[team]=leadRangePrev[team];
        enemies[team].clear();
        for(auto&& e:manager->getAssets([&](const std::shared_ptr<const Asset>& asset)->bool {
            return asset->getTeam()!=team && isinstance<Fighter>(asset);
        })){
            auto f=getShared<Fighter>(e);
            enemies[team].push_back(f);
        }
        friendMsls[team].clear();
        for(auto&& e:manager->getAssets([&](const std::shared_ptr<const Asset>& asset)->bool {
            return asset->getTeam()==team && isinstance<Missile>(asset);
        })){
            auto m=getShared<Missile>(e);
            friendMsls[team].push_back(m);
        }
        numMissiles[team]=friendMsls[team].size();
        biteFlag[team]=Eigen::VectorX<bool>::Constant(numMissiles[team],false);
        memoryTrackFlag[team]=Eigen::VectorX<bool>::Constant(numMissiles[team],false);
    }
}
void R7ContestRewardSample01::onInnerStepEnd(){
    nl::json track=nl::json::array();
    auto ruler_=getShared<Ruler,Ruler>(manager->getRuler());
    auto crs=ruler_->getLocalCRS();
    for(auto&& team:target){
		//(1)Biteへの加点、(2)メモリトラック落ちへの減点
        int i=0;
        for(auto&& m_:friendMsls[team]){
            auto m=m_.lock();
            if(m->hasLaunched && m->isAlive()){
                if(m->mode==Missile::Mode::SELF && !biteFlag[team](i)){
					reward[team]+=pBite;
					biteFlag[team](i)=true;
                }
				if(m->mode==Missile::Mode::MEMORY && !memoryTrackFlag[team](i)){
					reward[team]-=pMemT;
					memoryTrackFlag[team](i)=true;
                }
            }
            ++i;
        }
		//(3)敵探知への加点(生存中の敵の何%を探知できているか)(DL前提)
        track=nl::json::array();
        for(auto&& f_:friends[team]){
            if(f_.lock()->isAlive()){
                track=f_.lock()->observables.at("/sensor/track"_json_pointer);
                break;
            }
        }
        int numAlive=0;
        int numTracked=0;
        for(auto&& f_:enemies[team]){
            auto f=f_.lock();
			if(f->isAlive()){
				numAlive+=1;
                for(auto&& t_:track){
                    if(t_.get<Track3D>().isSame(f)){
					    numTracked+=1;
                        break;
                    }
                }
            }
        }
		if(numAlive>0){
			reward[team]+=(1.0*numTracked/numAlive)*pDetect*interval[SimPhase::ON_INNERSTEP_END]*manager->getBaseTimeStep();;
        }
        double ene=0;
        std::vector<double> tmp;
        tmp.push_back(-dLine);
        for(auto&& f_:friends[team]){
            auto f=f_.lock();
            if(f->isAlive()){
			    //(4)過剰な機動への減点(角速度ノルムに対してL2、高度方向速度に対してL1正則化)
                double vAlt=f->motion.relPtoH(f->vel(),"ENU")(2);
			    reward[team]+=(-pVel*abs(vAlt)-(f->omega().squaredNorm())*pOmega)*interval[SimPhase::ON_INNERSTEP_END]*manager->getBaseTimeStep();
			    //(5)前進・後退への更なる加減点
                tmp.push_back(forwardAx[team].dot(f->pos(crs).block<2,1>(0,0,2,1)));
            }
            //(6)保持している力学的エネルギー(回転を除く)の増減による加減点
            //被撃墜・墜落後も状態量としては位置・速度等の値を保持しているためRewardからは参照可能(observableから消えるだけ)
            ene+=f->vel().squaredNorm()/2+gravity*f->getHeight();
        }
        leadRange[team]=*std::max_element(tmp.begin(),tmp.end());
        if(pLineAsPeak){
            //最高到達点で前進の加点をする場合
            if(leadRange[team]>leadRangePrev[team]){
    		    reward[team]+=(leadRange[team]-leadRangePrev[team])*pLine;
                leadRangePrev[team]=leadRange[team];
            }
        }else{
            //都度前進・後退の加減点をする場合
		    reward[team]+=(leadRange[team]-leadRangePrev[team])*pLine;
            leadRangePrev[team]=leadRange[team];
        }
        reward[team]+=(ene-totalEnergy[team])*pEnergy;
        totalEnergy[team]=ene;
    }
}

void exportR7ContestRewardSample01(py::module& m, const std::shared_ptr<asrc::core::FactoryHelper>& factoryHelper)
{
    using namespace pybind11::literals;
    expose_entity_subclass<R7ContestRewardSample01>(m,"R7ContestRewardSample01",py::module_local()) // 投稿時はpy::module_local()を引数に加えてビルドしたものを用いること！
    DEF_READWRITE(R7ContestRewardSample01,pBite)
    DEF_READWRITE(R7ContestRewardSample01,pMemT)
    DEF_READWRITE(R7ContestRewardSample01,pDetect)
    DEF_READWRITE(R7ContestRewardSample01,pVel)
    DEF_READWRITE(R7ContestRewardSample01,pOmega)
    DEF_READWRITE(R7ContestRewardSample01,pLine)
    DEF_READWRITE(R7ContestRewardSample01,pEnergy)
    DEF_READWRITE(R7ContestRewardSample01,westSider)
    DEF_READWRITE(R7ContestRewardSample01,eastSider)
    DEF_READWRITE(R7ContestRewardSample01,forwardAx)
    DEF_READWRITE(R7ContestRewardSample01,numMissiles)
    DEF_READWRITE(R7ContestRewardSample01,biteFlag)
    DEF_READWRITE(R7ContestRewardSample01,memoryTrackFlag)
    DEF_READWRITE(R7ContestRewardSample01,friends)
    DEF_READWRITE(R7ContestRewardSample01,enemies)
    DEF_READWRITE(R7ContestRewardSample01,friendMsls)
    ;
    FACTORY_ADD_CLASS(Reward,R7ContestRewardSample01)
}

ASRC_PLUGIN_NAMESPACE_END
