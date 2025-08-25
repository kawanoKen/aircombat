// Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#include "WinLoseReward.h"
#include <algorithm>
#include "Utility.h"
#include "Units.h"
#include "SimulationManager.h"
#include "Asset.h"
#include "Fighter.h"
#include "Agent.h"
#include "Ruler.h"

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)
using namespace util;

void WinLoseReward::initialize(){
    BaseType::initialize();
    win=getValueFromJsonKRD(modelConfig,"win",randomGen,1.0);
    lose=getValueFromJsonKRD(modelConfig,"lose",randomGen,-1.0);
    draw=getValueFromJsonKRD(modelConfig,"draw",randomGen,0.0);
}
void WinLoseReward::serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full){
    BaseType::serializeInternalState(archive,full);

    if(full){
        ASRC_SERIALIZE_NVP(archive
            ,win
            ,lose
            ,draw
        );

        if(asrc::core::util::isInputArchive(archive)){
            auto ruler_=getShared<Ruler,Ruler>(manager->getRuler());
            auto o=ruler_->observables;
            westSider=o.at("westSider");
            eastSider=o.at("eastSider");
        }
    }
}
void WinLoseReward::onEpisodeBegin(){
    j_target="All";
    this->TeamReward::onEpisodeBegin();
    auto ruler_=getShared<Ruler,Ruler>(manager->getRuler());
    auto o=ruler_->observables;
    westSider=o.at("westSider");
    eastSider=o.at("eastSider");
}
void WinLoseReward::onStepEnd(){
    auto ruler_=getShared<Ruler,Ruler>(manager->getRuler());
    if(ruler_->dones["__all__"]){
        if(ruler_->winner==westSider){
            reward[westSider]+=win;
            reward[eastSider]+=lose;
        }else if(ruler_->winner==eastSider){
            reward[eastSider]+=win;
            reward[westSider]+=lose;
        }else{//Draw
            reward[eastSider]+=draw;
            reward[westSider]+=draw;
        }
    }
    this->TeamReward::onStepEnd();
}

void exportWinLoseReward(py::module &m, const std::shared_ptr<asrc::core::FactoryHelper>& factoryHelper)
{
    using namespace pybind11::literals;
    expose_entity_subclass<WinLoseReward>(m,"WinLoseReward")
    DEF_READWRITE(WinLoseReward,win)
    DEF_READWRITE(WinLoseReward,lose)
    DEF_READWRITE(WinLoseReward,draw)
    ;
    FACTORY_ADD_CLASS(Reward,WinLoseReward)
    ;
}

ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
