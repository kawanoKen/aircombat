// Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#include "GUIDataFrameCreator.h"
#include "SimulationManager.h"
#include "Fighter.h"
#include "Missile.h"
#include "Sensor.h"
#include "Agent.h"
#include "Ruler.h"

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)
ASRC_INLINE_NAMESPACE_BEGIN(util)

GUIDataFrameCreator::GUIDataFrameCreator(){
}
nl::json GUIDataFrameCreator::makeHeader(const std::shared_ptr<SimulationManagerAccessorForCallback>& manager){
    nl::json header={
        {"uuid",manager->getUUID()},
        {"epoch",manager->getEpoch()},
        {"epoch_deltas",Epoch::getDeltas()},
        {"coordinateReferenceSystem",{
            {"root",manager->getRootCRS()}
        }},
        {"factory",manager->getFactoryModelConfig(true)},
        {"worker_index",manager->worker_index()},
        {"vector_index",manager->vector_index()},
        {"episode_index",manager->episode_index()}
    };
    std::map<std::string,EntityConstructionInfo> crsReconstructionInfos;
    std::vector<std::pair<EntityIdentifier,bool>> crsInternalStateIndices;
    std::vector<InternalStateSerializer<std::shared_ptr<CoordinateReferenceSystem>>> crsInternalStates;
    serializedAt.clear();
    for(auto&& e: manager->getCRSes()){
        auto crs=e.lock();
        auto fullname=crs->getFullName();
        auto id=crs->getEntityID();
        if(!crs->isEpisodic()){
            crsReconstructionInfos[fullname]=manager->getEntityConstructionInfo(crs,id);
            crsInternalStateIndices.emplace_back(id,true);
            crsInternalStates.emplace_back(std::move(crs),true);
            serializedAt[id]=manager->getTickCount();
        }
    }
    header["/coordinateReferenceSystem/preset"_json_pointer]=std::move(crsReconstructionInfos);
    header["/coordinateReferenceSystem/internalStateIndices"_json_pointer]=std::move(crsInternalStateIndices);
    header["/coordinateReferenceSystem/internalStates"_json_pointer]=std::move(crsInternalStates);
    tickAtLastSerialization=manager->getTickCount();
    return header;
}
nl::json GUIDataFrameCreator::makeFrame(const std::shared_ptr<SimulationManagerAccessorForCallback>& manager){

    std::vector<EntityConstructionInfo> crsReconstructionInfos;
    std::vector<std::pair<EntityIdentifier,bool>> crsInternalStateIndices;
    std::vector<InternalStateSerializer<std::shared_ptr<CoordinateReferenceSystem>>> crsInternalStates;
    for(auto&& e: manager->getCRSes()){
        auto crs=e.lock();
        auto fullname=crs->getFullName();
        auto id=crs->getEntityID();
        if(serializedAt.find(id)==serializedAt.end()){
            crsReconstructionInfos.push_back(manager->getEntityConstructionInfo(crs,id));
            crsInternalStateIndices.emplace_back(id,true);
            crsInternalStates.emplace_back(std::move(crs),true);
            serializedAt[id]=manager->getTickCount();
        }
    }

    nl::json frame={
        {"time",manager->getTime()},
        {"elapsedTime",manager->getElapsedTime()},
        {"tick",manager->getTickCount()},
        {"ruler",manager->getRuler().lock()->observables},
        {"scores",manager->scores()},
        {"totalRewards",manager->totalRewards()},
        {"agents",nl::json::object()},
        {"fighters",nl::json::object()},
        {"missiles",nl::json::object()},
        {"crses",{
            {"reconstructionInfos",crsReconstructionInfos},
            {"internalStateIndices",crsInternalStateIndices},
            {"internalStates",crsInternalStates},
            {"internalMotionStates",manager->getInternalMotionStatesOfAffineCRSes()}
        }}
    };
    auto rootCRS=manager->getRootCRS();
    for(auto&& e:manager->getAssets()){
        if(isinstance<Fighter>(e)){
            auto f=getShared<Fighter>(e);
            frame["fighters"][f->getFullName()]={
                {"isAlive",f->isAlive()},
                {"target",f->target},
                {"team",f->getTeam()},
                {"name",f->getName()},
                {"pos",f->pos(rootCRS)},
                {"vel",f->vel(rootCRS)},
                {"remMsls",f->remMsls},
                {"motion",f->motion},
                {"ex",f->relBtoA(Eigen::Vector3d(1,0,0),rootCRS)},
                {"ey",f->relBtoA(Eigen::Vector3d(0,1,0),rootCRS)},
                {"ez",f->relBtoA(Eigen::Vector3d(0,0,1),rootCRS)},
                {"observables",f->observables}
            };
            if(!f->agent.expired()){
                frame["fighters"][f->getFullName()]["agent"]=f->agent.lock()->getFullName();
            }
            if(!f->radar.expired()){
                frame["fighters"][f->getFullName()]["radar"]={{"Lref",f->radar.lock()->Lref}};
            }
        }else if(isinstance<Missile>(e)){
            auto m=getShared<Missile>(e);
            frame["missiles"][m->getFullName()]={
                {"isAlive",m->isAlive()},
                {"team",m->getTeam()},
                {"name",m->getName()},
                {"pos",m->pos(rootCRS)},
                {"vel",m->vel(rootCRS)},
                {"motion",m->motion},
                {"ex",m->relBtoA(Eigen::Vector3d(1,0,0),rootCRS)},
                {"ey",m->relBtoA(Eigen::Vector3d(0,1,0),rootCRS)},
                {"ez",m->relBtoA(Eigen::Vector3d(0,0,1),rootCRS)},
                {"hasLaunched",m->hasLaunched},
                {"mode",m->mode},
                {"estTPos",m->estTPos(rootCRS)},
                {"sensor",{
                    {"isActive",m->sensor.lock()->isActive},
                    {"Lref",m->sensor.lock()->Lref},
                    {"thetaFOR",m->sensor.lock()->thetaFOR}
                }}
            };
        }
    }
    for(auto&& e:manager->getAgents()){
        auto a=e.lock();
        frame["agents"][a->getFullName()]={
            {"team",a->getTeam()},
            {"repr",a->repr()},
            {"parents",a->parents},
            {"observables",a->observables}
        };
    }
    tickAtLastSerialization=manager->getTickCount();
    return frame;
}

void GUIDataFrameCreator::serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full){
    if(full){
        ASRC_SERIALIZE_NVP(archive
            ,serializedAt
        );
    }else{
        std::map<EntityIdentifier,std::uint64_t> updated;
        if(isOutputArchive(archive)){
            for(auto&& [id,tick] : serializedAt){
                if(tick > tickAtLastSerialization){
                    updated[id]=tick;
                }
            }
        }
        ASRC_SERIALIZE_NVP(archive
            ,updated
        );
        if(isInputArchive(archive)){
            for(auto&& [id,tick] : updated){
                serializedAt[id]=tick;
            }
        }
    }
}

void exportGUIDataFrameCreator(py::module &m)
{
    using namespace pybind11::literals;
    expose_common_class<GUIDataFrameCreator>(m,"GUIDataFrameCreator")
    .def(py_init<>())
    DEF_FUNC(GUIDataFrameCreator,makeHeader)
    DEF_FUNC(GUIDataFrameCreator,makeFrame)
    DEF_FUNC(GUIDataFrameCreator,serializeInternalState)
    ;
}

ASRC_NAMESPACE_END(util)
ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
