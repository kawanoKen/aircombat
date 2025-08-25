// Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#include <queue>
#include <fstream>
#include <chrono>
#include <future>
#include <atomic>
#include <magic_enum/magic_enum.hpp>
#include "SimulationManager.h"
#include "Utility.h"
#include "Asset.h"
#include "PhysicalAsset.h"
#include "Agent.h"
#include "Controller.h"
#include "CommunicationBuffer.h"
#include "Callback.h"
#include "Ruler.h"
#include "Reward.h"
#include "Viewer.h"

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)
using namespace util;

bool OrderComparer::operator()(const OrderComparer::Type& lhs,const OrderComparer::Type& rhs) const{
    //(nextTick,priority)
    if(lhs.first==rhs.first){
        return lhs.second<rhs.second;
    }else{
        return lhs.first<rhs.first;
    }
}

void SimulationManagerBase::serializeBeforeEntityReconstructionInfo(asrc::core::util::AvailableArchiveTypes & archive, bool full, const nl::json& serializationConfig_){
    BaseType::serializeBeforeEntityReconstructionInfo(archive,full,serializationConfig_);

    // 時刻管理
    if(full){
        if(asrc::core::util::isOutputArchive(archive)){
            call_operator(archive,
                CEREAL_NVP(epoch),
                cereal::make_nvp("epoch_deltas",Epoch::getDeltas())
            );
        }else{
            nl::json epoch_deltas;
            ASRC_SERIALIZE_NVP(archive
                ,epoch
                ,epoch_deltas
            );
            Epoch::setDeltas(epoch_deltas);
        }
    }
    if(asrc::core::util::isOutputArchive(archive)){
        call_operator(archive,cereal::make_nvp("elapsedTime",getElapsedTime()));
    }else{
        double elapsedTime_;
        call_operator(archive,cereal::make_nvp("elapsedTime",elapsedTime_));
    }
}
void SimulationManagerBase::serializeAfterEntityReconstructionInfo(asrc::core::util::AvailableArchiveTypes & archive, bool full, const nl::json& serializationConfig_){
    BaseType::serializeAfterEntityReconstructionInfo(archive,full,serializationConfig_);

    // 座標系管理
    if(full){
        ASRC_SERIALIZE_NVP(archive
            ,crsConfig
            ,rootCRS
        );
    }
    ASRC_SERIALIZE_NVP(archive
        ,crses
    );
}
Epoch SimulationManagerBase::getEpoch() const{
    return epoch;
}
void SimulationManagerBase::setEpoch(const nl::json& j){
    epoch.load_from_json(j);
}
Time SimulationManagerBase::getTime() const{
    return epoch+getElapsedTime();
}
void SimulationManagerBase::setupCRSes(const nl::json& j){
    for(auto it = crses.begin(); it != crses.end();){
        auto [id,crs] = *it;
        it=crses.erase(it);
        removeEntity(crs);
    }
    rootCRS=nullptr;
    crsConfig=j;
    if(j.contains("preset")){
        for(auto& e:j.at("preset").items()){
            std::string name=e.key();
            nl::json elem=e.value();
            elem["isEpisodic"]=false;
            elem["baseName"]="CoordinateReferenceSystem";
            elem["entityFullName"]=name;
            auto crs=createOrGetEntity<CoordinateReferenceSystem>(elem);
            assert(crs);
            crses[crs->getEntityID()]=crs;
        }
    }
    if(j.contains("root")){
        rootCRS=createOrGetEntity<CoordinateReferenceSystem>(j.at("root"));
    }else{
        rootCRS=createEntity<CoordinateReferenceSystem>(nl::json({
            {"isEpisodic",false},
            {"baseName","CoordinateReferenceSystem"},
            {"modelName","Flat_NED"},
            {"instanceConfig",nl::json::object()}
        }));
    }
    if(!rootCRS){
        throw std::runtime_error("rootCRS could not be setup.");
    }
    crses[rootCRS->getEntityID()]=rootCRS;
}
std::vector<std::pair<EntityIdentifier,MotionState>> SimulationManagerBase::getInternalMotionStatesOfAffineCRSes() const{
    std::vector<std::pair<EntityIdentifier,MotionState>> ret;
    for(auto&& [id, crs] : crses){
        if(auto affine=std::dynamic_pointer_cast<AffineCRS>(crs)){
            ret.emplace_back(id,affine->getInternalMotionState());
        }
    }
    return std::move(ret);
}
void SimulationManagerBase::setInternalMotionStatesOfAffineCRSes(const std::vector<std::pair<EntityIdentifier,MotionState>>& states){
    for(auto&& [id, motion] : states){
        auto entity=std::dynamic_pointer_cast<AffineCRS>(getEntityByID(id));
        if(entity){
            entity->updateByMotionState(motion);
        }else{
            throw std::runtime_error("Entity with id="+id.toString()+" is not AffineCRS.");
        }
    }
}
std::shared_ptr<CoordinateReferenceSystem> SimulationManagerBase::getRootCRS() const{
    return rootCRS;
}
nl::json SimulationManagerBase::getCRSConfig() const{
    return crsConfig;
}

SimulationManager::SimulationManager(const nl::json& config_,const std::int32_t& worker_index_,const std::int32_t& vector_index_,
    std::function<nl::json(const nl::json&,const std::int32_t&,const std::int32_t&)> overrider_
):
SimulationManagerBase(vector_index_),
assetPhases{SimPhase::VALIDATE,SimPhase::PERCEIVE,SimPhase::CONTROL,SimPhase::BEHAVE},
agentPhases{SimPhase::AGENT_STEP},
callbackPhases{SimPhase::ON_GET_OBSERVATION_SPACE,SimPhase::ON_GET_ACTION_SPACE,SimPhase::ON_MAKE_OBS,SimPhase::ON_DEPLOY_ACTION,SimPhase::ON_EPISODE_BEGIN,SimPhase::ON_VALIDATION_END,SimPhase::ON_STEP_BEGIN,SimPhase::ON_INNERSTEP_BEGIN,SimPhase::ON_INNERSTEP_END,SimPhase::ON_STEP_END,SimPhase::ON_EPISODE_END},
pool(1)
{
    EntityManager::setWorkerIndex(worker_index_);
    isEpisodeActive=false;
    tickCount=0;
    lastAssetPhase=SimPhase::NONE;
    lastGroup1CallbackPhase=SimPhase::NONE;
    lastGroup2CallbackPhase=SimPhase::NONE;
    baseTimeStep=0.1;
    randomGen=std::mt19937(std::random_device()());
    isConfigured=false;
    isReconfigureRequested=false;
    if(overrider_){
        configOverrider=overrider_;
    }else{
        configOverrider=[](const nl::json& j,const std::int32_t& w,const std::int32_t& v){return j;};
    }
    try{
        nl::json fullConfig=parseConfig(config_);
        fullConfig=configOverrider(
            {
                {"Manager", fullConfig.contains("Manager") ? parseConfig(fullConfig.at("Manager")) : nl::json::object()},
                {"Factory", fullConfig.contains("Factory") ? parseConfig(fullConfig.at("Factory")) : nl::json::object()}
            },
            worker_index(),
            vector_index()
        );
        managerConfig = fullConfig.contains("Manager") ? fullConfig.at("Manager") : nl::json::object();
        if(fullConfig.contains("Factory")){
            addModelsFromJson(fullConfig.at("Factory"));
        }
        if(managerConfig.contains("uuid")){
            managerConfig.at("uuid").get_to(uuid);
        }
        if(managerConfig.contains("episode_index")){
            managerConfig.at("episode_index").get_to(_episode_index);
        }
    }catch(std::exception& ex){
        std::cout<<"In SimulationManager::ctor, parsing the config failed."<<std::endl;
        std::cout<<ex.what()<<std::endl;
        std::cout<<"config_="<<config_<<std::endl;
        std::cout<<"worker_index_="<<worker_index_<<std::endl;
        std::cout<<"vector_index_="<<vector_index_<<std::endl;
        throw ex;
    }
    _isVirtual=false;
}

//
// コンフィグ管理
//
nl::json SimulationManager::getManagerConfig() const{
    return managerConfig;
}
nl::json SimulationManager::getFactoryModelConfig(bool withDefaultModels) const{
    return this->EntityManager::getModelConfigs(withDefaultModels);
}
void SimulationManager::requestReconfigure(const nl::json& fullConfigPatch_){
    isReconfigureRequested=true;
    nl::json fullConfigPatch=parseConfig(fullConfigPatch_);
    if(fullConfigPatch.contains("Manager")){
        managerConfigPatchForReconfigure.merge_patch(parseConfig(fullConfigPatch.at("Manager")));
    }
    if(fullConfigPatch.contains("Factory")){
        factoryConfigPatchForReconfigure.merge_patch(parseConfig(fullConfigPatch.at("Factory")));
    }
}
void SimulationManager::requestReconfigure(const nl::json& managerConfigPatch,const nl::json& factoryConfigPatch){
    isReconfigureRequested=true;
    managerConfigPatchForReconfigure.merge_patch(parseConfig(managerConfigPatch));
    factoryConfigPatchForReconfigure.merge_patch(parseConfig(factoryConfigPatch));
}
nl::json SimulationManager::parseConfig(const nl::json& config_){
    nl::json ret=nl::json::object();
    if(config_.is_object()){
        ret=config_;
    }else if(config_.is_array()){
        for(auto& e:config_){
            if(e.is_object()){
                ret.merge_patch(e);
            }else if(e.is_string()){
                std::ifstream ifs(e.get<std::string>());
                nl::json sub;
                ifs>>sub;
                ret.merge_patch(sub);
            }else{
                throw std::runtime_error("invalid config type.");
            }
        }
    }else{
        throw std::runtime_error("invalid config type.");
    }
    return ret;
}
void SimulationManager::setViewerType(const std::string& viewerType){
    if(viewer){
        removeEntity(viewer);
        viewer=nullptr;
    }
    viewer=createEntity<Viewer>(false,"Viewer",viewerType,nl::json({
        {"entityFullName","Viewer/Viewer"},
    }));
}

void SimulationManager::configure(){
    unsigned int seed_;
    try{
        seed_=managerConfig.at("seed");
    }catch(...){
        seed_=std::random_device()();
    }
    setSeed(seed_);
    tickCount=0;
    lastAssetPhase=SimPhase::NONE;
    lastGroup1CallbackPhase=SimPhase::NONE;
    lastGroup2CallbackPhase=SimPhase::NONE;
    measureTime=getValueFromJsonKRD(managerConfig,"measureTime",randomGen,false);
    numThreads=getValueFromJsonKRD(managerConfig,"numThreads",randomGen,1);
    skipNoAgentStep=getValueFromJsonKRD(managerConfig,"skipNoAgentStep",randomGen,true);
    enableStructuredReward=getValueFromJsonKRD(managerConfig,"enableStructuredReward",randomGen,false);
    pool.reset(numThreads);
    exposeDeadAgentReward=getValueFromJsonKRD(managerConfig,"exposeDeadAgentReward",randomGen,true);
    delayLastObsForDeadAgentUntilAllDone=getValueFromJsonKRD(managerConfig,"delayLastObsForDeadAgentUntilAllDone",randomGen,true);
    setEpoch(getValueFromJsonKRD(managerConfig,"Epoch",randomGen,nl::json::object()));
    setupCRSes(getValueFromJsonKRD(managerConfig,"CoordinateReferenceSystem",randomGen,nl::json::object()));
    baseTimeStep=getValueFromJsonKRD(managerConfig,"/TimeStep/baseTimeStep"_json_pointer,randomGen,0.1);
    defaultAgentStepInterval=getValueFromJsonKRD(managerConfig,"/TimeStep/defaultAgentStepInterval"_json_pointer,randomGen,1);
    try{
        if(managerConfig.contains("ViewerType")){
            setViewerType(managerConfig.at("ViewerType"));
        }else{
            setViewerType("None");
        }
    }catch(std::exception& ex){
        std::cout<<"setup of the Viewer failed."<<std::endl;
        std::cout<<ex.what()<<std::endl;
        throw ex;
    }
    eventHandlers.clear();
    //callbacks.clear();
    //loggers.clear();
    if(ruler){
        removeEntity(ruler);
        ruler=nullptr;
    }
    try{
        if(managerConfig.at("Ruler").is_object()){
            nl::json j=managerConfig.at("Ruler");
            j["isEpisodic"]=false;
            j["baseName"]="Ruler";
            j["/instanceConfig/name"_json_pointer]="Ruler";
            if(!j.contains("entityFullName")){
                j["entityFullName"]="Ruler/Ruler";
            }
            ruler=createEntity<Ruler>(j);
        }else if(managerConfig.at("Ruler").is_string()){
            // 後方互換性維持のため。for backward compatibility
            ruler=createEntity<Ruler>(false,"Ruler",managerConfig.at("Ruler"),nl::json({
                {"entityFullName","Ruler/Ruler"},
            }));
        }
        ruler->validate();
    }catch(std::exception& ex){
        std::cout<<"setup of the Ruler failed."<<std::endl;
        std::cout<<ex.what()<<std::endl;
        throw ex;
    }
    for(auto it = rewardGenerators.begin(); it != rewardGenerators.end();){
        auto r = *it;
        it=rewardGenerators.erase(it);
        removeEntity(r);
    }
    rewardGenerators.clear();
    rewardGroups.clear();
    nl::json subConfig;
    if(managerConfig.contains("Rewards")){
        subConfig=managerConfig.at("Rewards");
    }else{
        subConfig={{{"model","ScoreReward"},{"target","All"}}};
    }
    for(std::size_t i=0;i<subConfig.size();++i){
        nl::json elem=subConfig.at(i);
        try{
            auto rewardGenerator=createEntity<Reward>(false,"Reward",elem.at("model"),nl::json({
                {"entityFullName","Reward/Reward"+std::to_string(i+1)},
                {"target",elem.at("target")}
            }));
            rewardGenerator->validate();
            rewardGenerators.push_back(rewardGenerator);
            std::string group="Default";
            if(elem.contains("group")){
                group=elem.at("group");
            }
            rewardGroups.push_back(group);
        }catch(std::exception& ex){
            std::cout<<"Creation of a reward failed. config="<<elem<<std::endl;
            std::cout<<ex.what()<<std::endl;
            throw ex;
        }
    }
    try{
        assetConfigDispatcher.initialize(managerConfig.at("AssetConfigDispatcher"));
    }catch(std::exception& ex){
        std::cout<<"initialization of assetConfigDispatcher failed."<<std::endl;
        std::cout<<"config="<<managerConfig.at("AssetConfigDispatcher")<<std::endl;
        std::cout<<ex.what()<<std::endl;
        throw ex;
    }
    try{
        agentConfigDispatcher.initialize(managerConfig.at("AgentConfigDispatcher"));
    }catch(std::exception& ex){
        std::cout<<"initialization of agentConfigDispatcher failed."<<std::endl;
        std::cout<<"config="<<managerConfig.at("AgentConfigDispatcher")<<std::endl;
        std::cout<<ex.what()<<std::endl;
        throw ex;
    }
    if(managerConfig.contains("Callbacks")){
        subConfig=managerConfig.at("Callbacks");
    }else{
        subConfig=nl::json::object();
    }
    for(auto& e:subConfig.items()){
        try{
            auto found=callbacks.find(e.key());
            if(found==callbacks.end() || callbacks[e.key()]->acceptReconfigure){
                if(found!=callbacks.end()){
                    callbacks.erase(found);
                    removeEntity(found->second);
                }
                if(e.value().contains("class")){
                    //classでの指定(configはmodelConfig)
                    callbacks[e.key()]=createEntityByClassName<Callback>(false,"Callback",e.value()["class"],e.value()["config"],nl::json({
                        {"entityFullName","Callback/"+e.key()},
                    }));
                }else if(e.value().contains("model")){
                    //modelでの指定(configはinstanceConfig)
                    nl::json ic=e.value()["config"];
                    ic.merge_patch({
                        {"entityFullName","Callback/"+e.key()},
                    });
                    callbacks[e.key()]=createEntity<Callback>(false,"Callback",e.value()["model"],ic);
                }else{
                    throw std::runtime_error("A config for callbacks must contain 'class' or 'model' key.");
                }
                callbacks[e.key()]->validate();
            }
        }catch(std::exception& ex){
            std::cout<<"Creation of a callback failed. config={"<<e.key()<<":"<<e.value()<<"}"<<std::endl;
            std::cout<<ex.what()<<std::endl;
            throw ex;
        }
    }
    if(managerConfig.contains("Loggers")){
        subConfig=managerConfig.at("Loggers");
    }else{
        subConfig=nl::json::object();
    }
    for(auto& e:subConfig.items()){
        try{
            auto found=loggers.find(e.key());
            if(found==loggers.end() || loggers[e.key()]->acceptReconfigure){
                if(found!=loggers.end()){
                    loggers.erase(found);
                    removeEntity(found->second);
                }
                if(e.value().contains("class")){
                    //classでの指定(configはmodelConfig)
                    loggers[e.key()]=createEntityByClassName<Callback>(false,"Callback",e.value()["class"],e.value()["config"],nl::json({
                        {"entityFullName","Logger/"+e.key()},
                    }));
                }else if(e.value().contains("model")){
                    //modelでの指定(configはinstanceConfig)
                    nl::json ic=e.value()["config"];
                    ic.merge_patch({
                        {"entityFullName","Logger/"+e.key()},
                    });
                    loggers[e.key()]=createEntity<Callback>(false,"Callback",e.value()["model"],ic);
                }else{
                    throw std::runtime_error("A config for loggers must contain 'class' or 'model' key.");
                }
                loggers[e.key()]->validate();
            }
        }catch(std::exception& ex){
            std::cout<<"Creation of a logger failed. config={"<<e.key()<<":"<<e.value()<<"}"<<std::endl;
            std::cout<<ex.what()<<std::endl;
            throw ex;
        }
    }
    isConfigured=true;
    isReconfigureRequested=false;
    managerConfigPatchForReconfigure=nl::json();
    factoryConfigPatchForReconfigure=nl::json();
}
void SimulationManager::configure(const nl::json& fullConfig_){
    //replace
    nl::json fullConfig=parseConfig(fullConfig_);
    configure(
        fullConfig.contains("Manager") ? fullConfig.at("Manager") : nl::json::object(),
        fullConfig.contains("Factory") ? fullConfig.at("Factory") : nl::json::object()
    );
}
void SimulationManager::configure(const nl::json& managerConfig_, const nl::json& factoryConfig_){
    //replace
    nl::json fullConfig=configOverrider(
        {
            {"Manager", parseConfig(managerConfig_)},
            {"Factory", parseConfig(factoryConfig_)}
        },
        worker_index(),
        vector_index()
    );
    managerConfig = fullConfig.contains("Manager") ? fullConfig.at("Manager") : nl::json::object();
    clearModels();
    if(fullConfig.contains("Factory")){
        addModelsFromJson(fullConfig.at("Factory"));
    }
    configure();
}
void SimulationManager::reconfigureIfRequested(){
    //merge_patch
    if(isReconfigureRequested){
        reconfigure(managerConfigPatchForReconfigure,factoryConfigPatchForReconfigure);
    }
}
void SimulationManager::reconfigure(const nl::json& fullConfigPatch_){
    //merge_patch
    nl::json fullConfigPatch=parseConfig(fullConfigPatch_);
    reconfigure(
        fullConfigPatch.contains("Manager") ? fullConfigPatch.at("Manager") : nl::json::object(),
        fullConfigPatch.contains("Factory") ? fullConfigPatch.at("Factory") : nl::json::object()
    );
}
void SimulationManager::reconfigure(const nl::json& managerConfigPatch_, const nl::json& factoryConfigPatch_){
    //merge_patch
    managerConfig.merge_patch(parseConfig(managerConfigPatch_));
    nl::json factoryConfig=getFactoryModelConfig(false);
    factoryConfig.merge_patch(parseConfig(factoryConfigPatch_));
    nl::json fullConfig=configOverrider(
        {
            {"Manager", managerConfig},
            {"Factory", factoryConfig}
        },
        worker_index(),
        vector_index()
    );
    managerConfig = fullConfig.contains("Manager") ? fullConfig.at("Manager") : nl::json::object();
    if(fullConfig.contains("Factory")){
        reconfigureModelConfigs(fullConfig.at("Factory"));
    }
    configure();
}

//
// 実行状態管理
//
std::int32_t SimulationManager::worker_index() const{
    return getWorkerIndex();
}
std::int32_t SimulationManager::vector_index() const{
    return getVectorIndex();
}
std::int32_t SimulationManager::episode_index() const{
    return getEpisodeIndex();
}

//
// 内部状態のシリアライゼーション
//
SimulationManager::CommunicationBufferConstructionInfo::CommunicationBufferConstructionInfo(){}
SimulationManager::CommunicationBufferConstructionInfo::CommunicationBufferConstructionInfo(const std::shared_ptr<CommunicationBuffer>& buffer){
    if(buffer){
        j_participants=buffer->j_participants;
        j_inviteOnRequest=buffer->j_inviteOnRequest;
    }
}
void SimulationManager::CommunicationBufferConstructionInfo::serialize(asrc::core::util::AvailableArchiveTypes& archive){
    ASRC_SERIALIZE_NVP(archive
        ,j_participants
        ,j_inviteOnRequest
    );
}
void SimulationManager::serializeBeforeEntityReconstructionInfo(asrc::core::util::AvailableArchiveTypes & archive, bool full, const nl::json& serializationConfig_){
    if(full){
        if(asrc::core::util::isInputArchive(archive)){
            eventHandlers.clear();
        }
    }

    BaseType::serializeBeforeEntityReconstructionInfo(archive,full,serializationConfig_);

    //コンフィグ管理
    // TODO configOverriderはシリアライズ不可
    if(full){
        ASRC_SERIALIZE_NVP(archive
            ,managerConfig
            ,isReconfigureRequested
            ,managerConfigPatchForReconfigure
            ,factoryConfigPatchForReconfigure
        );
    }
    
    ASRC_SERIALIZE_NVP(archive
        ,isConfigured
        ,isEpisodeActive
    );
    if(isConfigured){
        if(full){
            ASRC_SERIALIZE_NVP(archive
                ,numThreads
                ,measureTime
                ,skipNoAgentStep
                ,enableStructuredReward
                ,baseTimeStep
                ,defaultAgentStepInterval
                ,exposeDeadAgentReward
                ,delayLastObsForDeadAgentUntilAllDone
            );
            if(asrc::core::util::isInputArchive(archive)){
                pool.reset(numThreads);

                try{
                    assetConfigDispatcher.initialize(managerConfig.at("AssetConfigDispatcher"));
                }catch(std::exception& ex){
                    std::cout<<"initialization of assetConfigDispatcher failed."<<std::endl;
                    std::cout<<"config="<<managerConfig.at("AssetConfigDispatcher")<<std::endl;
                    std::cout<<ex.what()<<std::endl;
                    throw ex;
                }
                try{
                    agentConfigDispatcher.initialize(managerConfig.at("AgentConfigDispatcher"));
                }catch(std::exception& ex){
                    std::cout<<"initialization of agentConfigDispatcher failed."<<std::endl;
                    std::cout<<"config="<<managerConfig.at("AgentConfigDispatcher")<<std::endl;
                    std::cout<<ex.what()<<std::endl;
                    throw ex;
                }
            }
        }
        ASRC_SERIALIZE_NVP(archive
            ,tickCount
            ,lastAssetPhase
            ,lastGroup1CallbackPhase
            ,lastGroup2CallbackPhase
            ,internalStepCount
            ,exposedStepCount
            ,nextTick
            ,nextTickCB
            ,nextTickP
            ,dones
            ,stepDones
            ,prevDones
            ,scores
            ,rewards
            ,totalRewards
            ,numLearners
            ,numExperts
            ,numClones
            ,manualDone
            ,seedValue
            ,randomGen
        );

        if(measureTime){
            ASRC_SERIALIZE_NVP(archive
                ,maxProcessTime
                ,minProcessTime
                ,processCount
                ,meanProcessTime
            );
        }

        ASRC_SERIALIZE_NVP(archive
            ,rewardGroups
            ,teams
        );
        ASRC_SERIALIZE_NVP(archive
            ,observation
            ,action
            ,lastObservations
            ,observation_space
            ,action_space
        );
    }
}
void SimulationManager::serializeEntityReconstructionInfo(asrc::core::util::AvailableArchiveTypes & archive, bool full, const nl::json& serializationConfig_){
    BaseType::serializeEntityReconstructionInfo(archive,full,serializationConfig_);
    if(isConfigured){
        // CommunicationBuffer
        if(full){
            if(asrc::core::util::isInputArchive(archive)){
                communicationBuffers.clear();
            }
        }
        std::map<std::string,CommunicationBufferConstructionInfo> communicationBufferConstructionInfos;

        if(asrc::core::util::isOutputArchive(archive)){
            for(auto& [name, communicationBuffer] : communicationBuffers){
                bool full_sub=(full || !communicationBuffer->hasBeenSerialized);
                if(full_sub){
                    communicationBufferConstructionInfos[name]=CommunicationBufferConstructionInfo(communicationBuffer);
                };
            }
        }

        ASRC_SERIALIZE_NVP(archive,communicationBufferConstructionInfos);

        if(asrc::core::util::isInputArchive(archive)){
            auto casted_this=std::static_pointer_cast<SimulationManager>(this->shared_from_this());
            for(auto& [name, info] : communicationBufferConstructionInfos){
                communicationBuffers[name]=CommunicationBuffer::create(
                    casted_this,
                    name,
                    info.j_participants,
                    info.j_inviteOnRequest
                );
            }
        }
    }
}
void SimulationManager::serializeAfterEntityReconstructionInfo(asrc::core::util::AvailableArchiveTypes & archive, bool full, const nl::json& serializationConfig_){
    BaseType::serializeAfterEntityReconstructionInfo(archive,full,serializationConfig_);

    if(isConfigured){
        // Entity
        ASRC_SERIALIZE_NVP(archive
            ,assets
            ,viewer
            ,ruler
            ,agents
            ,controllers
            ,callbacks
            ,loggers
            ,rewardGenerators
            ,hasDependencyUpdated
            ,asset_dependency_graph
            ,asset_indices
            ,callback_dependency_graph
            ,callback_indices
            ,asset_process_queue
            ,agent_process_queue
            ,group1_callback_process_queue
            ,group2_callback_process_queue
            ,asset_next_ticks
            ,callback_next_ticks
            ,killRequests
            ,agentsToAct
            ,agentsActed
            ,experts
        );

        // CommunicationBuffer
        std::map<std::string,bool> full_sub_cbuf;
        if(asrc::core::util::isOutputArchive(archive)){
            for(auto&& [name, communicationBuffer] : communicationBuffers){
                full_sub_cbuf[name]=(full || !communicationBuffer->hasBeenSerialized);
            }
        }
        ASRC_SERIALIZE_NVP(archive,full_sub_cbuf);

        std::map<std::string,InternalStateSerializer<std::shared_ptr<CommunicationBuffer>>> communicationBufferInternalStateSerializers;
        for(auto [name, communicationBuffer] : communicationBuffers){
            communicationBufferInternalStateSerializers.emplace(
                std::piecewise_construct,
                std::forward_as_tuple(name),
                std::forward_as_tuple(std::move(communicationBuffer),full_sub_cbuf.at(name))
            );
        }
        call_operator(archive
            ,cereal::make_nvp("communicationBufferInternalStateSerializers",makeInternalStateSerializer(communicationBufferInternalStateSerializers,full))
        );
    }
}
void SimulationManager::serializeAfterEntityInternalStates(asrc::core::util::AvailableArchiveTypes & archive, bool full, const nl::json& serializationConfig_){
    BaseType::serializeAfterEntityInternalStates(archive,full,serializationConfig_);

    if(asrc::core::util::isInputArchive(archive)){
        if(isConfigured){
            // dependency graph
            // 頻繁にロードすることはないと想定し、graph全体を再構築する。
            // 一括でsortされるため、シリアライズ前と順序が変わる可能性がある。
            // そのため、順序を参照する変数はgraph再構築後に新たな順序を反映する必要がある。
            buildDependencyGraph();

            // process queue
            // ロードされたasset_next_ticks, callback_next_ticksは有効なのでそれをもとに復元する
            asset_process_queue.clear();
            agent_process_queue.clear();
            for(auto&& [phase,next_ticks] : asset_next_ticks){
                for(auto&& [id,next_tick] : next_ticks){
                    auto asset=getEntityByID<Asset>(id);
                    auto index=asset_dependency_graph[phase].get_index(asset);
                    if(assetPhases.contains(phase)){
                        asset_process_queue[phase][next_tick][index]=asset;
                    }
                    if(agentPhases.contains(phase)){
                        agent_process_queue[phase][next_tick][index]=std::static_pointer_cast<Agent>(asset);
                    }
                }
            }

            group1_callback_process_queue.clear();
            group2_callback_process_queue.clear();
            for(auto&& [phase,next_ticks] : callback_next_ticks){
                for(auto&& [id,next_tick] : next_ticks){
                    auto callback=getEntityByID<Callback>(id);
                    auto index=callback_dependency_graph[phase].get_index(callback);
                    bool isGroup1 = (
                        isinstance<Ruler>(callback)
                        || isinstance<Reward>(callback)
                    );
                    if(isGroup1){
                        group1_callback_process_queue[phase][next_tick][index]=callback;
                    }else{
                        group2_callback_process_queue[phase][next_tick][index]=callback;
                    }
                }
            }

            // agentsToAct, agentsActed
            // indexの変化を反映する必要がある。
            auto phase=SimPhase::AGENT_STEP;
            {
                auto it=agentsToAct.begin();
                std::map<std::size_t,std::weak_ptr<Agent>> updated;
                while(it!=agentsToAct.end()){
                    auto current_index=it->first;
                    auto new_index=asset_dependency_graph[phase].get_index(it->second);
                    if(new_index!=current_index){
                        updated[new_index]=it->second;
                        it=agentsToAct.erase(it);
                    }else{
                        ++it;
                    }
                }
                agentsToAct.merge(updated);
            }
            {
                auto it=agentsActed.begin();
                std::map<std::size_t,std::weak_ptr<Agent>> updated;
                while(it!=agentsActed.end()){
                    auto current_index=it->first;
                    auto new_index=asset_dependency_graph[phase].get_index(it->second);
                    if(new_index!=current_index){
                        updated[new_index]=it->second;
                        it=agentsActed.erase(it);
                    }else{
                        ++it;
                    }
                }
                agentsActed.merge(updated);
            }

        }
    }
}

//
// 時刻管理
//
double SimulationManager::getElapsedTime() const{
    return baseTimeStep*tickCount;
}
std::uint64_t SimulationManager::getTickCount() const{
    return tickCount;
}
std::uint64_t SimulationManager::getTickCountAt(const Time& time) const{
    double dt=time-epoch;
    if(dt<0){
        throw std::runtime_error("SimulationManager::getTickCountAt(time) can e called only when the 'time' is after the epoch.");
    }
    return std::max<std::uint64_t>(0,std::lround(dt/baseTimeStep));
}
std::uint64_t SimulationManager::getDefaultAgentStepInterval() const{
    return defaultAgentStepInterval;
}
std::uint64_t SimulationManager::getInternalStepCount() const{
    return internalStepCount;
}
std::uint64_t SimulationManager::getExposedStepCount() const{
    return exposedStepCount;
}
std::uint64_t SimulationManager::getAgentStepCount(const std::string& fullName_) const{
    return getAgentStepCount(getAgent(fullName_).lock());
}
std::uint64_t SimulationManager::getAgentStepCount(const std::shared_ptr<Agent>& agent) const{
    return agent->getStepCount();
}
std::uint64_t SimulationManager::getPossibleNextTickCount(const SimPhase& phase) const{
    // 現在の処理状況(tickCountと直近のphase)において、
    // 引数で与えられたphaseが次に実行される可能性のある時刻(tick)を返す。
    // phaseの処理中には呼び出さないこと。
    switch(phase){
        case SimPhase::PERCEIVE:
            //inReset=trueを除くとつねにtickCount+1
            return tickCount+1;
        case SimPhase::CONTROL:
            if(lastAssetPhase==SimPhase::CONTROL || lastAssetPhase==SimPhase::BEHAVE){
                return tickCount+1;
            }else{
                //NONE,VALIDATE,PERCEIVE
                return tickCount;
            }
        case SimPhase::BEHAVE:
            if(lastAssetPhase==SimPhase::BEHAVE){
                return tickCount+1;
            }else{
                //NONE,VALIDATE,PERCEIVE,CONTROL
                return tickCount;
            }
        case SimPhase::AGENT_STEP:
            if(lastAssetPhase==SimPhase::CONTROL || lastAssetPhase==SimPhase::BEHAVE){
                return tickCount+1;
            }else{
                //NONE,VALIDATE,PERCEIVE
                return tickCount;
            }
        case SimPhase::ON_INNERSTEP_BEGIN:
            if(lastAssetPhase==SimPhase::PERCEIVE){
                if(lastGroup1CallbackPhase==SimPhase::ON_INNERSTEP_BEGIN){
                    return tickCount+1;
                }else{
                    return tickCount;
                }
            }else{
                return tickCount+1;
            }
        case SimPhase::ON_INNERSTEP_END:
            if(lastAssetPhase==SimPhase::PERCEIVE){
                // onInnerStepEndはつねにonInnerStepBeginと対になって実行される
                if(lastGroup1CallbackPhase==SimPhase::ON_INNERSTEP_BEGIN){
                    return tickCount;
                }else{
                    return tickCount+1;
                }
            }else{
                return tickCount+1;
            }
        default: // non-periodic phase
            return tickCount;
    }
}

//
//処理順序管理
//
void SimulationManager::addDependency(const SimPhase& phase,const std::shared_ptr<Entity>& predecessor, const std::shared_ptr<Entity>& successor){
    if(isinstance<Asset>(predecessor)){
        assert(asset_dependency_graph.contains(phase));
        assert(
            isinstance<Asset>(successor)
            && asset_dependency_graph.at(phase).contains(std::static_pointer_cast<Asset>(predecessor))
            && asset_dependency_graph.at(phase).contains(std::static_pointer_cast<Asset>(successor))
        );
        hasDependencyUpdated[phase]=asset_dependency_graph.at(phase).add_dependency(
            std::static_pointer_cast<Asset>(predecessor),
            std::static_pointer_cast<Asset>(successor)
        );
    }else if(isinstance<Callback>(predecessor)){
        assert(callback_dependency_graph.contains(phase));
        assert(
            isinstance<Callback>(successor)
            && callback_dependency_graph.at(phase).contains(std::static_pointer_cast<Callback>(predecessor))
            && callback_dependency_graph.at(phase).contains(std::static_pointer_cast<Callback>(successor))
        );
        hasDependencyUpdated[phase]=callback_dependency_graph.at(phase).add_dependency(
            std::static_pointer_cast<Callback>(predecessor),
            std::static_pointer_cast<Callback>(successor)
        );
    }else{
        throw std::runtime_error("The given predecessor is not Asset or Callback.");
    }
}

void SimulationManager::removeDependency(const SimPhase& phase,const std::shared_ptr<Entity>& predecessor, const std::shared_ptr<Entity>& successor){
    if(isinstance<Asset>(predecessor)){
        assert(asset_dependency_graph.contains(phase));
        assert(
            isinstance<Asset>(successor)
            && asset_dependency_graph.at(phase).contains(std::static_pointer_cast<Asset>(predecessor))
            && asset_dependency_graph.at(phase).contains(std::static_pointer_cast<Asset>(successor))
        );
        asset_dependency_graph.at(phase).remove_dependency(
            std::static_pointer_cast<Asset>(predecessor),
            std::static_pointer_cast<Asset>(successor)
        );
    }else if(isinstance<Callback>(predecessor)){
        assert(callback_dependency_graph.contains(phase));
        assert(
            isinstance<Callback>(successor)
            && callback_dependency_graph.at(phase).contains(std::static_pointer_cast<Callback>(predecessor))
            && callback_dependency_graph.at(phase).contains(std::static_pointer_cast<Callback>(successor))
        );
        callback_dependency_graph.at(phase).remove_dependency(
            std::static_pointer_cast<Callback>(predecessor),
            std::static_pointer_cast<Callback>(successor)
        );
    }else{
        throw std::runtime_error("The given predecessor is not Asset or Callback.");
    }
}

void SimulationManager::addDependencyGenerator(std::function<void(const std::shared_ptr<Asset>&)> generator){
    //Agent以外のAsset用
    assetDependencyGenerators.push_back(generator);
}
void SimulationManager::addDependencyGenerator(std::function<void(const std::shared_ptr<Callback>&)> generator){
    //Callback用
    callbackDependencyGenerators.push_back(generator);
}

void SimulationManager::calcFirstTick(const SimPhase& phase,const std::shared_ptr<Entity>& entity){
    updateProcessQueueIfNeeded(phase);
    if(auto asset=std::dynamic_pointer_cast<Asset>(entity)){
        assert(
            assetPhases.contains(phase)
            || (
                agentPhases.contains(phase)
                && isinstance<Agent>(entity)
            )
        );
        std::uint64_t first=asset->getFirstTick(phase);
        if(phase==SimPhase::PERCEIVE && first==0){
            first=asset->getNextTick(phase,first);
            assert(first>0);
        }
        if(isEpisodeActive){
            // episode途中での追加だった場合は初回処理時刻を算出する必要あり
            auto minimum=getPossibleNextTickCount(phase);
            if(first<minimum){
                first=asset->getNextTick(phase,minimum-1); // minimum-1より後の時刻で最初の処理時刻(つまり、minimum以上の値)が得られる
            }
        }else{
            while(first<0){// uint64なのでここには入らないが、今後signedに変更する可能性もあるので残す
                std::uint64_t tmp=asset->getNextTick(phase,first);
                assert(tmp>first);
                first=tmp;
            }
        }
        auto index=asset_dependency_graph.at(phase).get_index(asset);
        if(assetPhases.contains(phase)){
            asset_process_queue[phase][first][index]=asset;
        }
        if(agentPhases.contains(phase)){
            agent_process_queue[phase][first][index]=std::static_pointer_cast<Agent>(asset);
        }
        asset_next_ticks[phase][asset->getEntityID()]=first;
    }else if(auto callback=std::dynamic_pointer_cast<Callback>(entity)){
        assert(
            callbackPhases.contains(phase)
        );
        bool isGroup1 = (
            isinstance<Ruler>(callback)
            || isinstance<Reward>(callback)
        );
        std::uint64_t first=callback->getFirstTick(phase);
        auto index=callback_dependency_graph.at(phase).get_index(callback);
        if(isGroup1){
            group1_callback_process_queue[phase][first][index]=callback;
        }else{
            group2_callback_process_queue[phase][first][index]=callback;
        }
        callback_next_ticks[phase][callback->getEntityID()]=first;
    }else{
        throw std::runtime_error("The given entity is not Asset or Callback.");
    }
}

void SimulationManager::calcNextTick(const SimPhase& phase,const std::shared_ptr<Entity>& entity){
    updateProcessQueueIfNeeded(phase);
    if(auto asset=std::dynamic_pointer_cast<Asset>(entity)){
        assert(
            assetPhases.contains(phase)
            || (
                agentPhases.contains(phase)
                && isinstance<Agent>(entity)
            )
        );
        if(asset_dependency_graph.at(phase).contains(asset)){
            auto index=asset_dependency_graph.at(phase).get_index(asset);
            std::uint64_t next;
            if(asset->isAlive() || phase==SimPhase::AGENT_STEP){
                // AGENT_STEPはaliveでなくても周期処理の対象から外さない(exposeDeadAgentReward等のフラグによりstepのreturnに含まれる可能性があるため)
                // 専用のメンバ変数agentsToAct,agentsActedを用いて制御する
                next=asset->getNextTick(phase,tickCount);
                assert(next>tickCount);
            }else{
                next=std::numeric_limits<std::uint64_t>::max();
            }
            if(assetPhases.contains(phase)){
                asset_process_queue[phase][next][index]=asset;
            }
            if(agentPhases.contains(phase)){
                agent_process_queue[phase][next][index]=std::static_pointer_cast<Agent>(asset);
            }
            asset_next_ticks[phase][asset->getEntityID()]=next;
        }
    }else if(auto callback=std::dynamic_pointer_cast<Callback>(entity)){
        assert(
            callbackPhases.contains(phase)
        );
        bool isGroup1 = (
            isinstance<Ruler>(callback)
            || isinstance<Reward>(callback)
        );
        if(callback_dependency_graph.at(phase).contains(callback)){
            auto index=callback_dependency_graph.at(phase).get_index(callback);
            std::uint64_t next=callback->getNextTick(phase,tickCount);
            if(isGroup1){
                group1_callback_process_queue[phase][next][index]=callback;
            }else{
                group2_callback_process_queue[phase][next][index]=callback;
            }
            callback_next_ticks[phase][callback->getEntityID()]=next;
        }
    }else{
        throw std::runtime_error("The given entity is not Asset or Callback.");
    }
}

void SimulationManager::calcNextTick(const SimPhase& phase){
    updateProcessQueueIfNeeded(phase);
    if(assetPhases.contains(phase)){
        if(asset_process_queue[phase].size()>0){
            nextTick[phase]=asset_process_queue[phase].begin()->first;
        }else{
            nextTick[phase]=std::numeric_limits<std::uint64_t>::max();
        }
    }
    if(agentPhases.contains(phase)){
        if(agent_process_queue[phase].size()>0){
            nextTick[phase]=agent_process_queue[phase].begin()->first;
        }else{
            if(tickCount==0){
                nextTick[phase]=1;//std::numeric_limits<std::uint64_t>::max();
            }else{
                nextTick[phase]=tickCount+defaultAgentStepInterval;//std::numeric_limits<std::uint64_t>::max();
            }
        }
    }
    if(callbackPhases.contains(phase)){
        nextTick[phase]=std::min<std::uint64_t>(
            (
                group1_callback_process_queue[phase].size()>0
                ? group1_callback_process_queue[phase].begin()->first
                : std::numeric_limits<std::uint64_t>::max()
            ),
            (
                group2_callback_process_queue[phase].size()>0
                ? group2_callback_process_queue[phase].begin()->first
                : std::numeric_limits<std::uint64_t>::max()
            )
        );
    }
}

void SimulationManager::buildDependencyGraph(){
    assetDependencyGenerators.clear();
    callbackDependencyGenerators.clear();
    hasDependencyUpdated.clear();
    // instance(vertex)の登録
    asset_dependency_graph.clear();
    for(auto&& phase : assetPhases){
        asset_dependency_graph[phase].auto_reorder=false;

        for(auto&& [fullName, physicalAsset] : assets){
            asset_dependency_graph[phase].add_instance(physicalAsset);
        }
        for(auto&& [fullName, controller] : controllers){
            asset_dependency_graph[phase].add_instance(controller);
        }
        for(auto&& [fullName, agent] : agents){
            asset_dependency_graph[phase].add_instance(agent);
        }
    }
    for(auto&& phase : agentPhases){
        asset_dependency_graph[phase].auto_reorder=false;

        for(auto&& [fullName, agent] : agents){
            asset_dependency_graph[phase].add_instance(agent);
        }
    }

    callback_dependency_graph.clear();
    for(auto&& phase : callbackPhases){
        callback_dependency_graph[phase].auto_reorder=false;

        callback_dependency_graph[phase].add_instance(ruler);
        for(auto&& r : rewardGenerators){
            callback_dependency_graph[phase].add_instance(r);
        }
        for(auto&& [fullName, callback] : callbacks){
            callback_dependency_graph[phase].add_instance(callback);
        }
        callback_dependency_graph[phase].add_instance(viewer);
        for(auto&& [fullName, logger] : loggers){
            callback_dependency_graph[phase].add_instance(logger);
        }
    }

    // dependency(edge)の登録
    for(auto&& [fullName, physicalAsset] : assets){
        physicalAsset->setDependency();
    }
    for(auto&& [fullName, controller] : controllers){
        controller->setDependency();
    }
    for(auto&& [fullName, agent] : agents){
        agent->setDependency();
    }

    ruler->setDependency();
    for(auto&& r : rewardGenerators){
        r->setDependency();
        for(auto&& phase : callbackPhases){
            addDependency(phase,ruler,r);
        }
    }
    for(auto&& [fullName, callback] : callbacks){
        callback->setDependency();
        for(auto&& phase : callbackPhases){
            addDependency(phase,callback,viewer);
        }
    }
    viewer->setDependency();
    for(auto&& [fullName, logger] : loggers){
        logger->setDependency();
        for(auto&& phase : callbackPhases){
            addDependency(phase,viewer,logger);
        }
    }

    for(auto&& generator : assetDependencyGenerators){
        for(auto&& [fullName, physicalAsset] : assets){
            generator(physicalAsset);
        }
        for(auto&& [fullName, controller] : controllers){
            generator(controller);
        }
        for(auto&& [fullName, agent] : agents){
            generator(agent);
        }
    }
    for(auto&& generator : callbackDependencyGenerators){
        generator(ruler);
        for(auto&& r : rewardGenerators){
            generator(r);
        }
        for(auto&& [fullName, callback] : callbacks){
            generator(callback);
        }
        generator(viewer);
        for(auto&& [fullName, logger] : loggers){
            generator(logger);
        }
    }

    // graphのソート
    for(auto&& phase : assetPhases){
        asset_dependency_graph[phase].reorder();
        asset_dependency_graph[phase].auto_reorder=true;
        hasDependencyUpdated[phase]=false;
    }
    for(auto&& phase : agentPhases){
        asset_dependency_graph[phase].reorder();
        asset_dependency_graph[phase].auto_reorder=true;
        hasDependencyUpdated[phase]=false;
    }
    for(auto&& phase : callbackPhases){
        callback_dependency_graph[phase].reorder();
        callback_dependency_graph[phase].auto_reorder=true;
        hasDependencyUpdated[phase]=false;
    }
}

void SimulationManager::buildProcessQueue(){
    asset_process_queue.clear();
    agent_process_queue.clear();
    asset_next_ticks.clear();
    asset_indices.clear();
    for(auto&& phase : assetPhases){
        std::size_t index=0;
        for(auto&& asset : asset_dependency_graph[phase].get_range()){
            asset_indices[phase][asset->getEntityID()]=index;
            calcFirstTick(phase,asset);

            if(measureTime){
                std::string phaseName=enumToStr(phase);
                std::string fullName=asset->getFullName();
                maxProcessTime[phaseName][fullName]=0;
                minProcessTime[phaseName][fullName]=std::numeric_limits<std::uint64_t>::max();
                meanProcessTime[phaseName][fullName]=0;
                processCount[phaseName][fullName]=0;
            }
            ++index;
        }
        calcNextTick(phase);
    }
    for(auto&& phase : agentPhases){
        std::size_t index=0;
        for(auto&& asset : asset_dependency_graph[phase].get_range()){
            asset_indices[phase][asset->getEntityID()]=index;
            calcFirstTick(phase,asset);

            if(measureTime){
                std::string phaseName=enumToStr(phase);
                std::string fullName=asset->getFullName();
                maxProcessTime[phaseName][fullName]=0;
                minProcessTime[phaseName][fullName]=std::numeric_limits<std::uint64_t>::max();
                meanProcessTime[phaseName][fullName]=0;
                processCount[phaseName][fullName]=0;
            }
            ++index;
        }
        calcNextTick(phase);
    }

    agentsActed.clear();
    agentsToAct.clear();
    for(auto&& [index,agent] : agent_process_queue[SimPhase::AGENT_STEP][nextTick[SimPhase::AGENT_STEP]]){
        agentsToAct[index]=agent;
    }

    group1_callback_process_queue.clear();
    group2_callback_process_queue.clear();
    callback_next_ticks.clear();
    callback_indices.clear();
    for(auto&& phase : callbackPhases){
        std::size_t index=0;
        for(auto&& callback : callback_dependency_graph[phase].get_range()){
            callback_indices[phase][callback->getEntityID()]=index;
            calcFirstTick(phase,callback);
            if(measureTime){
                std::string phaseName=enumToStr(phase);
                std::string fullName=callback->getFullName();
                maxProcessTime[phaseName][fullName]=0;
                minProcessTime[phaseName][fullName]=std::numeric_limits<std::uint64_t>::max();
                meanProcessTime[phaseName][fullName]=0;
                processCount[phaseName][fullName]=0;
            }
            ++index;
        }
        calcNextTick(phase);
    }
}

void SimulationManager::updateProcessQueueIfNeeded(const SimPhase& phase){
    if(hasDependencyUpdated[phase]){
        if(assetPhases.contains(phase)){
            auto& graph = asset_dependency_graph[phase];
            if(!graph.is_sorted()){
                graph.reorder();
            }
            //reindex
            for(auto&& [tick, queue] : asset_process_queue.at(phase)){
                auto it=queue.begin();
                std::map<std::size_t,std::shared_ptr<Asset>> updated;
                while(it!=queue.end()){
                    auto current_index=it->first;
                    auto new_index=graph.get_index(it->second);
                    if(new_index!=current_index){
                        updated[new_index]=it->second;
                        asset_indices[phase][it->second->getEntityID()]=new_index;
                        it=queue.erase(it);
                    }else{
                        ++it;
                    }
                }
                queue.merge(updated);
            }
        }
        if(agentPhases.contains(phase)){
            auto& graph = asset_dependency_graph[phase];
            if(!graph.is_sorted()){
                graph.reorder();
            }
            //reindex
            for(auto&& [tick, queue] : agent_process_queue.at(phase)){
                auto it=queue.begin();
                std::map<std::size_t,std::shared_ptr<Agent>> updated;
                while(it!=queue.end()){
                    auto current_index=it->first;
                    auto new_index=graph.get_index(it->second);
                    if(new_index!=current_index){
                        updated[new_index]=it->second;
                        asset_indices[phase][it->second->getEntityID()]=new_index;
                        it=queue.erase(it);
                    }else{
                        ++it;
                    }
                }
                queue.merge(updated);
            }
            if(phase==SimPhase::AGENT_STEP){
                {
                    auto it=agentsToAct.begin();
                    std::map<std::size_t,std::weak_ptr<Agent>> updated;
                    while(it!=agentsToAct.end()){
                        auto current_index=it->first;
                        auto new_index=graph.get_index(it->second);
                        if(new_index!=current_index){
                            updated[new_index]=it->second;
                            it=agentsToAct.erase(it);
                        }else{
                            ++it;
                        }
                    }
                    agentsToAct.merge(updated);
                }
                {
                    auto it=agentsActed.begin();
                    std::map<std::size_t,std::weak_ptr<Agent>> updated;
                    while(it!=agentsActed.end()){
                        auto current_index=it->first;
                        auto new_index=graph.get_index(it->second);
                        if(new_index!=current_index){
                            updated[new_index]=it->second;
                            it=agentsActed.erase(it);
                        }else{
                            ++it;
                        }
                    }
                    agentsActed.merge(updated);
                }
            }
        }
        if(callbackPhases.contains(phase)){
            auto& graph = callback_dependency_graph[phase];
            if(!graph.is_sorted()){
                graph.reorder();
            }
            //reindex
            for(auto&& [tick, queue] : group1_callback_process_queue.at(phase)){
                auto it=queue.begin();
                std::map<std::size_t,std::shared_ptr<Callback>> updated;
                while(it!=queue.end()){
                    auto current_index=it->first;
                    auto new_index=graph.get_index(it->second);
                    if(new_index!=current_index){
                        updated[new_index]=it->second;
                        callback_indices[phase][it->second->getEntityID()]=new_index;
                        it=queue.erase(it);
                    }else{
                        ++it;
                    }
                }
                queue.merge(updated);
            }
            for(auto&& [tick, queue] : group2_callback_process_queue.at(phase)){
                auto it=queue.begin();
                std::map<std::size_t,std::shared_ptr<Callback>> updated;
                while(it!=queue.end()){
                    auto current_index=it->first;
                    auto new_index=graph.get_index(it->second);
                    if(new_index!=current_index){
                        updated[new_index]=it->second;
                        callback_indices[phase][it->second->getEntityID()]=new_index;
                        it=queue.erase(it);
                    }else{
                        ++it;
                    }
                }
                queue.merge(updated);
            }
        }
        hasDependencyUpdated[phase]=false;
    }
}

void SimulationManager::printOrderedAssets(){
    for(auto&& phase : assetPhases){
        std::cout<<"=========================="<<std::endl;
        std::cout<<"asset_dependency_graph["<<enumToStr(phase)<<"]="<<std::endl;
        for(auto&& asset : asset_dependency_graph[phase].get_range()){
            std::cout<<asset->getFullName()<<std::endl;
            for(auto&& predecessor : asset_dependency_graph[phase].get_predecessors(asset)){
                std::cout<<"    depends on "<<predecessor->getFullName()<<std::endl;
            }
        }
    }
}

//
// エピソード管理
//
void SimulationManager::setSeed(const unsigned int& seed_){
    seedValue=seed_;
    randomGen.seed(seedValue);
    assetConfigDispatcher.seed(seedValue);
    agentConfigDispatcher.seed(seedValue);
}
py::tuple SimulationManager::reset(std::optional<unsigned int> seed_,const std::optional<py::dict>& options){
    if(seed_.has_value()){
        setSeed(seed_.value());
    }
    if(!isConfigured){
        if(isReconfigureRequested){
            reconfigure(managerConfigPatchForReconfigure,factoryConfigPatchForReconfigure);
        }else{
            configure();
        }
    }
    cleanupEpisode(isEpisodeActive);
    tickCount=0;
    lastAssetPhase=SimPhase::NONE;
    internalStepCount=0;
    exposedStepCount=0;
    nextTick.clear();
    eventHandlers.clear();
    communicationBuffers.clear();
    maxProcessTime.clear();
    minProcessTime.clear();
    meanProcessTime.clear();
    processCount.clear();
    reconfigureIfRequested();
    createAssets();
    generateCommunicationBuffers();
    buildDependencyGraph();
    buildProcessQueue();
    isEpisodeActive=true;
    get_observation_space();
    get_action_space();
    runCallbackPhaseFunc(SimPhase::ON_EPISODE_BEGIN,true);
    manualDone=false;
    dones.clear();
    prevDones.clear();
    stepDones.clear();
    gatherDones();
    scores=ruler->score;
    gatherRewards();
    totalRewards.clear();
    gatherTotalRewards();
    for(auto&& comm:communicationBuffers){
        comm.second->validate();
    }
    std::chrono::system_clock::time_point before,after;
    std::chrono::microseconds difference;
    for(auto&& asset : asset_dependency_graph[SimPhase::VALIDATE].get_range()){
        try{
            if(measureTime){
                before=std::chrono::system_clock::now();
            }
            asset->validate();
            if(measureTime){
                std::string name=asset->getFullName();
                after=std::chrono::system_clock::now();
                difference=std::chrono::duration_cast<std::chrono::microseconds>(after-before);
                std::string phaseName=std::string(magic_enum::enum_name(SimPhase::VALIDATE));
                maxProcessTime[phaseName][name]=std::max<std::uint64_t>(maxProcessTime[phaseName][name],difference.count());
                minProcessTime[phaseName][name]=std::min<std::uint64_t>(minProcessTime[phaseName][name],difference.count());
                processCount[phaseName][name]++;
                meanProcessTime[phaseName][name]+=((double)(difference.count())-meanProcessTime[phaseName][name])/processCount[phaseName][name];
            }
        }catch(std::exception& ex){
            std::cout<<"asset.validate() failed. asset="<<asset->getFullName()<<std::endl;
            std::cout<<ex.what()<<std::endl;
            throw ex;
        }
    }
    lastAssetPhase=SimPhase::VALIDATE;
    setupNewEntities();
    runCallbackPhaseFunc(SimPhase::ON_VALIDATION_END,true);
    runCallbackPhaseFunc(SimPhase::ON_VALIDATION_END,false);
    for(auto&& asset : asset_dependency_graph[SimPhase::PERCEIVE].get_range()){
        try{
            if(asset->isAlive()){
                if(measureTime){
                    before=std::chrono::system_clock::now();
                }
                asset->perceive(true);
                if(measureTime){
                    std::string name=asset->getFullName();
                    after=std::chrono::system_clock::now();
                    std::string phaseName=std::string(magic_enum::enum_name(SimPhase::PERCEIVE));
                    difference=std::chrono::duration_cast<std::chrono::microseconds>(after-before);
                    maxProcessTime[phaseName][name]=std::max<std::uint64_t>(maxProcessTime[phaseName][name],difference.count());
                    minProcessTime[phaseName][name]=std::min<std::uint64_t>(minProcessTime[phaseName][name],difference.count());
                    processCount[phaseName][name]++;
                    meanProcessTime[phaseName][name]+=((double)(difference.count())-meanProcessTime[phaseName][name])/processCount[phaseName][name];
                }
            }
        }catch(std::exception& ex){
            std::cout<<"asset.perceive(true) failed. asset="<<asset->getFullName()<<std::endl;
            std::cout<<ex.what()<<std::endl;
            throw ex;
        }
    }
    lastAssetPhase=SimPhase::PERCEIVE;
    setupNewEntities();
    py::gil_scoped_acquire acquire;
    lastObservations.clear();
    auto obs=makeObs();
    runCallbackPhaseFunc(SimPhase::ON_EPISODE_BEGIN,false);
    py::dict infos,info;
    info["score"]=util::todict(ruler->score);
    info["w"]=worker_index();
    info["v"]=vector_index();
    if(dones["__all__"]){
        info["endReason"]=ruler->endReason;
    }
    for(auto&& e:obs){
        infos[e.first]=info;
    }
    auto copy=py::module::import("copy");
    return py::make_tuple(
        copy.attr("deepcopy")(obs),
        infos
    );
}
py::tuple SimulationManager::step(const py::dict& action){
    updateAgentsActed();
    deployAction(action);
    updateAgentsToAct();
    manualDone=false;
    runCallbackPhaseFunc(SimPhase::ON_STEP_BEGIN,true);
    runCallbackPhaseFunc(SimPhase::ON_STEP_BEGIN,false);
    while(tickCount<nextTick[SimPhase::AGENT_STEP]){
        //std::cout<<"tickCount="<<tickCount<<",nextAgentTick="<<nextTick[SimPhase::AGENT_STEP]<<std::endl;
        nextTickCB=std::min(nextTick[SimPhase::ON_INNERSTEP_BEGIN],std::min(nextTick[SimPhase::CONTROL],nextTick[SimPhase::BEHAVE]));
        nextTickP=std::min(nextTick[SimPhase::PERCEIVE],nextTick[SimPhase::ON_INNERSTEP_END]);
        innerStep();
    }
    internalStepCount++;
    runCallbackPhaseFunc(SimPhase::ON_STEP_END,true);
    scores=ruler->score;
    gatherRewards();
    gatherTotalRewards();
    py::gil_scoped_acquire acquire;
    auto obs=makeObs();
    py::dict infos,info;
    info["score"]=util::todict(ruler->score);
    info["w"]=worker_index();
    info["v"]=vector_index();
    if(dones["__all__"]){
        info["endReason"]=ruler->endReason;
    }
    for(auto&& e:obs){
        infos[e.first]=info;
    }
    runCallbackPhaseFunc(SimPhase::ON_STEP_END,false);
    if(dones["__all__"]){
        runCallbackPhaseFunc(SimPhase::ON_EPISODE_END,true);
        runCallbackPhaseFunc(SimPhase::ON_EPISODE_END,false);
        if(measureTime){
            nl::json tmp=minProcessTime;
            std::cout<<"minProcessTime="<<tmp.dump(1)<<std::endl;
            tmp=maxProcessTime;
            std::cout<<"maxProcessTime="<<tmp.dump(1)<<std::endl;
            tmp=meanProcessTime;
            std::cout<<"meanProcessTime="<<tmp.dump(1)<<std::endl;
        }
    }
    if(obs.size()>0 || !skipNoAgentStep || dones["__all__"]){
        exposedStepCount++;
        auto copy=py::module::import("copy");
        if(enableStructuredReward){
            py::dict dictRewards;
            for(auto&& [fullName, val] :rewards){
                dictRewards[fullName.c_str()]=util::todict(val);
            }
            return py::make_tuple(
                copy.attr("deepcopy")(obs),
                dictRewards,
                util::todict(stepDones),//terminated
                util::todict(stepDones),//truncated
                infos
            );
        }else{
            py::dict scalarRewards;
            for(auto&& [fullName, val] :rewards){
                scalarRewards[fullName.c_str()]=val["Default"];
            }
            return py::make_tuple(
                copy.attr("deepcopy")(obs),
                scalarRewards,
                util::todict(stepDones),//terminated
                util::todict(stepDones),//truncated
                infos
            );
        }
    }else{
        return step(py::dict());
    }
}
py::tuple SimulationManager::getReturnsOfTheLastStep(){
    py::dict infos,info;
    info["score"]=util::todict(ruler->score);
    info["w"]=worker_index();
    info["v"]=vector_index();
    if(dones["__all__"]){
        info["endReason"]=ruler->endReason;
    }
    for(auto&& e:observation){
        infos[e.first]=info;
    }
    auto copy=py::module::import("copy");
    if(enableStructuredReward){
        py::dict dictRewards;
        for(auto&& [fullName, val] :rewards){
            dictRewards[fullName.c_str()]=util::todict(val);
        }
        return py::make_tuple(
            copy.attr("deepcopy")(observation),
            dictRewards,
            util::todict(stepDones),//terminated
            util::todict(stepDones),//truncated
            infos
        );
    }else{
        py::dict scalarRewards;
        for(auto&& [fullName, val] :rewards){
            scalarRewards[fullName.c_str()]=val["Default"];
        }
        return py::make_tuple(
            copy.attr("deepcopy")(observation),
            scalarRewards,
            util::todict(stepDones),//terminated
            util::todict(stepDones),//truncated
            infos
        );
    }
}
void SimulationManager::stopEpisodeExternally(void){
    dones["__all__"]=true;
    runCallbackPhaseFunc(SimPhase::ON_EPISODE_END,true);
    runCallbackPhaseFunc(SimPhase::ON_EPISODE_END,false);
}
void SimulationManager::cleanupEpisode(bool increment){
    for(auto it = crses.begin(); it != crses.end();){
        auto [id,crs] = *it;
        if(crs->isEpisodic()){
            it=crses.erase(it);
            removeEntity(crs);
        }else{
            ++it;
        }
    }
    this->EntityManager::cleanupEpisode(increment);
    isEpisodeActive=false;
}
void SimulationManager::runAssetPhaseFunc(const SimPhase& phase){
    //後の処理で各phaseごとの関数呼び出し形式を共通化しておく
    static const std::map<SimPhase,std::function<void(const std::shared_ptr<Asset>&)>> phaseFunc={
        {SimPhase::PERCEIVE,[](const std::shared_ptr<Asset>& a){a->perceive(false);}},
        {SimPhase::CONTROL,[](const std::shared_ptr<Asset>& a){a->control();}},
        {SimPhase::BEHAVE,[](const std::shared_ptr<Asset>& a){a->behave();}}
    };
    assert(phaseFunc.count(phase)>0);
    //Asset::kill()の待機リストを初期化
    killRequests.clear();
    //phaseFunc実行中のエラー監視用フラグ。子スレッドからの例外送出を拾わないため必要。
    std::atomic_bool success=true;

    updateProcessQueueIfNeeded(phase);
    //この時刻のこのphaseで処理すべきAssetの範囲を取得する
    auto itr = asset_process_queue[phase].find(tickCount);
    if(itr==asset_process_queue[phase].end()){
        //この時刻の処理対象なし。
        calcNextTick(phase);
        return;
    }
    auto toBeProcessed = std::move(itr->second);
    asset_process_queue[phase].erase(itr);
    if(numThreads==1){
        //シングルスレッドの場合、先頭から順番に処理すればよい
        std::chrono::system_clock::time_point before,after;
        std::chrono::microseconds difference;
        std::string phaseName=enumToStr(phase);
        for(auto&& [order,asset] : toBeProcessed){
            std::string name=asset->getFullName();
            if(asset->isAlive()){
                try{
                    if(measureTime){
                        before=std::chrono::system_clock::now();
                    }
                    phaseFunc.at(phase)(asset);
                    if(measureTime){
                        after=std::chrono::system_clock::now();
                        difference=std::chrono::duration_cast<std::chrono::microseconds>(after-before);
                        maxProcessTime[phaseName][name]=std::max<std::uint64_t>(maxProcessTime[phaseName][name],difference.count());
                        minProcessTime[phaseName][name]=std::min<std::uint64_t>(minProcessTime[phaseName][name],difference.count());
                        processCount[phaseName][name]++;
                        meanProcessTime[phaseName][name]+=((double)(difference.count())-meanProcessTime[phaseName][name])/processCount[phaseName][name];
                    }
                }catch(std::exception& ex){
                    std::cout<<"asset."<<phaseName<<"() failed. asset="<<name<<std::endl;
                    std::cout<<ex.what()<<std::endl;
                    std::cout<<"w="<<worker_index()<<","<<"v="<<vector_index()<<std::endl;
                    success=false;
                    throw ex;
                }
            }
        }
    }else{
        //並列化する場合、処理できるようになったものから処理を実施
        py::gil_scoped_release release;
        std::set<std::size_t> waitings;
        for(auto&& index : toBeProcessed | std::views::keys){
            waitings.insert(index);
        }

        //処理状況監視用の変数の初期化
        std::map<std::size_t,std::atomic_bool> futures;//各Assetの処理状況を格納する。終了時にtrueとなる。
        std::map<std::size_t,std::vector<std::size_t>> checklists;//各Assetが依存するAssetのindexリスト。trueになった後に何度も確認してしまわないように要素を削除しながら回す。
        for(auto&& index : waitings){
            futures[index]=false;
            checklists[index];
            for(auto&& predecessor : asset_dependency_graph[phase].get_predecessors_by_index(index)){
                if(toBeProcessed.contains(predecessor)){
                    checklists[index].push_back(predecessor);
                }
            }
        }

        //処理を順番に実行する
        auto wit=waitings.begin();
        while(wit!=waitings.end()){
            std::size_t index=*wit;
            bool ready=true;
            auto dit=checklists[index].begin();
            while(dit!=checklists[index].end()){
                if(futures.find(*dit)==futures.end() || futures[*dit]){
                    dit=checklists[index].erase(dit);
                }else{
                    ready=false;
                    dit=checklists[index].end();
                }
            }
            if(ready){
                auto asset=toBeProcessed.at(index);
                std::string name=asset->getFullName();
                if(asset->isAlive()){
                    int w=worker_index();
                    int v=vector_index();
                    pool.detach_task([&futures,&success,phase,index,name,asset,w,v,this]{
                        std::chrono::system_clock::time_point before,after;
                        std::chrono::microseconds difference;
                        try{
                            if(measureTime){
                                before=std::chrono::system_clock::now();
                            }
                            phaseFunc.at(phase)(asset);
                            if(measureTime){
                                std::string phaseName=enumToStr(phase);
                                after=std::chrono::system_clock::now();
                                difference=std::chrono::duration_cast<std::chrono::microseconds>(after-before);
                                maxProcessTime.at(phaseName).at(name)=std::max<std::uint64_t>(maxProcessTime.at(phaseName).at(name),difference.count());
                                minProcessTime.at(phaseName).at(name)=std::min<std::uint64_t>(minProcessTime.at(phaseName).at(name),difference.count());
                                processCount.at(phaseName).at(name)++;
                                meanProcessTime.at(phaseName).at(name)+=((double)(difference.count())-meanProcessTime.at(phaseName).at(name))/processCount.at(phaseName).at(name);
                            }
                        }catch(std::exception& ex){
                            std::cout<<"asset."<<enumToStr(phase)<<"() failed. asset="<<name<<std::endl;
                            std::cout<<ex.what()<<std::endl;
                            std::cout<<"w="<<w<<","<<"v="<<v<<std::endl;
                            success=false;
                            throw ex;
                        }
                        futures[index]=true;
                    });
                }else{
                    //not alive
                    futures[index]=true;
                }
                wit=waitings.erase(wit);
            }else{
                ++wit;
            }
            if(wit==waitings.end()){
                wit=waitings.begin();
            }
            if(!success){
                throw std::runtime_error("SimulationManager::runAssetPhaseFunc("+enumToStr(phase)+") failed.");
            }
        }
        pool.wait();
    }
    lastAssetPhase=phase;
    killAndRemoveAssets();
    setupNewEntities();
    //次回処理時刻の算出
    for(auto&& [index,asset] : toBeProcessed){
        calcNextTick(phase,asset);
    }
    calcNextTick(phase);
}
void SimulationManager::runCallbackPhaseFunc(const SimPhase& phase,bool group1){
    static const std::map<SimPhase,std::function<void(const std::shared_ptr<Callback>&)>> phaseFunc={
        {SimPhase::ON_GET_OBSERVATION_SPACE,[](const std::shared_ptr<Callback>& c){c->onGetObservationSpace();}},
        {SimPhase::ON_GET_ACTION_SPACE,[](const std::shared_ptr<Callback>& c){c->onGetActionSpace();}},
        {SimPhase::ON_MAKE_OBS,[](const std::shared_ptr<Callback>& c){c->onMakeObs();}},
        {SimPhase::ON_DEPLOY_ACTION,[](const std::shared_ptr<Callback>& c){c->onDeployAction();}},
        {SimPhase::ON_EPISODE_BEGIN,[](const std::shared_ptr<Callback>& c){c->onEpisodeBegin();}},
        {SimPhase::ON_VALIDATION_END,[](const std::shared_ptr<Callback>& c){c->onValidationEnd();}},
        {SimPhase::ON_STEP_BEGIN,[](const std::shared_ptr<Callback>& c){c->onStepBegin();}},
        {SimPhase::ON_INNERSTEP_BEGIN,[](const std::shared_ptr<Callback>& c){c->onInnerStepBegin();}},
        {SimPhase::ON_INNERSTEP_END,[](const std::shared_ptr<Callback>& c){c->onInnerStepEnd();}},
        {SimPhase::ON_STEP_END,[](const std::shared_ptr<Callback>& c){c->onStepEnd();}},
        {SimPhase::ON_EPISODE_END,[](const std::shared_ptr<Callback>& c){c->onEpisodeEnd();}}
    };
    assert(phaseFunc.count(phase)>0);
    updateProcessQueueIfNeeded(phase);
    auto& group_queue = group1 ? group1_callback_process_queue.at(phase) : group2_callback_process_queue.at(phase);
    std::chrono::system_clock::time_point before,after;
    std::chrono::microseconds difference;
    std::string phaseName=enumToStr(phase);
    killRequests.clear();
    if(phase==SimPhase::ON_GET_OBSERVATION_SPACE ||
        phase==SimPhase::ON_GET_ACTION_SPACE ||
        phase==SimPhase::ON_MAKE_OBS ||
        phase==SimPhase::ON_DEPLOY_ACTION || 
        phase==SimPhase::ON_EPISODE_BEGIN || 
        phase==SimPhase::ON_EPISODE_END || 
        phase==SimPhase::ON_VALIDATION_END || 
        phase==SimPhase::ON_STEP_BEGIN || 
        phase==SimPhase::ON_STEP_END
    ){
        // 周期処理ではないもの
        for(auto&& callback : callback_dependency_graph[phase].get_range()){
            if(group1 != (isinstance<Ruler>(callback) || isinstance<Reward>(callback))){
                continue;
            }
            std::string name=callback->getFullName();
            try{
                if(measureTime){
                    before=std::chrono::system_clock::now();
                }
                phaseFunc.at(phase)(callback);
                if(measureTime){
                    after=std::chrono::system_clock::now();
                    difference=std::chrono::duration_cast<std::chrono::microseconds>(after-before);
                    maxProcessTime[phaseName][name]=std::max<std::uint64_t>(maxProcessTime[phaseName][name],difference.count());
                    minProcessTime[phaseName][name]=std::min<std::uint64_t>(minProcessTime[phaseName][name],difference.count());
                    processCount[phaseName][name]++;
                    meanProcessTime[phaseName][name]+=((double)(difference.count())-meanProcessTime[phaseName][name])/processCount[phaseName][name];
                }
                if(phase==SimPhase::ON_STEP_END && callback==ruler){
                    //RulerのonStepEndの後、Rewardの計算前にdonesを処理してagentsToActを上書き
                    gatherDones();
                }
            }catch(std::exception& ex){
                std::cout<<"callback."<<magic_enum::enum_name(phase)<<"() failed. callback="<<name<<std::endl;
                std::cout<<ex.what()<<std::endl;
                std::cout<<"w="<<worker_index()<<","<<"v="<<vector_index()<<std::endl;
                throw ex;
            }
        }
    }else{ // SimPhase::ON_INNERSTEP_BEGIN || SimPhase::ON_INNERSTEP_END
        // 周期処理のもの
        //この時刻のこのphaseで処理すべきCallbackの範囲を取得する
        auto itr = group_queue.find(tickCount);
        if(itr==group_queue.end()){
            //この時刻の処理対象なし。
            calcNextTick(phase);
            return;
        }
        auto toBeProcessed = std::move(itr->second);
        group_queue.erase(itr);
        for(auto&& [index,callback] : toBeProcessed){
            std::string name=callback->getFullName();
            try{
                if(measureTime){
                    before=std::chrono::system_clock::now();
                }
                phaseFunc.at(phase)(callback);
                if(measureTime){
                    after=std::chrono::system_clock::now();
                    difference=std::chrono::duration_cast<std::chrono::microseconds>(after-before);
                    maxProcessTime[phaseName][name]=std::max<std::uint64_t>(maxProcessTime[phaseName][name],difference.count());
                    minProcessTime[phaseName][name]=std::min<std::uint64_t>(minProcessTime[phaseName][name],difference.count());
                    processCount[phaseName][name]++;
                    meanProcessTime[phaseName][name]+=((double)(difference.count())-meanProcessTime[phaseName][name])/processCount[phaseName][name];
                }
                calcNextTick(phase,callback);
            }catch(std::exception& ex){
                std::cout<<"callback."<<phaseName<<"() failed. callback="<<name<<std::endl;
                std::cout<<ex.what()<<std::endl;
                std::cout<<"w="<<worker_index()<<","<<"v="<<vector_index()<<std::endl;
                throw ex;
            }
        }
        calcNextTick(phase);
    }
    if(group1){
        lastGroup1CallbackPhase=phase;
    }else{
        lastGroup2CallbackPhase=phase;
    }
    killAndRemoveAssets();
    setupNewEntities();
}
void SimulationManager::innerStep(){
    if(nextTickCB<std::min(nextTickP,nextTick[SimPhase::AGENT_STEP])){
        tickCount=nextTickCB;
        runCallbackPhaseFunc(SimPhase::ON_INNERSTEP_BEGIN,true);
        runCallbackPhaseFunc(SimPhase::ON_INNERSTEP_BEGIN,false);
        runAssetPhaseFunc(SimPhase::CONTROL);
        runAssetPhaseFunc(SimPhase::BEHAVE);
    }else if(nextTickP<=nextTick[SimPhase::AGENT_STEP]){
        tickCount=nextTickP;
        runAssetPhaseFunc(SimPhase::PERCEIVE);
        runCallbackPhaseFunc(SimPhase::ON_INNERSTEP_END,true);
        scores=ruler->score;
        gatherRewards();
        gatherTotalRewards();
        runCallbackPhaseFunc(SimPhase::ON_INNERSTEP_END,false);
    }else{
        tickCount=nextTick[SimPhase::AGENT_STEP];
    }
}
py::dict SimulationManager::makeObs(){
    py::gil_scoped_acquire acquire;
    observation=py::dict();
    for(auto& [index,wAgent] : agentsToAct){
        auto agent=wAgent.lock();
        auto fullName=agent->getFullName();
        try{
            if(agent->isAlive()){
                py::object obs=agent->makeObs();
                lastObservations[fullName.c_str()]=obs;
                observation[fullName.c_str()]=obs;
            }else if(!prevDones[fullName]){
                observation[fullName.c_str()]=lastObservations[fullName.c_str()];
            }
        }catch(std::exception& ex){
            std::cout<<"agent.makeObs() failed. agent="<<fullName<<std::endl;
            std::cout<<ex.what()<<std::endl;
            std::cout<<"w="<<worker_index()<<","<<"v="<<vector_index()<<std::endl;
            throw ex;
        }
    }
    runCallbackPhaseFunc(SimPhase::ON_MAKE_OBS,true);
    runCallbackPhaseFunc(SimPhase::ON_MAKE_OBS,false);
    for(auto& [index,wAgent] : agentsToAct){
        auto agent=wAgent.lock();
        auto fullName=agent->getFullName();
        if(observation.contains(fullName)){
            if(isinstance<ExpertWrapper>(agent)){
                auto ex=getShared<ExpertWrapper>(agent);
                ex->imitatorObs=py::cast<py::tuple>(observation[fullName.c_str()])[0];
                ex->expertObs=py::cast<py::tuple>(observation[fullName.c_str()])[1];
                if(ex->whichExpose=="Expert"){
                    observation[fullName.c_str()]=ex->expertObs;
                }else if(ex->whichExpose=="Imitator"){
                    observation[fullName.c_str()]=ex->imitatorObs;
                }else{//if(ex->whichExpose=="Both"){
                    //そのまま
                }
            }
        }
    }
    return observation;
}
void SimulationManager::updateAgentsActed(){
    agentsActed.clear();
    SimPhase phase=SimPhase::AGENT_STEP;
    updateProcessQueueIfNeeded(phase);
    //この時刻のこのphaseで処理すべきAssetの範囲を取得する
    auto itr = agent_process_queue[phase].find(tickCount);
    if(itr==agent_process_queue[phase].end()){
        //この時刻の処理対象なし。
        calcNextTick(phase);
        return;
    }
    auto toBeProcessed = std::move(itr->second);
    agent_process_queue[phase].erase(itr);
    for(auto&& [index,agent] : toBeProcessed){
        auto fullName=agent->getFullName();
        if(delayLastObsForDeadAgentUntilAllDone && dones[fullName]){
            //全体の終了前にdoneになったAgentはagentsActedに入れない
            continue;
        }
        agentsActed[index]=agent;
    }

    //次回処理時刻の算出
    for(auto&& [index,agent] : toBeProcessed){
        calcNextTick(phase,agent);
    }
    calcNextTick(phase);
}
void SimulationManager::deployAction(const py::dict& action_){
    py::gil_scoped_acquire acquire;
    SimPhase phase=SimPhase::AGENT_STEP;
    action=action_;
    runCallbackPhaseFunc(SimPhase::ON_DEPLOY_ACTION,true);
    runCallbackPhaseFunc(SimPhase::ON_DEPLOY_ACTION,false);

    for(auto&& [index,wAgent] : agentsActed){
        auto agent=wAgent.lock();
        auto fullName=agent->getFullName();
        try{
            if(!dones[fullName] && agent->isAlive()){
                if(action.contains(fullName)){
                    agent->deploy(action[fullName.c_str()]);
                }else{
                    agent->deploy(py::none());
                }
            }
        }catch(std::exception& ex){
            std::cout<<"agent.deploy() failed. agent="<<fullName<<std::endl;
            std::cout<<ex.what()<<std::endl;
            std::cout<<"w="<<worker_index()<<","<<"v="<<vector_index()<<std::endl;
            throw ex;
        }
    }
}
void SimulationManager::updateAgentsToAct(){
    agentsToAct.clear();
    SimPhase phase=SimPhase::AGENT_STEP;

    updateProcessQueueIfNeeded(phase);
    //次回のこのphaseで処理すべきAssetの範囲を取得する
    auto itr = agent_process_queue[phase].find(nextTick[phase]);
    if(itr==agent_process_queue[phase].end()){
        return;
    }
    for(auto&& [index,agent] : itr->second){
        agentsToAct[index]=agent;
    }
}
void SimulationManager::gatherDones(){
    for(auto&& [key, value]: stepDones){
        prevDones[key]=value;
    }
    for(auto&& [fullName,agent] : agents){
        if(agent->dontExpose){
            //dontExposeがtrueのものは外へ出力しない
            continue;
        }
        dones[fullName]=!agent->isAlive() || ruler->dones[fullName] || manualDone;
        if(delayLastObsForDeadAgentUntilAllDone && exposeDeadAgentReward){
            auto index=asset_dependency_graph[SimPhase::AGENT_STEP].get_index(agent);
            auto found=agentsToAct.find(index);
            if(dones[fullName] && found!=agentsToAct.end()){
                //全体の終了前にdoneになったAgentはagentsToActから外す
                agentsToAct.erase(found);
            }
        }
    }
    dones["__all__"]=ruler->dones["__all__"] || manualDone;
    stepDones.clear();
    for(auto&& [index,wAgent] : agentsToAct){
        auto agent=wAgent.lock();
        auto fullName=agent->getFullName();
        if(!prevDones[fullName]){
            stepDones[fullName]=dones[fullName];
        }
    }
    stepDones["__all__"]=dones["__all__"];
    //全体の終了時は全AgentをagentsToActに入れる(dontExposeがtrueのものを除く)
    bool allDone=ruler->dones["__all__"] || manualDone;
    if(allDone){
        agentsToAct.clear();
        for(auto&& [fullName,agent] : agents){
            if(agent->dontExpose){
                continue;
            }
            auto index=asset_dependency_graph[SimPhase::AGENT_STEP].get_index(agent);
            agentsToAct[index]=agent;
            if(!prevDones[fullName]){
                stepDones[fullName]=true;
            }
        }
    }
}
void SimulationManager::gatherRewards(){
    rewards.clear();
    for(auto&& [index,wAgent] : agentsToAct){
        auto agent=wAgent.lock();
        auto fullName=agent->getFullName();
        std::map<std::string,double> val;
        int rIdx=0;
        for(auto& r:rewardGenerators){
            try{
                std::string group;
                if(enableStructuredReward){
                    group=rewardGroups[rIdx];
                }else{
                    group="Default";
                }
                if(val.find(group)==val.end()){
                    val[group]=0.0;
                }
                val[group]+=r->getReward(agent);
            }catch(std::exception& ex){
                std::cout<<"reward.getReward(agent) failed. reward="<<r->getFullName()<<",agent="<<fullName<<std::endl;
                std::cout<<ex.what()<<std::endl;
                throw ex;
            }
            rIdx++;
        }
        if(val.size()==0){
            val["Default"]=0.0;
        }
        if(exposeDeadAgentReward || (agent->isAlive() || !prevDones[fullName])){
            rewards[fullName]=std::move(val);
        }
    }
}
void SimulationManager::gatherTotalRewards(){
    for(auto&& [fullName,agent] : agents){
        std::map<std::string,double> val;
        int rIdx=0;
        for(auto&& r : rewardGenerators){
            try{
                std::string group;
                if(enableStructuredReward){
                    group=rewardGroups[rIdx];
                }else{
                    group="Default";
                }
                if(val.find(group)==val.end()){
                    val[group]=0.0;
                }
                val[group]+=r->getTotalReward(agent);
            }catch(std::exception& ex){
                std::cout<<"reward.getTotalReward(agent) failed. reward="<<r->getFullName()<<",agent="<<fullName<<std::endl;
                std::cout<<ex.what()<<std::endl;
                throw ex;
            }
            rIdx++;
        }
        if(val.size()==0){
            val["Default"]=0.0;
        }
        totalRewards[fullName]=std::move(val);
    }
}
py::dict SimulationManager::get_observation_space(){
    py::gil_scoped_acquire acquire;
    if(!isEpisodeActive){
        reset(std::nullopt,std::nullopt);
    }
    observation_space=py::dict();
    for(auto&& [fullName,agent] : agents){
        if(agent->dontExpose){
            //dontExposeがtrueのものは外へ出力しない
            continue;
        }
        try{
            observation_space[fullName.c_str()]=agent->observation_space();
        }catch(std::exception& ex){
            std::cout<<"agent.get_observation_space() failed. agent="<<fullName<<std::endl;
            std::cout<<ex.what()<<std::endl;
            throw ex;
        }
    }
    runCallbackPhaseFunc(SimPhase::ON_GET_OBSERVATION_SPACE,true);
    runCallbackPhaseFunc(SimPhase::ON_GET_OBSERVATION_SPACE,false);
    for(auto&& [fullName,agent] : agents){
        if(agent->dontExpose){
            //dontExposeがtrueのものは外へ出力しない
            continue;
        }
        if(isinstance<ExpertWrapper>(agent)){
            auto ex=getShared<ExpertWrapper>(agent);
            auto exFullName=ex->getFullName();
            if(ex->whichExpose=="Imitator"){
                observation_space[exFullName.c_str()]=py::cast<py::tuple>(observation_space[exFullName.c_str()])[0];
            }else if(ex->whichExpose=="Expert"){
                observation_space[exFullName.c_str()]=py::cast<py::tuple>(observation_space[exFullName.c_str()])[1];
            }else{//if(ex->whichExpose=="Both"){
                //そのまま
            }
        }
    }
    return observation_space;
}
py::dict SimulationManager::get_action_space(){
    py::gil_scoped_acquire acquire;
    if(!isEpisodeActive){
        reset(std::nullopt,std::nullopt);
    }
    action_space=py::dict();
    for(auto&& [fullName,agent] : agents){
        if(agent->dontExpose){
            //dontExposeがtrueのものは外へ出力しない
            continue;
        }
        try{
            action_space[fullName.c_str()]=agent->action_space();
        }catch(std::exception& ex){
            std::cout<<"agent.get_action_space() failed. agent="<<fullName<<std::endl;
            std::cout<<ex.what()<<std::endl;
            throw ex;
        }
    }
    runCallbackPhaseFunc(SimPhase::ON_GET_ACTION_SPACE,true);
    runCallbackPhaseFunc(SimPhase::ON_GET_ACTION_SPACE,false);
    for(auto&& [fullName,agent] : agents){
        if(agent->dontExpose){
            //dontExposeがtrueのものは外へ出力しない
            continue;
        }
        if(isinstance<ExpertWrapper>(agent)){
            auto ex=getShared<ExpertWrapper>(agent);
            auto exFullName=ex->getFullName();
            if(ex->whichExpose=="Imitator"){
                action_space[exFullName.c_str()]=py::cast<py::tuple>(action_space[exFullName.c_str()])[0];
            }else if(ex->whichExpose=="Expert"){
                action_space[exFullName.c_str()]=py::cast<py::tuple>(action_space[exFullName.c_str()])[1];
            }else{//if(ex->whichExpose=="Both"){
                //そのまま
            }
        }
    }
    return action_space;
}
void SimulationManager::addEventHandler(const std::string& name,std::function<void(const nl::json&)> handler){
    eventHandlers[name].push_back(handler);
}
void SimulationManager::triggerEvent(const std::string& name, const nl::json& args){
    if(eventHandlers.count(name)>0){
        for(auto& e:eventHandlers[name]){
            try{
                e(args);
            }catch(std::exception& ex){
                std::cout<<"triggerEvent(args) failed. name="<<name<<", args="<<args<<std::endl;
                std::cout<<ex.what()<<std::endl;
                throw ex;
            }
        }
    }else{
        //std::cout<<"no handler"<<std::endl;
    }
}

std::shared_ptr<Entity> SimulationManager::createEntityImpl(bool isManaged, bool isEpisodic, const std::string& baseName,const std::string& className,const std::string& modelName,const nl::json& modelConfig,const nl::json& instanceConfig_, const FactoryHelperChain& helperChain, const std::shared_ptr<const Entity>& caller){
    nl::json instanceConfig=instanceConfig_;
    bool isDummy = modelConfig.is_null() && instanceConfig.is_null();
    if(!isDummy){
        instanceConfig["seed"]=randomGen();
    }
    return EntityManager::createEntityImpl(isManaged, isEpisodic, baseName, className, modelName, modelConfig, instanceConfig, helperChain, caller);
}
std::weak_ptr<Agent> SimulationManager::createAgent(const nl::json& agentConfig,const std::string& agentName,const std::map<std::string,std::shared_ptr<PhysicalAssetAccessor>>& parents, const FactoryHelperChain& helperChain, const std::shared_ptr<const Entity>&  caller){
    nl::json instanceConfig=nl::json::object();
    try{
        if(agentConfig.contains("instanceConfig")){
            instanceConfig=agentConfig.at("instanceConfig");
        }
        instanceConfig["name"]=agentName;
        instanceConfig["parents"]=parents;
        instanceConfig["seed"]=randomGen();
        std::string type=agentConfig.at("type");
        std::string modelName;
        instanceConfig["type"]=type;
        if(agentConfig.contains("model")){
            modelName=instanceConfig["model"]=agentConfig.at("model");
        }else{
            assert(type=="ExpertE" || type=="ExpertI" || type=="ExpertBE" || type=="ExpertBI");
            instanceConfig["model"]=agentConfig.at("expertModel");
            modelName=type;
        }
        if(agentConfig.contains("policy")){
            instanceConfig["policy"]=agentConfig.at("policy");
        }else{
            instanceConfig["policy"]="Internal";
        }
        if(type=="Internal"){
            instanceConfig["policy"]="Internal";
        }else if(type=="External"){
            //nothing special
        }else if(type=="ExpertE" || type=="ExpertI" || type=="ExpertBE" || type=="ExpertBI"){
            instanceConfig["imitatorModelName"]=agentConfig.at("imitatorModel");
            instanceConfig["expertModelName"]=agentConfig.at("expertModel");
            instanceConfig["model"]=instanceConfig.at("expertModelName");
            if(agentConfig.contains("expertPolicy")){
                //external expert
                instanceConfig["expertPolicyName"]=agentConfig.at("expertPolicy");
                instanceConfig["policy"]=agentConfig.at("expertPolicy");
            }else{
                //internal expert
                instanceConfig["expertPolicyName"]=instanceConfig.at("policy");
                instanceConfig["policy"]="Internal";
            }
            if(agentConfig.contains("identifier")){
                instanceConfig["identifier"]=agentConfig.at("identifier");
            }else{
                instanceConfig["identifier"]=instanceConfig.at("imitatorModelName");
            }
        }
        instanceConfig["entityFullName"]=agentName+":"+instanceConfig["model"].get<std::string>()+":"+instanceConfig["policy"].get<std::string>();
        return std::dynamic_pointer_cast<Agent>(createEntityImpl(true,true,"Agent","",modelName,nl::json(),instanceConfig,helperChain,caller));
    }catch(std::exception& ex){
        std::cout<<"createAgent(agentConfig,agentName,parents) failed."<<std::endl;
        std::cout<<"agentConfig="<<agentConfig<<std::endl;
        std::cout<<"agentName="<<agentName<<std::endl;
        std::cout<<"parents={"<<std::endl;
        for(auto&& p:parents){
            std::cout<<"  "<<p.first<<":"<<p.second->getFullName()<<","<<std::endl;
        }
        std::cout<<"}"<<std::endl;
        std::cout<<ex.what()<<std::endl;
        throw ex;
    }
}
bool SimulationManager::generateCommunicationBuffer(const std::string& name_,const nl::json& participants_,const nl::json& inviteOnRequest_){
    if(communicationBuffers.count(name_)==0){
        communicationBuffers[name_]=CommunicationBuffer::create(std::static_pointer_cast<SimulationManager>(this->shared_from_this()),name_,participants_,inviteOnRequest_);
        
        if(isEpisodeActive && !isDuringDeserialization){
            communicationBuffers[name_]->validate();
        }
        return true;
    }else{
        return false;
    }
}
bool SimulationManager::requestInvitationToCommunicationBuffer(const std::string& bufferName,const std::shared_ptr<Asset>& asset){
    if(communicationBuffers.count(bufferName)>0){
        return communicationBuffers[bufferName]->requestInvitation(asset);
    }else{
        return false;
    }
}
void SimulationManager::createAssets(){
    assetConfigDispatcher.reset();
    agentConfigDispatcher.reset();
    assets.clear();
    agents.clear();
    controllers.clear();
    teams.clear();
    numExperts=0;
    experts.clear();
    nl::json agentBuffer=nl::json::object();
    std::map<std::string,nl::json> assetConfigs;
    try{
        assetConfigs=RecursiveJsonExtractor::run(
            assetConfigDispatcher.run(managerConfig.at("Assets")),
            [](const nl::json& node){
                if(node.is_object()){
                    if(node.contains("type")){
                        return node["type"]!="group" && node["type"]!="broadcast";
                    }
                }
                return false;
            }
        );
    }catch(std::exception& ex){
        std::cout<<"dispatch of asset config failed."<<std::endl;
        if(managerConfig.is_object() && managerConfig.contains("Assets")){
            std::cout<<"config="<<managerConfig.at("Assets")<<std::endl;
        }else{
            std::cout<<"config doesn't have 'Assets' as a key."<<std::endl;
        }
        std::cout<<ex.what()<<std::endl;
        throw ex;
    }
    std::map<std::string,int> dummy;
    for(auto& e:assetConfigs){
        try{
            std::string assetName=e.first.substr(1);//remove "/" at head.
            std::string assetType=e.second.at("type");
            std::string modelName=e.second.at("model");
            nl::json instanceConfig=e.second.at("instanceConfig");
            instanceConfig["entityFullName"]=assetName;
            std::shared_ptr<Asset> asset=createEntity<Asset>(true,assetType,modelName,instanceConfig);
            dummy[asset->getTeam()]=0;
            if(e.second.contains("Agent")){
                nl::json agentConfig=agentConfigDispatcher.run(e.second["Agent"]);
                std::string agentName=agentConfig["name"];
                std::string agentPort;
                if(agentConfig.contains("port")){
                    agentPort=agentConfig["port"];
                }else{
                    agentPort="0";
                }
                if(agentBuffer.contains(agentName)){
                    if(agentBuffer[agentName]["parents"].contains(agentPort)){
                        throw std::runtime_error("Duplicated agent port: agent="+agentName+", port="+agentPort);
                    }else{
                        agentBuffer[agentName]["parents"][agentPort]=asset->getAccessor();
                    }
                }else{
                    agentBuffer[agentName]={
                        {"config",agentConfig},
                        {"parents",{{agentPort,asset->getAccessor()}}}
                    };
                }
            }
        }catch(std::exception& ex){
            std::cout<<"creation of assets failed."<<std::endl;
            std::cout<<"assetConfig={"<<e.first<<":"<<e.second<<"}"<<std::endl;
            std::cout<<ex.what()<<std::endl;
            throw ex;
        }
    }
    for(auto& e:dummy){
        teams.push_back(e.first);
    }
    for(auto& e:agentBuffer.items()){
        std::string agentName=e.key();
        std::shared_ptr<Agent> agent=createAgent(e.value()["config"],agentName,e.value()["parents"],FactoryHelperChain(),nullptr).lock();
        for(auto& p:agent->parents){
            getEntityFromAccessor<PhysicalAsset>(p.second)->setAgent(agent,p.first);
        }
        if(agent->type=="ExpertE" || agent->type=="ExpertI" || agent->type=="ExpertBE" || agent->type=="ExpertBI"){
            numExperts++;
            experts[agentName]=agent;
        }
    }
}
void SimulationManager::generateCommunicationBuffers(){
    if(!managerConfig.contains("CommunicationBuffers")){
        return;
    }
    assert(managerConfig.at("CommunicationBuffers").is_object());
    for(auto&& e:managerConfig.at("CommunicationBuffers").items()){
        try{
            nl::json participants=e.value().contains("participants") ? e.value().at("participants") : nl::json::array();
            nl::json inviteOnRequest=e.value().contains("inviteOnRequest") ? e.value().at("inviteOnRequest") : nl::json::array();
            generateCommunicationBuffer(e.key(),participants,inviteOnRequest);
        }catch(std::exception& ex){
            std::cout<<"generateCommunicationBuffer() failed."<<std::endl;
            std::cout<<"name="<<e.key()<<std::endl;
            std::cout<<"participants="<<e.value()<<std::endl;
            std::cout<<ex.what()<<std::endl;
            throw ex;
        }
    }
}
void SimulationManager::registerEntity(bool isManaged,const std::shared_ptr<Entity>& entity){
    BaseType::registerEntity(isManaged,entity);
    if(isManaged){
        std::lock_guard<std::recursive_mutex> lock(mtx);
        if(auto asset=std::dynamic_pointer_cast<Asset>(entity)){
            if(auto physicalAsset=std::dynamic_pointer_cast<PhysicalAsset>(entity)){
                assets[physicalAsset->getFullName()]=physicalAsset;
            }else if(auto controller=std::dynamic_pointer_cast<Controller>(entity)){
                controllers[controller->getFullName()]=controller;
            }else if(auto agent=std::dynamic_pointer_cast<Agent>(entity)){
                agents[agent->getFullName()]=agent;
            }

            if(isEpisodeActive && !isDuringDeserialization){
                addedAssets.push_back(asset);
                // vertexだけはここで追加しておく
                for(auto&& phase:assetPhases){
                    asset_dependency_graph[phase].add_instance(asset);
                }
                if(auto agent=std::dynamic_pointer_cast<Agent>(asset)){
                    for(auto&& phase : agentPhases){
                        asset_dependency_graph[phase].add_instance(asset);
                    }
                }
            }
        }else if(auto callback=std::dynamic_pointer_cast<Callback>(entity)){
            // TODO Loggerとの識別

            if(isEpisodeActive && !isDuringDeserialization){
                addedCallbacks.push_back(callback);
                // vertexだけはここで追加しておく
                for(auto&& phase:callbackPhases){
                    callback_dependency_graph[phase].add_instance(callback);
                }
            }
        }else if(auto crs=std::dynamic_pointer_cast<CoordinateReferenceSystem>(entity)){
            crses[crs->getEntityID()]=crs;
        }else{
            throw std::runtime_error("SimulationManager cannot manage Entity of baseName='"+entity->getFactoryBaseName()+"'.");
        }
    }
}

void SimulationManager::removeEntity(const std::shared_ptr<Entity>& entity){
    if(entity && entity->isManaged()){
        std::lock_guard<std::recursive_mutex> lock(mtx);
        if(auto asset=std::dynamic_pointer_cast<Asset>(entity)){
            if(auto physicalAsset=std::dynamic_pointer_cast<PhysicalAsset>(entity)){
                auto found=assets.find(physicalAsset->getFullName());
                if(found!=assets.end()){
                    assets.erase(found);
                }
            }else if(auto controller=std::dynamic_pointer_cast<Controller>(entity)){
                auto found=controllers.find(controller->getFullName());
                if(found!=controllers.end()){
                    controllers.erase(found);
                }
            }else if(auto agent=std::dynamic_pointer_cast<Agent>(entity)){
                auto found=agents.find(agent->getFullName());
                if(found!=agents.end()){
                    agents.erase(found);
                }
            }

            for(auto&& phase : assetPhases){
                auto& graph = asset_dependency_graph[phase];
                auto id=asset->getEntityID();
                if(asset_indices[phase].contains(id)){
                    auto index=asset_indices[phase][id];
                    auto next=asset_next_ticks[phase][id];

                    asset_next_ticks[phase].erase(id);
                    asset_indices[phase].erase(id);

                    if(assetPhases.contains(phase)){
                        if(asset_process_queue.at(phase).contains(next)){
                            auto& queue=asset_process_queue.at(phase).at(next);
                            auto found=queue.find(index);
                            if(found!=queue.end()){
                                queue.erase(found);
                            }
                        }
                    }
                    if(agentPhases.contains(phase)){
                        if(agent_process_queue.at(phase).contains(next)){
                            auto& queue=agent_process_queue.at(phase).at(next);
                            auto found=queue.find(index);
                            if(found!=queue.end()){
                                queue.erase(found);
                            }
                        }
                    }
                    
                    graph.remove_instance(asset);
                    hasDependencyUpdated[phase]=true;
                }
            }
            for(auto&& phase : agentPhases){
                auto id=asset->getEntityID();
                auto& graph = asset_dependency_graph[phase];
                if(asset_indices[phase].contains(id)){
                    auto index=asset_indices[phase][id];
                    auto next=asset_next_ticks[phase][id];

                    asset_next_ticks[phase].erase(id);
                    asset_indices[phase].erase(id);

                    if(agent_process_queue.at(phase).contains(next)){
                        auto& queue=agent_process_queue.at(phase).at(next);
                        auto found=queue.find(index);
                        if(found!=queue.end()){
                            queue.erase(found);
                        }
                    }
                    
                    graph.remove_instance(asset);
                    hasDependencyUpdated[phase]=true;
                }
            }
        }else if(auto callback=std::dynamic_pointer_cast<Callback>(entity)){
            {
                auto found=callbacks.find(callback->getFullName());
                if(found!=callbacks.end()){
                    callbacks.erase(found);
                }
            }
            {
                auto found=loggers.find(callback->getFullName());
                if(found!=loggers.end()){
                    loggers.erase(found);
                }
            }

            for(auto&& phase:callbackPhases){
                auto& graph = callback_dependency_graph[phase];
                auto id=callback->getEntityID();
                if(callback_indices[phase].contains(id)){
                    auto index=callback_indices[phase][id];
                    auto next=callback_next_ticks[phase][id];

                    callback_next_ticks[phase].erase(id);
                    callback_indices[phase].erase(id);

                    bool isGroup1 = (
                        isinstance<Ruler>(callback)
                        || isinstance<Reward>(callback)
                    );
                    if(isGroup1){
                        if(group1_callback_process_queue.at(phase).contains(next)){
                            auto& queue=group1_callback_process_queue.at(phase).at(next);
                            auto found=queue.find(index);
                            if(found!=queue.end()){
                                queue.erase(found);
                            }
                        }
                    }else{
                        if(group2_callback_process_queue.at(phase).contains(next)){
                            auto& queue=group2_callback_process_queue.at(phase).at(next);
                            auto found=queue.find(index);
                            if(found!=queue.end()){
                                queue.erase(found);
                            }
                        }
                    }
                    
                    graph.remove_instance(callback);
                    hasDependencyUpdated[phase]=true;
                }
            }
        }else if(auto crs=std::dynamic_pointer_cast<CoordinateReferenceSystem>(entity)){
            auto found=crses.find(crs->getEntityID());
            if(found!=crses.end()){
                crses.erase(found);
            }
        }else{
            throw std::runtime_error("SimulationManager cannot manage Entity of baseName='"+entity->getFactoryBaseName()+"'.");
        }
    }
    BaseType::removeEntity(entity);
}

void SimulationManager::removeEntity(const std::weak_ptr<Entity>& entity,const EntityIdentifier& id,const std::string& name){
    BaseType::removeEntity(entity,id,name);
}

std::shared_ptr<EntityManagerAccessor> SimulationManager::createAccessorFor(const std::shared_ptr<const Entity>& entity){
    if(entity){
        auto casted_this=std::static_pointer_cast<SimulationManager>(this->shared_from_this());
        auto baseName=entity->getFactoryBaseName();
        if(baseName=="PhysicalAsset" || baseName=="Controller"){
            return EntityManagerAccessor::create<SimulationManagerAccessorForPhysicalAsset>(casted_this);
        }else if(baseName=="Agent"){
            return EntityManagerAccessor::create<SimulationManagerAccessorForAgent>(casted_this);
        }else if(baseName=="Callback" || baseName=="Ruler" || baseName=="Reward" || baseName=="Viewer"){
            return EntityManagerAccessor::create<SimulationManagerAccessorForCallback>(casted_this);
        }else{
            return nullptr;
        }
    }else{
        return nullptr;
    }
}

void SimulationManager::requestToKillAsset(const std::shared_ptr<Asset>& asset){
    killRequests.push_back(asset);
}

void SimulationManager::killAndRemoveAssets(){
    for(auto&& e:killRequests){
        if(e->isAlive()){
            e->kill();
        }
    }
    std::vector<std::shared_ptr<Entity>> tmp;
    {
        auto it=assets.begin();
        while(it!=assets.end()){
            auto asset=it->second;
            if(!asset->isAlive() && asset->removeWhenKilled){
                tmp.push_back(asset);
                it=assets.erase(it);
                removeEntity(asset);
            }else{
                ++it;
            }
        }
    }
    {
        auto it=controllers.begin();
        while(it!=controllers.end()){
            auto controller=it->second;
            if(!controller->isAlive() && controller->removeWhenKilled){
                tmp.push_back(controller);
                it=controllers.erase(it);
                removeEntity(controller);
            }else{
                ++it;
            }
        }
    }
    {
        auto it=agents.begin();
        while(it!=agents.end()){
            auto agent=it->second;
            if(!agent->isAlive() && agent->removeWhenKilled){
                tmp.push_back(agent);
                it=agents.erase(it);
                removeEntity(agent);
            }else{
                ++it;
            }
        }
    }
    tmp.clear();
}

void SimulationManager::setupNewEntities(){
    if(addedAssets.size()>0){
        bool areThereAgents=false;
        for(auto&& asset : addedAssets){
            for(auto&& phase:assetPhases){
                if(measureTime){
                    std::string phaseName=enumToStr(phase);
                    std::string fullName=asset->getFullName();
                    maxProcessTime[phaseName][fullName]=0;
                    minProcessTime[phaseName][fullName]=std::numeric_limits<std::uint64_t>::max();
                    meanProcessTime[phaseName][fullName]=0;
                    processCount[phaseName][fullName]=0;
                }
            }
            if(auto agent=std::dynamic_pointer_cast<Agent>(asset)){
                areThereAgents=true;
                for(auto&& phase : agentPhases){
                    if(measureTime){
                        std::string phaseName=enumToStr(phase);
                        std::string fullName=asset->getFullName();
                        maxProcessTime[phaseName][fullName]=0;
                        minProcessTime[phaseName][fullName]=std::numeric_limits<std::uint64_t>::max();
                        meanProcessTime[phaseName][fullName]=0;
                        processCount[phaseName][fullName]=0;
                    }
                }
            }
        }

        for(auto&& asset : addedAssets){
            asset->setDependency();
            for(auto&& generator : assetDependencyGenerators){
                generator(asset);
            }
        }

        for(auto&& phase : assetPhases){
            updateProcessQueueIfNeeded(phase);
            for(auto&& asset : addedAssets){
                asset_indices[phase][asset->getEntityID()]=asset_dependency_graph.at(phase).get_index(asset);
                calcFirstTick(phase,asset);
            }
            calcNextTick(phase);
        }
        if(areThereAgents){
            for(auto&& phase:agentPhases){
                updateProcessQueueIfNeeded(phase);
                for(auto&& asset : addedAssets){
                    if(auto agent=std::dynamic_pointer_cast<Agent>(asset)){
                        asset_indices[phase][asset->getEntityID()]=asset_dependency_graph.at(phase).get_index(asset);
                        calcFirstTick(phase,asset);
                    }
                }
                calcNextTick(phase);
            }
        }

        // validateとperceive(true)
        std::chrono::system_clock::time_point before,after;
        std::chrono::microseconds difference;
		std::sort(addedAssets.begin(),addedAssets.end(),
			[this](const std::shared_ptr<Asset>& lhs,const std::shared_ptr<Asset>& rhs)->bool {
                return asset_dependency_graph[SimPhase::VALIDATE].get_index(lhs)
                    < asset_dependency_graph[SimPhase::VALIDATE].get_index(rhs);
		});
        for(auto&& asset : addedAssets){
            try{
                if(measureTime){
                    before=std::chrono::system_clock::now();
                }
                asset->validate();
                if(measureTime){
                    std::string name=asset->getFullName();
                    after=std::chrono::system_clock::now();
                    difference=std::chrono::duration_cast<std::chrono::microseconds>(after-before);
                    std::string phaseName=std::string(magic_enum::enum_name(SimPhase::VALIDATE));
                    maxProcessTime[phaseName][name]=std::max<std::uint64_t>(maxProcessTime[phaseName][name],difference.count());
                    minProcessTime[phaseName][name]=std::min<std::uint64_t>(minProcessTime[phaseName][name],difference.count());
                    processCount[phaseName][name]++;
                    meanProcessTime[phaseName][name]+=((double)(difference.count())-meanProcessTime[phaseName][name])/processCount[phaseName][name];
                }
            }catch(std::exception& ex){
                std::cout<<"asset.validate() failed. asset="<<asset->getFullName()<<std::endl;
                std::cout<<ex.what()<<std::endl;
                throw ex;
            }
        }

		std::sort(addedAssets.begin(),addedAssets.end(),
			[this](const std::shared_ptr<Asset>& lhs,const std::shared_ptr<Asset>& rhs)->bool {
                return asset_dependency_graph[SimPhase::PERCEIVE].get_index(lhs)
                    < asset_dependency_graph[SimPhase::PERCEIVE].get_index(rhs);
		});
        for(auto&& asset : addedAssets){
            try{
                if(asset->isAlive()){
                    if(measureTime){
                        before=std::chrono::system_clock::now();
                    }
                    asset->perceive(true);
                    if(measureTime){
                        std::string name=asset->getFullName();
                        after=std::chrono::system_clock::now();
                        std::string phaseName=std::string(magic_enum::enum_name(SimPhase::PERCEIVE));
                        difference=std::chrono::duration_cast<std::chrono::microseconds>(after-before);
                        maxProcessTime[phaseName][name]=std::max<std::uint64_t>(maxProcessTime[phaseName][name],difference.count());
                        minProcessTime[phaseName][name]=std::min<std::uint64_t>(minProcessTime[phaseName][name],difference.count());
                        processCount[phaseName][name]++;
                        meanProcessTime[phaseName][name]+=((double)(difference.count())-meanProcessTime[phaseName][name])/processCount[phaseName][name];
                    }
                }
            }catch(std::exception& ex){
                std::cout<<"asset.perceive(true) failed. asset="<<asset->getFullName()<<std::endl;
                std::cout<<ex.what()<<std::endl;
                throw ex;
            }
        }
    }
    addedAssets.clear();

    if(addedCallbacks.size()>0){
        for(auto&& callback : addedCallbacks){
            for(auto&& phase:callbackPhases){
                if(measureTime){
                    std::string phaseName=enumToStr(phase);
                    std::string fullName=callback->getFullName();
                    maxProcessTime[phaseName][fullName]=0;
                    minProcessTime[phaseName][fullName]=std::numeric_limits<std::uint64_t>::max();
                    meanProcessTime[phaseName][fullName]=0;
                    processCount[phaseName][fullName]=0;
                }
            }
        }

        for(auto&& callback : addedCallbacks){
            callback->setDependency();
            for(auto&& generator : callbackDependencyGenerators){
                generator(callback);
            }
        }

        for(auto&& phase : callbackPhases){
            updateProcessQueueIfNeeded(phase);
            for(auto&& callback : addedCallbacks){
                callback_indices[phase][callback->getEntityID()]=callback_dependency_graph.at(phase).get_index(callback);
                calcFirstTick(phase,callback);
            }
            calcNextTick(phase);
        }
    }
    addedCallbacks.clear();
}
std::vector<std::string> SimulationManager::getTeams() const{
    return teams;
}

//
// 仮想シミュレータの生成
//
bool SimulationManager::isVirtual() const{
    return _isVirtual;
}
nl::json SimulationManager::createVirtualSimulationManagerConfig(const nl::json& option){
    if(!isConfigured){
        throw std::runtime_error("Configure the SimulationManager before creating a virtual one!");
    }

    nl::json ret={
        {"Manager",{
            {"TimeStep",{
                {"baseTimeStep",baseTimeStep},
                {"defaultAgentStepInterval",getDefaultAgentStepInterval()}
            }},
            {"Epoch",getEpoch()},
            {"CoordinateReferenceSystem",{
                {"preset",nl::json::object()},
                {"root",nl::json::object()}
            }},
            {"Ruler",{
				{"class","Ruler"},
				{"modelConfig",{
				    {"instanceConfig",{
					    {"crs",ruler->getLocalCRS()}
                    }}
				}}
			}},
            {"ViewerType","None"},
            {"Assets",nl::json::object()},
            {"AssetConfigDispatcher",nl::json::object()},
            {"AgentConfigDispatcher",nl::json::object()},
            {"skipNoAgentStep",skipNoAgentStep},
            {"enableStructuredReward",enableStructuredReward},
            {"exposeDeadAgentReward",exposeDeadAgentReward},
            {"delayLastObsForDeadAgentUntilAllDone",delayLastObsForDeadAgentUntilAllDone}
        }},
        {"Factory",getFactoryModelConfig(false)}
    };

    auto& preset=ret["/Manager/CoordinateReferenceSystem/preset"_json_pointer];
    for(auto&& [id, crs] : crses){
        if(!crs->isEpisodic()){
            preset[crs->getFullName()]=crs;
        }
    }
    ret["/Manager/CoordinateReferenceSystem/root"_json_pointer]=getRootCRS();

    if(option.contains("patch")){
        ret.merge_patch(option.at("patch"));
    }

    return ret;
}
void SimulationManager::postprocessAfterVirtualSimulationManagerCreation(const std::shared_ptr<SimulationManager>& sim, const nl::json& option){
    sim->_isVirtual=true;
}

Epoch SimulationManagerAccessorBase::getEpoch() const{
    return manager.lock()->getEpoch();
}
Time SimulationManagerAccessorBase::getTime() const{
    return manager.lock()->getTime();
}
double SimulationManagerAccessorBase::getElapsedTime() const{
    return manager.lock()->getElapsedTime();
}
double SimulationManagerAccessorBase::getBaseTimeStep() const{
    return manager.lock()->baseTimeStep;
}
std::uint64_t SimulationManagerAccessorBase::getTickCount() const{
    return manager.lock()->tickCount;
}
std::uint64_t SimulationManagerAccessorBase::getTickCountAt(const Time& time) const{
    return manager.lock()->getTickCountAt(time);
}
std::uint64_t SimulationManagerAccessorBase::getDefaultAgentStepInterval() const{
    return manager.lock()->defaultAgentStepInterval;
}
std::uint64_t SimulationManagerAccessorBase::getInternalStepCount() const{
    return manager.lock()->getInternalStepCount();
}
std::uint64_t SimulationManagerAccessorBase::getExposedStepCount() const{
    return manager.lock()->getExposedStepCount();
}
std::shared_ptr<CoordinateReferenceSystem> SimulationManagerAccessorBase::getRootCRS() const{
    return  manager.lock()->getRootCRS();
}
nl::json SimulationManagerAccessorBase::getCRSConfig() const{
    return manager.lock()->getCRSConfig();
}
std::vector<std::string> SimulationManagerAccessorBase::getTeams() const{
    return std::move(manager.lock()->getTeams());
}
SimulationManagerAccessorBase::SimulationManagerAccessorBase(const std::shared_ptr<SimulationManager>& manager_)
:EntityManagerAccessor(manager_),
manager(manager_){
}
SimulationManagerAccessorBase::SimulationManagerAccessorBase(const std::shared_ptr<SimulationManagerAccessorBase>& original_)
:EntityManagerAccessor(original_){
    if(original_){
        manager=original_->manager;
    }
}

std::vector<std::pair<EntityIdentifier,MotionState>> SimulationManagerAccessorForCallback::getInternalMotionStatesOfAffineCRSes() const{
    return manager.lock()->getInternalMotionStatesOfAffineCRSes();
}
void SimulationManagerAccessorForCallback::setInternalMotionStatesOfAffineCRSes(const std::vector<std::pair<EntityIdentifier,MotionState>>& states){
    manager.lock()->setInternalMotionStatesOfAffineCRSes(states);
}
boost::uuids::uuid SimulationManagerAccessorForCallback::getUUID() const{
    return manager.lock()->getUUID();
}
boost::uuids::uuid SimulationManagerAccessorForCallback::getUUIDOfEntity(const std::shared_ptr<const Entity>& entity) const{
    return manager.lock()->getUUIDOfEntity(entity);
}
EntityConstructionInfo SimulationManagerAccessorForCallback::getEntityConstructionInfo(const std::shared_ptr<Entity>& entity,EntityIdentifier id) const{
    return manager.lock()->getEntityConstructionInfo(entity,id);
}
void SimulationManagerAccessorForCallback::requestReconfigure(const nl::json& fullConfigPatch){
    manager.lock()->requestReconfigure(fullConfigPatch);
}
void SimulationManagerAccessorForCallback::requestReconfigure(const nl::json& managerConfigPatch,const nl::json& factoryConfigPatch){
    manager.lock()->requestReconfigure(managerConfigPatch,factoryConfigPatch);
}
void SimulationManagerAccessorForCallback::addEventHandler(const std::string& name,std::function<void(const nl::json&)> handler){
    manager.lock()->addEventHandler(name,handler);
}
void SimulationManagerAccessorForCallback::triggerEvent(const std::string& name, const nl::json& args){
    manager.lock()->triggerEvent(name,args);
}
void SimulationManagerAccessorForCallback::requestToKillAsset(const std::shared_ptr<Asset>& asset){
    manager.lock()->requestToKillAsset(asset);
}
void SimulationManagerAccessorForCallback::addDependency(const SimPhase& phase,const std::shared_ptr<Entity>& predecessor, const std::shared_ptr<Entity>& successor){
    manager.lock()->addDependency(phase,predecessor,successor);
}
void SimulationManagerAccessorForCallback::removeDependency(const SimPhase& phase,const std::shared_ptr<Entity>& predecessor, const std::shared_ptr<Entity>& successor){
    manager.lock()->removeDependency(phase,predecessor,successor);
}
void SimulationManagerAccessorForCallback::addDependencyGenerator(std::function<void(const std::shared_ptr<Callback>&)> generator){
    manager.lock()->addDependencyGenerator(generator);
}
py::dict& SimulationManagerAccessorForCallback::observation_space() const{
    return manager.lock()->observation_space;
}
py::dict& SimulationManagerAccessorForCallback::action_space() const{
    return manager.lock()->action_space;
}
py::dict& SimulationManagerAccessorForCallback::observation() const{
    return manager.lock()->observation;
}
py::dict& SimulationManagerAccessorForCallback::action() const{
    return manager.lock()->action;
}
std::map<std::string,bool>& SimulationManagerAccessorForCallback::dones() const{
    return manager.lock()->dones;
}
std::map<std::string,double>& SimulationManagerAccessorForCallback::scores() const{
    return manager.lock()->scores;
}
std::map<std::string,std::map<std::string,double>>& SimulationManagerAccessorForCallback::rewards() const{
    return manager.lock()->rewards;
}
std::map<std::string,std::map<std::string,double>>& SimulationManagerAccessorForCallback::totalRewards() const{
    return manager.lock()->totalRewards;
}
const std::map<std::string,std::weak_ptr<Agent>>& SimulationManagerAccessorForCallback::experts() const{
    return manager.lock()->experts;
}
const std::map<std::size_t,std::weak_ptr<Agent>>& SimulationManagerAccessorForCallback::agentsActed() const{
    return manager.lock()->agentsActed;
}
const std::map<std::size_t,std::weak_ptr<Agent>>& SimulationManagerAccessorForCallback::agentsToAct() const{
    return manager.lock()->agentsToAct;
}
std::int32_t SimulationManagerAccessorForCallback::worker_index() const{
    return manager.lock()->worker_index();
}
std::int32_t SimulationManagerAccessorForCallback::vector_index() const{
    return manager.lock()->vector_index();
}
std::int32_t SimulationManagerAccessorForCallback::episode_index() const{
    return manager.lock()->episode_index();
}
bool& SimulationManagerAccessorForCallback::manualDone(){
    return manager.lock()->manualDone;
}
std::uint64_t SimulationManagerAccessorForCallback::getAgentStepCount(const std::string& fullName_) const{
    return manager.lock()->getAgentStepCount(fullName_);
}
std::uint64_t SimulationManagerAccessorForCallback::getAgentStepCount(const std::shared_ptr<Agent>& agent) const{
    return manager.lock()->getAgentStepCount(agent);
}
void SimulationManagerAccessorForCallback::setManualDone(const bool& b){
    manager.lock()->manualDone=b;
}
nl::json SimulationManagerAccessorForCallback::getManagerConfig() const{
    return manager.lock()->getManagerConfig();
}
nl::json SimulationManagerAccessorForCallback::getFactoryModelConfig(bool withDefaultModels) const{
    return manager.lock()->getFactoryModelConfig(withDefaultModels);
}

void SimulationManagerAccessorForPhysicalAsset::triggerEvent(const std::string& name, const nl::json& args){
    manager.lock()->triggerEvent(name,args);
}
void SimulationManagerAccessorForPhysicalAsset::requestToKillAsset(const std::shared_ptr<Asset>& asset){
    manager.lock()->requestToKillAsset(asset);
}
void SimulationManagerAccessorForPhysicalAsset::addDependency(const SimPhase& phase,const std::shared_ptr<Entity>& predecessor, const std::shared_ptr<Entity>& successor){
    manager.lock()->addDependency(phase,predecessor,successor);
}
void SimulationManagerAccessorForPhysicalAsset::removeDependency(const SimPhase& phase,const std::shared_ptr<Entity>& predecessor, const std::shared_ptr<Entity>& successor){
    manager.lock()->removeDependency(phase,predecessor,successor);
}
void SimulationManagerAccessorForPhysicalAsset::addDependencyGenerator(std::function<void(const std::shared_ptr<Asset>&)> generator){
    manager.lock()->addDependencyGenerator(generator);
}
bool SimulationManagerAccessorForPhysicalAsset::generateCommunicationBuffer(const std::string& name_,const nl::json& participants_,const nl::json& inviteOnRequest_){
    return manager.lock()->generateCommunicationBuffer(name_,participants_,inviteOnRequest_);
}
bool SimulationManagerAccessorForPhysicalAsset::requestInvitationToCommunicationBuffer(const std::string& bufferName,const std::shared_ptr<Asset>& asset){
    return manager.lock()->requestInvitationToCommunicationBuffer(bufferName,asset);
}

bool SimulationManagerAccessorForAgent::requestInvitationToCommunicationBuffer(const std::string& bufferName,const std::shared_ptr<Asset>& asset){
    return manager.lock()->requestInvitationToCommunicationBuffer(bufferName,asset);
}
std::weak_ptr<RulerAccessor> SimulationManagerAccessorForAgent::getRuler() const{
    return manager.lock()->ruler->getAccessor<RulerAccessor>();
}

void exportSimulationManager(py::module &m)
{
    using namespace pybind11::literals;
    expose_enum_value_helper(
        expose_enum_class<SimPhase>(m,"SimPhase")
        ,"NONE"
        ,"VALIDATE"
        ,"PERCEIVE"
        ,"CONTROL"
        ,"BEHAVE"
        ,"AGENT_STEP"
        ,"ON_GET_OBSERVATION_SPACE"
        ,"ON_GET_ACTION_SPACE"
        ,"ON_MAKE_OBS"
        ,"ON_EPISODE_BEGIN"
        ,"ON_VALIDATION_END"
        ,"ON_STEP_BEGIN"
        ,"ON_INNERSTEP_BEGIN"
        ,"ON_INNERSTEP_END"
        ,"ON_STEP_END"
        ,"ON_EPISODE_END"
    );

    expose_dependency_graph_name<EntityDependencyGraph<Asset>>(m,"AssetDependencyGraph");
    expose_dependency_graph_name<EntityDependencyGraph<Callback>>(m,"CallbackDependencyGraph");

    py::class_<MapIterable<CoordinateReferenceSystem,EntityIdentifier,CoordinateReferenceSystem>>(m,"MapIterable<CoordinateReferenceSystem>")
    .def("__iter__",&MapIterable<CoordinateReferenceSystem,EntityIdentifier,CoordinateReferenceSystem>::iter,py::keep_alive<0,1>())
    .def("__next__",&MapIterable<CoordinateReferenceSystem,EntityIdentifier,CoordinateReferenceSystem>::next)
    ;
    py::class_<MapIterable<PhysicalAsset,std::string,PhysicalAsset>>(m,"MapIterable<PhysicalAsset>")
    .def("__iter__",&MapIterable<PhysicalAsset,std::string,PhysicalAsset>::iter,py::keep_alive<0,1>())
    .def("__next__",&MapIterable<PhysicalAsset,std::string,PhysicalAsset>::next)
    ;
    py::class_<MapIterable<Controller,std::string,Controller>>(m,"MapIterable<Controller>")
    .def("__iter__",&MapIterable<Controller,std::string,Controller>::iter,py::keep_alive<0,1>())
    .def("__next__",&MapIterable<Controller,std::string,Controller>::next)
    ;
    py::class_<MapIterable<Agent,std::string,Agent>>(m,"MapIterable<Agent>")
    .def("__iter__",&MapIterable<Agent,std::string,Agent>::iter,py::keep_alive<0,1>())
    .def("__next__",&MapIterable<Agent,std::string,Agent>::next)
    ;
    py::class_<VectorIterable<Reward,Reward>>(m,"VectorIterable<Reward>")
    .def("__iter__",&VectorIterable<Reward,Reward>::iter,py::keep_alive<0,1>())
    .def("__next__",&VectorIterable<Reward,Reward>::next)
    ;
    py::class_<MapIterable<Callback,std::string,Callback>>(m,"MapIterable<Callback>")
    .def("__iter__",&MapIterable<Callback,std::string,Callback>::iter,py::keep_alive<0,1>())
    .def("__next__",&MapIterable<Callback,std::string,Callback>::next)
    ;

    expose_common_class<SimulationManagerBase>(m,"SimulationManagerBase")
    DEF_FUNC(SimulationManagerBase,getEpoch)
    DEF_FUNC(SimulationManagerBase,getTime)
    DEF_FUNC(SimulationManagerBase,getElapsedTime)
    DEF_FUNC(SimulationManagerBase,getRootCRS)
    DEF_FUNC(SimulationManagerBase,getCRSConfig)
    .def("getCRSes",py::overload_cast<>(&SimulationManagerBase::getCRSes<>,py::const_))
    .def("getCRSes",[](SimulationManagerBase& v,const py::object& matcher){
        MapIterable<CoordinateReferenceSystem,EntityIdentifier,CoordinateReferenceSystem>::MatcherType wrapped=[matcher](MapIterable<CoordinateReferenceSystem,EntityIdentifier,CoordinateReferenceSystem>::ConstSharedType p)->bool {
            std::shared_ptr<CoordinateReferenceSystem> non_const=std::const_pointer_cast<CoordinateReferenceSystem>(p);
            return py::cast<bool>(matcher(non_const));
        };
        return v.getCRSes<>(wrapped);
    },py::keep_alive<0,2>())
    .def("getCRS",&SimulationManagerBase::getCRS<>)
    DEF_FUNC(SimulationManagerBase,getInternalMotionStatesOfAffineCRSes)
    DEF_FUNC(SimulationManagerBase,setInternalMotionStatesOfAffineCRSes)
    ;

    expose_common_class<SimulationManager>(m,"SimulationManager")
    .def(py_init<const nl::json&,const std::int32_t&,const std::int32_t&,typename SimulationManager::ConfigOverriderType>()
        ,"config_"_a,"worker_index_"_a=0,"vector_index"_a=0,"overrider"_a=py::none(),py::keep_alive<5,1>()
    )
    //
    // コンフィグ管理
    //
    DEF_FUNC(SimulationManager,getManagerConfig)
    DEF_FUNC(SimulationManager,getFactoryModelConfig, "withDefaultModels"_a=true)
    .def("requestReconfigure",py::overload_cast<const nl::json&>(&SimulationManager::requestReconfigure))
    .def("requestReconfigure",py::overload_cast<const nl::json&,const nl::json&>(&SimulationManager::requestReconfigure))
    DEF_STATIC_FUNC(SimulationManager,parseConfig)
    DEF_FUNC(SimulationManager,setViewerType)
    .def("configure",py::overload_cast<>(&SimulationManager::configure))
    .def("configure",py::overload_cast<const nl::json&>(&SimulationManager::configure))
    .def("configure",py::overload_cast<const nl::json&,const nl::json&>(&SimulationManager::configure))
    .def("reconfigureIfRequested",py::overload_cast<>(&SimulationManager::reconfigureIfRequested))
    .def("reconfigure",py::overload_cast<const nl::json&>(&SimulationManager::reconfigure))
    .def("reconfigure",py::overload_cast<const nl::json&,const nl::json&>(&SimulationManager::reconfigure))
    //
    // 実行状態管理
    //
    .def_property_readonly("worker_index",&SimulationManager::worker_index)
    .def_property_readonly("vector_index",&SimulationManager::vector_index)
    .def_property_readonly("episode_index",&SimulationManager::episode_index)
    DEF_READONLY(SimulationManager,skipNoAgentStep)
    //
    // 時刻管理
    //
    DEF_FUNC(SimulationManager,getTickCount)
    DEF_FUNC(SimulationManager,getTickCountAt)
    DEF_FUNC(SimulationManager,getDefaultAgentStepInterval)
    DEF_FUNC(SimulationManager,getInternalStepCount)
    DEF_FUNC(SimulationManager,getExposedStepCount)
    .def("getAgentStepCount",py::overload_cast<const std::string&>(&SimulationManager::getAgentStepCount,py::const_))
    .def("getAgentStepCount",py::overload_cast<const std::shared_ptr<Agent>&>(&SimulationManager::getAgentStepCount,py::const_))
    //
    // エピソード管理
    //
    DEF_FUNC(SimulationManager,get_observation_space)
    DEF_FUNC(SimulationManager,get_action_space)
    DEF_FUNC(SimulationManager,reset,py::kw_only(),"seed"_a=std::nullopt,"options"_a=std::nullopt)
    DEF_FUNC(SimulationManager,step)
    DEF_FUNC(SimulationManager,getReturnsOfTheLastStep)
    DEF_FUNC(SimulationManager,stopEpisodeExternally)
    DEF_FUNC(SimulationManager,cleanupEpisode)
    DEF_FUNC(SimulationManager,getTeams)
    DEF_READONLY(SimulationManager,dones)
    DEF_READONLY(SimulationManager,scores)
    DEF_READONLY(SimulationManager,rewards)
    DEF_READONLY(SimulationManager,totalRewards)
    DEF_READONLY(SimulationManager,experts)
    DEF_READWRITE(SimulationManager,manualDone)
    //
    // 仮想シミュレータの生成
    //
    DEF_FUNC(SimulationManager,isVirtual)
    DEF_FUNC(SimulationManager,createVirtualSimulationManagerConfig)
    DEF_FUNC(SimulationManager,postprocessAfterVirtualSimulationManagerCreation)
    .def("createVirtualSimulationManager",&SimulationManager::createVirtualSimulationManager<>)
    .def("getAsset",&SimulationManager::getAsset<>)
    .def("getAssets",py::overload_cast<>(&SimulationManager::getAssets<>,py::const_))
    .def("getAssets",[](SimulationManager& v,const py::object& matcher){
        MapIterable<PhysicalAsset,std::string,PhysicalAsset>::MatcherType wrapped=[matcher](MapIterable<PhysicalAsset,std::string,PhysicalAsset>::ConstSharedType p)->bool {
            std::shared_ptr<PhysicalAsset> non_const=std::const_pointer_cast<PhysicalAsset>(p);
            return py::cast<bool>(matcher(non_const));
        };
        return v.getAssets<>(wrapped);
    },py::keep_alive<0,2>())
    .def("getAgent",&SimulationManager::getAgent<>)
    .def("getAgents",py::overload_cast<>(&SimulationManager::getAgents<>,py::const_))
    .def("getAgents",[](SimulationManager& v,const py::object& matcher){
        MapIterable<Agent,std::string,Agent>::MatcherType wrapped=[matcher](MapIterable<Agent,std::string,Agent>::ConstSharedType p)->bool {
            std::shared_ptr<Agent> non_const=std::const_pointer_cast<Agent>(p);
            return py::cast<bool>(matcher(non_const));
        };
        return v.getAgents<>(wrapped);
    },py::keep_alive<0,2>())
    .def("getController",&SimulationManager::getController<>)
    .def("getControllers",py::overload_cast<>(&SimulationManager::getControllers<>,py::const_))
    .def("getControllers",[](SimulationManager& v,const py::object& matcher){
        MapIterable<Controller,std::string,Controller>::MatcherType wrapped=[matcher](MapIterable<Controller,std::string,Controller>::ConstSharedType p)->bool {
            std::shared_ptr<Controller> non_const=std::const_pointer_cast<Controller>(p);
            return py::cast<bool>(matcher(non_const));
        };
        return v.getControllers<>(wrapped);
    },py::keep_alive<0,2>())
    .def("getRuler",&SimulationManager::getRuler<>)
    .def("getViewer",&SimulationManager::getViewer<>)
    .def("getRewardGenerator",[](const SimulationManager& v,const int& idx){return v.getRewardGenerator(idx);})
    .def("getRewardGenerators",py::overload_cast<>(&SimulationManager::getRewardGenerators<>,py::const_))
    .def("getRewardGenerators",[](SimulationManager& v,const py::object& matcher){
        VectorIterable<Reward,Reward>::MatcherType wrapped=[matcher](VectorIterable<Reward,Reward>::ConstSharedType p)->bool {
            std::shared_ptr<Reward> non_const=std::const_pointer_cast<Reward>(p);
            return py::cast<bool>(matcher(non_const));
        };
        return v.getRewardGenerators<>(wrapped);
    },py::keep_alive<0,2>())
    .def("getCallback",&SimulationManager::getCallback<>)
    .def("getCallbacks",py::overload_cast<>(&SimulationManager::getCallbacks<>,py::const_))
    .def("getCallbacks",[](SimulationManager& v,const py::object& matcher){
        MapIterable<Callback,std::string,Callback>::MatcherType wrapped=[matcher](MapIterable<Callback,std::string,Callback>::ConstSharedType p)->bool {
            std::shared_ptr<Callback> non_const=std::const_pointer_cast<Callback>(p);
            return py::cast<bool>(matcher(non_const));
        };
        return v.getCallbacks<>(wrapped);
    },py::keep_alive<0,2>())
    .def("getLogger",&SimulationManager::getLogger<>)
    .def("getLoggers",py::overload_cast<>(&SimulationManager::getLoggers<>,py::const_))
    .def("getLoggers",[](SimulationManager& v,const py::object& matcher){
        MapIterable<Callback,std::string,Callback>::MatcherType wrapped=[matcher](MapIterable<Callback,std::string,Callback>::ConstSharedType p)->bool {
            std::shared_ptr<Callback> non_const=std::const_pointer_cast<Callback>(p);
            return py::cast<bool>(matcher(non_const));
        };
        return v.getLoggers<>(wrapped);
    },py::keep_alive<0,2>())
    DEF_FUNC(SimulationManager,printOrderedAssets)
    DEF_READONLY(SimulationManager,observation_space)
    DEF_READONLY(SimulationManager,action_space)
    ;

    expose_common_class<SimulationManagerAccessorBase>(m,"SimulationManagerAccessorBase")
    .def(py_init<const std::shared_ptr<SimulationManager>&>())
    .def(py_init<const std::shared_ptr<SimulationManagerAccessorBase>&>())
    DEF_FUNC(SimulationManagerAccessorBase,getEpoch)
    DEF_FUNC(SimulationManagerAccessorBase,getTime)
    DEF_FUNC(SimulationManagerAccessorBase,getElapsedTime)
    DEF_FUNC(SimulationManagerAccessorBase,getBaseTimeStep)
    DEF_FUNC(SimulationManagerAccessorBase,getTickCount)
    DEF_FUNC(SimulationManagerAccessorBase,getTickCountAt)
    DEF_FUNC(SimulationManagerAccessorBase,getDefaultAgentStepInterval)
    DEF_FUNC(SimulationManagerAccessorBase,getInternalStepCount)
    DEF_FUNC(SimulationManagerAccessorBase,getExposedStepCount)
    .def("createVirtualSimulationManager",py::overload_cast<const nl::json&>(&SimulationManagerAccessorBase::createVirtualSimulationManager<>))
    DEF_FUNC(SimulationManagerAccessorBase,getRootCRS)
    DEF_FUNC(SimulationManagerAccessorBase,getCRSConfig)
    .def("getCRSes",py::overload_cast<>(&SimulationManagerAccessorBase::getCRSes<>,py::const_))
    .def("getCRSes",[](SimulationManagerAccessorBase& v,const py::object& matcher){
        MapIterable<CoordinateReferenceSystem,EntityIdentifier,CoordinateReferenceSystem>::MatcherType wrapped=[matcher](MapIterable<CoordinateReferenceSystem,EntityIdentifier,CoordinateReferenceSystem>::ConstSharedType p)->bool {
            std::shared_ptr<CoordinateReferenceSystem> non_const=std::const_pointer_cast<CoordinateReferenceSystem>(p);
            return py::cast<bool>(matcher(non_const));
        };
        return v.getCRSes<>(wrapped);
    },py::keep_alive<0,2>())
    .def("getCRS",&SimulationManagerAccessorBase::getCRS<>)
    DEF_FUNC(SimulationManagerAccessorBase,getTeams)
    ;

    expose_common_class<SimulationManagerAccessorForCallback>(m,"SimulationManagerAccessorForCallback")
    DEF_FUNC(SimulationManagerAccessorForCallback,getInternalMotionStatesOfAffineCRSes)
    DEF_FUNC(SimulationManagerAccessorForCallback,setInternalMotionStatesOfAffineCRSes)
    DEF_FUNC(SimulationManagerAccessorForCallback,getUUID)
    DEF_FUNC(SimulationManagerAccessorForCallback,getUUIDOfEntity)
    DEF_FUNC(SimulationManagerAccessorForCallback,getEntityConstructionInfo)
    .def("requestReconfigure",py::overload_cast<const nl::json&>(&SimulationManagerAccessorForCallback::requestReconfigure))
    .def("requestReconfigure",py::overload_cast<const nl::json&,const nl::json&>(&SimulationManagerAccessorForCallback::requestReconfigure))
    DEF_FUNC(SimulationManagerAccessorForCallback,addEventHandler)
    .def("addEventHandler",[](SimulationManagerAccessorForCallback& v,const std::string& name,std::function<void(const nl::json&)> handler){
        v.addEventHandler(name,handler);
    })
    DEF_FUNC(SimulationManagerAccessorForCallback,triggerEvent)
    .def("triggerEvent",[](SimulationManagerAccessorForCallback& v,const std::string& name, const nl::json& args){
        v.triggerEvent(name,args);
    })
    .def_property("observation_space",&SimulationManagerAccessorForCallback::observation_space,&SimulationManagerAccessorForCallback::observation_space)
    .def_property("action_space",&SimulationManagerAccessorForCallback::action_space,&SimulationManagerAccessorForCallback::action_space)
    .def_property("observation",&SimulationManagerAccessorForCallback::observation,&SimulationManagerAccessorForCallback::observation)
    .def_property("action",&SimulationManagerAccessorForCallback::action,&SimulationManagerAccessorForCallback::action)
    .def_property("dones",&SimulationManagerAccessorForCallback::dones,&SimulationManagerAccessorForCallback::dones)
    .def_property("scores",&SimulationManagerAccessorForCallback::scores,&SimulationManagerAccessorForCallback::scores)
    .def_property("rewards",&SimulationManagerAccessorForCallback::rewards,&SimulationManagerAccessorForCallback::rewards)
    .def_property("totalRewards",&SimulationManagerAccessorForCallback::totalRewards,&SimulationManagerAccessorForCallback::totalRewards)
    .def_property_readonly("experts",&SimulationManagerAccessorForCallback::experts)
    .def_property_readonly("agentsActed",&SimulationManagerAccessorForCallback::agentsActed)
    .def_property_readonly("agentsToAct",&SimulationManagerAccessorForCallback::agentsToAct)
    .def_property_readonly("worker_index",&SimulationManagerAccessorForCallback::worker_index)
    .def_property_readonly("vector_index",&SimulationManagerAccessorForCallback::vector_index)
    .def_property_readonly("episode_index",&SimulationManagerAccessorForCallback::episode_index)
    .def_property("manualDone",&SimulationManagerAccessorForCallback::manualDone,&SimulationManagerAccessorForCallback::setManualDone)
    .def("getAgentStepCount",py::overload_cast<const std::string&>(&SimulationManagerAccessorForCallback::getAgentStepCount,py::const_))
    .def("getAgentStepCount",py::overload_cast<const std::shared_ptr<Agent>&>(&SimulationManagerAccessorForCallback::getAgentStepCount,py::const_))
    DEF_FUNC(SimulationManagerAccessorForCallback,setManualDone)
    DEF_FUNC(SimulationManagerAccessorForCallback,requestToKillAsset)
    DEF_FUNC(SimulationManagerAccessorForCallback,addDependency)
    DEF_FUNC(SimulationManagerAccessorForCallback,removeDependency)
    DEF_FUNC(SimulationManagerAccessorForCallback,addDependencyGenerator)
    .def("getAsset",&SimulationManagerAccessorForCallback::getAsset<>)
    .def("getAssets",py::overload_cast<>(&SimulationManagerAccessorForCallback::getAssets<>,py::const_))
    .def("getAssets",[](SimulationManagerAccessorForCallback& v,const py::object& matcher){
        MapIterable<PhysicalAsset,std::string,PhysicalAsset>::MatcherType wrapped=[matcher](MapIterable<PhysicalAsset,std::string,PhysicalAsset>::ConstSharedType p)->bool {
            std::shared_ptr<PhysicalAsset> non_const=std::const_pointer_cast<PhysicalAsset>(p);
            return py::cast<bool>(matcher(non_const));
        };
        return v.getAssets<>(wrapped);
    },py::keep_alive<0,2>())
    .def("getAgent",&SimulationManagerAccessorForCallback::getAgent<>)
    .def("getAgents",py::overload_cast<>(&SimulationManagerAccessorForCallback::getAgents<>,py::const_))
    .def("getAgents",[](SimulationManagerAccessorForCallback& v,const py::object& matcher){
        MapIterable<Agent,std::string,Agent>::MatcherType wrapped=[matcher](MapIterable<Agent,std::string,Agent>::ConstSharedType p)->bool {
            std::shared_ptr<Agent> non_const=std::const_pointer_cast<Agent>(p);
            return py::cast<bool>(matcher(non_const));
        };
        return v.getAgents<>(wrapped);
    },py::keep_alive<0,2>())
    .def("getController",&SimulationManagerAccessorForCallback::getController<>)
    .def("getControllers",py::overload_cast<>(&SimulationManagerAccessorForCallback::getControllers<>,py::const_))
    .def("getControllers",[](SimulationManagerAccessorForCallback& v,const py::object& matcher){
        MapIterable<Controller,std::string,Controller>::MatcherType wrapped=[matcher](MapIterable<Controller,std::string,Controller>::ConstSharedType p)->bool {
            std::shared_ptr<Controller> non_const=std::const_pointer_cast<Controller>(p);
            return py::cast<bool>(matcher(non_const));
        };
        return v.getControllers<>(wrapped);
    },py::keep_alive<0,2>())
    .def("getRuler",&SimulationManagerAccessorForCallback::getRuler<>)
    .def("getViewer",&SimulationManagerAccessorForCallback::getViewer<>)
    .def("getRewardGenerator",[](SimulationManagerAccessorForCallback& v,const int& idx){return v.getRewardGenerator(idx);})
    .def("getRewardGenerators",py::overload_cast<>(&SimulationManagerAccessorForCallback::getRewardGenerators<>,py::const_))
    .def("getRewardGenerators",[](SimulationManagerAccessorForCallback& v,const py::object& matcher){
        VectorIterable<Reward,Reward>::MatcherType wrapped=[matcher](VectorIterable<Reward,Reward>::ConstSharedType p)->bool {
            std::shared_ptr<Reward> non_const=std::const_pointer_cast<Reward>(p);
            return py::cast<bool>(matcher(non_const));
        };
        return v.getRewardGenerators<>(wrapped);
    },py::keep_alive<0,2>())
    .def("getCallback",&SimulationManagerAccessorForCallback::getCallback<>)
    .def("getCallbacks",py::overload_cast<>(&SimulationManagerAccessorForCallback::getCallbacks<>,py::const_))
    .def("getCallbacks",[](SimulationManagerAccessorForCallback& v,const py::object& matcher){
        MapIterable<Callback,std::string,Callback>::MatcherType wrapped=[matcher](MapIterable<Callback,std::string,Callback>::ConstSharedType p)->bool {
            std::shared_ptr<Callback> non_const=std::const_pointer_cast<Callback>(p);
            return py::cast<bool>(matcher(non_const));
        };
        return v.getCallbacks<>(wrapped);
    },py::keep_alive<0,2>())
    .def("getLogger",&SimulationManagerAccessorForCallback::getLogger<>)
    .def("getLoggers",py::overload_cast<>(&SimulationManagerAccessorForCallback::getLoggers<>,py::const_))
    .def("getLoggers",[](SimulationManagerAccessorForCallback& v,const py::object& matcher){
        MapIterable<Callback,std::string,Callback>::MatcherType wrapped=[matcher](MapIterable<Callback,std::string,Callback>::ConstSharedType p)->bool {
            std::shared_ptr<Callback> non_const=std::const_pointer_cast<Callback>(p);
            return py::cast<bool>(matcher(non_const));
        };
        return v.getLoggers<>(wrapped);
    },py::keep_alive<0,2>())
    DEF_FUNC(SimulationManagerAccessorForCallback,getManagerConfig)
    .def("getFactoryModelConfig", &SimulationManagerAccessorForCallback::getFactoryModelConfig, py::arg("withDefaultModels")=true)
    ;

    expose_common_class<SimulationManagerAccessorForPhysicalAsset>(m,"SimulationManagerAccessorForPhysicalAsset")
    DEF_FUNC(SimulationManagerAccessorForPhysicalAsset,triggerEvent)
    DEF_FUNC(SimulationManagerAccessorForPhysicalAsset,requestToKillAsset)
    DEF_FUNC(SimulationManagerAccessorForPhysicalAsset,addDependency)
    DEF_FUNC(SimulationManagerAccessorForPhysicalAsset,removeDependency)
    DEF_FUNC(SimulationManagerAccessorForPhysicalAsset,addDependencyGenerator)
    .def("getAsset",&SimulationManagerAccessorForPhysicalAsset::getAsset<>)
    .def("getAssets",py::overload_cast<>(&SimulationManagerAccessorForPhysicalAsset::getAssets<>,py::const_))
    .def("getAssets",py::overload_cast<MapIterable<PhysicalAsset,std::string,PhysicalAsset>::MatcherType>(&SimulationManagerAccessorForPhysicalAsset::getAssets<>,py::const_),py::keep_alive<0,2>())
    .def("getAgent",&SimulationManagerAccessorForPhysicalAsset::getAgent<>)
    .def("getAgents",py::overload_cast<>(&SimulationManagerAccessorForPhysicalAsset::getAgents<>,py::const_))
    .def("getAgents",py::overload_cast<MapIterable<Agent,std::string,Agent>::MatcherType>(&SimulationManagerAccessorForPhysicalAsset::getAgents<>,py::const_),py::keep_alive<0,2>())
    .def("getController",&SimulationManagerAccessorForPhysicalAsset::getController<>)
    .def("getControllers",py::overload_cast<>(&SimulationManagerAccessorForPhysicalAsset::getControllers<>,py::const_))
    .def("getControllers",py::overload_cast<MapIterable<Controller,std::string,Controller>::MatcherType>(&SimulationManagerAccessorForPhysicalAsset::getControllers<>,py::const_),py::keep_alive<0,2>())
    .def("getRuler",&SimulationManagerAccessorForPhysicalAsset::getRuler<>)
    .def("createAgent",&SimulationManagerAccessorForPhysicalAsset::createAgent<>)
    DEF_FUNC(SimulationManagerAccessorForPhysicalAsset,requestInvitationToCommunicationBuffer)
    DEF_FUNC(SimulationManagerAccessorForPhysicalAsset,generateCommunicationBuffer)
    ;

    expose_common_class<SimulationManagerAccessorForAgent>(m,"SimulationManagerAccessorForAgent")
    DEF_FUNC(SimulationManagerAccessorForAgent,requestInvitationToCommunicationBuffer)
    .def("getRuler",&SimulationManagerAccessorForAgent::getRuler)
    ;
}

ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
