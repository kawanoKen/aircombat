// Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#include "Entity.h"
#include "Utility.h"
#include "SimulationManager.h"
#include <limits>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>
ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)
using namespace util;

//
// class EntityManager
//

std::shared_mutex EntityManager::static_mtx;
std::map<std::int32_t,std::weak_ptr<EntityManager>> EntityManager::managerInstances;
std::map<std::int32_t,boost::uuids::uuid> EntityManager::uuidOfManagerInstances;
std::map<boost::uuids::uuid,std::int32_t> EntityManager::uuidOfManagerInstancesR;
std::int32_t EntityManager::_worker_index=0;

EntityManager::EntityManager(const std::int32_t& vector_index_){
    isDuringDeserialization=false;
    uuid=boost::uuids::random_generator()();
    _vector_index=vector_index_;
    _episode_index=0;
    nextEpisodeIndependentEntityIndex=0;
    nextEntityIndex=0;
    factory=Factory::create();
}
EntityManager::~EntityManager(){
    std::lock_guard<std::shared_mutex> lock(static_mtx);
    auto found=managerInstances.find(_vector_index);
    if(found!=managerInstances.end() && uuidOfManagerInstances.at(_vector_index)==uuid){
        managerInstances.erase(found);
        uuidOfManagerInstances.erase(uuidOfManagerInstances.find(_vector_index));
        uuidOfManagerInstancesR.erase(uuidOfManagerInstancesR.find(uuid));
    }
}
void EntityManager::addInstance(const std::shared_ptr<EntityManager>& manager){
    if(manager){
        auto vector_index=manager->getVectorIndex();
        std::lock_guard<std::shared_mutex> lock(static_mtx);
        {
            auto found=managerInstances.find(vector_index);
            if(found!=managerInstances.end()){
                if(found->second.expired()){
                    managerInstances.erase(found);
                    auto uuid=uuidOfManagerInstances.at(vector_index);
                    uuidOfManagerInstances.erase(uuidOfManagerInstances.find(vector_index));
                    uuidOfManagerInstancesR.erase(uuidOfManagerInstancesR.find(uuid));
                }else{
                    throw std::runtime_error(
                        "EntityManager instance with vector_index="+std::to_string(vector_index)+" already exists."
                    );
                }
            }
        }
        managerInstances[vector_index]=manager;
        {
            auto found=uuidOfManagerInstancesR.find(manager->uuid);
            if(found!=uuidOfManagerInstancesR.end()){
                if(found->second!=vector_index){
                    throw std::runtime_error("There is already an EntityManager instance with uuid="+boost::uuids::to_string(manager->uuid)+".");
                }
            }
        }
        uuidOfManagerInstances[vector_index]=manager->uuid;
        uuidOfManagerInstancesR[manager->uuid]=vector_index;
    }
}
std::shared_ptr<EntityManager> EntityManager::getManagerInstance(std::int32_t vector_index){
    auto found=managerInstances.find(vector_index);
    if(found!=managerInstances.end()){
        if(found->second.expired()){
            throw std::runtime_error("The EntityManager instance with vector_index="+std::to_string(vector_index)+" is expired.");
        }else{
            return found->second.lock();
        }
    }else{
        throw std::runtime_error("There is no EntityManager instance with vector_index="+std::to_string(vector_index)+".");
    }
}
std::shared_ptr<EntityManager> EntityManager::getManagerInstance(const boost::uuids::uuid& uuid){
    auto found=uuidOfManagerInstancesR.find(uuid);
    if(found!=uuidOfManagerInstancesR.end()){
        return getManagerInstance(uuidOfManagerInstancesR.at(uuid));
    }else{
        throw std::runtime_error("There is no EntityManager instance with uuid="+boost::uuids::to_string(uuid)+".");
    }
}
boost::uuids::uuid EntityManager::getUUID() const{
    return uuid;
}
void EntityManager::setUUID(boost::uuids::uuid new_uuid){
    std::lock_guard<std::shared_mutex> static_lock(static_mtx);
    std::lock_guard<std::recursive_mutex> lock(mtx);
    if(uuidOfManagerInstancesR.find(new_uuid)==uuidOfManagerInstancesR.end()){
        boost::uuids::uuid old_uuid=uuid;
        auto found=uuidOfManagerInstancesR.find(old_uuid);
        if(found!=uuidOfManagerInstancesR.end()){
            uuidOfManagerInstancesR.erase(found);
        }
    }else{
        throw std::runtime_error("There is already an EntityManager instance with uuid="+boost::uuids::to_string(new_uuid)+".");
    }
    uuidOfManagerInstancesR[new_uuid]=_vector_index;
    uuidOfManagerInstances[_vector_index]=new_uuid;
    uuid=new_uuid;
}
boost::uuids::uuid EntityManager::getUUIDOfEntity(const std::shared_ptr<const Entity>& entity) const{
    assert(entity);
    auto id=entity->getEntityID();
    auto found=uuidOfEntity.find(id);
    if(found!=uuidOfEntity.end()){
        return found->second;
    }else{
        throw std::runtime_error("The given Entity instance does not have the UUID. It is not monitored.");
    }
}
void EntityManager::setWorkerIndex(const std::int32_t& w){
    std::lock_guard<std::shared_mutex> lock(static_mtx);
    if(_worker_index!=w){
        for(auto mit = managerInstances.begin(); mit != managerInstances.end();){
            auto [v,wManager] = *mit;
            auto manager=wManager.lock();
            std::lock_guard<std::recursive_mutex> lock(manager->mtx);
            manager->_worker_index=w;
            std::map<EntityIdentifier,std::shared_ptr<Entity>> copiedEntities=manager->entities;
            manager->entities.clear();
            for(auto it = copiedEntities.begin(); it != copiedEntities.end();){
                auto [id, entity] = *it;
                entity->entityID.worker_index=w;
                manager->entities[entity->entityID]=entity;
                ++it;
            }
            std::map<EntityIdentifier,std::weak_ptr<Entity>> copiedUnmanagedEntities=manager->unmanagedEntities;
            manager->unmanagedEntities.clear();
            for(auto it = copiedUnmanagedEntities.begin(); it != copiedUnmanagedEntities.end();){
                auto [id, wEntity] = *it;
                wEntity.lock()->entityID.worker_index=w;
                manager->unmanagedEntities[wEntity.lock()->entityID]=wEntity;
                ++it;
            }
            for(auto it = manager->entityNames.begin(); it != manager->entityNames.end();){
                it->second.worker_index=w;
                ++it;
            }
            std::set<EntityIdentifier> copiedCreatedEntitiesSinceLastSerializations=manager->createdEntitiesSinceLastSerialization;
            manager->createdEntitiesSinceLastSerialization.clear();
            for(auto id : copiedCreatedEntitiesSinceLastSerializations){
                manager->createdEntitiesSinceLastSerialization.insert(
                    EntityIdentifier(w,id.vector_index,id.episode_index,id.entity_index)
                );
            }
            std::set<EntityIdentifier> copiedRemovedEntitiesSinceLastSerializations=manager->removedEntitiesSinceLastSerialization;
            manager->removedEntitiesSinceLastSerialization.clear();
            for(auto& id : copiedRemovedEntitiesSinceLastSerializations){
                manager->removedEntitiesSinceLastSerialization.insert(
                    EntityIdentifier(w,id.vector_index,id.episode_index,id.entity_index)
                );
            }
            ++mit;
        }
    }
    _worker_index=w;
}
std::int32_t EntityManager::getWorkerIndex(){
    std::shared_lock<std::shared_mutex> lock(static_mtx);
    return _worker_index;
}
std::int32_t EntityManager::getUnusedVectorIndex(){
    std::shared_lock<std::shared_mutex> lock(static_mtx);
    std::int32_t ret=-1;
    while(ret<std::numeric_limits<std::int32_t>::max()){
        ret++;
        if(managerInstances.find(ret)==managerInstances.end()){
            return ret;
        }
    }
    throw std::runtime_error("There is no unused vector_index for worker_index="+std::to_string(_worker_index)+".");
}
std::int32_t EntityManager::getVectorIndex() const{
    std::lock_guard<std::recursive_mutex> lock(mtx);
    return _vector_index;
}
std::int32_t EntityManager::getEpisodeIndex() const{
    std::lock_guard<std::recursive_mutex> lock(mtx);
    return _episode_index;
}
std::int32_t EntityManager::getNextEpisodeIndependentEntityIndex(const std::int32_t& preference,bool increment){
    std::lock_guard<std::recursive_mutex> lock(mtx);
    if(preference!=EntityIdentifier::invalid_entity_index){
        if(entities.find(EntityIdentifier(
            _worker_index,
            _vector_index,
            EntityIdentifier::invalid_episode_index,
            preference
        ))==entities.end()){
            return preference;
        }
    }
    --nextEpisodeIndependentEntityIndex;
    while(nextEpisodeIndependentEntityIndex<=std::numeric_limits<std::int32_t>::max()){
        ++nextEpisodeIndependentEntityIndex;
        if(entities.find(EntityIdentifier(
            _worker_index,
            _vector_index,
            EntityIdentifier::invalid_episode_index,
            nextEpisodeIndependentEntityIndex
        ))==entities.end()){
            if(increment){
                return nextEpisodeIndependentEntityIndex++;
            }else{
                return nextEpisodeIndependentEntityIndex;
            }
        }
    }
    throw std::runtime_error(
        "There is no unused non-episodic entity_index for w="+std::to_string(_worker_index)
        +", v="+std::to_string(_vector_index)+"."
    );
}
std::int32_t EntityManager::getNextEntityIndex(const std::int32_t& preference,bool increment){
    std::lock_guard<std::recursive_mutex> lock(mtx);
    if(preference!=EntityIdentifier::invalid_entity_index){
        if(entities.find(EntityIdentifier(
            _worker_index,
            _vector_index,
            _episode_index,
            preference
        ))==entities.end()){
            return preference;
        }
    }
    --nextEntityIndex;
    while(nextEntityIndex<=std::numeric_limits<std::int32_t>::max()){
        ++nextEntityIndex;
        if(entities.find(EntityIdentifier(
            _worker_index,
            _vector_index,
            _episode_index,
            nextEntityIndex
        ))==entities.end()){
            if(increment){
                return nextEntityIndex++;
            }else{
                return nextEntityIndex;
            }
        }
    }
    throw std::runtime_error(
        "There is no unused episodic entity_index for w="+std::to_string(_worker_index)
        +", v="+std::to_string(_vector_index)+"."
    );
}
EntityIdentifier EntityManager::getNextEpisodeIndependentEntityIdentifier(const EntityIdentifier& preference,bool increment){
    return EntityIdentifier(
        getWorkerIndex(),
        _vector_index,
        EntityIdentifier::invalid_episode_index,
        getNextEpisodeIndependentEntityIndex(preference.entity_index,increment)
    );
}
EntityIdentifier EntityManager::getNextEntityIdentifier(const EntityIdentifier& preference,bool increment){
    return EntityIdentifier(
        getWorkerIndex(),
        _vector_index,
        _episode_index,
        getNextEntityIndex(preference.entity_index,increment)
    );
}
void EntityManager::purge(){
    std::lock_guard<std::shared_mutex> lock(static_mtx);
    auto found=managerInstances.find(_vector_index);
    if(found!=managerInstances.end()){
        if(!found->second.expired() && found->second.lock()==shared_from_this()){
            managerInstances.erase(found);
            uuidOfManagerInstances.erase(uuidOfManagerInstances.find(_vector_index));
            uuidOfManagerInstancesR.erase(uuidOfManagerInstancesR.find(uuid));
        }
    }
}
void EntityManager::clearModels(){
    factory->reset();
}
void EntityManager::addModel(const std::string& baseName,const std::string& modelName,const nl::json& modelConfig_){
    factory->addModel(baseName, modelName, modelConfig_);
}
void EntityManager::addModelsFromJson(const nl::json& j){
    factory->addModelsFromJson(j);
}
void EntityManager::addModelsFromJsonFile(const std::string& filePath){
    factory->addModelsFromJsonFile(filePath);
}
nl::json EntityManager::getModelConfigs(bool withDefaultModels) const{
    return factory->getModelConfigs(withDefaultModels);
}
void EntityManager::reconfigureModelConfigs(const nl::json& j){
    factory->reconfigureModelConfigs(j);
}

// Entityの生成
std::shared_ptr<Entity> EntityManager::createEntityImpl(bool isManaged, bool isEpisodic, const std::string& baseName,const std::string& className,const std::string& modelName,const nl::json& modelConfig,const nl::json& instanceConfig, const FactoryHelperChain& helperChain, const std::shared_ptr<const Entity>& caller){
    nl::json ic=instanceConfig;
    bool isDummy = modelConfig.is_null() && instanceConfig.is_null();
    EntityIdentifier id;
    if(!isDummy){
        EntityIdentifier preference;
        if(ic.contains("entityIdentifier")){
            preference.load_from_json(ic.at("entityIdentifier"));
            if(isEpisodic && preference.episode_index!=getEpisodeIndex()){
                throw std::runtime_error(
                    "EntityManager::createEntityImpl failed. Mismatch of 'episode_index' with the given EntityIdentifier preference."
                );
            }else if(!isEpisodic && preference.episode_index!=EntityIdentifier::invalid_episode_index){
                throw std::runtime_error(
                    "EntityManager::createEntityImpl failed. Mismatch of 'episode_index' with the given EntityIdentifier preference."
                );
            }
        }
        if(isEpisodic){
            id=getNextEntityIdentifier(preference,true);
        }else{
            if(caller && caller->isEpisodic()){
                throw std::runtime_error(
                    "EntityManager::createEntityImpl failed. Non-episodic entity can be created only by non-episodic entities. caller="+std::string(caller ? "{'entityFullName':"+caller->getFullName()+",'entityIdentifier':"+caller->getEntityID().toString() : "nullptr")
                    +", baseName="+baseName+", className="+className+", modelName="+modelName
                    +", modelConfig="+modelConfig.dump()
                    +", instanceConfig="+instanceConfig.dump()
                );
            }
            id=getNextEpisodeIndependentEntityIdentifier(preference,true);
        }
    }
    auto ret=factory->createImpl(baseName, className, modelName, modelConfig, ic, helperChain, caller);
    ret->setEntityManager(this->shared_from_this());
    ret->setEntityID(id);
    if(ic.contains("entityFullName")){
        ret->setFullName(ic.at("entityFullName"));
    }
    ret->setManagerAccessor(createAccessorFor(ret));
    if(!isDummy){
        registerEntity(isManaged,ret);
        if(!isDuringDeserialization){
            try{
                ret->initialize();
            }catch(std::exception& ex){
                throw std::runtime_error(ret->getDemangledName()+"::initialize() failed. [reason='"+std::string(ex.what())+"']");
            }
            try{
                ret->makeChildren();
            }catch(std::exception& ex){
                throw std::runtime_error(ret->getDemangledName()+"::makeChildren() failed. [reason='"+std::string(ex.what())+"']");
            }
        }
    }
    return ret;
}
std::shared_ptr<Entity> EntityManager::createUnmanagedEntityOverloader(bool isEpisodic, const std::string& baseName,const std::string& className,const std::string& modelName,const nl::json& modelConfig,const nl::json& instanceConfig, const FactoryHelperChain& helperChain){
    return createEntityImpl(false, isEpisodic, baseName, className, modelName, modelConfig, instanceConfig, helperChain, nullptr);
}
std::shared_ptr<Entity> EntityManager::createUnmanagedEntityOverloader(bool isEpisodic, const std::string& baseName,const std::string& className,const std::string& modelName,const nl::json& modelConfig,const nl::json& instanceConfig, const std::shared_ptr<const Entity>& caller){
    if(caller){
        return createEntityImpl(false, isEpisodic, baseName, className, modelName, modelConfig, instanceConfig, caller->getFactoryHelperChain(), caller);
    }else{
        return createEntityImpl(false, isEpisodic, baseName, className, modelName, modelConfig, instanceConfig, FactoryHelperChain(), caller);
    }
}
std::shared_ptr<Entity> EntityManager::createUnmanagedEntityOverloader(bool isEpisodic, const std::string& baseName,const std::string& modelName,const nl::json& instanceConfig, const FactoryHelperChain& helperChain){
    return createUnmanagedEntityOverloader(isEpisodic,baseName,"",modelName,nl::json(),instanceConfig,helperChain);
}
std::shared_ptr<Entity> EntityManager::createUnmanagedEntityOverloader(bool isEpisodic, const std::string& baseName,const std::string& modelName,const nl::json& instanceConfig, const std::shared_ptr<const Entity>& caller){
    return createUnmanagedEntityOverloader(isEpisodic,baseName,"",modelName,nl::json(),instanceConfig,caller);
}
std::shared_ptr<Entity> EntityManager::createUnmanagedEntityByClassNameOverloader(bool isEpisodic, const std::string& baseName,const std::string& className,const nl::json& modelConfig,const nl::json& instanceConfig, const FactoryHelperChain& helperChain){
    return createUnmanagedEntityOverloader(isEpisodic,baseName,className,"",modelConfig,instanceConfig,helperChain);
}
std::shared_ptr<Entity> EntityManager::createUnmanagedEntityByClassNameOverloader(bool isEpisodic, const std::string& baseName,const std::string& className,const nl::json& modelConfig,const nl::json& instanceConfig, const std::shared_ptr<const Entity>& caller){
   return createUnmanagedEntityOverloader(isEpisodic,baseName,className,"",modelConfig,instanceConfig,caller);
}
std::shared_ptr<Entity> EntityManager::createEntityImpl(bool isManaged, const nl::json& j, const FactoryHelperChain& helperChain, const std::shared_ptr<const Entity>& caller){
    if(!j.is_object()){
        throw std::runtime_error("create Entity with single json can only accept a json object.");
    }
    bool isEpisodic=false;
    if(j.contains("isEpisodic")){
        isEpisodic=j.at("isEpisodic");
    }

    std::string baseName;
    if(j.contains("baseName")){
        baseName=j.at("baseName");
    }else{
        throw std::runtime_error("The key 'baseName' is missing in the json given to EntityManager::createEntityImpl");
    }

    std::string className;
    if(j.contains("class")){
        className=j.at("class");
    }else if(j.contains("className")){
        className=j.at("className");
    }

    std::string modelName;
    if(j.contains("model")){
        modelName=j.at("model");
    }else if(j.contains("modelName")){
        modelName=j.at("modelName");
    }

    nl::json jsonNull;
    const nl::json& modelConfig = (
        j.contains("config") ? j.at("config") : (
            j.contains("modelConfig") ? j.at("modelConfig") : (
                jsonNull
            )
        )
    );

    nl::json jsonEmptyObject=nl::json::object();
    nl::json instanceConfig = (
        j.contains("instanceConfig") ? j.at("instanceConfig") : (
            jsonEmptyObject
        )
    );
    if(j.contains("entityIdentifier")){
        instanceConfig["entityIdentifier"]=j.at("entityIdentifier");
    }
    if(j.contains("entityFullName")){
        instanceConfig["entityFullName"]=j.at("entityFullName");
    }
    if(j.contains("uuid")){
        instanceConfig["uuid"]=j.at("uuid");
    }
    if(j.contains("uuidForAccessor")){
        instanceConfig["uuidForAccessor"]=j.at("uuidForAccessor");
    }
    if(j.contains("uuidForTrack")){
        instanceConfig["uuidForTrack"]=j.at("uuidForTrack");
    }

    return createEntityImpl(isManaged, isEpisodic, baseName, className, modelName, modelConfig, instanceConfig, helperChain, caller);

}
std::shared_ptr<Entity> EntityManager::createUnmanagedEntityOverloader(const nl::json& j, const FactoryHelperChain& helperChain){
    return createEntityImpl(false, j, helperChain, nullptr);

}
std::shared_ptr<Entity> EntityManager::createUnmanagedEntityOverloader(const nl::json& j, const std::shared_ptr<const Entity>& caller){
    if(caller){
        return createEntityImpl(false, j, caller->getFactoryHelperChain(), caller);
    }else{
        return createEntityImpl(false, j, FactoryHelperChain(), caller);
    }
}
std::shared_ptr<Entity> EntityManager::createOrGetEntityImpl(bool isManaged, const nl::json& j, const FactoryHelperChain& helperChain, const std::shared_ptr<const Entity>& caller){
    if(j.is_object()){
        // get if exists
        try{
            auto ret=Entity::from_json(j);
            if(ret){
                return ret;
            }
        }catch(...){
        }
        if(j.contains("entityIdentifier")){
            EntityIdentifier id=j.at("entityIdentifier");
            auto ret=getEntityByIDImpl(id,caller);
            if(ret){
                return ret;
            }
        }
        if(j.contains("entityFullName")){
            std::string name=j.at("entityFullName");
            auto ret=getEntityByNameImpl(name,caller);
            if(ret){
                return ret;
            }
        }
        // create
        return createEntityImpl(isManaged, j, helperChain, caller);
    }else{
        throw std::runtime_error("EntityManager::createOrGetEntityImpl only accepts json object. j="+j.dump());
    }
}
std::shared_ptr<Entity> EntityManager::createOrGetUnmanagedEntityOverloader(const nl::json& j, const FactoryHelperChain& helperChain){
    return createOrGetEntityImpl(false, j, helperChain, nullptr);
}
std::shared_ptr<Entity> EntityManager::createOrGetUnmanagedEntityOverloader(const nl::json& j, const std::shared_ptr<const Entity>& caller){
    if(caller){
        return createOrGetEntityImpl(false, j, caller->getFactoryHelperChain(), caller);
    }else{
        return createOrGetEntityImpl(false, j, FactoryHelperChain(), caller);
    }
}
std::shared_ptr<Entity> EntityManager::createEntityOverloader(bool isEpisodic, const std::string& baseName,const std::string& className,const std::string& modelName,const nl::json& modelConfig,const nl::json& instanceConfig, const FactoryHelperChain& helperChain){
    return createEntityImpl(true, isEpisodic, baseName, className, modelName, modelConfig, instanceConfig, helperChain, nullptr);
}
std::shared_ptr<Entity> EntityManager::createEntityOverloader(bool isEpisodic, const std::string& baseName,const std::string& className,const std::string& modelName,const nl::json& modelConfig,const nl::json& instanceConfig, const std::shared_ptr<const Entity>& caller){
    if(caller){
        return createEntityImpl(true, isEpisodic, baseName, className, modelName, modelConfig, instanceConfig, caller->getFactoryHelperChain(), caller);
    }else{
        return createEntityImpl(true, isEpisodic, baseName, className, modelName, modelConfig, instanceConfig, FactoryHelperChain(), caller);
    }
}
std::shared_ptr<Entity> EntityManager::createEntityOverloader(bool isEpisodic, const std::string& baseName,const std::string& modelName,const nl::json& instanceConfig, const FactoryHelperChain& helperChain){
    return createEntityOverloader(isEpisodic,baseName,"",modelName,nl::json(),instanceConfig,helperChain);
}
std::shared_ptr<Entity> EntityManager::createEntityOverloader(bool isEpisodic, const std::string& baseName,const std::string& modelName,const nl::json& instanceConfig, const std::shared_ptr<const Entity>& caller){
    return createEntityOverloader(isEpisodic,baseName,"",modelName,nl::json(),instanceConfig,caller);
}
std::shared_ptr<Entity> EntityManager::createEntityByClassNameOverloader(bool isEpisodic, const std::string& baseName,const std::string& className,const nl::json& modelConfig,const nl::json& instanceConfig, const FactoryHelperChain& helperChain){
    return createEntityOverloader(isEpisodic,baseName,className,"",modelConfig,instanceConfig,helperChain);
}
std::shared_ptr<Entity> EntityManager::createEntityByClassNameOverloader(bool isEpisodic, const std::string& baseName,const std::string& className,const nl::json& modelConfig,const nl::json& instanceConfig, const std::shared_ptr<const Entity>& caller){
    return createEntityOverloader(isEpisodic,baseName,className,"",modelConfig,instanceConfig,caller);
}
std::shared_ptr<Entity> EntityManager::createEntityOverloader(const nl::json& j, const FactoryHelperChain& helperChain){
    return createEntityImpl(true, j, helperChain, nullptr);
}
std::shared_ptr<Entity> EntityManager::createEntityOverloader(const nl::json& j, const std::shared_ptr<const Entity>& caller){
    if(caller){
        return createEntityImpl(true, j, caller->getFactoryHelperChain(), caller);
    }else{
        return createEntityImpl(true, j, FactoryHelperChain(), caller);
    }
}
std::shared_ptr<Entity> EntityManager::createOrGetEntityOverloader(const nl::json& j, const FactoryHelperChain& helperChain){
    return createOrGetEntityImpl(true, j, helperChain, nullptr);
}
std::shared_ptr<Entity> EntityManager::createOrGetEntityOverloader(const nl::json& j, const std::shared_ptr<const Entity>& caller){
    if(caller){
        return createOrGetEntityImpl(true, j, caller->getFactoryHelperChain(), caller);
    }else{
        return createOrGetEntityImpl(true, j, FactoryHelperChain(), caller);
    }
}

// Entity生成処理を派生クラスでカスタマイズするためのフック
void EntityManager::registerEntity(bool isManaged,const std::shared_ptr<Entity>& entity){
    std::lock_guard<std::recursive_mutex> lock(mtx);
    EntityIdentifier id=entity->getEntityID();
    if(getEntityByID(id)){
        throw std::runtime_error(
            "Entity with the given id "+id.toString()+" is already registered."
        );
    }
    if(isManaged){
        entities[id]=entity;
    }else{
        unmanagedEntities[id]=entity;
    }
    entity->isMonitored=true;
    auto fullName=entity->getFullName();
    if(fullName!=""){
        if(getEntityByName(fullName)){
            throw std::runtime_error(
                "Entity with the given name '"+fullName+"' is already registered."
            );
        }
        entityNames[fullName]=id;
    }
    auto uuid_=entity->getUUID();
    if(uuidOfEntityR.find(uuid_)!=uuidOfEntityR.end()){
        throw std::runtime_error(
            "Entity with the given uuid '"+boost::uuids::to_string(uuid_)+"' is already registered."
        );
    }
    uuidOfEntity[id]=uuid_;
    uuidOfEntityR[uuid_]=id;
    uuid_=entity->getUUIDForAccessor();
    if(uuidOfEntityForEntityAccessorsR.find(uuid_)!=uuidOfEntityForEntityAccessorsR.end()){
        throw std::runtime_error(
            "Entity with the given uuid (for accessor) '"+boost::uuids::to_string(uuid_)+"' is already registered."
        );
    }
    uuidOfEntityForEntityAccessors[id]=uuid_;
    uuidOfEntityForEntityAccessorsR[uuid_]=id;
    uuid_=entity->getUUIDForTrack();
    if(uuidOfEntityForTracksR.find(uuid_)!=uuidOfEntityForTracksR.end()){
        throw std::runtime_error(
            "Entity with the given uuid (for track) '"+boost::uuids::to_string(uuid_)+"' is already registered."
        );
    }
    uuidOfEntityForTracks[id]=uuid_;
    uuidOfEntityForTracksR[uuid_]=id;
    entityCreationOrder.push_back(id);
    createdEntitiesSinceLastSerialization.insert(id);
}
std::shared_ptr<EntityManagerAccessor> EntityManager::createAccessorFor(const std::shared_ptr<const Entity>& entity){
    if(entity){
        return EntityManagerAccessor::create<EntityManagerAccessor>(this->shared_from_this());
    }else{
        return nullptr;
    }
}

// Entityの取得、削除
void EntityManager::removeUUIDOfEntity(const EntityIdentifier& id){
    std::lock_guard<std::recursive_mutex> lock(mtx);
    {
        auto found=uuidOfEntityForEntityAccessors.find(id);
        if(found!=uuidOfEntityForEntityAccessors.end()){
            uuidOfEntityForEntityAccessorsR.erase(uuidOfEntityForEntityAccessorsR.find(found->second));
            uuidOfEntityForEntityAccessors.erase(found);
        }
    }
    {
        auto found=uuidOfEntityForTracks.find(id);
        if(found!=uuidOfEntityForTracks.end()){
            uuidOfEntityForTracksR.erase(uuidOfEntityForTracksR.find(found->second));
            uuidOfEntityForTracks.erase(found);
        }
    }
    {
        auto found=uuidOfEntity.find(id);
        if(found!=uuidOfEntity.end()){
            uuidOfEntityR.erase(uuidOfEntityR.find(found->second));
            uuidOfEntity.erase(found);
        }
    }
}
void EntityManager::cleanupEpisode(bool increment){
    std::lock_guard<std::recursive_mutex> lock(mtx);
    for(auto it = unmanagedEntities.begin(); it != unmanagedEntities.end();){
        auto [id, wEntity] = *it;
        if(id.episode_index != EntityIdentifier::invalid_episode_index){
            if(!wEntity.expired()){
                auto entity=wEntity.lock();
                entity->isMonitored=false;
                auto name=entity->getFullName();
                auto nFound=entityNames.find(name);
                if(name!="" && nFound!=entityNames.end()){
                    if(nFound->second==id){
                        entityNames.erase(nFound);
                    }
                }
            }
            it = unmanagedEntities.erase(it);
            removeUUIDOfEntity(id);
            entityCreationOrder.erase(
                std::remove(entityCreationOrder.begin(),entityCreationOrder.end(),id),
                entityCreationOrder.end()
            );
            removedEntitiesSinceLastSerialization.insert(id);
        }else if(wEntity.expired()){
            it = unmanagedEntities.erase(it);
            removeUUIDOfEntity(id);
            entityCreationOrder.erase(
                std::remove(entityCreationOrder.begin(),entityCreationOrder.end(),id),
                entityCreationOrder.end()
            );
            removedEntitiesSinceLastSerialization.insert(id);
        }else{
            ++it;
        }
    }
    for(auto it = entities.begin(); it != entities.end();){
        auto [id, entity] = *it;
        if(entity->isEpisodic()){
            entity->isMonitored=false;
            it = entities.erase(it);
            removeUUIDOfEntity(id);
            entityCreationOrder.erase(
                std::remove(entityCreationOrder.begin(),entityCreationOrder.end(),id),
                entityCreationOrder.end()
            );
            removedEntitiesSinceLastSerialization.insert(id);
        }else{
            ++it;
        }
    }
    nextEntityIndex=0;
    if(increment){
        _episode_index++;
    }
}
void EntityManager::removeEntity(const std::shared_ptr<Entity>& entity){
    if(entity){
        std::lock_guard<std::recursive_mutex> lock(mtx);
        bool isMonitored=entity->isMonitored;
        if(isMonitored){
            entity->isMonitored=false;
            auto id=entity->getEntityID();
            auto name=entity->getFullName();
            auto nFound=entityNames.find(name);
            if(name!="" && nFound!=entityNames.end()){
                if(nFound->second==id){
                    entityNames.erase(nFound);
                }
            }
            auto found=entities.find(id);
            if(found!=entities.end()){
                if(found->second == entity){
                    entities.erase(found);
                }
            }else{
                auto wFound=unmanagedEntities.find(id);
                if(wFound!=unmanagedEntities.end()){
                    if(wFound->second.expired()){
                        unmanagedEntities.erase(wFound);
                    }else{
                        auto found=wFound->second.lock();
                        if(found == entity){
                            unmanagedEntities.erase(wFound);
                        }
                    }
                }
            }
            removeUUIDOfEntity(id);
            entityCreationOrder.erase(
                std::remove(entityCreationOrder.begin(),entityCreationOrder.end(),id),
                entityCreationOrder.end()
            );
            removedEntitiesSinceLastSerialization.insert(id);
        }
    }
}
void EntityManager::removeEntity(const std::weak_ptr<Entity>& entity,const EntityIdentifier& id,const std::string& name){
    if(entity.expired()){
        std::lock_guard<std::recursive_mutex> lock(mtx);
        auto wFound=unmanagedEntities.find(id);
        if(wFound!=unmanagedEntities.end()){
            if(wFound->second.expired()){
                if(name==""){
                    unmanagedEntities.erase(wFound);
                }else{
                    auto nFound=entityNames.find(name);
                    if(name!="" && nFound!=entityNames.end()){
                        if(nFound->second==id){
                            unmanagedEntities.erase(wFound);
                            entityNames.erase(nFound);
                        }
                    }
                }
            }
        }
        removeUUIDOfEntity(id);
        entityCreationOrder.erase(
            std::remove(entityCreationOrder.begin(),entityCreationOrder.end(),id),
            entityCreationOrder.end()
        );
        removedEntitiesSinceLastSerialization.insert(id);
    }else{
        removeEntity(entity.lock());
    }
}
std::shared_ptr<Entity> EntityManager::getEntityByIDImpl(const EntityIdentifier& id) const{
    std::lock_guard<std::recursive_mutex> lock(mtx);
    auto found=entities.find(id);
    if(found!=entities.end()){
        return found->second;
    }else{
        auto wFound=unmanagedEntities.find(id);
        if(wFound!=unmanagedEntities.end()){
            if(wFound->second.expired()){
                unmanagedEntities.erase(wFound);
                return nullptr; 
            }else{
                return wFound->second.lock();
            }
        }else{
            return nullptr;
        }
    }
}
std::shared_ptr<Entity> EntityManager::getEntityByNameImpl(const std::string& name) const{
    std::lock_guard<std::recursive_mutex> lock(mtx);
    auto found=entityNames.find(name);
    if(found!=entityNames.end()){
        auto ret=getEntityByIDImpl(found->second);
        if(!ret){
            entityNames.erase(found);
        }
        return ret;
    }else{
        return nullptr;
    }
}
std::shared_ptr<Entity> EntityManager::getEntityByIDImpl(const EntityIdentifier& id, const std::shared_ptr<const Entity>& caller) const{
    auto ret=getEntityByIDImpl(id);
    if(ret && caller){
        if(Factory::checkPermissionForGet(caller,ret->getFactoryBaseName())){
            return ret;
        }else{
            throw std::runtime_error("Getting an entity of '"+ret->getFactoryBaseName()+"' by '"+caller->getFactoryBaseName()+"' is not allowed.");
        }
    }
    return ret;
}
std::shared_ptr<Entity> EntityManager::getEntityByNameImpl(const std::string& name, const std::shared_ptr<const Entity>& caller) const{
    auto ret=getEntityByNameImpl(name);
    if(ret && caller){
        if(Factory::checkPermissionForGet(caller,ret->getFactoryBaseName())){
            return ret;
        }else{
            throw std::runtime_error("Getting an entity of '"+ret->getFactoryBaseName()+"' by '"+caller->getFactoryBaseName()+"' is not allowed.");
        }
    }
    return ret;
}
std::shared_ptr<EntityAccessor> EntityManager::getEntityAccessorImpl(const boost::uuids::uuid& uuid_) const{
    auto found=uuidOfEntityForEntityAccessorsR.find(uuid_);
    if(found!=uuidOfEntityForEntityAccessorsR.end()){
        auto entity=getEntityByID(found->second);
        if(entity){
            return entity->getAccessor();
        }else{
            throw std::runtime_error("There is no Entity with uuidForAccessor="+boost::uuids::to_string(uuid_)+".");
        }
    }else{
        throw std::runtime_error("There is no Entity with uuidForAccessor="+boost::uuids::to_string(uuid_)+".");
    }
}
bool EntityManager::isManaged(const std::shared_ptr<const Entity>& entity) const{
    if(!entity){
        return false;
    }
    auto id=entity->getEntityID();
    return entities.find(id)!=entities.end() && entities.at(id)==entity;
}

// 内部状態のシリアライゼーション
EntityConstructionInfo::EntityConstructionInfo():removed(true),entityIdentifier(){}
EntityConstructionInfo::EntityConstructionInfo(const std::shared_ptr<Entity>& entity,EntityIdentifier id){
    removed=!(bool(entity));
    if(removed){
        entityIdentifier=id;
    }else{
        entityIdentifier=entity->getEntityID();
        assert(id==entityIdentifier);
        isManaged=entity->isManaged();
        factoryHelperChain=entity->getFactoryHelperChain();
        config={
            {"isEpisodic",entity->isEpisodic()},
            {"baseName",entity->getFactoryBaseName()},
            {"className",entity->getFactoryClassName()},
            {"modelName",entity->getFactoryModelName()},
            {"modelConfig",entity->modelConfig},
            {"instanceConfig",entity->instanceConfig},
            {"entityIdentifier",entityIdentifier},
            {"entityFullName",entity->getFullName()},
            {"uuid",entity->getUUID()},
            {"uuidForAccessor",entity->getUUIDForAccessor()},
            {"uuidForTrack",entity->getUUIDForTrack()}
        };
    }
}
void EntityConstructionInfo::serialize(asrc::core::util::AvailableArchiveTypes& archive){
    ASRC_SERIALIZE_NVP(archive
        ,removed
        ,entityIdentifier
    )
    if(!removed){
        ASRC_SERIALIZE_NVP(archive
            ,isManaged
            ,factoryHelperChain
            ,config
        )
    }
}
void EntityManager::serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full, const nl::json& serializationConfig_){
    std::lock_guard<std::recursive_mutex> lock(mtx);
    if(asrc::core::util::isInputArchive(archive)){
        isDuringDeserialization=true;
    }
    try{
        serializeBeforeEntityReconstructionInfo(archive,full,serializationConfig_);
        serializeEntityReconstructionInfo(archive,full,serializationConfig_);
        serializeAfterEntityReconstructionInfo(archive,full,serializationConfig_);
        serializeEntityInternalStates(archive,full,serializationConfig_);
        serializeAfterEntityInternalStates(archive,full,serializationConfig_);
        createdEntitiesSinceLastSerialization.clear();
        removedEntitiesSinceLastSerialization.clear();
    }catch(std::exception& ex){
        throw std::runtime_error("EntityManager::serializeInternalState() failed. [reason='"+std::string(ex.what())+"']");
    }
    if(asrc::core::util::isInputArchive(archive)){
        isDuringDeserialization=false;
    }
}
void EntityManager::serializeBeforeEntityReconstructionInfo(asrc::core::util::AvailableArchiveTypes & archive, bool full, const nl::json& serializationConfig_){
    bool isFullSource=full;
    if(asrc::core::util::isOutputArchive(archive)){
        if(!factory){
            throw std::runtime_error("EntityManager instance not initialized.");
        }
    }
    call_operator(archive,cereal::make_nvp("isFull",isFullSource));

    if(full){
        if(asrc::core::util::isInputArchive(archive)){
            if(!isFullSource){
                throw std::runtime_error("Attempted full deserialization, but the state is not a full state.");
            }
        }

        call_operator(archive,cereal::make_nvp("worker_index",_worker_index));
        //if(Factory::getWorkerIndex()!=_worker_index){
        //    throw std::runtime_error("worker_index mismatch.");
        //}

        if(asrc::core::util::isInputArchive(archive)){
            if(!isFullSource){
                throw std::runtime_error("Attempted full deserialization, but the state is not a full state.");
            }
            //内部状態の初期化(Factoryのstaticメンバの操作を含む。)
            if(factory){
                for(auto&& it = entities.begin(); it != entities.end();){
                    auto&& [id, entity] = *it;
                    it = entities.erase(it);
                    removeEntity(entity);
                }
                for(auto&& it = unmanagedEntities.begin(); it != unmanagedEntities.end();){
                    auto&& [id, wEntity] = *it;
                    it = unmanagedEntities.erase(it);
                    if(!wEntity.expired()){
                        removeEntity(wEntity.lock());
                    }
                }
            }
            factory=nullptr;
            purge();
        }

        boost::uuids::uuid uuid_=uuid;
        call_operator(archive,
            cereal::make_nvp("vector_index",_vector_index),
            cereal::make_nvp("uuid",uuid_),
            cereal::make_nvp("episode_index",_episode_index),
            CEREAL_NVP(nextEpisodeIndependentEntityIndex),
            CEREAL_NVP(nextEntityIndex)
        );

        if(asrc::core::util::isInputArchive(archive)){
            managerInstances[_vector_index]=this->shared_from_this();
            setUUID(uuid_);
            factory=Factory::create();
        }
    }

    call_operator(archive,
        cereal::make_nvp("factory",VariantArchiveInternalStateSerializerWrapper(
            [this](AvailableArchiveTypes& a, bool f){
                factory->serializeInternalState(a,f);
            },
            full
        ))
    );
}
void EntityManager::serializeEntityReconstructionInfo(asrc::core::util::AvailableArchiveTypes & archive, bool full, const nl::json& serializationConfig_){
    std::vector<EntityConstructionInfo> entityConstructionInfos;

    if(asrc::core::util::isOutputArchive(archive)){
        for(auto& id : removedEntitiesSinceLastSerialization){
            entityConstructionInfos.emplace_back(nullptr,id);
        }
        for(auto& id :entityCreationOrder){
            if(removedEntitiesSinceLastSerialization.find(id)==removedEntitiesSinceLastSerialization.end()){
                // 削除されていなければ内部状態を出力する
                auto entity=getEntityByIDImpl(id);
                bool full_sub=(
                    full || 
                    createdEntitiesSinceLastSerialization.find(id)!=createdEntitiesSinceLastSerialization.end()
                );
                if(full_sub){
                    // full出力時か、新規生成Entityならばコンストラクト用情報を出力する
                    entityConstructionInfos.emplace_back(entity,id);
                }
            }
        }
    }
    call_operator(archive,CEREAL_NVP(entityConstructionInfos));

    if(asrc::core::util::isInputArchive(archive)){
        reconstructEntities(entityConstructionInfos);
    }
}
void EntityManager::serializeAfterEntityReconstructionInfo(asrc::core::util::AvailableArchiveTypes & archive, bool full, const nl::json& serializationConfig_){
}
void EntityManager::serializeEntityInternalStates(asrc::core::util::AvailableArchiveTypes & archive, bool full, const nl::json& serializationConfig_){
    std::vector<std::pair<EntityIdentifier,bool>> entityInternalStateIndices;
    if(asrc::core::util::isOutputArchive(archive)){
        for(auto& id :entityCreationOrder){
            if(removedEntitiesSinceLastSerialization.find(id)==removedEntitiesSinceLastSerialization.end()){
                // 削除されていなければ内部状態を出力する
                auto entity=getEntityByIDImpl(id);
                bool full_sub=(
                    full || 
                    createdEntitiesSinceLastSerialization.find(id)!=createdEntitiesSinceLastSerialization.end()
                );
                if(entity){
                    // 初回出力時には必ずfullの内部状態とする
                    entityInternalStateIndices.emplace_back(id,full_sub);
                }
            }
        }
    }
    ASRC_SERIALIZE_NVP(archive,entityInternalStateIndices);

    serializeInternalStateOfEntity(entityInternalStateIndices,archive);

    if(asrc::core::util::isInputArchive(archive)){
        tmp_entities.clear();
    }
}
void EntityManager::serializeAfterEntityInternalStates(asrc::core::util::AvailableArchiveTypes & archive, bool full, const nl::json& serializationConfig_){
}
EntityConstructionInfo EntityManager::getEntityConstructionInfo(const std::shared_ptr<Entity>& entity,EntityIdentifier id) const{
    if(entity){
        return {entity,id};
    }else{
        return {nullptr,id};
    }
}
std::shared_ptr<Entity> EntityManager::reconstructEntity(const EntityConstructionInfo& info){
    bool prev=isDuringDeserialization;
    isDuringDeserialization=true;
    if(info.removed){
        auto entity=getEntityByID(info.entityIdentifier);
        if(entity){
            removeEntity(entity);
        }
        isDuringDeserialization=prev;
        return nullptr;
    }else{
        auto entity=createEntityImpl(info.isManaged,info.config,info.factoryHelperChain,nullptr);
        if(!entity){
            throw std::runtime_error("Entity instance could not be obtained.");
        }
        isDuringDeserialization=prev;
        return entity;
    }
}
void EntityManager::reconstructEntities(const std::vector<EntityConstructionInfo>& infos){
    bool prev=isDuringDeserialization;
    isDuringDeserialization=true;
    tmp_entities.clear();
    for(auto&& info : infos){
        try{
            tmp_entities.push_back(reconstructEntity(info));
        }catch(std::exception& ex){
            throw std::runtime_error(
                "EntityManager::reconstructEntity failed. [reason='"+std::string(ex.what())+"']"
            );
        }
    }
    isDuringDeserialization=prev;
}
void EntityManager::reconstructEntities(asrc::core::util::AvailableInputArchiveTypes & archive, const std::string& name){
    std::vector<EntityConstructionInfo> entityConstructionInfos;
    call_operator(archive,cereal::make_nvp(name.c_str(),entityConstructionInfos));
    reconstructEntities(entityConstructionInfos);
}

void EntityManager::serializeInternalStateOfEntity(const std::vector<std::pair<EntityIdentifier,bool>>& indices,asrc::core::util::AvailableArchiveTypes & archive, const std::string& name){
    bool prev=isDuringDeserialization;
    isDuringDeserialization=true;
    std::vector<InternalStateSerializer<std::shared_ptr<Entity>>> entityInternalStateSerializers;
    for(auto && [id,full] : indices){
        auto entity=getEntityByIDImpl(id);
        if(entity){
            entityInternalStateSerializers.emplace_back(std::move(entity),full);
        }
    }
    call_operator(archive
        ,cereal::make_nvp(name.c_str(),makeInternalStateSerializer(entityInternalStateSerializers,true))
    );
    isDuringDeserialization=prev;
}

//
// class EntityManagerAccessor
//

EntityManagerAccessor::EntityManagerAccessor(const std::shared_ptr<EntityManager>& manager_)
:manager(manager_){
}
EntityManagerAccessor::EntityManagerAccessor(const std::shared_ptr<EntityManagerAccessor>& original_){
    if(original_){
        manager=original_->manager;
    }
}
bool EntityManagerAccessor::expired() const noexcept{
    return manager.expired();
}
bool EntityManagerAccessor::isSame(const std::shared_ptr<EntityManager>& manager_) const{
    return !expired() && manager.lock()==manager_;
}
bool EntityManagerAccessor::isSame(const std::shared_ptr<EntityManagerAccessor>& original_) const{
    return !expired() && original_ && !original_->expired() && manager.lock()==original_->manager.lock();
}

//
// class Entity
//

const std::string Entity::baseName="Entity";

Entity::Entity(const nl::json& modelConfig_,const nl::json& instanceConfig_)
:modelConfig(modelConfig_),instanceConfig(instanceConfig_),isDummy(modelConfig_.is_null() && instanceConfig_.is_null()){
    isMonitored=false;
    fullName="";
    if(instanceConfig.contains("uuid")){
        instanceConfig.at("uuid").get_to(uuid);
    }else{
        uuid=boost::uuids::random_generator()();
    }
    if(instanceConfig.contains("uuidForAccessor")){
        instanceConfig.at("uuidForAccessor").get_to(uuidForAccessor);
    }else{
        uuidForAccessor=boost::uuids::random_generator()();
    }
    if(instanceConfig.contains("uuidForTrack")){
        instanceConfig.at("uuidForTrack").get_to(uuidForTrack);
    }else{
        uuidForTrack=boost::uuids::random_generator()();
    }
    if(isDummy){return;}
    unsigned int seed_;
    try{
        seed_=instanceConfig.at("seed");
    }catch(...){
        seed_=std::random_device()();
        instanceConfig["seed"]=seed_;
    }
    seed(seed_);
}
Entity::~Entity(){
    if(!entityManager.expired() && isMonitored){
        entityManager.lock()->removeEntity(this->weak_from_this(),entityID,fullName);
    }
}
void Entity::setFactoryCreationMetadata(const std::string& baseName, const std::string& className, const std::string& modelName, const FactoryHelperChain& helperChain){
    factoryBaseName=baseName;
    factoryClassName=className;
    factoryModelName=modelName;
    factoryHelperChain=helperChain;
}
void Entity::setEntityManager(const std::shared_ptr<EntityManager>& em){
    entityManager=em;
}
void Entity::setManagerAccessor(const std::shared_ptr<EntityManagerAccessor>& ema){
    manager=ema;
}
void Entity::setEntityID(const EntityIdentifier& id){
    entityID=id;
}
EntityIdentifier Entity::getEntityID() const{
    return entityID;
}
boost::uuids::uuid Entity::getUUID() const{
    return uuid;
}
boost::uuids::uuid Entity::getUUIDForAccessor() const{
    return uuidForAccessor;
}
boost::uuids::uuid Entity::getUUIDForTrack() const{
    return uuidForTrack;
}
bool Entity::isManaged() const{
    if(entityManager.expired()){
        return false;
    }else{
        return entityManager.lock()->isManaged(this->shared_from_this());
    }
}
bool Entity::isEpisodic() const{
    return entityID.episode_index != EntityIdentifier::invalid_episode_index;
}
void Entity::seed(const unsigned int& seed_){
    randomGen.seed(seed_);
}
std::string Entity::getFactoryBaseName() const{
    //FactoryにおけるbaseNameを返す。
    return factoryBaseName;
}
std::string Entity::getFactoryClassName() const{
    //Factoryにおけるクラス名を返す。
    return factoryClassName;
}
std::string Entity::getFactoryModelName() const{
    //Factoryにおけるモデル名を返す。
    return factoryModelName;
}
std::string Entity::getFullName() const{
    return fullName;
}
std::shared_ptr<EntityAccessor> Entity::getAccessorImpl(){
    return std::make_shared<EntityAccessor>(this->shared_from_this());
}
void Entity::setFullName(const std::string& fullName_){
    fullName=fullName_;
}
void Entity::initialize(){
}
void Entity::makeChildren(){
}
void Entity::validate(){
}
std::uint64_t Entity::getFirstTick(const SimPhase& phase) const{
    return firstTick.at(phase);
}
std::uint64_t Entity::getInterval(const SimPhase& phase) const{
    return interval.at(phase);
}
std::uint64_t Entity::getNextTick(const SimPhase& phase,const std::uint64_t now){
    std::uint64_t first=getFirstTick(phase);
    if(now<first){
        return first;
    }
    std::uint64_t delta=getInterval(phase);
    std::uint64_t recent=now-(now-first)%delta;
    if(std::numeric_limits<std::uint64_t>::max()-recent>=delta){
        return recent+delta;
    }else{
        return std::numeric_limits<std::uint64_t>::max();
    }
}
bool Entity::isTickToRunPhaseFunc(const SimPhase& phase,const std::uint64_t now){
    if(now==0){
        return getFirstTick(phase)==now;
    }else{
        return getNextTick(phase,now-1)==now;
    }
}
bool Entity::isSame(const std::shared_ptr<Entity>&  other){
    return this==other.get();
}
FactoryHelperChain Entity::getFactoryHelperChain() const{
    return factoryHelperChain;
}
std::string Entity::resolveClassName(const std::string& baseName, const std::string& className) const{
    return factoryHelperChain.resolveClassName(baseName,className);
}
std::string Entity::resolveModelName(const std::string& baseName, const std::string& modelName) const{
    if(entityManager.expired()){
        return factoryHelperChain.resolveModelName(baseName,modelName);
    }else{
        return factoryHelperChain.resolveModelName(baseName,modelName,entityManager.lock()->factory);
    }
}
std::filesystem::path Entity::resolveFilePath(const std::string& path) const{
    return factoryHelperChain.resolveFilePath(path);
}
std::filesystem::path Entity::resolveFilePath(const std::filesystem::path& path) const{
    return factoryHelperChain.resolveFilePath(path);
}
// 内部状態のシリアライゼーション
// デシリアライズ時のインスタンス生成はEntityManagerが行うものとする。
// ただし、デシリアライズ時に参照する内部状態は共通とする。
void Entity::serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full){
    ASRC_SERIALIZE_NVP(archive
        ,randomGen
        ,firstTick
        ,interval
    )
}

//
// class EntityAccessor
//

EntityAccessor::EntityAccessor(const std::shared_ptr<Entity>& entity_){
    entity=entity_;
}
bool EntityAccessor::expired() const noexcept{
    return entity.expired();
}
bool EntityAccessor::isSame(const std::shared_ptr<Entity>& entity_) const{
    return !expired() && entity.lock()==entity_;
}
bool EntityAccessor::isSame(const std::shared_ptr<EntityAccessor>& original_) const{
    return !expired() && original_ && !original_->expired() && entity.lock()->isSame(original_->entity.lock());
}
EntityIdentifier EntityAccessor::getEntityID() const{
    return entity.lock()->getEntityID();
}
boost::uuids::uuid EntityAccessor::getUUIDForAccessor() const{
    return entity.lock()->getUUIDForAccessor();
}
boost::uuids::uuid EntityAccessor::getUUIDForTrack() const{
    return entity.lock()->getUUIDForTrack();
}
std::string EntityAccessor::getFactoryBaseName() const{
    return entity.lock()->getFactoryBaseName();
}
std::string EntityAccessor::getFactoryClassName() const{
    return entity.lock()->getFactoryClassName();
}
std::string EntityAccessor::getFactoryModelName() const{
    return entity.lock()->getFactoryModelName();
}
std::string EntityAccessor::getFullName() const{
    return entity.lock()->getFullName();
}
bool EntityAccessor::isinstancePY(const py::object& cls){
    return py::isinstance(py::cast(entity.lock()),cls);
}

void exportEntity(py::module &m)
{
    using namespace pybind11::literals;

    bind_stl_container<std::vector<std::shared_ptr<Entity>>>(m,py::module_local(false));
    bind_stl_container<std::vector<std::shared_ptr<EntityConstructionInfo>>>(m,py::module_local(false));
    bind_stl_container<std::map<std::int32_t,std::weak_ptr<EntityManager>>>(m,py::module_local(false));
    bind_stl_container<std::map<EntityIdentifier,std::shared_ptr<Entity>>>(m,py::module_local(false));
    bind_stl_container<std::map<EntityIdentifier, std::weak_ptr<Entity>>>(m,py::module_local(false));
    bind_stl_container<std::map<SimPhase,std::uint64_t>>(m,py::module_local(false));

    Factory::addBaseName(
        Entity::baseName,
        "", //parent's baseName
        {"AnyOthers"}, // permissionForCreate,
        {"AnyOthers"} // permissionForGet
    );

    expose_common_class<EntityConstructionInfo>(m,"EntityConstructionInfo")
    .def(py_init<>())
    .def(py_init<const std::shared_ptr<Entity>&,EntityIdentifier>())
    ;

    expose_common_class<EntityManager>(m,"EntityManager")
    .def(py_init<const std::int32_t&>())
    DEF_STATIC_FUNC(EntityManager,setWorkerIndex)
    DEF_STATIC_FUNC(EntityManager,getWorkerIndex)
    DEF_STATIC_FUNC(EntityManager,getUnusedVectorIndex)
    DEF_FUNC(EntityManager,getVectorIndex)
    DEF_FUNC(EntityManager,getEpisodeIndex)
    DEF_FUNC(EntityManager,getNextEpisodeIndependentEntityIndex)
    DEF_FUNC(EntityManager,getNextEntityIndex)
    DEF_FUNC(EntityManager,getNextEpisodeIndependentEntityIdentifier)
    DEF_FUNC(EntityManager,getNextEntityIdentifier)
    DEF_FUNC(EntityManager,purge)
    DEF_FUNC(EntityManager,clearModels)
    DEF_FUNC(EntityManager,addModel)
    DEF_FUNC(EntityManager,addModelsFromJson)
    DEF_FUNC(EntityManager,addModelsFromJsonFile)
    DEF_FUNC(EntityManager,getModelConfigs)
    DEF_FUNC(EntityManager,reconfigureModelConfigs)
    .def("resolveModelConfig",&EntityManager::resolveModelConfig<const std::string&,const std::string&,const std::string&,const nl::json&,const FactoryHelperChain&,bool>
        ,"baseName"_a,"className"_a,"modelName"_a,"modelConfig"_a,"chain"_a,"withDefaultModels"_a=true
    )
    .def("resolveModelConfig",&EntityManager::resolveModelConfig<const std::string&,const std::string&,const FactoryHelperChain&,bool>
        ,"baseName"_a,"modelName"_a,"chain"_a,"withDefaultModels"_a=true
    )
    .def("resolveModelConfig",&EntityManager::resolveModelConfig<const std::string&,const nl::json&,const FactoryHelperChain&,bool>
        ,"baseName"_a,"j"_a,"chain"_a,"withDefaultModels"_a=true
    )
    .def("resolveModelConfig",&EntityManager::resolveModelConfig<const nl::json&,const FactoryHelperChain&,bool>
        ,"j"_a,"chain"_a,"withDefaultModels"_a=true
    )
    .def("resolveModelConfig",&EntityManager::resolveModelConfig<const std::string&,const std::string&,const std::string&,const nl::json&,bool>
        ,"baseName"_a,"className"_a,"modelName"_a,"modelConfig"_a,"withDefaultModels"_a=true
    )
    .def("resolveModelConfig",&EntityManager::resolveModelConfig<const std::string&,const std::string&,bool>
        ,"baseName"_a,"modelName"_a,"withDefaultModels"_a=true
    )
    .def("resolveModelConfig",&EntityManager::resolveModelConfig<const std::string&,const nl::json&,bool>
        ,"baseName"_a,"j"_a,"withDefaultModels"_a=true
    )
    .def("resolveModelConfig",&EntityManager::resolveModelConfig<const nl::json&,bool>
        ,"j"_a,"withDefaultModels"_a=true
    )
    .def("createEntityImpl",
        py::overload_cast<
            bool, bool, const std::string&, const std::string&, const std::string&, const nl::json&, const nl::json&, const FactoryHelperChain&, const std::shared_ptr<const Entity>&
        >(&EntityManager::createEntityImpl)
    )
    .def("createUnmanagedEntityOverloader",
        py::overload_cast<
            bool, const std::string&, const std::string&, const std::string&, const nl::json&, const nl::json&, const FactoryHelperChain&
        >(&EntityManager::createUnmanagedEntityOverloader)
        ,"isEpisodic"_a,"baseName"_a,"className"_a,"modelName"_a,"modelConfig"_a,"instanceConfig"_a,"helperChain"_a=FactoryHelperChain()
    )
    .def("createUnmanagedEntityOverloader",
        py::overload_cast<
            bool, const std::string&, const std::string&, const std::string&, const nl::json&, const nl::json&, const std::shared_ptr<const Entity>&
        >(&EntityManager::createUnmanagedEntityOverloader)
    )
    .def("createUnmanagedEntityOverloader",
        py::overload_cast<
            bool, const std::string&, const std::string&, const nl::json&, const FactoryHelperChain&
        >(&EntityManager::createUnmanagedEntityOverloader)
        ,"isEpisodic"_a,"baseName"_a,"modelName"_a,"instanceConfig"_a,"helperChain"_a=FactoryHelperChain()
    )
    .def("createUnmanagedEntityOverloader",
        py::overload_cast<
            bool, const std::string&, const std::string&, const nl::json&, const std::shared_ptr<const Entity>&
        >(&EntityManager::createUnmanagedEntityOverloader)
    )
    .def("createUnmanagedEntityByClassNameOverloader",
        py::overload_cast<
            bool, const std::string&, const std::string&, const nl::json&, const nl::json&, const FactoryHelperChain&
        >(&EntityManager::createUnmanagedEntityByClassNameOverloader)
        ,"isEpisodic"_a,"baseName"_a,"className"_a,"modelConfig"_a,"instanceConfig"_a,"helperChain"_a=FactoryHelperChain()
    )
    .def("createUnmanagedEntityByClassNameOverloader",
        py::overload_cast<
            bool, const std::string&, const std::string&, const nl::json&, const nl::json&, const std::shared_ptr<const Entity>&
        >(&EntityManager::createUnmanagedEntityByClassNameOverloader)
    )
    .def("createEntityImpl",
        py::overload_cast<
            bool, const nl::json&, const FactoryHelperChain&, const std::shared_ptr<const Entity>&
        >(&EntityManager::createEntityImpl)
    )
    .def("createUnmanagedEntityOverloader",
        py::overload_cast<const nl::json&, const FactoryHelperChain&>(&EntityManager::createUnmanagedEntityOverloader)
        ,"j"_a,"helperChain"_a=FactoryHelperChain()
    )
    .def("createUnmanagedEntityOverloader",
        py::overload_cast<const nl::json&, const std::shared_ptr<const Entity>&>(&EntityManager::createUnmanagedEntityOverloader)
    )
    .def("createOrGetEntityImpl",
        py::overload_cast<
            bool, const nl::json&, const FactoryHelperChain&, const std::shared_ptr<const Entity>&
        >(&EntityManager::createOrGetEntityImpl)
    )
    .def("createOrGetUnmanagedEntityOverloader",
        py::overload_cast<const nl::json&, const FactoryHelperChain&>(&EntityManager::createOrGetUnmanagedEntityOverloader)
        ,"j"_a,"helperChain"_a=FactoryHelperChain()
    )
    .def("createOrGetUnmanagedEntityOverloader",
        py::overload_cast<const nl::json&, const std::shared_ptr<const Entity>&>(&EntityManager::createOrGetUnmanagedEntityOverloader)
    )
    .def("createUnmanagedEntity",
        &EntityManager::createUnmanagedEntity<Entity,
            bool, const std::string&, const std::string&, const std::string&, const nl::json&, const nl::json&, const FactoryHelperChain&
        >
        ,"isEpisodic"_a,"baseName"_a,"className"_a,"modelName"_a,"modelConfig"_a,"instanceConfig"_a,"helperChain"_a=FactoryHelperChain()
    )
    .def("createUnmanagedEntity",
        &EntityManager::createUnmanagedEntity<Entity,
            bool, const std::string&, const std::string&, const std::string&, const nl::json&, const nl::json&, const std::shared_ptr<const Entity>&
        >
    )
    .def("createUnmanagedEntity",
        &EntityManager::createUnmanagedEntity<Entity,
            bool, const std::string&, const std::string&, const nl::json&, const FactoryHelperChain&
        >
        ,"isEpisodic"_a,"baseName"_a,"modelName"_a,"instanceConfig"_a,"helperChain"_a=FactoryHelperChain()
    )
    .def("createUnmanagedEntity",
        &EntityManager::createUnmanagedEntity<Entity,
            bool, const std::string&, const std::string&, const nl::json&, const std::shared_ptr<const Entity>&
        >
    )
    .def("createUnmanagedEntityByClassName",
        &EntityManager::createUnmanagedEntityByClassName<Entity,
            bool, const std::string&, const std::string&, const nl::json&, const nl::json&, const FactoryHelperChain&
        >
        ,"isEpisodic"_a,"baseName"_a,"className"_a,"modelConfig"_a,"instanceConfig"_a,"helperChain"_a=FactoryHelperChain()
    )
    .def("createUnmanagedEntityByClassName",
        &EntityManager::createUnmanagedEntityByClassName<Entity,
            bool, const std::string&, const std::string&, const nl::json&, const nl::json&, const std::shared_ptr<const Entity>&
        >
    )
    .def("createUnmanagedEntity",
        &EntityManager::createUnmanagedEntity<Entity, const nl::json&, const FactoryHelperChain&>
        ,"j"_a,"helperChain"_a=FactoryHelperChain()
    )
    .def("createUnmanagedEntity",
        &EntityManager::createUnmanagedEntity<Entity, const nl::json&, const std::shared_ptr<const Entity>&>
    )
    .def("createOrGetUnmanagedEntity",
        &EntityManager::createOrGetUnmanagedEntity<Entity, const nl::json&, const FactoryHelperChain&>
        ,"j"_a,"helperChain"_a=FactoryHelperChain()
    )
    .def("createOrGetUnmanagedEntity",
        &EntityManager::createOrGetUnmanagedEntity<Entity, const nl::json&, const std::shared_ptr<const Entity>&>
    )
    .def("createEntityOverloader",
        py::overload_cast<
            bool, const std::string&, const std::string&, const std::string&, const nl::json&, const nl::json&, const FactoryHelperChain&
        >(&EntityManager::createEntityOverloader)
        ,"isEpisodic"_a,"baseName"_a,"className"_a,"modelName"_a,"modelConfig"_a,"instanceConfig"_a,"helperChain"_a=FactoryHelperChain()
    )
    .def("createEntityOverloader",
        py::overload_cast<
            bool, const std::string&, const std::string&, const std::string&, const nl::json&, const nl::json&, const std::shared_ptr<const Entity>&
        >(&EntityManager::createEntityOverloader)
    )
    .def("createEntityOverloader",
        py::overload_cast<
            bool, const std::string&, const std::string&, const nl::json&, const FactoryHelperChain&
        >(&EntityManager::createEntityOverloader)
        ,"isEpisodic"_a,"baseName"_a,"modelName"_a,"instanceConfig"_a,"helperChain"_a=FactoryHelperChain()
    )
    .def("createEntityOverloader",
        py::overload_cast<
            bool, const std::string&, const std::string&, const nl::json&, const std::shared_ptr<const Entity>&
        >(&EntityManager::createEntityOverloader)
    )
    .def("createEntityByClassNameOverloader",
        py::overload_cast<
            bool, const std::string&, const std::string&, const nl::json&, const nl::json&, const FactoryHelperChain&
        >(&EntityManager::createEntityByClassNameOverloader)
        ,"isEpisodic"_a,"baseName"_a,"className"_a,"modelConfig"_a,"instanceConfig"_a,"helperChain"_a=FactoryHelperChain()
    )
    .def("createEntityByClassNameOverloader",
        py::overload_cast<
            bool, const std::string&, const std::string&, const nl::json&, const nl::json&, const std::shared_ptr<const Entity>&
        >(&EntityManager::createEntityByClassNameOverloader)
    )
    .def("createEntityOverloader",
        py::overload_cast<const nl::json&, const FactoryHelperChain&>(&EntityManager::createEntityOverloader)
        ,"j"_a,"helperChain"_a=FactoryHelperChain()
    )
    .def("createEntityOverloader",
        py::overload_cast<const nl::json&, const std::shared_ptr<const Entity>&>(&EntityManager::createEntityOverloader)
    )
    .def("createOrGetEntityOverloader",
        py::overload_cast<const nl::json&, const FactoryHelperChain&>(&EntityManager::createOrGetEntityOverloader)
        ,"j"_a,"helperChain"_a=FactoryHelperChain()
    )
    .def("createOrGetEntityOverloader",
        py::overload_cast<const nl::json&, const std::shared_ptr<const Entity>&>(&EntityManager::createOrGetEntityOverloader)
    )
    .def("createEntity",
        &EntityManager::createEntity<Entity,
            bool, const std::string&, const std::string&, const std::string&, const nl::json&, const nl::json&, const FactoryHelperChain&
        >
        ,"isEpisodic"_a,"baseName"_a,"className"_a,"modelName"_a,"modelConfig"_a,"instanceConfig"_a,"helperChain"_a=FactoryHelperChain()
    )
    .def("createEntity",
        &EntityManager::createEntity<Entity,
            bool, const std::string&, const std::string&, const std::string&, const nl::json&, const nl::json&, const std::shared_ptr<const Entity>&
        >
    )
    .def("createEntity",
        &EntityManager::createEntity<Entity,
            bool, const std::string&, const std::string&, const nl::json&, const FactoryHelperChain&
        >
        ,"isEpisodic"_a,"baseName"_a,"modelName"_a,"instanceConfig"_a,"helperChain"_a=FactoryHelperChain()
    )
    .def("createEntity",
        &EntityManager::createEntity<Entity,
            bool, const std::string&, const std::string&, const nl::json&, const std::shared_ptr<const Entity>&
        >
    )
    .def("createEntityByClassName",
        &EntityManager::createEntityByClassName<Entity,
            bool, const std::string&, const std::string&, const nl::json&, const nl::json&, const FactoryHelperChain&
        >
        ,"isEpisodic"_a,"baseName"_a,"className"_a,"modelConfig"_a,"instanceConfig"_a,"helperChain"_a=FactoryHelperChain()
    )
    .def("createEntityByClassName",
        &EntityManager::createEntityByClassName<Entity,
            bool, const std::string&, const std::string&, const nl::json&, const nl::json&, const std::shared_ptr<const Entity>&
        >
    )
    .def("createEntity",
        &EntityManager::createEntity<Entity, const nl::json&, const FactoryHelperChain&>
        ,"j"_a,"helperChain"_a=FactoryHelperChain()
    )
    .def("createEntity",
        &EntityManager::createEntity<Entity, const nl::json&, const std::shared_ptr<const Entity>&>
    )
    .def("createOrGetEntity",
        &EntityManager::createOrGetEntity<Entity, const nl::json&, const FactoryHelperChain&>
        ,"j"_a,"helperChain"_a=FactoryHelperChain()
    )
    .def("createOrGetEntity",
        &EntityManager::createOrGetEntity<Entity, const nl::json&, const std::shared_ptr<const Entity>&>
    )
    DEF_FUNC(EntityManager,registerEntity)
    DEF_FUNC(EntityManager,createAccessorFor)
    .def("cleanupEpisode",&EntityManager::cleanupEpisode,"increment"_a=true)
    .def("removeEntity",py::overload_cast<const std::shared_ptr<Entity>&>(&EntityManager::removeEntity))
    .def("removeEntity",py::overload_cast<const std::weak_ptr<Entity>&,const EntityIdentifier&,const std::string&>(&EntityManager::removeEntity))
    .def("getEntityByID",py::overload_cast<const EntityIdentifier&>(&EntityManager::getEntityByIDImpl,py::const_))
    .def("getEntityByName",py::overload_cast<const std::string&>(&EntityManager::getEntityByNameImpl,py::const_))
    .def("getEntityByID",py::overload_cast<const EntityIdentifier&, const std::shared_ptr<const Entity>&>(&EntityManager::getEntityByIDImpl,py::const_))
    .def("getEntityByName",py::overload_cast<const std::string&, const std::shared_ptr<const Entity>&>(&EntityManager::getEntityByNameImpl,py::const_))
    .def("getEntityAccessor",&EntityManager::getEntityAccessor<>)
    .def("getEntityFromAccessor",&EntityManager::getEntityFromAccessor<>)
    DEF_FUNC(EntityManager,isManaged)
    DEF_FUNC(EntityManager,serializeInternalState)
    DEF_FUNC(EntityManager,serializeBeforeEntityReconstructionInfo)
    DEF_FUNC(EntityManager,serializeEntityReconstructionInfo)
    DEF_FUNC(EntityManager,serializeAfterEntityReconstructionInfo)
    DEF_FUNC(EntityManager,serializeEntityInternalStates)
    DEF_FUNC(EntityManager,serializeAfterEntityInternalStates)
    .def("saveInternalStateToJson",[](EntityManager& v, bool full, const nl::json& serializationConfig_){
        nl::ordered_json ret;
        {
            ::asrc::core::util::NLJSONOutputArchive archive(ret,true);
            ::asrc::core::util::AvailableArchiveTypes wrappedArchive(std::ref(archive));

            auto saver=[&](AvailableArchiveTypes& ar){
                v.serializeInternalState(ar,full,serializationConfig_);
            };
            ::asrc::core::util::VariantArchiveSerializerWrapper wrappedValue(saver);

            call_operator(wrappedArchive,wrappedValue);
        }
        return std::move(ret);
    })
    .def("loadInternalStateFromJson",[](EntityManager& v, const nl::ordered_json& j, bool full, const nl::json& serializationConfig_){
        ::asrc::core::util::NLJSONInputArchive archive(j,true);
        ::asrc::core::util::AvailableArchiveTypes wrappedArchive(std::ref(archive));

        auto loader=[&](AvailableArchiveTypes& ar){
            v.serializeInternalState(ar,full,serializationConfig_);
        };
        ::asrc::core::util::VariantArchiveSerializerWrapper wrappedValue(loader);

        call_operator(wrappedArchive,wrappedValue);
    })
    .def("saveInternalStateToBinary",[](EntityManager& v, bool full, const nl::json& serializationConfig_){
        std::stringstream oss;
        {
            cereal::PortableBinaryOutputArchive archive(oss);
            ::asrc::core::util::AvailableArchiveTypes wrappedArchive(std::ref(archive));
            auto saver=[&](AvailableArchiveTypes& ar){
                v.serializeInternalState(ar,full,serializationConfig_);
            };
            ::asrc::core::util::VariantArchiveSerializerWrapper wrappedValue(saver);
            call_operator(wrappedArchive,wrappedValue);
        }
        return py::bytes(oss.str());
    })
    .def("loadInternalStateFromBinary",[](EntityManager& v, const py::bytes& str, bool full, const nl::json& serializationConfig_){
        std::istringstream iss(str.cast<std::string>());
        cereal::PortableBinaryInputArchive archive(iss);
        ::asrc::core::util::AvailableArchiveTypes wrappedArchive(std::ref(archive));

        auto loader=[&](AvailableArchiveTypes& ar){
            v.serializeInternalState(ar,full,serializationConfig_);
        };
        ::asrc::core::util::VariantArchiveSerializerWrapper wrappedValue(loader);

        call_operator(wrappedArchive,wrappedValue);
    })
    .def("reconstructEntities",py::overload_cast<const std::string&>(&EntityManager::reconstructEntities<std::string>)
        ,"binary"_a
    )
    .def("reconstructEntities",py::overload_cast<const nl::ordered_json&>(&EntityManager::reconstructEntities<nl::ordered_json>)
        ,"j"_a
    )
    .def("reconstructEntities",py::overload_cast<const nl::json&>(&EntityManager::reconstructEntities<nl::json>)
        ,"j"_a
    )
    DEF_FUNC(EntityManager,getEntityConstructionInfo)
    .def("serializeInternalStateOfEntity",&EntityManager::serializeInternalStateOfEntity
        ,"indices"_a,"archive"_a,"name"_a="entityInternalStateSerializers"
    )
    .def("loadInternalStateOfEntity",py::overload_cast<const std::vector<std::pair<EntityIdentifier,bool>>&,const std::string&>(&EntityManager::loadInternalStateOfEntity<std::string>)
        ,"indices"_a,"binary"_a
    )
    .def("loadInternalStateOfEntity",py::overload_cast<const std::vector<std::pair<EntityIdentifier,bool>>&,const nl::ordered_json&>(&EntityManager::loadInternalStateOfEntity<nl::ordered_json>)
        ,"indices"_a,"j"_a
    )
    .def("loadInternalStateOfEntity",py::overload_cast<const std::vector<std::pair<EntityIdentifier,bool>>&,const nl::json&>(&EntityManager::loadInternalStateOfEntity<nl::json>)
        ,"indices"_a,"j"_a
    )
    ;

    expose_common_class<EntityManagerAccessor>(m,"EntityManagerAccessor")
    .def(py_init<const std::shared_ptr<EntityManager>&>())
    .def(py_init<const std::shared_ptr<EntityManagerAccessor>&>())
    DEF_FUNC(EntityManagerAccessor,expired)
    .def("isSame",py::overload_cast<const std::shared_ptr<EntityManager>&>(&EntityManagerAccessor::isSame,py::const_))
    .def("isSame",py::overload_cast<const std::shared_ptr<EntityManagerAccessor>&>(&EntityManagerAccessor::isSame,py::const_))
    ;

    expose_entity_subclass<Entity>(m,"Entity")
    DEF_FUNC(Entity,seed)
    DEF_FUNC(Entity,getEntityID)
    DEF_FUNC(Entity,getUUIDForAccessor)
    DEF_FUNC(Entity,getUUIDForTrack)
    DEF_FUNC(Entity,isManaged)
    DEF_FUNC(Entity,isEpisodic)
    DEF_FUNC(Entity,getFactoryBaseName)
    DEF_FUNC(Entity,getFactoryClassName)
    DEF_FUNC(Entity,getFactoryModelName)
    DEF_FUNC(Entity,getFullName)
    DEF_FUNC(Entity,getAccessorImpl)
    .def("getAccessor",&Entity::getAccessor<EntityAccessor>)
    DEF_FUNC(Entity,initialize)
    DEF_FUNC(Entity,validate)
    DEF_FUNC(Entity,makeChildren)
    DEF_FUNC(Entity,getFirstTick)
    DEF_FUNC(Entity,getInterval)
    DEF_FUNC(Entity,getNextTick)
    DEF_FUNC(Entity,isTickToRunPhaseFunc)
    DEF_FUNC(Entity,isSame)
    DEF_FUNC(Entity,getFactoryHelperChain)
    DEF_FUNC(Entity,resolveClassName)
    DEF_FUNC(Entity,resolveModelName)
    .def("resolveFilePath",[](Entity& v,const std::string& path){return v.resolveFilePath(path).generic_string();})
    .def("resolveModelConfig",&Entity::resolveModelConfig</*const std::string&,*/const std::string&,const std::string&,const nl::json&>
        ,"baseName"_a,"className"_a,"modelName"_a,"modelConfig"_a
    )
    .def("resolveModelConfig",&Entity::resolveModelConfig</*const std::string&,*/const std::string&>
        ,"baseName"_a,"modelName"_a
    )
    .def("resolveModelConfig",&Entity::resolveModelConfig</*const std::string&,*/const nl::json&>
        ,"baseName"_a,"j"_a
    )
    .def("resolveModelConfig",[](const Entity& v,const nl::json& j){return v.resolveModelConfig(j);}
        ,"j"_a
    )
    .def("createUnmanagedEntity",&Entity::createUnmanagedEntity<Entity,
        bool, const std::string&, const std::string&, const std::string&, const nl::json&, const nl::json&
    >)
    .def("createUnmanagedEntity",&Entity::createUnmanagedEntity<Entity,
        bool, const std::string&, const std::string&, const std::string&, const nl::json&
    >)
    .def("createUnmanagedEntityByClassName",&Entity::createUnmanagedEntityByClassName<Entity,
        bool, const std::string&, const std::string&, const nl::json&, const nl::json&
    >)
    .def("createUnmanagedEntity",&Entity::createUnmanagedEntity<Entity, const nl::json&>)
    .def("createOrGetUnmanagedEntity",&Entity::createOrGetUnmanagedEntity<Entity, const nl::json&>)
    .def("createEntity",&Entity::createEntity<Entity,
        bool, const std::string&, const std::string&, const std::string&, const nl::json&, const nl::json&
    >)
    .def("createEntity",&Entity::createEntity<Entity,
        bool, const std::string&, const std::string&, const std::string&, const nl::json&
    >)
    .def("createEntityByClassName",&Entity::createEntityByClassName<Entity,
        bool, const std::string&, const std::string&, const nl::json&, const nl::json&
    >)
    .def("createEntity",&Entity::createEntity<Entity, const nl::json&>)
    .def("createOrGetEntity",&Entity::createOrGetEntity<Entity, const nl::json&>)
    .def("getEntityByID",&Entity::getEntityByID<>)
    .def("getEntityByName",&Entity::getEntityByName<>)
    DEF_FUNC(Entity,serializeInternalState)
    .def("saveInternalStateToJson",[](Entity& v, bool full){
        nl::ordered_json ret;
        {
            ::asrc::core::util::NLJSONOutputArchive archive(ret,true);
            ::asrc::core::util::AvailableArchiveTypes wrappedArchive(std::ref(archive));

            auto saver=[&](AvailableArchiveTypes& ar){
                v.serializeInternalState(ar,full);
            };
            ::asrc::core::util::VariantArchiveSerializerWrapper wrappedValue(saver);

            call_operator(wrappedArchive,wrappedValue);
        }
        return std::move(ret);
    })
    .def("loadInternalStateFromJson",[](Entity& v, const nl::ordered_json& j, bool full){
        ::asrc::core::util::NLJSONInputArchive archive(j,true);
        ::asrc::core::util::AvailableArchiveTypes wrappedArchive(std::ref(archive));

        auto loader=[&](AvailableArchiveTypes& ar){
            v.serializeInternalState(ar,full);
        };
        ::asrc::core::util::VariantArchiveSerializerWrapper wrappedValue(loader);

        call_operator(wrappedArchive,wrappedValue);
    })
    .def("saveInternalStateToBinary",[](Entity& v, bool full){
        std::stringstream oss;
        {
            cereal::PortableBinaryOutputArchive archive(oss);
            ::asrc::core::util::AvailableArchiveTypes wrappedArchive(std::ref(archive));
            auto saver=[&](AvailableArchiveTypes& ar){
                v.serializeInternalState(ar,full);
            };
            ::asrc::core::util::VariantArchiveSerializerWrapper wrappedValue(saver);
            call_operator(wrappedArchive,wrappedValue);
        }
        return py::bytes(oss.str());
    })
    .def("loadInternalStateFromBinary",[](Entity& v, const py::bytes& str, bool full){
        std::istringstream iss(str.cast<std::string>());
        cereal::PortableBinaryInputArchive archive(iss);
        ::asrc::core::util::AvailableArchiveTypes wrappedArchive(std::ref(archive));

        auto loader=[&](AvailableArchiveTypes& ar){
            v.serializeInternalState(ar,full);
        };
        ::asrc::core::util::VariantArchiveSerializerWrapper wrappedValue(loader);

        call_operator(wrappedArchive,wrappedValue);
    })
    DEF_READONLY(Entity,isDummy)
    DEF_READONLY(Entity,firstTick)
    DEF_READONLY(Entity,interval)
    DEF_READWRITE(Entity,modelConfig)
    DEF_READWRITE(Entity,instanceConfig)
    DEF_READONLY(Entity,randomGen)
    DEF_READONLY(Entity,manager)
    ;

    expose_common_class<EntityAccessor>(m,"EntityAccessor")
    .def(py_init<const std::shared_ptr<Entity>&>())
    DEF_FUNC(EntityAccessor,expired)
    .def("isSame",py::overload_cast<const std::shared_ptr<Entity>&>(&EntityAccessor::isSame,py::const_))
    .def("isSame",py::overload_cast<const std::shared_ptr<EntityAccessor>&>(&EntityAccessor::isSame,py::const_))
    DEF_FUNC(EntityAccessor,getEntityID)
    DEF_FUNC(EntityAccessor,getUUIDForAccessor)
    DEF_FUNC(EntityAccessor,getUUIDForTrack)
    DEF_FUNC(EntityAccessor,getFactoryBaseName)
    DEF_FUNC(EntityAccessor,getFactoryClassName)
    DEF_FUNC(EntityAccessor,getFullName)
    DEF_FUNC(EntityAccessor,getFactoryModelName)
    .def("isinstance",&EntityAccessor::isinstancePY)
    ;
}

ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
