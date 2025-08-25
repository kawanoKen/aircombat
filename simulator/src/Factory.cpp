// Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#include "Factory.h"
#include <fstream>
#include <limits>
#include "Entity.h"

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)

std::shared_mutex Factory::static_mtx;
std::map<std::string,std::map<std::string,Factory::creatorType>> Factory::creators;
std::map<std::string,std::map<std::string,nl::json>> Factory::defaultModelConfigs;
std::map<std::string,std::shared_ptr<FactoryHelper>> Factory::helpers;
#define FACTORY_NOERROR_FOR_DUPLICATION
#define FACTORY_ACCEPT_PACKAGE_NAME_OMISSION

#ifdef FACTORY_ACCEPT_OVERWRITE
const bool Factory::acceptOverwriteDefaults=true;
#else
const bool Factory::acceptOverwriteDefaults=false;
#endif
#ifdef FACTORY_NOERROR_FOR_DUPLICATION
const bool Factory::noErrorForDuplication=true;
#else
const bool Factory::noErrorForDuplication=false;
#endif
#ifdef FACTORY_ACCEPT_PACKAGE_NAME_OMISSION
const bool Factory::acceptPackageNameOmission=true;
#else
const bool Factory::acceptPackageNameOmission=false;
#endif
Factory::Factory(){
}
std::shared_ptr<Factory> Factory::create(){
    return std::make_shared<make_shared_enabler<Factory>>();
}
void Factory::clear(){
    std::lock_guard<std::shared_mutex> lock(static_mtx);
    creators.clear();
    defaultModelConfigs.clear();
}
void Factory::reset(){
    std::lock_guard<std::recursive_mutex> lock(mtx);
    for(auto& [baseName, configs] : modelConfigs){
        for(auto& [modelName, modelConfig] : configs){
            removedModelConfigsSinceLastSerialization[baseName].insert(modelName);
        }
    }
    modelConfigs.clear();
    createdModelConfigsSinceLastSerialization.clear();
}
void Factory::addClass(const std::string& baseName,const std::string& className,creatorType fn){
    std::lock_guard<std::shared_mutex> lock(static_mtx);
    if(acceptOverwriteDefaults){
        creators[baseName][className]=fn;
    }else{
        if(creators[baseName].count(className)==0){
            creators[baseName][className]=fn;
        }else{
            std::string txt=className+" is already registered in creators["+baseName+"].";
            if(noErrorForDuplication){
                std::cout<<"Warning! "<<txt<<std::endl;
            }else{
                throw std::runtime_error(txt);
            }
        }
    }
}
void Factory::addPythonClass(const std::string& baseName,const std::string& className,py::object clsObj){
    creatorType fn=[clsObj](const nl::json& modelConfig,const nl::json& instanceConfig){
        // Pythonオブジェクトを生成した後C++オブジェクトとしてのみの管理を行うとPython側で定義した派生クラスの情報が抜け落ちる。
        // そのため、Pythonオブジェクトをshared_ptrとしてPtrBaseTypeへのalias constructorでその寿命を紐付ける。
        std::shared_ptr<py::object> obj = std::make_shared<py::object>(clsObj(modelConfig,instanceConfig));
        auto ret=std::shared_ptr<Entity>(obj,obj->cast<std::shared_ptr<Entity>>().get());
        return std::move(ret);
    };
    addClass(baseName,className,fn);
}
void Factory::addDefaultModel(const std::string& baseName,const std::string& modelName,const nl::json& modelConfig_){
    std::lock_guard<std::shared_mutex> lock(static_mtx);
    if(acceptOverwriteDefaults){
        defaultModelConfigs[baseName][modelName]=modelConfig_;
    }else{
        if(defaultModelConfigs[baseName].count(modelName)==0){
            defaultModelConfigs[baseName][modelName]=modelConfig_;
        }else{
            std::string txt=modelName+" is already registered in defaultModelConfigs["+baseName+"].";
            if(noErrorForDuplication){
                std::cout<<"Warning! "<<txt<<std::endl;
            }else{
                throw std::runtime_error(txt);
            }
        }
    }
}
void Factory::addDefaultModelsFromJson(const nl::json& j){
    assert(j.is_object());
    for(auto& base:j.items()){
        std::string baseName=base.key();
        for(auto& model:base.value().items()){
            addDefaultModel(baseName,model.key(),model.value());
        }
    }
}
void Factory::addDefaultModelsFromJsonFile(const std::string& filePath){
    std::ifstream ifs(filePath);
    nl::json j;
    ifs>>j;
    addDefaultModelsFromJson(j.at("Factory"));//requires "Factory" key in the root node.
}
void Factory::addModel(const std::string& baseName,const std::string& modelName,const nl::json& modelConfig_){
    std::lock_guard<std::recursive_mutex> lock(mtx);
    if(!modelConfig_.is_object()){
        throw std::runtime_error("A model definition json for Factory needs to be an object.");
    }
    modelConfigs[baseName][modelName]=modelConfig_;
    createdModelConfigsSinceLastSerialization[baseName].insert(modelName);
}
void Factory::addModelsFromJson(const nl::json& j){
    assert(j.is_object());
    for(auto& base:j.items()){
        std::string baseName=base.key();
        for(auto& model:base.value().items()){
            addModel(baseName,model.key(),model.value());
        }
    }
}
void Factory::addModelsFromJsonFile(const std::string& filePath){
    std::ifstream ifs(filePath);
    nl::json j;
    ifs>>j;
    addModelsFromJson(j.at("Factory"));//requires "Factory" key in the root node.
}
bool Factory::hasClass(const std::string& baseName,const std::string& className){
    std::shared_lock<std::shared_mutex> lock(static_mtx);
    auto& constCreators=std::as_const(creators);
    if(constCreators.count(baseName)==0){
        return false;
    }
    return constCreators.at(baseName).count(className)>0;
}
bool Factory::hasDefaultModel(const std::string& baseName,const std::string& modelName){
    std::shared_lock<std::shared_mutex> lock(static_mtx);
    auto& constDefaultModelConfigs=std::as_const(defaultModelConfigs);
    if(constDefaultModelConfigs.count(baseName)==0){
        return false;
    }
    return constDefaultModelConfigs.at(baseName).count(modelName)>0;
}
bool Factory::hasModel(const std::string& baseName,const std::string& modelName) const{
    std::lock_guard<std::recursive_mutex> lock(mtx);
    if(hasDefaultModel(baseName,modelName)){
        return true;
    }
    if(modelConfigs.count(baseName)==0){
        return false;
    }
    return modelConfigs.at(baseName).count(modelName)>0;
}
nl::json Factory::getModelConfig(const std::string& baseName,const std::string& modelName,bool withDefaultModels) const{
    std::lock_guard<std::recursive_mutex> lock(mtx);
    if(modelConfigs.count(baseName)>0){
        auto sub=modelConfigs.at(baseName);
        auto found=sub.find(modelName);
        if(found!=sub.end()){
            return found->second;
        }
    }
    if(withDefaultModels){
        std::shared_lock<std::shared_mutex> lock(static_mtx);
        auto& constDefaultModelConfigs=std::as_const(defaultModelConfigs);
        if(constDefaultModelConfigs.count(baseName)>0){
            auto sub=constDefaultModelConfigs.at(baseName);
            auto found=sub.find(modelName);
            if(found!=sub.end()){
                return found->second;
            }
        }
    }
    return nl::json();
}
nl::json Factory::getModelConfigs(bool withDefaultModels) const{
    std::lock_guard<std::recursive_mutex> lock(mtx);
    if(withDefaultModels){
        nl::json ret=defaultModelConfigs;
        ret.merge_patch(modelConfigs);
        return ret;
    }else{
        return modelConfigs;
    }
}
std::tuple<std::string,std::string,nl::json,FactoryHelperChain> Factory::resolveModelConfig(const std::string& baseName,const std::string& className,const std::string& modelName,const nl::json& modelConfig,const FactoryHelperChain& helperChain,bool withDefaultModels) const{
    auto&& [resolvedClassName,resolvedModelName,resolvedModelConfig,resolvedPrefixChain] = resolveModelConfigSub(baseName,className,modelName,modelConfig,FactoryHelperChain(),helperChain,withDefaultModels);
    return {resolvedClassName,resolvedModelName,resolvedModelConfig,FactoryHelperChain(resolvedPrefixChain,helperChain)};
}
std::tuple<std::string,std::string,nl::json,FactoryHelperChain> Factory::resolveModelConfig(const std::string& baseName,const std::string& modelName,const FactoryHelperChain& helperChain,bool withDefaultModels) const{
    auto&& [resolvedClassName,resolvedModelName,resolvedModelConfig,resolvedPrefixChain] = resolveModelConfigSub(baseName,modelName,FactoryHelperChain(),helperChain,withDefaultModels);
    return {resolvedClassName,resolvedModelName,resolvedModelConfig,FactoryHelperChain(resolvedPrefixChain,helperChain)};
}
std::tuple<std::string,std::string,nl::json,FactoryHelperChain> Factory::resolveModelConfig(const std::string& baseName,const nl::json& j,const FactoryHelperChain& helperChain,bool withDefaultModels) const{
    if(j.is_string()){
        return resolveModelConfig(baseName,j.get<std::string>(),helperChain,withDefaultModels);
    }
    if(!j.is_object()){
        throw std::runtime_error("A json argument for Factory::resolveModelConfig needs to be an object-type json.");
    }
    auto&& [resolvedClassName,resolvedModelName,resolvedModelConfig,resolvedPrefixChain] = resolveModelConfigSub(baseName,j,FactoryHelperChain(),helperChain,withDefaultModels);
    return {resolvedClassName,resolvedModelName,resolvedModelConfig,FactoryHelperChain(resolvedPrefixChain,helperChain)};
}
std::tuple<std::string,std::string,nl::json,FactoryHelperChain> Factory::resolveModelConfig(const nl::json& j,const FactoryHelperChain& helperChain,bool withDefaultModels) const{
    if(!j.is_object()){
        throw std::runtime_error("A json argument for Factory::resolveModelConfig needs to be an object-type json.");
    }
    std::string baseName;
    if(j.contains("baseName")){
        baseName=j.at("baseName");
    }else{
        throw std::runtime_error("The key 'baseName' is missing in the json given to Factory::resolveModelConfig");
    }
    return resolveModelConfig(baseName,j,helperChain,withDefaultModels);
}
std::tuple<std::string,std::string,nl::json,FactoryHelperChain> Factory::resolveModelConfig(const std::string& baseName,const std::string& className,const std::string& modelName,const nl::json& modelConfig,bool withDefaultModels) const{
    return resolveModelConfig(baseName,className,modelName,modelConfig,FactoryHelperChain(),withDefaultModels);
}
std::tuple<std::string,std::string,nl::json,FactoryHelperChain> Factory::resolveModelConfig(const std::string& baseName,const std::string& modelName,bool withDefaultModels) const{
    return resolveModelConfig(baseName,modelName,FactoryHelperChain(),withDefaultModels);
}
std::tuple<std::string,std::string,nl::json,FactoryHelperChain> Factory::resolveModelConfig(const std::string& baseName,const nl::json& j,bool withDefaultModels) const{
    return resolveModelConfig(baseName,j,FactoryHelperChain(),withDefaultModels);
}
std::tuple<std::string,std::string,nl::json,FactoryHelperChain> Factory::resolveModelConfig(const nl::json& j,bool withDefaultModels) const{
    return resolveModelConfig(j,FactoryHelperChain(),withDefaultModels);
}
std::tuple<std::string,std::string,nl::json,FactoryHelperChain> Factory::resolveModelConfigSub(const std::string& baseName,const std::string& modelName,const FactoryHelperChain& prefixChain,const FactoryHelperChain& originalChain,bool withDefaultModels) const{
    return resolveModelConfigSub(baseName,"",modelName,nl::json(),prefixChain,originalChain,withDefaultModels);
}
std::tuple<std::string,std::string,nl::json,FactoryHelperChain> Factory::resolveModelConfigSub(const std::string& baseName,const std::string& className,const std::string& modelName,const nl::json& modelConfig,const FactoryHelperChain& prefixChain,const FactoryHelperChain& originalChain,bool withDefaultModels) const{
    std::lock_guard<std::recursive_mutex> lock(mtx);
    bool isClassNameGiven=(className!="");
    bool isModelNameGiven=(modelName!="");
    bool isModelConfigGiven=modelConfig.is_object();
    std::string resolvedClassName;
    std::string resolvedModelName;
    nl::json resolvedModelConfig;
    FactoryHelperChain resolvedPrefixChain=FactoryHelperChain(prefixChain);
    if(isModelNameGiven){
        // model (refers to a pair of class and config)
        resolvedModelName=FactoryHelperChain(prefixChain,originalChain).resolveModelName(baseName,modelName,this->shared_from_this());
        nl::json registeredModel=getModelConfig(baseName,resolvedModelName,true);
        if(registeredModel.is_null()){
            //見つからなかったらmodelName='Default'として探す。
            std::cout<<"Warning: "+resolvedModelName+" is not registered as a model name in the Factory with baseName='"+baseName+"'."<<std::endl;
            if(modelName!="Default"){
                try{
                    return resolveModelConfigSub(baseName,className,"Default",modelConfig,prefixChain,originalChain,withDefaultModels);
                }catch(...){
                    throw std::runtime_error("Neither 'Default' nor '"+resolvedModelName+"' is not registered as a model name in the Factory with baseName='"+baseName+"'.");
                }
            }else{
                throw std::runtime_error("Neither 'Default' nor '"+resolvedModelName+"' is not registered as a model name in the Factory with baseName='"+baseName+"'.");
            }
        }
        auto modelHelper=getHelperFromClassOrModelName(resolvedModelName);
        resolvedPrefixChain.push_back(modelHelper);

        if(isClassNameGiven){
            resolvedClassName=FactoryHelperChain(prefixChain,modelHelper,originalChain).resolveClassName(baseName,className);
            auto classHelper=getHelperFromClassOrModelName(resolvedClassName);
            resolvedPrefixChain.push_back(classHelper);
        }

        auto&& [subClassName,subModelName,subModelConfig,subPrefixChain]=resolveModelConfigSub(baseName,registeredModel,resolvedPrefixChain,originalChain,withDefaultModels);

        if(!isClassNameGiven){
            resolvedClassName=subClassName;
        }
        if(isModelConfigGiven){
            subModelConfig.merge_patch(modelConfig);
        }
        resolvedModelConfig=subModelConfig;
        resolvedPrefixChain=subPrefixChain;
    }else{
        // class and config directly
        resolvedModelName="";
        if(!isClassNameGiven){
            throw std::runtime_error("At least one of 'model', 'modelConfig', 'class', or 'className' is required as a key in the json given to Factory::resolveModelConfigSub");
        }
        resolvedClassName=FactoryHelperChain(prefixChain,originalChain).resolveClassName(baseName,className);
        resolvedModelConfig=modelConfig;
        auto classHelper=getHelperFromClassOrModelName(resolvedClassName);
        resolvedPrefixChain.push_back(classHelper);
    }
    return {resolvedClassName,resolvedModelName,resolvedModelConfig,resolvedPrefixChain};
}
std::tuple<std::string,std::string,nl::json,FactoryHelperChain> Factory::resolveModelConfigSub(const std::string& baseName,const nl::json& j,const FactoryHelperChain& prefixChain,const FactoryHelperChain& originalChain,bool withDefaultModels) const{
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

    return resolveModelConfigSub(baseName,className,modelName,modelConfig,prefixChain,originalChain,withDefaultModels);
}
void Factory::reconfigureModelConfigs(const nl::json& j){
    std::lock_guard<std::recursive_mutex> lock(mtx);
    nl::json tmp=modelConfigs;
    tmp.merge_patch(j);
    modelConfigs=tmp.get<std::map<std::string,std::map<std::string,nl::json>>>();
    for(auto& [baseName, configs] : j.items()){
        for(auto& [modelName, modelConfig] : configs.items()){
            if(modelConfig.is_null()){
                removedModelConfigsSinceLastSerialization[baseName].insert(modelName);
                auto it=createdModelConfigsSinceLastSerialization[baseName].find(modelName);
                if(it!=createdModelConfigsSinceLastSerialization[baseName].end()){
                    createdModelConfigsSinceLastSerialization[baseName].erase(it);
                }
            }else{
                createdModelConfigsSinceLastSerialization[baseName].insert(modelName);
                auto it=removedModelConfigsSinceLastSerialization[baseName].find(modelName);
                if(it!=removedModelConfigsSinceLastSerialization[baseName].end()){
                    removedModelConfigsSinceLastSerialization[baseName].erase(it);
                }
            }
        }
    }
}
bool Factory::issubclassof(py::object baseCls,const std::string& derivedBaseName,const std::string& derivedClassName){
    auto dummy=Factory::create()->createByClassName<Entity>(derivedBaseName,derivedClassName,nl::json(nullptr),nl::json(nullptr));
    return py::isinstance(py::cast(dummy),baseCls);
}
Factory::creatorType Factory::getCreator(const std::string& baseName,const std::string& className){
    std::shared_lock<std::shared_mutex> lock(static_mtx);
    auto& constCreators=std::as_const(creators);
    if(constCreators.count(baseName)==0){
        return nullptr;
    }
    auto sub=constCreators.at(baseName);
    auto found=sub.find(className);
    if(found!=sub.end()){
        return found->second;
    }else{
        return nullptr;
    }
}
std::shared_ptr<Entity> Factory::createImpl(const std::string& baseName,const std::string& className,const std::string& modelName,const nl::json& modelConfig,const nl::json& instanceConfig, const FactoryHelperChain& helperChain, const std::shared_ptr<const Entity>& caller){
    auto [resolvedClassName,resolvedModelName,resolvedModelConfig,resolvedChain]=resolveModelConfig(baseName,className,modelName,modelConfig,helperChain,true);
    if(!hasBaseName(baseName)){
        throw std::runtime_error("'"+baseName+"' is not registered as a base name in the Factory. Without base name registration, Factory allows class and model registration but does not allows instantiation of them.");
    }
    if(caller && !checkPermissionForCreate(caller,baseName)){
            throw std::runtime_error("Creation of '"+baseName+"' by '"+caller->getFactoryBaseName()+"' is not allowed.");
    }
    auto creator=getCreator(baseName,resolvedClassName);
    if(!creator){
        throw std::runtime_error(resolvedClassName+" is not registered as a class name in the Factory with baseName="+baseName+".");
    }
    auto ret=creator(resolvedModelConfig,instanceConfig);
    ret->setFactoryCreationMetadata(baseName,resolvedClassName,resolvedModelName,resolvedChain);
    return ret;
}
std::shared_ptr<Entity> Factory::createOverloader(const std::string& baseName,const std::string& className,const std::string& modelName,const nl::json& modelConfig,const nl::json& instanceConfig){
    return createImpl(baseName, className, modelName, modelConfig, instanceConfig, FactoryHelperChain(), nullptr);
}
std::shared_ptr<Entity> Factory::createOverloader(const std::string& baseName,const std::string& className,const std::string& modelName,const nl::json& modelConfig,const nl::json& instanceConfig, const FactoryHelperChain& helperChain){
    return createImpl(baseName, className, modelName, modelConfig, instanceConfig, helperChain, nullptr);
}
std::shared_ptr<Entity> Factory::createOverloader(const std::string& baseName,const std::string& className,const std::string& modelName,const nl::json& modelConfig,const nl::json& instanceConfig, const std::shared_ptr<const Entity>& caller){
    if(caller){
        return createImpl(baseName, className, modelName, modelConfig, instanceConfig, caller->getFactoryHelperChain(), caller);
    }else{
        return createImpl(baseName, className, modelName, modelConfig, instanceConfig, FactoryHelperChain(), caller);
    }
}
std::shared_ptr<Entity> Factory::createOverloader(const std::string& baseName,const std::string& modelName,const nl::json& instanceConfig){
    return createOverloader(baseName, "", modelName, nl::json(), instanceConfig);
}
std::shared_ptr<Entity> Factory::createOverloader(const std::string& baseName,const std::string& modelName,const nl::json& instanceConfig, const FactoryHelperChain& helperChain){
    return createOverloader(baseName, "", modelName, nl::json(), instanceConfig, helperChain);
}
std::shared_ptr<Entity> Factory::createOverloader(const std::string& baseName,const std::string& modelName,const nl::json& instanceConfig, const std::shared_ptr<const Entity>& caller){
    return createOverloader(baseName, "", modelName, nl::json(), instanceConfig, caller);
}
std::shared_ptr<Entity> Factory::createByClassNameOverloader(const std::string& baseName,const std::string& className,const nl::json& modelConfig,const nl::json& instanceConfig){
    return createOverloader(baseName, className, "", modelConfig, instanceConfig);
}
std::shared_ptr<Entity> Factory::createByClassNameOverloader(const std::string& baseName,const std::string& className,const nl::json& modelConfig,const nl::json& instanceConfig, const FactoryHelperChain& helperChain){
    return createOverloader(baseName, className, "", modelConfig, instanceConfig, helperChain);
}
std::shared_ptr<Entity> Factory::createByClassNameOverloader(const std::string& baseName,const std::string& className,const nl::json& modelConfig,const nl::json& instanceConfig, const std::shared_ptr<const Entity>& caller){
    return createOverloader(baseName, className, "", modelConfig, instanceConfig, caller);
}
std::shared_ptr<Entity> Factory::createImpl(const nl::json& j, const FactoryHelperChain& helperChain, const std::shared_ptr<const Entity>& caller){
    if(j.is_object()){
        std::lock_guard<std::recursive_mutex> lock(mtx);
        std::string baseName;
        if(j.contains("baseName")){
            baseName=j.at("baseName");
        }else{
            throw std::runtime_error("The key 'baseName' is missing in the json given to Factory::createImpl");
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

        if(!j.contains("instanceConfig")){
            throw std::runtime_error("Missing key 'instanceConfig' in Factory::createImpl");
        }

        return createImpl(baseName,className,modelName,modelConfig,j.at("instanceConfig"),helperChain,caller);
    }else{
        throw std::runtime_error("Factory::createImpl with single json only accepts json object.");
    }
}
std::shared_ptr<Entity> Factory::createOverloader(const nl::json& j){
    return createImpl(j, FactoryHelperChain(), nullptr);
}
std::shared_ptr<Entity> Factory::createOverloader(const nl::json& j, const FactoryHelperChain& helperChain){
    return createImpl(j, helperChain, nullptr);
}
std::shared_ptr<Entity> Factory::createOverloader(const nl::json& j, const std::shared_ptr<const Entity>& caller){
    if(caller){
        return createImpl(j, caller->getFactoryHelperChain(), caller);
    }else{
        return createImpl(j, FactoryHelperChain(), caller);
    }
}

std::pair<std::string,std::string> Factory::parseName(const std::string& fullName){
    // fullName: class name or model name
    auto dot=fullName.rfind(".");
    if(dot==std::string::npos){
        return {"", fullName};
    }
    return {fullName.substr(0,dot), fullName.substr(dot+1)};
}

void Factory::addHelper(std::shared_ptr<FactoryHelper> helper,const std::string& packageName){
    std::lock_guard<std::shared_mutex> lock(static_mtx);
    if(helpers.count(packageName)>0){
        if(packageName==""){
            return; // default helper without prefix.
        }else{
            throw std::runtime_error("The FactoryHelper for \""+packageName+"\" is already registered in the Factory.");
        }
    }
    helpers[packageName]=helper;
}

std::shared_ptr<FactoryHelper> Factory::getHelperFromPackageName(const std::string& packageName){
    {
        std::shared_lock<std::shared_mutex> lock(static_mtx);
        auto constHelpers=std::as_const(helpers);
        if(constHelpers.find(packageName) != constHelpers.end()){
            return constHelpers.at(packageName);
        }
        for(auto&& [name, helper]: constHelpers){
            if(helper->hasAlias(packageName)){
                return helper;
            }
        }
    }
    // 未インポートなだけかもしれないので、インポートを試みる。現バージョンではインポート時点でFactoryに登録されるものとしているので見つかるはず。
    // try to import base package. For the current version, all classes and models should be registered by import and can be found after import.
    try{
        auto module=py::module::import(packageName.c_str());
        std::shared_lock<std::shared_mutex> lock(static_mtx);
        auto constHelpers=std::as_const(helpers);
        if(constHelpers.find(packageName) != constHelpers.end()){
            return constHelpers.at(packageName);
        }
        for(auto&& [name, helper]: constHelpers){
            if(helper->hasAlias(packageName)){
                return helper;
            }
        }
    }catch(...){
    }
    return nullptr;
}
std::shared_ptr<FactoryHelper> Factory::getHelperFromClassOrModelName(const std::string& fullName){
    std::shared_lock<std::shared_mutex> lock(static_mtx);
    auto [packageName, remainder] = parseName(fullName);
    return getHelperFromPackageName(packageName);
}

void Factory::dummyCreationTest(){
    auto factory=Factory::create();
    for(auto&& e1:creators){
        std::string baseName=e1.first;
        for(auto&& e2:e1.second){
            std::string className=e2.first;
            try{
                auto dummy=factory->createByClassName<Entity>(baseName,className,nl::json(nullptr),nl::json(nullptr));
                std::cout<<"Success: "<<baseName<<","<<className<<std::endl;
            }catch(...){
                std::cout<<"Failure: "<<baseName<<","<<className<<std::endl;
            }
        }
    }
}


//baseNameの管理
std::map<std::string, std::string> Factory::baseNameTree;
std::map<std::string, std::set<std::string>> Factory::permissionsForCreate;
std::map<std::string, std::set<std::string>> Factory::permissionsForGet;
std::set<std::string> Factory::mergeParentPermission(const std::set<std::string>& child, const std::set<std::string>& parent){
    std::set<std::string> ret(child);
    for(auto&& key: parent){
        if(key=="AnyOthers"){
            if(ret.find("!AnyOthers")==ret.end()){
                ret.insert(key);
            }
        }else if(key=="!AnyOthers"){
            if(ret.find("AnyOthers")==ret.end()){
                ret.insert(key);
            }
        }else if(key!=""){
            if(key.at(0)=='!'){
                if(key.size()>1 && ret.find(key.substr(1))==ret.end()){
                    ret.insert(key);
                }
            }else{
                if(ret.find("!"+key)==ret.end()){
                    ret.insert(key);
                }
            }
        }
    }
    return std::move(ret);
}
bool Factory::checkPermissionImpl(const std::string& callerBaseName, const std::string& targetBaseName,const std::map<std::string, std::set<std::string>>& permissions){
    std::shared_lock<std::shared_mutex> lock(static_mtx);
    auto constBaseNameTree=std::as_const(baseNameTree);
    if(permissions.find(targetBaseName) != permissions.end()){
        auto&& permission=permissions.at(targetBaseName);
        if(permission.find(callerBaseName)!=permission.end()){
            //明示的許可
            return true;
        }else if(permission.find("!"+callerBaseName)!=permission.end()){
            //明示的禁止
            return false;
        }else if(permission.find("AnyOthers")!=permission.end()){
            //暗黙的許可の場合は親を辿って明示的に禁止されていなければ許可
            std::string nextBaseName=constBaseNameTree.at(callerBaseName);
            while(nextBaseName!=""){
                if(!checkPermissionImpl(nextBaseName,targetBaseName,permissions)){
                    //親が禁止されていたら禁止
                    return false;
                }
                nextBaseName=constBaseNameTree.at(nextBaseName);
            }
            return true;
        }else if(permission.find("!AnyOthers")!=permission.end()){
            //暗黙的禁止の場合は親を辿って明示的に許可されていなければ禁止
            std::string nextBaseName=constBaseNameTree.at(callerBaseName);
            while(nextBaseName!=""){
                if(checkPermissionImpl(nextBaseName,targetBaseName,permissions)){
                    //親が許可されていたら許可
                    return true;
                }
                nextBaseName=constBaseNameTree.at(nextBaseName);
            }
            return false;
        }else{
            //指定なしは許可とする
            return true;
        }
    }else{
        return false;
    }
}
void Factory::addBaseName(const std::string& baseName, const std::string& parentBaseName, const std::set<std::string>& permissionForCreate, const std::set<std::string>& permissionForGet){
    std::lock_guard<std::shared_mutex> lock(static_mtx);
    if(hasBaseName(baseName)){
        //既に登録済の場合、何もせずにreturnする。エラーにはしない。
        return;
    }
    if(parentBaseName!="" && !hasBaseName(parentBaseName)){
        // 親として指定したbaseNameが存在しない場合はエラーとする。
        // FACTORY_ADD_BASEマクロを利用していれば生じない。
        throw std::runtime_error(parentBaseName+" is not registered as Factory's base name.");
    }
    baseNameTree[baseName]=parentBaseName;
    if(parentBaseName==""){
        permissionsForCreate[baseName]=permissionForCreate;
        permissionsForGet[baseName]=permissionForGet;
    }else{
        permissionsForCreate[baseName]=mergeParentPermission(permissionForCreate,permissionsForCreate.at(parentBaseName));
        permissionsForGet[baseName]=mergeParentPermission(permissionForGet,permissionsForGet.at(parentBaseName));
    }
}
bool Factory::hasBaseName(const std::string& baseName){
    return baseNameTree.find(baseName)!=baseNameTree.end();
}
bool Factory::checkPermissionForCreate(const std::shared_ptr<const Entity>& caller, const std::string& targetBaseName){
    if(!caller){return false;}
    std::string callerBaseName = caller->getFactoryBaseName();
    if(!hasBaseName(targetBaseName)){
        throw std::runtime_error("'"+targetBaseName+"' is not registered as a base name in the Factory.");
    }
    return checkPermissionImpl(callerBaseName,targetBaseName,permissionsForCreate);
}
bool Factory::checkPermissionForGet(const std::shared_ptr<const Entity>& caller, const std::string& targetBaseName){
    if(!caller){return false;}
    std::string callerBaseName = caller->getFactoryBaseName();
    if(!hasBaseName(targetBaseName)){
        throw std::runtime_error("'"+targetBaseName+"' is not registered as a base name in the Factory.");
    }
    return checkPermissionImpl(callerBaseName,targetBaseName,permissionsForGet);
}

// 内部状態のシリアライゼーション(Entityを除く)
void Factory::serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full){
    if(asrc::core::util::isOutputArchive(archive)){
        // save
        if(full){
            call_operator(archive,cereal::make_nvp("modelConfigs",getModelConfigs(false)));
        }else{
            nl::json modelConfigsToBeSaved=nl::json::object();
            for(auto& [baseName, modelNames] : createdModelConfigsSinceLastSerialization){
                modelConfigsToBeSaved[baseName]=nl::json::object();
                for(auto& modelName : modelNames){
                    modelConfigsToBeSaved[baseName][modelName]=getModelConfig(baseName,modelName,false);
                }
            }
            for(auto& [baseName, modelNames] : removedModelConfigsSinceLastSerialization){
                if(!modelConfigsToBeSaved.contains(baseName)){
                    modelConfigsToBeSaved[baseName]=nl::json::object();
                }
                for(auto& modelName : modelNames){
                    modelConfigsToBeSaved[baseName][modelName]=nullptr;
                }
            }
            call_operator(archive,cereal::make_nvp("modelConfigs",modelConfigsToBeSaved));
        }
        createdModelConfigsSinceLastSerialization.clear();
        removedModelConfigsSinceLastSerialization.clear();
    }else{
        // load
        nl::json loadedModelConfigs;
        call_operator(archive,cereal::make_nvp("modelConfigs",loadedModelConfigs));
        if(full){
            reset();
            addModelsFromJson(loadedModelConfigs);
        }else{
            for(auto& [baseName, configs] : loadedModelConfigs.items()){
                for(auto& [modelName, modelConfig] : configs.items()){
                    if(modelConfig.is_null()){
                        auto it=modelConfigs.at(baseName).find(modelName);
                        if(it!=modelConfigs.at(baseName).end()){
                            modelConfigs.at(baseName).erase(it);
                        }
                    }else{
                        addModel(baseName,modelName,modelConfig);
                    }
                }
            }
        }
        createdModelConfigsSinceLastSerialization.clear();
        removedModelConfigsSinceLastSerialization.clear();
    }
}

FactoryHelper::FactoryHelper(const std::string& _moduleName){
    isValid=false;
    moduleName=_moduleName;
    aliases.insert(_moduleName);
    prefix="";
}
FactoryHelper::~FactoryHelper(){}
void FactoryHelper::validate(const std::string& _packageName, const std::string& _packagePath){
    packageName=_packageName;
    packagePath=_packagePath;
    if(_packageName==""){
        prefix="";
    }else{
        prefix=_packageName+".";
    }
    Factory::addHelper(this->shared_from_this(),_packageName);
    isValid=true;
    for(auto&& [baseName, sub]: creators){
        for(auto&& [className, fn]: sub){
            Factory::addClass(baseName, prefix+className, fn);
        }
        sub.clear();
    }
    creators.clear();
    for(auto&& [baseName, sub]: defaultModelConfigs){
        for(auto&& [modelName, modelConfig_]: sub){
            Factory::addDefaultModel(baseName, prefix+modelName, modelConfig_);
        }
        sub.clear();
    }
    defaultModelConfigs.clear();
    for(auto && [name, sub] : subHelpers){
        sub->validate(packageName+"."+name,packagePath.string());
    }
}
void FactoryHelper::addAliasPackageName(const std::string& _parent){
    std::lock_guard<std::recursive_mutex> lock(mtx);
    std::string alias_name = _parent+"."+moduleName;
    if(!hasAlias(alias_name)){
        aliases.insert(alias_name);
        // std::cout<<"An alias added. '"<<alias_name<<"' - > ' prefix="<<prefix<<std::endl;
        for(auto&& [name, helper]: dependingModulePrefixes){
            helper->addAliasPackageName(_parent);
        }
        for(auto && [name, sub] : subHelpers){
            sub->addAliasPackageName(alias_name);
        }
    }
}
bool FactoryHelper::hasAlias(const std::string& _packageName) const{
    std::lock_guard<std::recursive_mutex> lock(mtx);
    return aliases.find(_packageName) != aliases.end();
}
void FactoryHelper::addDependency(const std::string& name, std::shared_ptr<FactoryHelper> helper){
    dependingModulePrefixes[name]=helper;
    for(auto && [sName, sub] : subHelpers){
        sub->addDependency(name,helper);
    }
}
std::shared_ptr<FactoryHelper> FactoryHelper::createSubHelper(const std::string& name){
    auto ret=FactoryHelper::create(name);
    for(auto && [dName,d] : dependingModulePrefixes){
        ret->addDependency(dName,d);
    }
    subHelpers[name]=ret;
    return ret;
}
std::pair<std::string,bool> FactoryHelper::resolveClassName(const std::string& baseName, const std::string& className) const{
    std::lock_guard<std::recursive_mutex> lock(mtx);
    auto [base, remainder] = Factory::parseName(className);
    if(base==""){
        if(Factory::hasClass(baseName,prefix+className)){
            return {prefix+className,true};
        }else{
            return {className,false};
        }
    }
    if(aliases.find(base) != aliases.end()){
        return {prefix+remainder,true};
    }
    if(dependingModulePrefixes.count(base)!=0){
        return dependingModulePrefixes.at(base)->resolveClassName(baseName,className);
    }else{
        return {className,false};
    }
}
std::pair<std::string,bool> FactoryHelper::resolveModelName(const std::string& baseName, const std::string& modelName, std::shared_ptr<const Factory> factory) const{
    std::lock_guard<std::recursive_mutex> lock(mtx);
    auto [base, remainder] = Factory::parseName(modelName);
    if(base==""){
        bool existence=false;
        if(factory){
            existence=factory->hasModel(baseName,prefix+modelName);
        }else{
            existence=Factory::hasDefaultModel(baseName,prefix+modelName);
        }
        if(existence){
            return {prefix+modelName,true};
        }else{
            return {modelName,false};
        }
    }
    if(aliases.find(base) != aliases.end()){
        return {prefix+remainder,true};
    }
    if(dependingModulePrefixes.count(base)!=0){
        return dependingModulePrefixes.at(base)->resolveModelName(baseName,modelName,factory);
    }else{
        return {modelName,false};
    }
}
void FactoryHelper::addClass(const std::string& baseName,const std::string& className,Factory::creatorType fn){
    std::lock_guard<std::recursive_mutex> lock(mtx);
    std::string modifiedClassName=prefix+className;
    if(isValid){
        Factory::addClass(baseName, modifiedClassName, fn);
    }else{
        if(Factory::acceptOverwriteDefaults){
            creators[baseName][modifiedClassName]=fn;
        }else{
            if(creators[baseName].count(modifiedClassName)==0){
                creators[baseName][modifiedClassName]=fn;
            }else{
                std::string txt=modifiedClassName+" is already registered in creators["+baseName+"].";
                if(Factory::noErrorForDuplication){
                    std::cout<<"Warning! "<<txt<<std::endl;
                }else{
                    throw std::runtime_error(txt);
                }
            }
        }
    }
}
void FactoryHelper::addPythonClass(const std::string& baseName,const std::string& className,py::object clsObj){
    Factory::creatorType fn=[clsObj](const nl::json& modelConfig,const nl::json& instanceConfig){
        // 2.13.1時点のpybind11においては、
        // Pythonオブジェクトを生成した後C++オブジェクトとしてのみの管理を行うとPython側で定義した派生クラスの情報が抜け落ちる。
        // また、virtual継承を行った基底クラスのポインタを挟むとPythonオブジェクトにcastできなくなる(static_castで実現されているため。)
        // (1) Pythonオブジェクトのshared_ptrを作成してalias constructorで寿命を紐付ける。
        // (2) py::handleをメンバとして保持しておき、Python側に戻す時はそちらを呼ぶ。
        // TODO 上記を解決する可能性のあるPRが出ているので、マージされたらよりよい解決策を検討する。
        std::shared_ptr<py::object> obj = std::make_shared<py::object>(clsObj(modelConfig,instanceConfig));
        auto ret=std::shared_ptr<Entity>(obj,obj->cast<std::shared_ptr<Entity>>().get());
        ret->setPyHandle(py::handle(*(obj.get())));
        return std::move(ret);
    };
    addClass(baseName,className,fn);
}
void FactoryHelper::addDefaultModel(const std::string& baseName,const std::string& modelName,const nl::json& modelConfig_){
    std::lock_guard<std::recursive_mutex> lock(mtx);
    std::string modifiedModelName=prefix+modelName;
    if(isValid){
        Factory::addDefaultModel(baseName, modifiedModelName, modelConfig_);
    }else{
        if(Factory::acceptOverwriteDefaults){
            defaultModelConfigs[baseName][modifiedModelName]=modelConfig_;
        }else{
            if(defaultModelConfigs[baseName].count(modifiedModelName)==0){
                defaultModelConfigs[baseName][modifiedModelName]=modelConfig_;
            }else{
                std::string txt=modifiedModelName+" is already registered in defaultModelConfigs["+baseName+"].";
                if(Factory::noErrorForDuplication){
                    std::cout<<"Warning! "<<txt<<std::endl;
                }else{
                    throw std::runtime_error(txt);
                }
            }
        }
    }
}
void FactoryHelper::addDefaultModelsFromJson(const nl::json& j){
    assert(j.is_object());
    for(auto& base:j.items()){
        std::string baseName=base.key();
        for(auto& model:base.value().items()){
            addDefaultModel(baseName,model.key(),model.value());
        }
    }
}
void FactoryHelper::addDefaultModelsFromJsonFile(const std::string& filePath){
    std::ifstream ifs(filePath);
    nl::json j;
    ifs>>j;
    addDefaultModelsFromJson(j.at("Factory"));//requires "Factory" key in the root node.
}
std::filesystem::path FactoryHelper::getPackagePath() const{
    return packagePath;
}

FactoryHelperChain::FactoryHelperChain(){
}
FactoryHelperChain::FactoryHelperChain(std::shared_ptr<FactoryHelper> helper){
    if(helper){
        helpers.push_back(helper);
    }
}
FactoryHelperChain::FactoryHelperChain(const nl::json& chain_j){
    chain_j.get_to(helpers);
}
FactoryHelperChain::FactoryHelperChain(const std::vector<FactoryHelperChain>& chains){
    for(auto&& other: chains){
        for(auto&& helper: other.helpers){
            helpers.push_back(helper);
        }
    }
}
FactoryHelperChain::FactoryHelperChain(py::tuple args){
    for(auto&& arg : args){
        if(py::isinstance<FactoryHelper>(arg)){
            helpers.push_back(arg.cast<std::shared_ptr<FactoryHelper>>());
        }else if(py::isinstance<nl::json>(arg)){
            std::deque<std::shared_ptr<FactoryHelper>> sub;
            arg.cast<nl::json>().get_to(sub);
            for(auto&& s : sub){
                helpers.push_back(s);
            }
        }else if(py::isinstance<nl::ordered_json>(arg)){
            std::deque<std::shared_ptr<FactoryHelper>> sub;
            arg.cast<nl::ordered_json>().get_to(sub);
            for(auto&& s : sub){
                helpers.push_back(s);
            }
        }else if(py::isinstance<FactoryHelperChain>(arg)){
            FactoryHelperChain sub=arg.cast<FactoryHelperChain>();
            for(auto&& s : sub.helpers){
                helpers.push_back(s);
            }
        }else if(py::isinstance<py::tuple>(arg)){
            FactoryHelperChain sub(arg);
            for(auto&& s : sub.helpers){
                helpers.push_back(s);
            }
        }else{
            throw std::runtime_error("FactoryHelperChain can accept 'FactoryHelper', 'nljson', 'FactoryHelperChain' or tuple of them for __init__. Given "+py::repr(py::type::of(arg)).cast<std::string>());
        }
    }
}
FactoryHelperChain::~FactoryHelperChain(){
}
void FactoryHelperChain::push_front(std::shared_ptr<FactoryHelper> helper){
    if(helper){
        helpers.push_front(helper);
    }
}
void FactoryHelperChain::push_back(std::shared_ptr<FactoryHelper> helper){
    if(helper){
        helpers.push_back(helper);
    }
}
std::shared_ptr<FactoryHelper> FactoryHelperChain::front(){
    return helpers.front();
}
void FactoryHelperChain::pop_front(){
    helpers.pop_front();
}
std::string FactoryHelperChain::resolveClassName(const std::string& baseName, const std::string& className) const{
    // (1) chainの中から探す。
    for(auto &&helper: helpers){
        auto [resolvedName, resolved] = helper->resolveClassName(baseName, className);
        if(resolved){
            return resolvedName;
        }
    }

    // (2) Factoryに該当するhelperがあればそれを使う。
    auto [base, remainder] = Factory::parseName(className);
    if(base!=""){
        auto helper=Factory::getHelperFromPackageName(base);
        if(helper){
            auto [resolvedName, resolved] = helper->resolveClassName(baseName, className);
            if(resolved){
                return resolvedName;
            }
        }
    }

    // (3) Factoryの全helperから探す。
    if(Factory::acceptPackageNameOmission){
        for(auto&& [name, helper]: Factory::helpers){
            auto [resolvedName, resolved] = helper->resolveClassName(baseName, className);
            if(resolved){
                return resolvedName;
            }
        }
    }
    return className;
}
std::string FactoryHelperChain::resolveModelName(const std::string& baseName, const std::string& modelName, std::shared_ptr<const Factory> factory) const{
    // (1) chainの中から探す。
    for(auto &&helper: helpers){
        auto [resolvedName, resolved] = helper->resolveModelName(baseName, modelName, factory);
        if(resolved){
            return resolvedName;
        }
    }

    // (2) Factoryに該当するhelperがあればそれを使う。
    auto [base, remainder] = Factory::parseName(modelName);
    if(base!=""){
        auto helper=Factory::getHelperFromPackageName(base);
        if(helper){
            auto [resolvedName, resolved] = helper->resolveModelName(baseName, modelName, factory);
            if(resolved){
                return resolvedName;
            }
        }
    }

    // (3) Factoryの全helperから探す。
    if(Factory::acceptPackageNameOmission){
        for(auto&& [name, helper]: Factory::helpers){
            auto [resolvedName, resolved] = helper->resolveModelName(baseName, modelName, factory);
            if(resolved){
                return resolvedName;
            }
        }
    }
    return modelName;
}
std::filesystem::path FactoryHelperChain::resolveFilePath(const std::string& path) const{
    return resolveFilePath(std::filesystem::path(path));
}
std::filesystem::path FactoryHelperChain::resolveFilePath(const std::filesystem::path& path) const{
    if(path.is_absolute()){
        return path;
    }else{
        // (1) chainの中から探す。
        for(auto &&helper: helpers){
            auto base=helper->getPackagePath();
            auto full=base/path;
            if(std::filesystem::exists(full)){
                return full;
            }
        }
        // (2) Factoryの全helperから探す
        if(Factory::acceptPackageNameOmission){
            for(auto&& [name, helper]: Factory::helpers){
                auto base=helper->getPackagePath();
                auto full=base/path;
                if(std::filesystem::exists(full)){
                    return full;
                }
            }
        }
        return std::filesystem::current_path()/path;
    }
}

void exportFactory(py::module &m)
{
    using namespace pybind11::literals;
    py::class_<Factory,std::shared_ptr<Factory>>(m,"Factory")
    .def(py::init([](){return Factory::create();}))
    DEF_STATIC_FUNC(Factory,clear)
    DEF_FUNC(Factory,reset)
    DEF_STATIC_FUNC(Factory,addClass)
    .def("addPythonClass",&Factory::addPythonClass)
    DEF_STATIC_FUNC(Factory,addDefaultModel)
    DEF_STATIC_FUNC(Factory,addDefaultModelsFromJson)
    DEF_STATIC_FUNC(Factory,addDefaultModelsFromJsonFile)
    DEF_FUNC(Factory,addModel)
    DEF_FUNC(Factory,addModelsFromJson)
    DEF_FUNC(Factory,addModelsFromJsonFile)
    DEF_STATIC_FUNC(Factory,hasClass)
    DEF_STATIC_FUNC(Factory,hasDefaultModel)
    DEF_FUNC(Factory,hasModel)
    DEF_FUNC(Factory,getModelConfig,"baseName"_a,"modelName"_a,"withDefaultModels"_a=true)
    DEF_FUNC(Factory,getModelConfigs,"withDefaultModels"_a=true)
    .def("resolveModelConfig",py::overload_cast<const std::string&,const std::string&,const std::string&,const nl::json&,const FactoryHelperChain&,bool>(&Factory::resolveModelConfig,py::const_)
        ,"baseName"_a,"className"_a,"modelName"_a,"modelConfig"_a,"chain"_a,"withDefaultModels"_a=true
    )
    .def("resolveModelConfig",py::overload_cast<const std::string&,const std::string&,const FactoryHelperChain&,bool>(&Factory::resolveModelConfig,py::const_)
        ,"baseName"_a,"modelName"_a,"chain"_a,"withDefaultModels"_a=true
    )
    .def("resolveModelConfig",py::overload_cast<const std::string&,const nl::json&,const FactoryHelperChain&,bool>(&Factory::resolveModelConfig,py::const_)
        ,"baseName"_a,"j"_a,"chain"_a,"withDefaultModels"_a=true
    )
    .def("resolveModelConfig",py::overload_cast<const nl::json&,const FactoryHelperChain&,bool>(&Factory::resolveModelConfig,py::const_)
        ,"j"_a,"chain"_a,"withDefaultModels"_a=true
    )
    .def("resolveModelConfig",py::overload_cast<const std::string&,const std::string&,const std::string&,const nl::json&,bool>(&Factory::resolveModelConfig,py::const_)
        ,"baseName"_a,"className"_a,"modelName"_a,"modelConfig"_a,"withDefaultModels"_a=true
    )
    .def("resolveModelConfig",py::overload_cast<const std::string&,const std::string&,bool>(&Factory::resolveModelConfig,py::const_)
        ,"baseName"_a,"modelName"_a,"withDefaultModels"_a=true
    )
    .def("resolveModelConfig",py::overload_cast<const std::string&,const nl::json&,bool>(&Factory::resolveModelConfig,py::const_)
        ,"baseName"_a,"j"_a,"withDefaultModels"_a=true
    )
    .def("resolveModelConfig",py::overload_cast<const nl::json&,bool>(&Factory::resolveModelConfig,py::const_)
        ,"j"_a,"withDefaultModels"_a=true
    )
    DEF_FUNC(Factory,reconfigureModelConfigs)
    .def_static("issubclassof",[](py::object baseCls,const std::string& derivedBaseName,const std::string& derivedClassName){
        return Factory::issubclassof(baseCls,derivedBaseName,derivedClassName);
    })
    DEF_STATIC_FUNC(Factory,dummyCreationTest)
    .def("create",&Factory::create<Entity, const std::string&, const std::string&, const std::string&, const nl::json&, const nl::json&>)
    .def("create",&Factory::create<Entity, const std::string&, const std::string&, const std::string&, const nl::json&, const nl::json&, const FactoryHelperChain&>)
    .def("create",&Factory::create<Entity, const std::string&, const std::string&, const std::string&, const nl::json&, const nl::json&, const std::shared_ptr<const Entity>&>)
    .def("create",&Factory::create<Entity, const std::string&, const std::string&, const nl::json&>)
    .def("create",&Factory::create<Entity, const std::string&, const std::string&, const nl::json&, const FactoryHelperChain&>)
    .def("create",&Factory::create<Entity, const std::string&, const std::string&, const nl::json&, const std::shared_ptr<const Entity>&>)
    .def("createByClassName",&Factory::createByClassName<Entity, const std::string&, const std::string&, const nl::json&, const nl::json&>)
    .def("createByClassName",&Factory::createByClassName<Entity, const std::string&, const std::string&, const nl::json&, const nl::json&, const FactoryHelperChain&>)
    .def("createByClassName",&Factory::createByClassName<Entity, const std::string&, const std::string&, const nl::json&, const nl::json&, const std::shared_ptr<const Entity>&>)
    .def("create",&Factory::create<Entity, const nl::json&>)
    .def("create",&Factory::create<Entity, const nl::json&, const FactoryHelperChain&>)
    .def("create",&Factory::create<Entity, const nl::json&, const std::shared_ptr<const Entity>&>)
    DEF_STATIC_FUNC(Factory,addBaseName)
    DEF_STATIC_FUNC(Factory,hasBaseName)
    DEF_STATIC_FUNC(Factory,checkPermissionForCreate)
    DEF_STATIC_FUNC(Factory,checkPermissionForGet)
    ;

    expose_common_class<FactoryHelper>(m,"FactoryHelper")
    .def(py_init<const std::string&>())
    DEF_FUNC(FactoryHelper,addDependency)
    DEF_FUNC(FactoryHelper,createSubHelper)
    DEF_FUNC(FactoryHelper,addPythonClass)
    DEF_FUNC(FactoryHelper,addDefaultModel)
    DEF_FUNC(FactoryHelper,addDefaultModelsFromJson)
    DEF_FUNC(FactoryHelper,addDefaultModelsFromJsonFile)
    DEF_FUNC(FactoryHelper,resolveClassName)
    DEF_FUNC(FactoryHelper,resolveModelName,"baseName"_a,"modelName"_a,"factory"_a=nullptr)
    DEF_FUNC(FactoryHelper,validate)
    DEF_FUNC(FactoryHelper,addAliasPackageName)
    .def("getPackagePath",[](FactoryHelper& v){return v.getPackagePath().generic_string();})
    ;

    py::class_<FactoryHelperChain>(m,"FactoryHelperChain")
    .def(py::init([](py::args& args){
        return FactoryHelperChain(args.cast<py::tuple>());
    }))
    DEF_FUNC(FactoryHelperChain,resolveClassName)
    DEF_FUNC(FactoryHelperChain,resolveModelName,"baseName"_a,"modelName"_a,"factory"_a=nullptr)
    .def("resolveFilePath",[](FactoryHelperChain& v,const std::string& path){return v.resolveFilePath(path).generic_string();})
    .def("save",&::asrc::core::util::save_func_exposed_to_python<FactoryHelperChain>)
    .def("load",&::asrc::core::util::load_func_exposed_to_python<FactoryHelperChain>)
    .def("static_load",&::asrc::core::util::static_load_func_exposed_to_python<FactoryHelperChain>)
    .def_property_readonly_static("_allow_cereal_serialization_in_cpp",[](py::object /* self */){return true;})
    ;

    auto atexit=py::module_::import("atexit");
    atexit.attr("register")(py::cpp_function([](){
        Factory::clear();
    }));
}


ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
