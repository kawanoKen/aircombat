/**
 * Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
 * @file
 * 
 * @brief シミュレーション登場物の生成処理をjsonで記述しやすくするための機能を提供する、ConfigDispatcherクラス
 */
#pragma once
#include "Common.h"
#include <iostream>
#include <random>
#include <memory>
#include <pybind11/pybind11.h>
#include <nlohmann/json.hpp>

namespace py=pybind11;
namespace nl=nlohmann;

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)
ASRC_INLINE_NAMESPACE_BEGIN(util)

class ConfigDispatcher;
/**
 * @internal
 * @class ConfigDispatcherElement
 * @brief [For internal use] ConfigDispatcherの構成要素を表すクラス
 */
class PYBIND11_EXPORT ConfigDispatcherElement{
    nl::json config;
    public:
    std::map<std::string,nl::json> instances;
    ConfigDispatcherElement(const nl::json& j);
    ~ConfigDispatcherElement();
    void clear();
    bool isBuiltinType(const std::string& type);
    nl::json get(ConfigDispatcher& dispatcher,const std::string& instance="",const int& index=-1);
};

/**
 * @class ConfigDispatcher
 * @brief シミュレーション登場物の生成処理をjsonで記述しやすくするための機能を提供するクラス
 */
class PYBIND11_EXPORT ConfigDispatcher{
    friend class ConfigDispatcherElement;
    std::mt19937 randomGen;
    nl::json config;
    nl::json get(const nl::json& query);
    nl::json sanitize(const nl::json& src);
    public:
    std::map<std::string,std::shared_ptr<ConfigDispatcherElement>> aliases;
    ConfigDispatcher();
    ConfigDispatcher(const nl::json& j);
    ~ConfigDispatcher();
    void clear();
    void reset();
    void initialize(const nl::json& j);
    void seed(const unsigned int& seed_);
    nl::json run(const nl::json& query);
};

class PYBIND11_EXPORT RecursiveJsonExtractor{
    public:
    static std::map<std::string,nl::json> run(const nl::json& root,std::function<bool(const nl::json&)> checker);
    static std::map<std::string,nl::json> sub(const nl::json& node,const std::string& path,std::function<bool(const nl::json&)> checker);
};

ASRC_NAMESPACE_END(util)
ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
