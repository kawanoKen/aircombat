/**
 * Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
 * @file
 * 
 * @brief エピソードにおける各 Agent の報酬の計算を行うためのクラス
 */
#pragma once
#include <vector>
#include <map>
#include <pybind11/pybind11.h>
#include "Callback.h"

namespace py=pybind11;
namespace nl=nlohmann;

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)

class Agent;
class Asset;

/**
 * @class Reward
 * @brief エピソードにおける各 Agent の報酬の計算を行うためのクラス
 * 
 * @details 単一のエピソード中に複数存在させることができる。
 */
ASRC_DECLARE_DERIVED_CLASS_WITH_TRAMPOLINE(Reward,Callback)
    public:
    static const std::string baseName; // baseNameを上書き
    nl::json j_target;
    std::vector<std::string> target; //!< 報酬計算対象となっている陣営の名前または Agent のfull nameのリスト
    std::map<std::string,double> reward; //!< 前回のstep以降に得た報酬
    std::map<std::string,double> totalReward; //!< エピソード開始時からの累積報酬
    bool isTeamReward;
    //constructors & destructor
    using BaseType::BaseType;
    //functions
    virtual void initialize() override;
    virtual void serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full) override;
    virtual void onEpisodeBegin() override;
    /**
     * @attention 派生クラスでは最初に必ずこのクラスのonStepBeginを呼び出すか、同等の処理を記述すること。
     */
    virtual void onStepBegin() override;
    /**
     * @attention 派生クラスでは最後に必ずこのクラスのonStepEndを呼び出すか、同等の処理を記述すること。
     */
    virtual void onStepEnd() override;
    virtual void setupTarget(const std::string& query);
    virtual double getReward(const std::string &key); //!< key をfull nameとする Agent が前回のstep以降に得た報酬を返す。
    virtual double getTotalReward(const std::string &key); //!< key をfull nameとする Agent のエピソード開始時からの累積報酬を返す。
    virtual double getReward(const std::shared_ptr<Agent>& key); //!< key が前回のstep以降に得た報酬を返す。
    virtual double getTotalReward(const std::shared_ptr<Agent>& key); //!< key のエピソード開始時からの累積報酬を返す。
};
ASRC_DECLARE_DERIVED_TRAMPOLINE(Reward)
    //virtual functions
    virtual void setupTarget(const std::string& query) override{
        PYBIND11_OVERRIDE(void,Base,setupTarget,query);
    }
    virtual double getReward(const std::string &key) override{
        PYBIND11_OVERRIDE(double,Base,getReward,key);
    }
    virtual double getTotalReward(const std::string &key) override{
        PYBIND11_OVERRIDE(double,Base,getTotalReward,key);
    }
    virtual double getReward(const std::shared_ptr<Agent>& key) override{
        PYBIND11_OVERRIDE(double,Base,getReward,key);
    }
    virtual double getTotalReward(const std::shared_ptr<Agent>& key) override{
        PYBIND11_OVERRIDE(double,Base,getTotalReward,key);
    }
};

/**
 * @class AgentReward
 * @brief 各 Agent ごとの個別報酬の計算を行うためのクラス
 */
ASRC_DECLARE_DERIVED_CLASS_WITHOUT_TRAMPOLINE(AgentReward,Reward)
    public:
    //constructors & destructor
    using BaseType::BaseType;
    //functions
    virtual void initialize() override;
};

/**
 * @class TeamReward
 * @brief 陣営全体の共通報酬の計算を行うためのクラス
 */
ASRC_DECLARE_DERIVED_CLASS_WITHOUT_TRAMPOLINE(TeamReward,Reward)
    public:
    //constructors & destructor
    using BaseType::BaseType;
    //functions
    virtual void initialize() override;
};

/**
 * @class ScoreReward
 * @brief Ruler の score と連動する報酬の計算を行うためのクラス
 */
ASRC_DECLARE_DERIVED_CLASS_WITHOUT_TRAMPOLINE(ScoreReward,TeamReward)
    public:
    //constructors & destructor
    using BaseType::BaseType;
    //functions
    virtual void onStepEnd() override;
};
void exportReward(py::module &m, const std::shared_ptr<asrc::core::FactoryHelper>& factoryHelper);

ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
