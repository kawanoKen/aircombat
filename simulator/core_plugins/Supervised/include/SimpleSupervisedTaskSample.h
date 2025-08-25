// Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#pragma once
#include <iostream>
#include <vector>
#include <map>
#include <pybind11/pybind11.h>
#include <Eigen/Core>
#include <ASRCAISim1/MathUtility.h>
#include <ASRCAISim1/Utility.h>
#include <ASRCAISim1/Reward.h>
#include "SupervisedTaskMixer.h"
namespace py=pybind11;
namespace nl=nlohmann;

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)

ASRC_DECLARE_DERIVED_CLASS_WITH_TRAMPOLINE(SimpleSupervisedTaskSample01,SupervisedTaskMixer)
    /*教師あり学習用データセットをobservationに追加するサンプル。
    このサンプルクラスでは最も簡単な例の一つとして戦闘とは無関係に、
    「整数一つを入力してその整数が3の倍数か否かを当てる分類問題」をobservationに追加する。
    */
    public:
    //parameters
    int lower,upper;
    //internal variables
    std::map<std::string,int> inputs;//agentごとのinput
    //constructors & destructor
    using BaseType::BaseType;
    //functions
    virtual void initialize() override;
    virtual void serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full) override;
    virtual void onEpisodeEnd() override;
    virtual py::object calcTaskInputSpace(const std::shared_ptr<Agent>& agent) override;
    virtual py::object calcTaskTruthSpace(const std::shared_ptr<Agent>& agent) override;
    virtual py::object calcTaskInput(const std::shared_ptr<Agent>& agent) override;
    virtual std::pair<bool,py::object> calcTaskTruth(const std::shared_ptr<Agent>& agent) override;
};
ASRC_DECLARE_DERIVED_TRAMPOLINE(SimpleSupervisedTaskSample01)
    virtual py::object calcTaskInputSpace(const std::shared_ptr<Agent>& agent) override{
        PYBIND11_OVERRIDE(py::object,Base,calcTaskInputSpace,agent);
    }
    virtual py::object calcTaskInput(const std::shared_ptr<Agent>& agent) override{
        PYBIND11_OVERRIDE(py::object,Base,calcTaskInput,agent);
    }
    virtual py::object calcTaskTruthSpace(const std::shared_ptr<Agent>& agent) override{
        PYBIND11_OVERRIDE(py::object,Base,calcTaskTruthSpace,agent);
    }
    virtual std::pair<bool,py::object> calcTaskTruth(const std::shared_ptr<Agent>& agent) override{
        typedef std::pair<bool,py::object> retType;
        PYBIND11_OVERRIDE(retType,Base,calcTaskTruth,agent);
    }
};

ASRC_DECLARE_DERIVED_CLASS_WITH_TRAMPOLINE(SimpleSupervisedTaskSample02,SupervisedTaskMixer)
    /*教師あり学習用データセットをobservationに追加するサンプル。
    このサンプルクラスでは最も簡単な例の一つとして戦闘とは無関係に、
    「実数二つを入力してその積を出力する回帰問題」をobservationに追加する。
    */
    public:
    //parameters
    float lower,upper;
    //internal variables
    std::map<std::string,Eigen::Vector2f> inputs;//agentごとのinput
    //constructors & destructor
    using BaseType::BaseType;
    //functions
    virtual void initialize() override;
    virtual void serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full) override;
    virtual void onEpisodeEnd() override;
    virtual py::object calcTaskInputSpace(const std::shared_ptr<Agent>& agent) override;
    virtual py::object calcTaskTruthSpace(const std::shared_ptr<Agent>& agent) override;
    virtual py::object calcTaskInput(const std::shared_ptr<Agent>& agent) override;
    virtual std::pair<bool,py::object> calcTaskTruth(const std::shared_ptr<Agent>& agent) override;
};
ASRC_DECLARE_DERIVED_TRAMPOLINE(SimpleSupervisedTaskSample02)
    virtual py::object calcTaskInputSpace(const std::shared_ptr<Agent>& agent) override{
        PYBIND11_OVERRIDE(py::object,Base,calcTaskInputSpace,agent);
    }
    virtual py::object calcTaskInput(const std::shared_ptr<Agent>& agent) override{
        PYBIND11_OVERRIDE(py::object,Base,calcTaskInput,agent);
    }
    virtual py::object calcTaskTruthSpace(const std::shared_ptr<Agent>& agent) override{
        PYBIND11_OVERRIDE(py::object,Base,calcTaskTruthSpace,agent);
    }
    virtual std::pair<bool,py::object> calcTaskTruth(const std::shared_ptr<Agent>& agent) override{
        typedef std::pair<bool,py::object> retType;
        PYBIND11_OVERRIDE(retType,Base,calcTaskTruth,agent);
    }
};

void exportSimpleSupervisedTaskSample(py::module &m, const std::shared_ptr<asrc::core::FactoryHelper>& factoryHelper);

ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
