// Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#pragma once
#include <map>
#include <pybind11/pybind11.h>
#include <ASRCAISim1/Utility.h>
#include <ASRCAISim1/Agent.h>
#include <ASRCAISim1/ObservationModifierBase.h>
namespace py=pybind11;
namespace nl=nlohmann;

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)

ASRC_DECLARE_DERIVED_CLASS_WITH_TRAMPOLINE(UnsupervisedTaskMixer,ObservationModifierBase)
    /*教師なし学習用データセットをobservationに追加するための基底クラス。
    */
    public:
    //parameters
    std::string taskName;//タスク名
    std::map<std::string,std::vector<std::string>> targetAgentModels;//対象とするAgentのモデル名。陣営ごとに指定するDict[team,List[agentModelName]]の形で指定する。
    //constructors & destructor
    using BaseType::BaseType;
    //functions
    virtual void initialize() override;
    virtual void serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full) override;
    virtual py::object calcTaskInputSpace(const std::shared_ptr<Agent>& agent)=0;
    virtual py::object calcTaskInput(const std::shared_ptr<Agent>& agent)=0;
    virtual py::object overrideObservationSpaceImpl(const std::shared_ptr<Agent>& agent,py::object oldSpace) override;
    virtual py::object overrideObservationImpl(const std::shared_ptr<Agent>& agent,py::object oldObs) override;
    virtual bool isTarget(const std::shared_ptr<Agent>& agent) override;
};
ASRC_DECLARE_DERIVED_TRAMPOLINE(UnsupervisedTaskMixer)
    virtual py::object calcTaskInputSpace(const std::shared_ptr<Agent>& agent) override{
        PYBIND11_OVERRIDE_PURE(py::object,Base,calcTaskInputSpace,agent);
    }
    virtual py::object calcTaskInput(const std::shared_ptr<Agent>& agent) override{
        PYBIND11_OVERRIDE_PURE(py::object,Base,calcTaskInput,agent);
    }
    virtual py::object overrideObservationSpaceImpl(const std::shared_ptr<Agent>& agent,py::object oldSpace) override{
        PYBIND11_OVERRIDE(py::object,Base,overrideObservationSpaceImpl,agent,oldSpace);
    }
    virtual py::object overrideObservationImpl(const std::shared_ptr<Agent>& agent,py::object oldObs) override{
        PYBIND11_OVERRIDE(py::object,Base,overrideObservationImpl,agent,oldObs);
    }
    virtual bool isTarget(const std::shared_ptr<Agent>& agent) override{
        PYBIND11_OVERRIDE(bool,Base,isTarget,agent);
    }
};

ASRC_DECLARE_DERIVED_CLASS_WITH_TRAMPOLINE(SupervisedTaskMixer,UnsupervisedTaskMixer)
    /*教師あり学習用データセットをobservationに追加するための基底クラス。
    教師なし学習用のクラスに正解ラベルを追加する形で実装している。
    正解ラベルは必ずしも即時に計算できるとは限らないため、observationには含めずメンバ変数truthとして保持しておくこととし、
    エピソードの終了時までに各step、各agentの正解ラベルを計算しておくものとする。
    学習用のプログラムからこの変数を参照してデータセット化するものとする。
    */
    public:
    std::vector<std::map<std::string,py::object>> truths;//各stepの、各agentのtruthであり、例えばtruths[manager->getExposedStepCount()][agent->getFullName()]=valueのように使用する。
    std::map<std::string,py::object> truth_spaces;//各agentのtruthのspace
    //constructors & destructor
    using BaseType::BaseType;
    //functions
    virtual void initialize() override;
    virtual void serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full) override;
    virtual void onEpisodeBegin() override;
    virtual void onStepEnd() override;
    virtual py::object calcTaskTruthSpace(const std::shared_ptr<Agent>& agent)=0;
    virtual std::pair<bool,py::object> calcTaskTruth(const std::shared_ptr<Agent>& agent)=0;//即時計算できる場合のインターフェース
};
ASRC_DECLARE_DERIVED_TRAMPOLINE(SupervisedTaskMixer)
    virtual py::object calcTaskTruthSpace(const std::shared_ptr<Agent>& agent) override{
        PYBIND11_OVERRIDE_PURE(py::object,Base,calcTaskTruthSpace,agent);
    }
    virtual std::pair<bool,py::object> calcTaskTruth(const std::shared_ptr<Agent>& agent) override{
        typedef std::pair<bool,py::object> retType;
        PYBIND11_OVERRIDE_PURE(retType,Base,calcTaskTruth,agent);
    }
};

void exportSupervisedTaskMixer(py::module &m, const std::shared_ptr<asrc::core::FactoryHelper>& factoryHelper);

ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
