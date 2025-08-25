/**
 * Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
 * @file
 * 
 * @brief シミュレーション中に周期的に呼び出される処理を記述してシミュレーションの流れを制御するクラス
 */
#pragma once
#include <mutex>
#include <pybind11/pybind11.h>
#include "Entity.h"
#include "SimulationManager.h"

namespace py=pybind11;
namespace nl=nlohmann;

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)

/**
 * @class Callback
 * @brief シミュレーション中に周期的に呼び出される処理を記述してシミュレーションの流れを制御するクラス
 * 
 * @details SimulationManager によって生成され、エピソードを跨いで存在し続ける。
 */
ASRC_DECLARE_DERIVED_CLASS_WITH_TRAMPOLINE(Callback,Entity)
    friend class SimulationManager;
    public:
    std::mutex mtx;
    std::shared_ptr<SimulationManagerAccessorForCallback> manager;
    bool acceptReconfigure; //!< SimulationManager のreconfigure時に自身の再生成を受け入れるかどうか。
    std::string group;
    std::string name;
    public:
    static const std::string baseName; // baseNameを上書き
    //constructors & destructor
    using BaseType::BaseType;
    //functions
    virtual void initialize() override;
    virtual void serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full) override;
    virtual std::string getGroup() const;
    virtual std::string getName() const;
    virtual void setDependency(); //!< 他の Callback との処理順序の依存関係を設定する。
    virtual void onGetObservationSpace(); //!< [get_observation_space](\ref SimulationManager::get_observation_space)において戻り値を返す直前
    virtual void onGetActionSpace(); //!< [get_action_space](\ref SimulationManager::get_action_space) において戻り値を返す直前
    virtual void onMakeObs(); //!< [step](\ref SimulationManager::step)関数の戻り値として Observation を生成する直前に呼ばれる関数
    virtual void onDeployAction(); //!< [step](\ref SimulationManager::step)関数の開始時に、外部から与えられた Action を各 Agent に配分する前に呼ばれる関数
    virtual void onEpisodeBegin(); //!< 各エピソードの開始時(＝reset 関数の最後)に呼ばれる関数
    virtual void onValidationEnd(); //!< 各エピソードの[validate](\ref Entity::validate)終了時に呼ばれる関数
    virtual void onStepBegin(); //!< [step](\ref SimulationManager::step)関数の開始時に、外部から与えられた Action を Agent が受け取った直後に呼ばれる関数
    virtual void onInnerStepBegin(); //!< 各 tick の開始時(＝ Asset の [control](\ref Asset::control) の前)に呼ばれる関数
    virtual void onInnerStepEnd(); //!< 各 tick の終了時(＝ Asset の [perceive](\ref Asset::perceive) の後)に呼ばれる関数
    virtual void onStepEnd(); //!< [step](\ref SimulationManager::step)関数の終了時(＝[step](\ref SimulationManager::step)関数の戻り値生成の前または後)に呼ばれる関数
    virtual void onEpisodeEnd(); //!< 各エピソードの終了時([step](\ref SimulationManager::step)関数の戻り値生成後)に呼ばれる関数
    protected:
    virtual void setManagerAccessor(const std::shared_ptr<EntityManagerAccessor>& ema) override;
};

ASRC_DECLARE_DERIVED_TRAMPOLINE(Callback)
    virtual std::string getGroup() const override{
        PYBIND11_OVERRIDE(std::string,Base,getGroup);
    }
    virtual std::string getName() const override{
        PYBIND11_OVERRIDE(std::string,Base,getName);
    }
    virtual void setDependency() override{
        PYBIND11_OVERRIDE(void,Base,setDependency);
    }
    virtual void onGetObservationSpace() override{
        PYBIND11_OVERRIDE(void,Base,onGetObservationSpace);
    }
    virtual void onGetActionSpace() override{
        PYBIND11_OVERRIDE(void,Base,onGetActionSpace);
    }
    virtual void onMakeObs() override{
        PYBIND11_OVERRIDE(void,Base,onMakeObs);
    }
    virtual void onDeployAction() override{
        PYBIND11_OVERRIDE(void,Base,onDeployAction);
    }
    virtual void onEpisodeBegin() override{
        PYBIND11_OVERRIDE(void,Base,onEpisodeBegin);
    }
    virtual void onValidationEnd() override{
        PYBIND11_OVERRIDE(void,Base,onValidationEnd);
    }
    virtual void onStepBegin() override{
        PYBIND11_OVERRIDE(void,Base,onStepBegin);
    }
    virtual void onInnerStepBegin() override{
        PYBIND11_OVERRIDE(void,Base,onInnerStepBegin);
    }
    virtual void onInnerStepEnd() override{
        PYBIND11_OVERRIDE(void,Base,onInnerStepEnd);
    }
    virtual void onStepEnd() override{
        PYBIND11_OVERRIDE(void,Base,onStepEnd);
    }
    virtual void onEpisodeEnd() override{
        PYBIND11_OVERRIDE(void,Base,onEpisodeEnd);
    }
};

void exportCallback(py::module &m, const std::shared_ptr<asrc::core::FactoryHelper>& factoryHelper);

ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
