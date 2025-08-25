/**
 * Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
 * @file
 * 
 * @brief Asset のうち、行動判断を行う主体としてシミュレータ外部とObservation,Actionを入出力するものを表すクラス
 */
#pragma once
#include <map>
#include <pybind11/pybind11.h>
#include "Asset.h"
#include "Utility.h"
#include "SimulationManager.h"

namespace py=pybind11;
namespace nl=nlohmann;

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)

class PhysicalAssetAccessor;

/**
 * @class Agent
 * @brief Asset のうち、行動判断を行う主体としてシミュレータ外部とObservation,Actionを入出力するものを表すクラス
 * 
 * @details 必ず一つ以上の PhysicalAsset に従属してそれらの「子」として生成される
 */
ASRC_DECLARE_DERIVED_CLASS_WITH_TRAMPOLINE(Agent,Asset)
    friend class SimulationManager;
    public:
    static const std::string baseName; // baseNameを上書き
    std::shared_ptr<SimulationManagerAccessorForAgent> manager; //!< SimulationManager の Accessor
    std::string name,type,model,policy;
    std::map<std::string,std::shared_ptr<PhysicalAssetAccessor>> parents; //!< 親となる PhysicalAsset
    bool dontExpose; //!< trueにするとSimulationManagerのreset,stepの返り値から除かれる。 makeObs と deploy も呼ばれなくなるので、それらが不要な Agent に限りtrueとしてよい。
    protected:
    std::shared_ptr<CoordinateReferenceSystem> localCRS; //!< 自身の行動判断の基準とするローカル座標系。直接ではなく getLocalCRS() で参照することを想定。
    public:
    //constructors & destructor
    using BaseType::BaseType;
    //functions
    virtual void initialize() override;
    virtual void serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full) override;
    virtual bool isAlive() const override;
    virtual std::string getTeam() const override;
    virtual std::string getGroup() const override;
    virtual std::string getName() const override;
    virtual std::uint64_t getStepCount() const; //!< これまでに行動判断を行った回数を返す。
    virtual std::uint64_t getStepInterval() const; //!< 行動判断周期(tick単位)を返す。
    virtual std::string repr() const; //!< [Deprecated] 自身の素性を表現する文字列を返す。
    void setDependency() override; //!< disable for Agent. Agent's dependency should be set by the parent PhysicalAsset.
    virtual py::object observation_space(); //!< 観測空間を返す。
    virtual py::object makeObs(); //!< parents の observables 等から Observationを生成して返す。
    virtual py::object action_space(); //!< 行動空間を返す。
    virtual void deploy(py::object action); //!< SimulationManager::step に供給された Action を受け取り parents の commands 等に反映する。
    virtual void perceive(bool inReset) override;
    virtual void control() override;
    virtual void behave() override;
    /**
     * @brief 他の Agent が出力した行動を自身の行動空間における Action に変換して返す。
     * 
     * @sa [模倣学習のための機能](\ref page_training_imitation)
     */
    virtual py::object convertActionFromAnother(const nl::json& decision,const nl::json& command);
    /**
     * @brief 他の Agent が出力した行動を参照しつつ、 自身の command() に相当する処理を行う。
     * 
     * @sa [模倣学習のための機能](\ref page_training_imitation)
     */
    virtual void controlWithAnotherAgent(const nl::json& decision,const nl::json& command);
    virtual std::shared_ptr<CoordinateReferenceSystem> getLocalCRS(); //!< 自身のローカル座標系を返す。初回呼び出し時にメンバ変数localCRSの値を設定する。デフォルトでは rulerの localCRSを使用する。 派生クラスでカスタマイズしてよい。
    protected:
    virtual void setManagerAccessor(const std::shared_ptr<EntityManagerAccessor>& ema) override;
};

ASRC_DECLARE_DERIVED_TRAMPOLINE(Agent)
    virtual bool isAlive() const override{
        PYBIND11_OVERRIDE(bool,Base,isAlive);
    }
    virtual std::string getTeam() const override{
        PYBIND11_OVERRIDE(std::string,Base,getTeam);
    }
    virtual std::string getGroup() const override{
        PYBIND11_OVERRIDE(std::string,Base,getGroup);
    }
    virtual std::string getName() const override{
        PYBIND11_OVERRIDE(std::string,Base,getName);
    }
    virtual std::uint64_t getStepCount() const override{
        PYBIND11_OVERRIDE(std::uint64_t,Base,getStepCount);
    }
    virtual std::uint64_t getStepInterval() const override{
        PYBIND11_OVERRIDE(std::uint64_t,Base,getStepInterval);
    }
    virtual std::string repr() const override{
        PYBIND11_OVERRIDE_NAME(std::string,Base,"__repr__",repr);
    }
    virtual py::object observation_space() override{
        PYBIND11_OVERRIDE(py::object,Base,observation_space);
    }
    virtual py::object makeObs() override{
        PYBIND11_OVERRIDE(py::object,Base,makeObs);
    }
    virtual py::object action_space() override{
        PYBIND11_OVERRIDE(py::object,Base,action_space);
    }
    virtual void deploy(py::object action) override{
        PYBIND11_OVERRIDE(void,Base,deploy,action);
    }
    virtual py::object convertActionFromAnother(const nl::json& decision,const nl::json& command) override{
        PYBIND11_OVERRIDE(py::object,Base,convertActionFromAnother,decision,command);
    }
    virtual void controlWithAnotherAgent(const nl::json& decision,const nl::json& command) override{
        PYBIND11_OVERRIDE(void,Base,controlWithAnotherAgent,decision,command);
    }
    virtual std::shared_ptr<CoordinateReferenceSystem> getLocalCRS() override{
        PYBIND11_OVERRIDE(std::shared_ptr<CoordinateReferenceSystem>,Base,getLocalCRS);
    }
};

/**
 * @class ExpertWrapper
 * @brief expert(模倣される側)と imitator(模倣する側)の2つの Agent インスタンスを保持し、模倣学習のために Observation, Actionの適切な操作を行うクラス。
 * 
 * @sa [模倣学習のための機能](\ref page_training_imitation)
 */
ASRC_DECLARE_DERIVED_CLASS_WITH_TRAMPOLINE(ExpertWrapper,Agent)
    public:
    std::string imitatorModelName; //!< 模倣する側のFactory におけるモデル名を表す文字列
    std::string expertModelName; //!< 模倣される側のFactory におけるモデル名を表す文字列
    std::string expertPolicyName; //!< 模倣される側のポリシー名を表す文字列。省略時は"Internal"となる。
    std::string trajectoryIdentifier; //!< 模倣用の軌跡データを出力する際の識別名。省略時は imitatorModelName と同一となる。
    std::string whichOutput;//親Assetに対してImitatorとExpertのどちらの出力を見せるか。observablesは現状expertで固定
    std::string whichExpose;//外部に対してImitatorとExpertのどちらのobservation,spaceを見せるか(指定しなければ両方)
    std::shared_ptr<Agent> imitator; //!< 模倣する側の Agent
    std::shared_ptr<Agent> expert; //!< 模倣される側の Agent
    py::object expertObs; //!< 模倣される側の直近のObservation
    py::object imitatorObs; //!< 模倣する側の直近のObservation
    py::object imitatorAction; //!< 模倣する側の直近のAction
    bool isInternal,hasImitatorDeployed;
    //constructors & destructor
    using BaseType::BaseType;
    //functions
    virtual void initialize() override;
    virtual void serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full) override;
    virtual void makeChildren() override;
    virtual void validate() override;
    virtual std::uint64_t getFirstTick(const SimPhase& phase) const override;
    virtual std::uint64_t getInterval(const SimPhase& phase) const override;
    virtual std::uint64_t getNextTick(const SimPhase& phase,const std::uint64_t now) override;
    virtual std::uint64_t getStepCount() const override;
    virtual std::uint64_t getStepInterval() const override;
    virtual std::string repr() const override;
    virtual py::object observation_space() override;
    virtual py::object makeObs() override;
    virtual py::object action_space() override;
    virtual py::object expert_observation_space(); //!< 模倣される側の観測空間を返す。
    virtual py::object expert_action_space(); //!< 模倣される側の行動空間を返す。
    virtual py::object imitator_observation_space(); //!< 模倣する側の行動空間を返す。
    virtual py::object imitator_action_space(); //!< 模倣する側の行動空間を返す。
    virtual void deploy(py::object action) override;
    virtual void perceive(bool inReset) override;
    virtual void control() override;
    virtual void behave() override;
};
ASRC_DECLARE_DERIVED_TRAMPOLINE(ExpertWrapper)
    //virtual functions
    virtual py::object expert_observation_space(){
        PYBIND11_OVERRIDE(py::object,Base,expert_observation_space);
    }
    virtual py::object expert_action_space(){
        PYBIND11_OVERRIDE(py::object,Base,expert_action_space);
    }
    virtual py::object imitator_observation_space(){
        PYBIND11_OVERRIDE(py::object,Base,imitator_observation_space);
    }
    virtual py::object imitator_action_space(){
        PYBIND11_OVERRIDE(py::object,Base,imitator_action_space);
    }
};

/**
 * @class MultiPortCombiner
 * @brief 複数の Agent インスタンスを組み合わせて一つの Agent として扱うためのベースクラス
 * 
 * @details
 *      一つにまとめた後のObservationとActionはユーザーが定義し、
 *      派生クラスにおいてmakeObs,actionSplitter,observation_space,action_spaceの4つをオーバーライドする必要がある。
 */
ASRC_DECLARE_DERIVED_CLASS_WITH_TRAMPOLINE(MultiPortCombiner,Agent)
    public:
    std::map<std::string,std::map<std::string,std::string>> ports;
    std::map<std::string,std::shared_ptr<Agent>> children;
    //constructors & destructor
    using BaseType::BaseType;
    //functions
    virtual void serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full) override;
    virtual void makeChildren() override;
    virtual void validate() override;
    virtual std::uint64_t getFirstTick(const SimPhase& phase) const override;
    virtual std::uint64_t getInterval(const SimPhase& phase) const override;
    virtual std::uint64_t getNextTick(const SimPhase& phase,const std::uint64_t now) override;
    virtual std::uint64_t getStepCount() const override;
    virtual std::uint64_t getStepInterval() const override;
    virtual py::object observation_space() override;
    virtual py::object makeObs() override;
    virtual py::object action_space() override;
    virtual std::map<std::string,py::object> actionSplitter(py::object action);
    virtual void deploy(py::object action) override;
    virtual void perceive(bool inReset) override;
    virtual void control() override;
    virtual void behave() override;
};
ASRC_DECLARE_DERIVED_TRAMPOLINE(MultiPortCombiner)
    virtual std::map<std::string,py::object> actionSplitter(py::object action){
        typedef std::map<std::string,py::object> retType;
        PYBIND11_OVERRIDE(retType,Base,actionSplitter,action);
    }
};
/**
 * @class SimpleMultiPortCombiner
 * @brief 複数の Agent インスタンスを組み合わせて一つの Agent として扱うための最もシンプルなクラス
 * 
 * @details ObservationとActionはchildrenのそれぞれの要素をdictに格納したものとする。
 */
ASRC_DECLARE_DERIVED_CLASS_WITHOUT_TRAMPOLINE(SimpleMultiPortCombiner,MultiPortCombiner)
    public:
    std::map<std::string,py::object> lastChildrenObservations;
    //constructors & destructor
    using BaseType::BaseType;
    //functions
    virtual void serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full) override;
    virtual py::object observation_space() override;
    virtual py::object makeObs() override;
    virtual py::object action_space() override;
    virtual std::map<std::string,py::object> actionSplitter(py::object action) override;
    virtual py::object convertActionFromAnother(const nl::json& decision,const nl::json& command) override;
    virtual void controlWithAnotherAgent(const nl::json& decision,const nl::json& command) override;
};

ASRC_DECLARE_DERIVED_CLASS_WITHOUT_TRAMPOLINE(SingleAssetAgent,Agent)
    public:
    std::shared_ptr<PhysicalAssetAccessor> parent;
    std::string port;
    public:
    //constructors & destructor
    using BaseType::BaseType;
    //functions
    virtual void initialize() override;
    virtual void serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full) override;
};

void exportAgent(py::module &m, const std::shared_ptr<asrc::core::FactoryHelper>& factoryHelper);

ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
