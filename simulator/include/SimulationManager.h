/**
 * Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
 * @file
 * 
 * @brief シミュレーションの実行を管理するSimulationManagerクラスとそのAccessorを定義する。
 */
#pragma once
#include "Common.h"
#include <iostream>
#include <memory>
#include <random>
#include <iterator>
#include <set>
#include <vector>
#include <map>
#include <functional>
#include <optional>
#include <variant>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <nlohmann/json.hpp>
#include <thread-pool/BS_thread_pool.hpp>
#include "Utility.h"
#include "TimeSystem.h"
#include "crs/CoordinateReferenceSystemBase.h"
#include "Factory.h"
#include "Entity.h"
#include "ConfigDispatcher.h"
#include "CommunicationBuffer.h"
#include "FilteredWeakIterable.h"
namespace py=pybind11;
namespace nl=nlohmann;

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)

class Asset;
class PhysicalAsset;
class PhysicalAssetAccessor;
class Agent;
class Controller;
class Callback;
class Ruler;
class RulerAccessor;
class Reward;
class Viewer;
class SimulationManagerAccessorBase;

/**
 * @brief シミュレーション実行中の処理を分類した列挙体
 */
enum class SimPhase{
    NONE,
    VALIDATE, //!< Asset::validate
    PERCEIVE, //!< Asset::perceive
    CONTROL, //!< Asset::control
    BEHAVE, //!< Asset::behave
    AGENT_STEP, //!< Agent::makeObs, Agent::deploy
    ON_GET_OBSERVATION_SPACE, //!< Callback::onGetObservationSpace
    ON_GET_ACTION_SPACE, //!< Callback::onGetActionSpace
    ON_MAKE_OBS, //!< Callback::onMakeObs
    ON_DEPLOY_ACTION, //!< Callback::onDeployAction
    ON_EPISODE_BEGIN, //!< Callback::onEpisodeBegin
    ON_VALIDATION_END, //!< Callback::onValidationEnd
    ON_STEP_BEGIN, //!< Callback::onStepBegin
    ON_INNERSTEP_BEGIN, //!< Callback::onInnerStepBegin
    ON_INNERSTEP_END, //!< Callback::onInnerStepEnd
    ON_STEP_END, //!< Callback::onStepEnd
    ON_EPISODE_END //!< Callback::onEpisodeEnd
};

DEFINE_SERIALIZE_ENUM_AS_STRING(SimPhase)
inline std::ostream& operator<<(std::ostream& os, const SimPhase& phase){
    os << magic_enum::enum_name(phase);
    return os;
}

struct PYBIND11_EXPORT OrderComparer{
    public:
    typedef std::pair<std::uint64_t,std::size_t> Type;
    bool operator()(const Type& lhs,const Type& rhs) const;
};

/**
 * @class SimulationManagerBase
 * @brief シミュレーション実行管理クラスの基底クラス。
 * 
 * @details 実行処理に依存しない共通部分をこのクラスで実装する。
 * 
 *      元バージョンでは時刻管理と座標系管理をこのクラスで実装する。
 */
ASRC_DECLARE_DERIVED_CLASS_WITH_TRAMPOLINE(SimulationManagerBase,EntityManager)
    protected:
    using BaseType::BaseType;
    /** @name 内部状態のシリアライゼーション (Experimental)
     */
    ///@{
    virtual void serializeBeforeEntityReconstructionInfo(asrc::core::util::AvailableArchiveTypes & archive, bool full, const nl::json& serializationConfig_) override;
    virtual void serializeAfterEntityReconstructionInfo(asrc::core::util::AvailableArchiveTypes & archive, bool full, const nl::json& serializationConfig_) override;
    ///@}

    /** @name 時刻管理
     */
    ///@{
    protected:
    Epoch epoch; //!< 基準時刻(シミュレーション開始時点の時刻)
    void setEpoch(const nl::json& j); //!< 基準時刻をjsonから読み込んで設定する。
    public:
    Epoch getEpoch() const; //!< 基準時刻の情報を返す。
    Time getTime() const; //!< 現在のシミュレーション時刻を返す。
    virtual double getElapsedTime() const =0; //!< シミュレーション開始時点からの経過時間(秒)を返す。
    ///@}

    /** @name 座標系管理
     */
    ///@{
    protected:
    std::shared_ptr<CoordinateReferenceSystem> rootCRS; //!< シミュレーション中の基準座標系。直交座標軸を持つ PureFlatCRS 、 ECEF 又は ECI のいずれかとする。
    std::map<EntityIdentifier, std::shared_ptr<CoordinateReferenceSystem>> crses; //!< このシミュレーション中に使用されているCRSインスタンスの一覧。
    nl::json crsConfig; //!< configで与えられたCRS構成情報
    void setupCRSes(const nl::json& j); //!< configで与えられたCRS構成情報からCRSを生成する。
    public:
    std::shared_ptr<CoordinateReferenceSystem> getRootCRS() const; //!< 基準座標系を返す。
    nl::json getCRSConfig() const; //!< configで与えられたCRS構成情報を返す。

    /**
     * @brief 全てのCRSをweak_ptr<T>としてイテレートするイテラブルを返す。
     */
    template<std::derived_from<CoordinateReferenceSystem> T=CoordinateReferenceSystem>
    auto getCRSes() const{
        return std::move(MapIterable<T,EntityIdentifier,CoordinateReferenceSystem>(crses));
    }

    /**
     * @brief 全てのCRSを matcher でフィルタリングしつつ、条件を満たすもののみをweak_ptr<T>としてイテレートするイテラブルを返す。
     */
    template<std::derived_from<CoordinateReferenceSystem> T=CoordinateReferenceSystem>
    auto getCRSes(typename MapIterable<T,EntityIdentifier,CoordinateReferenceSystem>::MatcherType matcher) const{
        return std::move(MapIterable<T,EntityIdentifier,CoordinateReferenceSystem>(crses,matcher));
    }

    /**
     * @brief 指定したfull nameを持つCRSをweak_ptr<T>にキャストして返す。
     */
    template<std::derived_from<CoordinateReferenceSystem> T=CoordinateReferenceSystem>
    std::weak_ptr<T> getCRS(const std::string& name_) const{
        try{
            auto entity=getEntityByName<Entity>(name_);
            if(!entity || crses.find(entity->getEntityID())==crses.end()){
                throw std::runtime_error("This SimulationManagerBase has no CRS named '"+name_+"'.");
            }
            auto ret=std::dynamic_pointer_cast<T>(entity);
            if(!ret){
                throw std::runtime_error("The CRS named '"+name_+"' is not an instance of '"+py::type_id<T>()+"'");
            }
            return ret;
        }catch(std::exception& e){
            std::cout<<"getCRS failure."<<std::endl;
            DEBUG_PRINT_EXPRESSION(name_)
            throw e;
        }
    }
    /**
     * @brief [For internal use] AffineCRSの状態を表すMotionStateを取得する。
     */
    std::vector<std::pair<EntityIdentifier,MotionState>> getInternalMotionStatesOfAffineCRSes() const;
    /**
     * @brief [For internal use] AffineCRSの状態を表すMotionStateを変更する。
     */
    void setInternalMotionStatesOfAffineCRSes(const std::vector<std::pair<EntityIdentifier,MotionState>>& states);
    ///@}
};
ASRC_DECLARE_DERIVED_TRAMPOLINE(SimulationManagerBase)
    virtual double getElapsedTime() const override{
        PYBIND11_OVERRIDE_PURE(double,Base,getElapsedTime);
    }
};

/**
 * @class SimulationManager
 * @brief gymnasium(OpenAI gym)のEnvとしてのインターフェースを持ったシミュレーション実行管理クラスの具象クラス。
 */
ASRC_DECLARE_DERIVED_CLASS_WITH_TRAMPOLINE(SimulationManager,SimulationManagerBase)
    friend class CommunicationBuffer;
    friend class SimulationManagerAccessorBase;
    friend class SimulationManagerAccessorForCallback;
    friend class SimulationManagerAccessorForPhysicalAsset;
    friend class SimulationManagerAccessorForAgent;
    typedef std::function<nl::json(const nl::json&,const std::int32_t&,const std::int32_t&)> ConfigOverriderType;
    protected:
    /**
     * @param [in] config_       コンフィグ。json object(dict)で与える。
     * @param [in] worker_index_ worker indexを指定する。プロセス中の全 EntityManager インスタンスに影響する。
     * @param [in] vector_index_ vector indexを指定する。
     * @param [in] overrider_    config_ の内容を worker_index_ と vector_index_ の内容に応じてオーバーライドする関数オブジェクトを与える。省略可。
     * 
     * @details
     *   config_ の形式は、  doc/src/format/SimulationManager_config.json のとおりであるが、
     *   必ずしも完全な形のjson objectを直接与える必要はなく、
     *   以下のように細分化したjsonファイルやjson objectを組み合わせて自動構成することができる。
     * 
     *   - ファイルパスで指定
     * 
     *       jsonファイルの中身をそのまま採用する。
     * 
     *   - json objectで指定
     * 
     *       json objectをそのまま採用する。
     * 
     *   - ファイルパス又はjson objectのリストで指定
     * 
     *       先頭要素から順にmerge patchでマージしたものを採用する。
     * 
     *   マージ後の完全な形のconfig_は以下のような形となる。
     * @include{strip} doc/src/format/SimulationManager_config.json
     */
    SimulationManager(const nl::json& config_,const std::int32_t& worker_index_=0,const std::int32_t& vector_index_=0,ConfigOverriderType overrider_=nullptr);
    //
    // コンフィグ管理
    //
    public:
    nl::json getManagerConfig() const; //!< 現在のManagerコンフィグを返す
    /**
     * @brief 自身が持つ Factoryコンフィグ(各baseNameに対応するmodel configの一覧)を返す。
     * 
     * @param [in] withDefaultModels trueの場合、 Factory の共有モデルも合わせて返す。falseの場合、自身の固有のモデルのみ。
     */
    nl::json getFactoryModelConfig(bool withDefaultModels=true) const;
    /**
     * @brief コンフィグの変更を予約する。
     * 
     * @param [in] fullConfigPatch_ "Manager"と"Factory"をキーに持つjson object
     * 
     * @details 次回エピソードの開始時にmerge patch処理によって反映される。
     */
    void requestReconfigure(const nl::json& fullConfigPatch_);
    /**
     * @brief コンフィグの変更を予約する。
     * 
     * @param [in] managerConfigPatch Managerコンフィグを更新するためのjson object
     * @param [in] factoryConfigPatch Factoryコンフィグを更新するためのjson object
     * 
     * @details 次回エピソードの開始時にmerge patch処理によって反映される。
     */
    void requestReconfigure(const nl::json& managerConfigPatch,const nl::json& factoryConfigPatch);
    /**
     * @brief jsonファイルパスやjson objectのリストから完全な形のconfigを構成する。
     * 
     * @details
     *   - ファイルパスで指定
     * 
     *       jsonファイルの中身をそのまま採用する。
     * 
     *   - json objectで指定
     * 
     *       json objectをそのまま採用する。
     * 
     *   - ファイルパス又はjson objectのリストで指定
     * 
     *       先頭要素から順にmerge patchでマージしたものを採用する。
     */
    static nl::json parseConfig(const nl::json& config_);
    void setViewerType(const std::string& viewerType); //!< [For internal use] Viewer モデルを変更する。
    virtual void configure(); //!< 現在のconfigに基づき、エピソード非依存部分(=Asset 以外)の初期化処理を行う。
    /**
     * @brief configをfullConfig_に置換(merge patchではない)したうえで configure() を呼び出す。
     */
    virtual void configure(const nl::json& fullConfig_); 
    /**
     * @brief ManagerコンフィグをmanagerConfig_に、FactoryコンフィグをfactoryConfig_に置換(merge patchではない)したうえで configure() を呼び出す。
     */
    virtual void configure(const nl::json& managerConfig_, const nl::json& factoryConfig_);
    /**
     * @brief requestReconfigure が呼ばれていた場合、その変更内容を反映して、 configure() を呼び出す。
     */
    virtual void reconfigureIfRequested();
    /**
     * @brief configをfullConfig_にmerge patchしたうえで configure() を呼び出す。
     */
    virtual void reconfigure(const nl::json& fullConfigPatch_);
    /**
     * @brief ManagerコンフィグをmanagerConfig_に、FactoryコンフィグをfactoryConfig_にmerge patchしたうえで configure() を呼び出す。
     */
    virtual void reconfigure(const nl::json& managerConfigPatch_, const nl::json& factoryConfigPatch_);
    protected:
    ConfigOverriderType configOverrider;
    bool isConfigured;
    nl::json managerConfig;
    bool isReconfigureRequested;
    nl::json managerConfigPatchForReconfigure,factoryConfigPatchForReconfigure;

    /** @name 実行状態管理
     */
    ///@{
    public:
    const std::set<SimPhase> assetPhases,agentPhases,callbackPhases;
    std::int32_t worker_index() const; //!< worker indexを返す。
    std::int32_t vector_index() const; //!< vector indexを返す。
    std::int32_t episode_index() const; //!< episode indexを返す。
    bool isEpisodeActive; //!< エピソードの実行途中かどうか
    int numThreads; //!< スレッド数(基本的には1を推奨)
    bool measureTime; //!< 各phaseの処理時間を計測するかどうか
    bool skipNoAgentStep; //!< 誰も行動しないstepのreturnを省略するかどうか
    bool enableStructuredReward; //!< 報酬をいくつかのグループに分けたdictとするかどうか
    std::map<std::string,std::map<std::string,std::uint64_t>> maxProcessTime,minProcessTime,processCount;
    std::map<std::string,std::map<std::string,std::uint64_t>> meanProcessTime;
    ///@}

    /** @name 内部状態のシリアライゼーション (Experimental)
     */
    ///@{
    /**
     * @internal
     * @class CommunicationBufferConstructionInfo
     * @brief [For internal use] CommunicationBuffer の生成に用いられた情報を保持するクラス。
     * 
     * @details シリアライズ時に使用される。ユーザによる利用は想定していない。
     */
    struct PYBIND11_EXPORT CommunicationBufferConstructionInfo{
        nl::json j_participants;
        nl::json j_inviteOnRequest;
        CommunicationBufferConstructionInfo();
        CommunicationBufferConstructionInfo(const std::shared_ptr<CommunicationBuffer>& buffer);
        void serialize(asrc::core::util::AvailableArchiveTypes& archive);
        template<asrc::core::traits::cereal_archive Archive>
        void serialize(Archive& archive){
            asrc::core::util::AvailableArchiveTypes wrapped(std::ref(archive));
            serialize(wrapped);
        }
    };
    virtual void serializeBeforeEntityReconstructionInfo(asrc::core::util::AvailableArchiveTypes & archive, bool full, const nl::json& serializationConfig_) override;
    virtual void serializeEntityReconstructionInfo(asrc::core::util::AvailableArchiveTypes & archive, bool full, const nl::json& serializationConfig_) override;
    virtual void serializeAfterEntityReconstructionInfo(asrc::core::util::AvailableArchiveTypes & archive, bool full, const nl::json& serializationConfig_) override;
    virtual void serializeAfterEntityInternalStates(asrc::core::util::AvailableArchiveTypes & archive, bool full, const nl::json& serializationConfig_) override;
    ///@}

    /** @name 時刻管理
     */
    ///@{
    protected:
    double baseTimeStep; //!< 1tickの秒数
    std::uint64_t tickCount; //!< 現在のtick数
    SimPhase lastAssetPhase; //!< 最後に実行されたAsset phase
    SimPhase lastGroup1CallbackPhase; //!< 最後に実行されたCallback phase(group1)
    SimPhase lastGroup2CallbackPhase; //!< 最後に実行されたCallback phase(group2)
    std::uint64_t internalStepCount; //!< シミュレータ内部でのstep数(Callback向け)
    std::uint64_t exposedStepCount; //!< gym環境として実際にreturnしたstep数(RL向け)
    std::map<SimPhase,std::uint64_t> nextTick; //!< 各phaseを次に処理すべきtick
    std::uint64_t nextTickCB; //!< 次に Asset::control 又は Asset::behave を処理すべきtick
    std::uint64_t nextTickP; //!< 次に Asset::perceive 処理すべきtick
    std::uint64_t defaultAgentStepInterval; //!< Agentのデフォルトの行動判断周期(tick単位)
    public:
    double getElapsedTime() const override; //!< シミュレーション開始時点からの経過秒数を返す。
    std::uint64_t getTickCount() const; //!< 現在の時刻を[tick]単位で返す。
    std::uint64_t getTickCountAt(const Time& time) const; //!< 時刻timeに相当するtick数を返す。
    std::uint64_t getDefaultAgentStepInterval() const; //!< Agentのデフォルトの行動判断周期(tick単位)を返す。
    std::uint64_t getInternalStepCount() const; //!< シミュレータ内部での step 数を返す。(Callback 向け)
    std::uint64_t getExposedStepCount() const; //!< gymnasium 環境として実際に return した step 数を返す。(RL 向け)
    std::uint64_t getAgentStepCount(const std::string& fullName_) const; //!< 指定した Agent が経験したstep数を返す。
    std::uint64_t getAgentStepCount(const std::shared_ptr<Agent>& agent) const; //!< 指定した Agent が経験したstep数を返す。
    /**
     * @brief [For internal use] 現在の処理状況(tickCountと直近のphase)において、
     * 引数で与えられたphaseが次に実行される可能性のある時刻(tick)を返す。
     * 
     * @attention phaseの処理中には呼び出さないこと。
     */
    std::uint64_t getPossibleNextTickCount(const SimPhase& phase) const;
    ///@}

    /** @name 処理順序管理
     */
    ///@{
    protected:
    std::map<SimPhase,bool> hasDependencyUpdated; //!< 各phaseの未反映の処理順序変更があるかどうか
    std::map<SimPhase,EntityDependencyGraph<Asset>> asset_dependency_graph; //!< 各phaseのAssetの処理順序を管理するgraph
    std::map<SimPhase,std::map<EntityIdentifier,std::size_t>> asset_indices; //!< 各phaseのAssetの処理順序
    std::map<SimPhase,EntityDependencyGraph<Callback>> callback_dependency_graph; //!< 各phaseのCallbackの処理順序を管理するgraph
    std::map<SimPhase,std::map<EntityIdentifier,std::size_t>> callback_indices; //!< 各phaseのCallbackの処理順序
    std::map<SimPhase,std::map<std::uint64_t,std::map<std::size_t,std::shared_ptr<Asset>>>> asset_process_queue; //!< 各phase,tickにおけるPhysicalAssetの処理待機列 [phase,[tick,[index,asset]]]
    std::map<SimPhase,std::map<std::uint64_t,std::map<std::size_t,std::shared_ptr<Agent>>>> agent_process_queue; //!< 各phase,tickにおけるAgentの処理待機列 [phase,[tick,[index,asset]]]
    std::map<SimPhase,std::map<std::uint64_t,std::map<std::size_t,std::shared_ptr<Callback>>>> group1_callback_process_queue; //!< 各phase,tickにおけるRuler,Rewardの処理待機列 [phase,[tick,[index,asset]]]
    std::map<SimPhase,std::map<std::uint64_t,std::map<std::size_t,std::shared_ptr<Callback>>>> group2_callback_process_queue; //!< 各phase,tickにおけるRuler,Reward以外のCallbackの処理待機列 [phase,[tick,[index,asset]]]
    std::map<SimPhase,std::map<EntityIdentifier,std::uint64_t>> asset_next_ticks; //!< 各phaseにおけるAssetの次回処理時刻(tick) [phase,[id,tick]]
    std::map<SimPhase,std::map<EntityIdentifier,std::uint64_t>> callback_next_ticks; //!< 各phaseにおけるCallbackの次回処理時刻(tick) [phase,[id,tick]]
    std::map<std::size_t,std::weak_ptr<Agent>> agentsToAct; //!< 次の step 関数の終了時刻が行動判断タイミングとなっているAgent の一覧
    std::map<std::size_t,std::weak_ptr<Agent>> agentsActed; //!< 直近の step 関数又は reset 関数の終了時刻が行動判断タイミングとなっている Agent の一覧
    std::deque<std::function<void(const std::shared_ptr<Asset>&)>> assetDependencyGenerators; //!< エピソード途中に追加された Asset を引数にとって処理順序の依存関係を追加する関数オブジェクトのリスト
    std::deque<std::function<void(const std::shared_ptr<Callback>&)>> callbackDependencyGenerators; //!< configure の後に追加された Callback を引数にとって処理順序の依存関係を追加する関数オブジェクトのリスト
    /**
     * @brief phase の処理順序において predecessor が successor より前に来るように依存関係を追加する。
     */
    void addDependency(const SimPhase& phase,const std::shared_ptr<Entity>& predecessor, const std::shared_ptr<Entity>& successor);
    /**
     * @brief phase の処理順序において predecessor が successor より前に来るよという依存関係があった場合、その依存関係を削除する。
     */
    void removeDependency(const SimPhase& phase,const std::shared_ptr<Entity>& predecessor, const std::shared_ptr<Entity>& successor);
    /**
     * @brief エピソード途中に追加された Asset を引数にとって処理順序の依存関係を追加する関数オブジェクトを追加する。
     */
    void addDependencyGenerator(std::function<void(const std::shared_ptr<Asset>&)> generator);
    /**
     * @brief エピソード途中に追加された Callback を引数にとって処理順序の依存関係を追加する関数オブジェクトを追加する。
     */
    void addDependencyGenerator(std::function<void(const std::shared_ptr<Callback>&)> generator);
    void calcFirstTick(const SimPhase& phase,const std::shared_ptr<Entity>& entity); //!< [For internal use] entity にとっての phase の初回処理時刻(tick)を計算する。
    void calcNextTick(const SimPhase& phase,const std::shared_ptr<Entity>& entity); //!< [For internal use] entity にとっての phase の次回処理時刻(tick)を計算する。
    void calcNextTick(const SimPhase& phase); //!< [For internal use] 次に phase を処理すべき時刻(tick)を計算する。
    void buildDependencyGraph(); //!< [For internal use] 処理順序を管理するgraphを生成する。
    void buildProcessQueue(); //!< [For internal use] 処理待機列を生成する。
    void updateProcessQueueIfNeeded(const SimPhase& phase); //!< [For internal use] phase の処理順序の変更有無を見て必要に応じて処理待機列を更新する。
    public:
    void printOrderedAssets(); //!< [For debug] 各 phase の Asset の処理順序を標準出力に書き出す。
    ///@}

    /** @name エピソード管理
     */
    ///@{
    public:
    py::dict observation_space; //!< 現在のエピソードの観測空間。 Agent のfull name をキーとしたdictで保持する。
    py::dict action_space; //!< 現在のエピソードの行動空間。 Agent のfull name をキーとしたdictで保持する。
    py::dict get_observation_space(); //!< 現在のエピソードの観測空間。 Agent のfull name をキーとしたdictで返す。
    py::dict get_action_space(); //!< 現在のエピソードの行動空間。 Agent のfull name をキーとしたdictで返す。
    void setSeed(const unsigned int& seed_); //!< 乱数生成器のシードを設定する。
    /**
     * @brief gymnasium インターフェースに準拠した reset 関数。
     * 
     * @returns
     *      - observations
     * 
     *          この時刻に行動すべき各 Agent の observation 。full nameをキーとしたdictで返す。
     * 
     *      - info
     * 
     *          以下の要素を持ったdictを、observation の返却対象となっている全 Agent のfull nameをキーとしたdictで返す。
     *          - score 現在の得点。
     *          - w worker index
     *          - v vector index
     *          - endReason エピソード終了時のみ、終了理由のenum値を表す文字列。
     */
    py::tuple reset(std::optional<unsigned int> seed_,const std::optional<py::dict>& options);
    /**
     * @brief gymnasium インターフェースに準拠した step 関数。
     * 
     * @param [in] action この時刻に行動すべき各 Agent のfull nameをキーとした行動のdict。 Agent が外部からの行動供給を要しない場合は省略可。
     * 
     * @returns
     *      - observations
     * 
     *          この時刻に行動すべき各 Agent の observation 。full nameをキーとしたdictで返す。
     * 
     *      - rewards
     * 
     *          この時刻に行動すべき各 Agent の observation 。full nameをキーとしたdictで返す。
     * 
     *      - terminateds
     * 
     *          この時刻に行動すべき(はずだった)各 Agent にとって、このエピソードが既に終了したかどうかのフラグ。
     *          full nameをキーとしたdictで返す。また、全体の終了状況を"__all__"キーに入れる。
     * 
     *      - truncateds
     * 
     *          terminatedsと同じ値を返す。
     * 
     *      - info
     * 
     *          以下の要素を持ったdictを、observation の返却対象となっている全 Agent のfull nameをキーとしたdictで返す。
     *          - score 現在の得点。
     *          - w worker index
     *          - v vector index
     *          - endReason エピソード終了時のみ、終了理由のenum値を表す文字列。
     */
    py::tuple step(const py::dict& action);
    /**
     * @brief 前回の step 関数の返り値を返す。
     */
    py::tuple getReturnsOfTheLastStep();
    void stopEpisodeExternally(void); //!< エピソードを即座に停止する。
    virtual void cleanupEpisode(bool increment=true) override; //!< エピソード終了後のクリーンアップ処理を行う。次のエピソードの reset 関数内で呼ばれる。
    std::vector<std::string> getTeams() const; //!< このエピソード中に存在する Asset の陣営リストを返す。
    py::dict observation; //!< 直近の reset 又は step で返したobservation
    py::dict action; //!< 直近の step で与えられたaction
    std::map<std::string,bool> dones; //!< 現在の各 Agent の終了状況。
    std::map<std::string,bool> stepDones; //!< この step に各 Agent が終了したかどうか。
    std::map<std::string,bool> prevDones; //!< 前回の step までに各 Agent が終了となったかどうか。
    std::map<std::string,double> scores; //!< 現在の得点。陣営名をキーとする。
    std::map<std::string,std::map<std::string,double>> rewards; //!< 直近の step で返した 各 Agent の reward
    std::map<std::string,std::map<std::string,double>> totalRewards; //!< 現在までの各 Agent の累積報酬
    std::map<std::string,std::weak_ptr<Agent>> experts; //!< 現在存在している ExpertWrapper の一覧
    bool manualDone; //!< 手動でエピソード終了する際のフラグ。これをtrueにすると次の step でエピソードが終了される。
    protected:
    unsigned int seedValue; //!< 乱数生成器のシード値
    std::mt19937 randomGen; //!< 乱数生成器
    bool exposeDeadAgentReward; //!< 生存していないAgentへの報酬を出力し続けるかどうか
    bool delayLastObsForDeadAgentUntilAllDone; //!< 生存状態がfalseとなったAgentに関するstepのreturnをエピソード終了時まで遅延するかどうか
    BS::thread_pool<BS::tp::none> pool; //!< マルチスレッド処理を行う場合のスレッドプール
    protected:
    std::vector<std::shared_ptr<Asset>> killRequests;
    std::vector<std::shared_ptr<Asset>> addedAssets;
    std::vector<std::shared_ptr<Callback>> addedCallbacks;
    std::map<std::string,std::shared_ptr<PhysicalAsset>> assets;
    std::map<std::string,std::shared_ptr<Controller>> controllers;
    std::map<std::string,std::shared_ptr<Agent>> agents;
    std::map<std::string,std::shared_ptr<CommunicationBuffer>> communicationBuffers;
    std::shared_ptr<Ruler> ruler;
    std::shared_ptr<Viewer> viewer;
    std::map<std::string,std::shared_ptr<Callback>> callbacks,loggers;
    std::vector<std::shared_ptr<Reward>> rewardGenerators;
    std::vector<std::string> rewardGroups;
    std::vector<std::string> teams;
    std::map<std::string,std::vector<std::function<void(const nl::json&)>>> eventHandlers;
    ConfigDispatcher assetConfigDispatcher,agentConfigDispatcher;
    int numLearners,numExperts,numClones;
    py::dict lastObservations;
    void runAssetPhaseFunc(const SimPhase& phase);
    void runCallbackPhaseFunc(const SimPhase& phase,bool group1);
    void innerStep();
    py::dict makeObs();
    void updateAgentsActed();
    virtual void deployAction(const py::dict& action);
    void updateAgentsToAct();
    void gatherDones();
    void gatherRewards();
    void gatherTotalRewards();
    /**
     * @brief nameという名前のイベントに対するイベントハンドラを追加する。
     */
    void addEventHandler(const std::string& name,std::function<void(const nl::json&)> handler);
    /**
     * @brief nameという名前のイベントを、argsを引数として発生させる。
     */
    void triggerEvent(const std::string& name, const nl::json& args);
    using BaseType::createEntityImpl;
    virtual std::shared_ptr<Entity> createEntityImpl(bool isManaged, bool isEpisodic, const std::string& baseName,const std::string& className,const std::string& modelName,const nl::json& modelConfig,const nl::json& instanceConfig, const FactoryHelperChain& helperChain, const std::shared_ptr<const Entity>& caller) override;
    /**
     * @brief [For internal use] Agentを生成する。
     */
    std::weak_ptr<Agent> createAgent(const nl::json& agentConfig,const std::string& agentName,const std::map<std::string,std::shared_ptr<PhysicalAssetAccessor>>& parents, const FactoryHelperChain& helperChain, const std::shared_ptr<const Entity>&  caller);
    /**
     * @brief CommunicationBufferを生成する。
     */
    bool generateCommunicationBuffer(const std::string& name_,const nl::json& participants_,const nl::json& inviteOnRequest_);
    /**
     * @brief bufferNameという名前を持つCommunicationBufferに、assetの加入を要求する。
     */
    bool requestInvitationToCommunicationBuffer(const std::string& bufferName,const std::shared_ptr<Asset>& asset);
    void createAssets();
    void generateCommunicationBuffers();
    /**
     * @brief asset->kill() の呼び出しを予約する。
     */
    void requestToKillAsset(const std::shared_ptr<Asset>& asset);
    void killAndRemoveAssets();
    void setupNewEntities();
    public:
    virtual void registerEntity(bool isManaged,const std::shared_ptr<Entity>& entity) override;
    virtual void removeEntity(const std::shared_ptr<Entity>& entity) override;
    virtual void removeEntity(const std::weak_ptr<Entity>& entity,const EntityIdentifier& id,const std::string& name) override;
    virtual std::shared_ptr<EntityManagerAccessor> createAccessorFor(const std::shared_ptr<const Entity>& entity) override;
    ///@}

    /** @name 仮想シミュレータの生成(experimental)
     */
    ///@{
    protected:
    bool _isVirtual; //!< 自身が仮想シミュレータとして生成されたものか否か
    public:
    bool isVirtual() const; //!< 自身が仮想シミュレータとして生成されたものか否か
    virtual nl::json createVirtualSimulationManagerConfig(const nl::json& option);
    virtual void postprocessAfterVirtualSimulationManagerCreation(const std::shared_ptr<SimulationManager>& sim, const nl::json& option);
    /**
     * @brief 仮想シミュレータを生成する。
     * 
     * @details 時刻と座標系に関する設定は自身と同じものに自動的に設定される。
     *      特に、CRSは自身の持つインスタンスがそのまま共有される(複製ではない。)
     * 
     *      Factoryコンフィグは自身と同じものが複製される。
     *      
     *      Ruler は 自身の ruler と同じlocalCRSを持つだけの空のインスタンスが生成される。
     *      それ以外の Asset や Callback は何も生成されない。
     * 
     *      これらの振る舞いは option 引数によって変更可能である。
     * 
     * @param [in] option 仮想シミュレータのコンストラクタに渡すconfigを上書きするpatch。merge patch処理が行われる。
     */
    template<std::derived_from<SimulationManager> T=SimulationManager>
    std::shared_ptr<T> createVirtualSimulationManager(const nl::json& option){
        auto ret=SimulationManager::create<T>(
            createVirtualSimulationManagerConfig(option),
            getWorkerIndex(),
            getUnusedVectorIndex()
        );
        postprocessAfterVirtualSimulationManagerCreation(ret,option);
        return ret;
    }
    ///@}

    /** @name Entityの取得
     */
    ///@{
    /**
     * @brief 指定したfull nameを持つ PhysicalAsset をweak_ptr<T>にキャストして返す。
     */
    template<std::derived_from<PhysicalAsset> T=PhysicalAsset>
    std::weak_ptr<T> getAsset(const std::string& fullName_) const{
        try{
            if constexpr(std::is_same_v<T,PhysicalAsset>){
                return assets.at(fullName_);
            }else{
                return std::dynamic_pointer_cast<T>(assets.at(fullName_));
            }
        }catch(std::exception& e){
            std::cout<<"getAsset failure."<<std::endl;
            DEBUG_PRINT_EXPRESSION(fullName_)
            throw e;
        }
    }
    /**
     * @brief 全ての PhysicalAsset をweak_ptr<T>としてイテレート可能なイテラブルを返す。
     */
    template<std::derived_from<PhysicalAsset> T=PhysicalAsset>
    auto getAssets() const{
        return std::move(MapIterable<T,std::string,PhysicalAsset>(assets));
    }
    /**
     * @brief 全ての PhysicalAsset を matcher でフィルタリングしつつ、条件を満たすもののみをweak_ptr<T>としてイテレートするイテラブルを返す。
     */
    template<std::derived_from<PhysicalAsset> T=PhysicalAsset>
    auto getAssets(typename MapIterable<T,std::string,PhysicalAsset>::MatcherType matcher) const{
        return std::move(MapIterable<T,std::string,PhysicalAsset>(assets,matcher));
    }
    /**
     * @brief 指定したfull nameを持つ Agent をweak_ptr<T>にキャストして返す。
     */
    template<std::derived_from<Agent> T=Agent>
    std::weak_ptr<T> getAgent(const std::string& fullName_) const{
        try{
            if constexpr(std::is_same_v<T,Agent>){
                return agents.at(fullName_);
            }else{
                return std::dynamic_pointer_cast<T>(agents.at(fullName_));
            }
        }catch(std::exception& e){
            std::cout<<"getAgent failure."<<std::endl;
            DEBUG_PRINT_EXPRESSION(fullName_)
            throw e;
        }
    }
    /**
     * @brief 全ての Agent をweak_ptr<T>としてイテレート可能なイテラブルを返す。
     */
    template<std::derived_from<Agent> T=Agent>
    auto getAgents() const{
        return std::move(MapIterable<T,std::string,Agent>(agents));
    }
    /**
     * @brief 全ての Agent を matcher でフィルタリングしつつ、条件を満たすもののみをweak_ptr<T>としてイテレートするイテラブルを返す。
     */
    template<std::derived_from<Agent> T=Agent>
    auto getAgents(typename MapIterable<T,std::string,Agent>::MatcherType matcher) const{
        return std::move(MapIterable<T,std::string,Agent>(agents,matcher));
    }
    /**
     * @brief 指定したfull nameを持つ Controller をweak_ptr<T>にキャストして返す。
     */
    template<std::derived_from<Controller> T=Controller>
    std::weak_ptr<T> getController(const std::string& fullName_) const{
        try{
            if constexpr(std::is_same_v<T,Controller>){
                return controllers.at(fullName_);
            }else{
                return std::dynamic_pointer_cast<T>(controllers.at(fullName_));
            }
        }catch(std::exception& e){
            std::cout<<"getController failure."<<std::endl;
            DEBUG_PRINT_EXPRESSION(fullName_)
            throw e;
        }
    }
    /**
     * @brief 全ての Controller をweak_ptr<T>としてイテレート可能なイテラブルを返す。
     */
    template<std::derived_from<Controller> T=Controller>
    auto getControllers() const{
        return std::move(MapIterable<T,std::string,Controller>(controllers));
    }
    /**
     * @brief 全ての Controller を matcher でフィルタリングしつつ、条件を満たすもののみをweak_ptr<T>としてイテレートするイテラブルを返す。
     */
    template<std::derived_from<Controller> T=Controller>
    auto getControllers(typename MapIterable<T,std::string,Controller>::MatcherType matcher) const{
        return std::move(MapIterable<T,std::string,Controller>(controllers,matcher));
    }
    /**
     * @brief Ruler をweak_ptr<T>にキャストして返す。
     */
    template<std::derived_from<Ruler> T=Ruler>
    std::weak_ptr<T> getRuler() const{
        if constexpr(std::is_same_v<T,Ruler>){
            return ruler;
        }else{
            return std::dynamic_pointer_cast<T>(ruler);
        }
    }
    /**
     * @brief Viewer をweak_ptr<T>にキャストして返す。
     */
    template<std::derived_from<Viewer> T=Viewer>
    std::weak_ptr<T> getViewer() const{
        if constexpr(std::is_same_v<T,Viewer>){
            return viewer;
        }else{
            return std::dynamic_pointer_cast<T>(viewer);
        }
    }
    /**
     * @brief 指定したインデックスに対応する Reward をweak_ptr<T>にキャストして返す。
     */
    template<std::derived_from<Reward> T=Reward>
    std::weak_ptr<T> getRewardGenerator(const int& idx) const{
        try{
            if constexpr(std::is_same_v<T,Reward>){
                return rewardGenerators.at(idx);
            }else{
                return std::dynamic_pointer_cast<T>(rewardGenerators.at(idx));
            }
        }catch(std::exception& e){
            std::cout<<"getRewardGenerator failure."<<std::endl;
            DEBUG_PRINT_EXPRESSION(idx)
            throw e;
        }
    }
    /**
     * @brief 全ての Reward をweak_ptr<T>としてイテレート可能なイテラブルを返す。
     */
    template<std::derived_from<Reward> T=Reward>
    auto getRewardGenerators() const{
        return std::move(VectorIterable<T,Reward>(rewardGenerators));
    }
    /**
     * @brief 全ての Reward を matcher でフィルタリングしつつ、条件を満たすもののみをweak_ptr<T>としてイテレートするイテラブルを返す。
     */
    template<std::derived_from<Reward> T=Reward>
    auto getRewardGenerators(typename VectorIterable<T,Reward>::MatcherType matcher) const{
        return std::move(VectorIterable<T,Reward>(rewardGenerators,matcher));
    }
    /**
     * @brief 指定したfull nameを持つ Callback をweak_ptr<T>にキャストして返す。
     */
    template<std::derived_from<Callback> T=Callback>
    std::weak_ptr<T> getCallback(const std::string& name_) const{
        auto found=callbacks.find("Callback/"+name_);
        if(found==callbacks.end()){
            found=callbacks.find(name_);
        }
        if(found!=callbacks.end()){
            if constexpr(std::is_same_v<T,Callback>){
                return found->second;
            }else{
                return std::dynamic_pointer_cast<T>(found->second);
            }
        }else{
            throw std::runtime_error(
                "SimulationManager::getCallback() failed. name_="+name_
            );
        }
    }
    /**
     * @brief 全ての Callback をweak_ptr<T>としてイテレート可能なイテラブルを返す。
     */
    template<std::derived_from<Callback> T=Callback>
    auto getCallbacks() const{
        return std::move(MapIterable<T,std::string,Callback>(callbacks));
    }
    /**
     * @brief 全ての Callback を matcher でフィルタリングしつつ、条件を満たすもののみをweak_ptr<T>としてイテレートするイテラブルを返す。
     */
    template<std::derived_from<Callback> T=Callback>
    auto getCallbacks(typename MapIterable<T,std::string,Callback>::MatcherType matcher) const{
        return std::move(MapIterable<T,std::string,Callback>(callbacks,matcher));
    }
    /**
     * @brief 指定したfull nameを持つ Logger をweak_ptr<T>にキャストして返す。
     */
    template<std::derived_from<Callback> T=Callback>
    std::weak_ptr<T> getLogger(const std::string& name_) const{
        auto found=loggers.find("Logger/"+name_);
        if(found==loggers.end()){
            found=loggers.find(name_);
        }
        if(found!=loggers.end()){
            if constexpr(std::is_same_v<T,Callback>){
                return found->second;
            }else{
                return std::dynamic_pointer_cast<T>(found->second);
            }
        }else{
            throw std::runtime_error(
                "SimulationManager::getLogger() failed. name_="+name_
            );
        }
    }
    /**
     * @brief 全ての Logger をweak_ptr<T>としてイテレート可能なイテラブルを返す。
     */
    template<std::derived_from<Callback> T=Callback>
    auto getLoggers() const{
        return std::move(MapIterable<T,std::string,Callback>(loggers));
    }
    /**
     * @brief 全ての Logger を matcher でフィルタリングしつつ、条件を満たすもののみをweak_ptr<T>としてイテレートするイテラブルを返す。
     */
    template<std::derived_from<Callback> T=Callback>
    auto getLoggers(typename MapIterable<T,std::string,Callback>::MatcherType matcher) const{
        return std::move(MapIterable<T,std::string,Callback>(loggers,matcher));
    }
    ///@}
};
ASRC_DECLARE_DERIVED_TRAMPOLINE(SimulationManager)
    virtual double getElapsedTime() const override{
        PYBIND11_OVERRIDE(double,Base,getElapsedTime);
    }
    virtual nl::json createVirtualSimulationManagerConfig(const nl::json& option) override{
        PYBIND11_OVERRIDE(nl::json,Base,createVirtualSimulationManagerConfig,option);
    }
    virtual void postprocessAfterVirtualSimulationManagerCreation(const std::shared_ptr<SimulationManager>& sim, const nl::json& option) override{
        PYBIND11_OVERRIDE(void,Base,postprocessAfterVirtualSimulationManagerCreation,sim,option);
    }
};

/**
 * @class SimulationManagerAccessorBase
 * @brief SimulationManager とその派生クラスのメンバに対するアクセス制限を実現するためのAccessorクラス。
 * 
 * @details SimulationManager が Asset 又は Callback を生成した際、
 *      その種類によって以下の3種類の SimulationManagerAccessorBase の派生クラスのインスタンスが渡される。
 * 
 *      - PhysicalAsset 又は Controller の場合: SimulationManagerAccessorForPhysicalAsset
 *      - Agent の場合: SimulationManagerAccessorForAgent
 *      - Callback の場合: SimulationManagerAccessorForCallback
 * 
 *      Asset, Callback 側は渡されたAccessorインスタンスに提供されているメンバにのみアクセス可能となる。
 */
ASRC_DECLARE_DERIVED_CLASS_WITH_TRAMPOLINE(SimulationManagerAccessorBase,EntityManagerAccessor)
    friend class SimulationManager;
    public:
    virtual Epoch getEpoch() const; //!< SimulationManager::getEpoch() const
    virtual Time getTime() const; //!< SimulationManager::getTime() const
    virtual double getElapsedTime() const; //!< SimulationManager::getElapsedTime() const
    virtual double getBaseTimeStep() const; //!< SimulationManager::baseTimeStep
    virtual std::uint64_t getTickCount() const; //!< SimulationManager::getTickCount() const
    virtual std::uint64_t getTickCountAt(const Time& time) const; //!< SimulationManager::getTickCountAt(const Time&) const
    virtual std::uint64_t getDefaultAgentStepInterval() const; //!< SimulationManager::getDefaultAgentStepInterval() const
    virtual std::uint64_t getInternalStepCount() const; //!< SimulationManager::getInternalStepCount() const
    virtual std::uint64_t getExposedStepCount() const; //!< SimulationManager::getExposedStepCount() const
    /**
     * @brief SimulationManager::createVirtualSimulationManager<T>(const nl::json&)
     */
    template<std::derived_from<SimulationManager> T=SimulationManager>
    std::shared_ptr<T> createVirtualSimulationManager(const nl::json& option){
        return manager.lock()->createVirtualSimulationManager<T>(option);
    }
    std::shared_ptr<CoordinateReferenceSystem> getRootCRS() const; //!< SimulationManager::getRootCRS() const
    nl::json getCRSConfig() const; //!< SimulationManager::getCRSConfig() const
    /**
     * @brief SimulationManager::getCRSes<T>()
     */
    template<std::derived_from<CoordinateReferenceSystem> T=CoordinateReferenceSystem>
    auto getCRSes() const{
        return manager.lock()->getCRSes<T>();
    }
    /**
     * @brief SimulationManager::getCRSes<T>(typename MapIterable<T,EntityIdentifier,CoordinateReferenceSystem>::MatcherType) const
     */
    template<std::derived_from<CoordinateReferenceSystem> T=CoordinateReferenceSystem>
    auto getCRSes(typename MapIterable<T,EntityIdentifier,CoordinateReferenceSystem>::MatcherType matcher) const{
        return manager.lock()->getCRSes<T>(matcher);
    }
    /**
     * @brief SimulationManager::getCRS<T>(const std::string&) const
     */
    template<std::derived_from<CoordinateReferenceSystem> T=CoordinateReferenceSystem>
    std::weak_ptr<T> getCRS(const std::string& name_) const{
        return manager.lock()->getCRS<T>(name_);
    }
    virtual std::vector<std::string> getTeams() const; //!< SimulationManager::getTeams() const
    protected:
    SimulationManagerAccessorBase(const std::shared_ptr<SimulationManager>& manager_);
    SimulationManagerAccessorBase(const std::shared_ptr<SimulationManagerAccessorBase>& original_);
    protected:
    std::weak_ptr<SimulationManager> manager;
};
ASRC_DECLARE_DERIVED_TRAMPOLINE(SimulationManagerAccessorBase)
    virtual Epoch getEpoch() const override{
        PYBIND11_OVERRIDE(Epoch,Base,getEpoch);
    }
    virtual Time getTime() const override{
        PYBIND11_OVERRIDE(Time,Base,getTime);
    }
    virtual double getElapsedTime() const override{
        PYBIND11_OVERRIDE(double,Base,getElapsedTime);
    }
    virtual double getBaseTimeStep() const override{
        PYBIND11_OVERRIDE(double,Base,getBaseTimeStep);
    }
    virtual std::uint64_t getTickCount() const override{
        PYBIND11_OVERRIDE(std::uint64_t,Base,getTickCount);
    }
    virtual std::uint64_t getTickCountAt(const Time& time) const override{
        PYBIND11_OVERRIDE(std::uint64_t,Base,getTickCountAt,time);
    }
    virtual std::uint64_t getDefaultAgentStepInterval() const override{
        PYBIND11_OVERRIDE(std::uint64_t,Base,getDefaultAgentStepInterval);
    }
    virtual std::uint64_t getInternalStepCount() const override{
        PYBIND11_OVERRIDE(std::uint64_t,Base,getInternalStepCount);
    }
    virtual std::uint64_t getExposedStepCount() const override{
        PYBIND11_OVERRIDE(std::uint64_t,Base,getExposedStepCount);
    }
    virtual std::vector<std::string> getTeams() const override{
        PYBIND11_OVERRIDE(std::vector<std::string>,Base,getTeams);
    }
};

/**
 * @class SimulationManagerAccessorForCallback
 * @brief Callback 用の SimulationManagerAccessor
 */
ASRC_DECLARE_DERIVED_CLASS_WITHOUT_TRAMPOLINE(SimulationManagerAccessorForCallback,SimulationManagerAccessorBase)
    friend class SimulationManager;
    protected:
    using BaseType::BaseType;
    public:
    std::vector<std::pair<EntityIdentifier,MotionState>> getInternalMotionStatesOfAffineCRSes() const; //!< SimulationManager::getInternalMotionStatesOfAffineCRSes() const
    void setInternalMotionStatesOfAffineCRSes(const std::vector<std::pair<EntityIdentifier,MotionState>>& states); //!< SimulationManager::setInternalMotionStatesOfAffineCRSes(const std::vector<std::pair<EntityIdentifier,MotionState>>&)
    boost::uuids::uuid getUUID() const; //!< SimulationManager::getUUID() const
    boost::uuids::uuid getUUIDOfEntity(const std::shared_ptr<const Entity>& entity) const; //!< SimulationManager::getUUIDOfEntity(const std::shared_ptr<const Entity>&) const
    EntityConstructionInfo getEntityConstructionInfo(const std::shared_ptr<Entity>& entity,EntityIdentifier id) const; //!< SimulationManager::getEntityConstructionInfo(const std::shared_ptr<Entity>&,EntityIdentifier) const
    void requestReconfigure(const nl::json& fullConfigPatch); //!< SimulationManager::requestReconfigure(const nl::json&)
    void requestReconfigure(const nl::json& managerConfigPatch,const nl::json& factoryConfigPatch); //!< SimulationManager::requestReconfigure(const nl::json&,const nl::json&)
    void addEventHandler(const std::string& name,std::function<void(const nl::json&)> handler); //!< SimulationManager::addEventHandler(const std::string&,std::function<void(const nl::json&)>)
    void triggerEvent(const std::string& name, const nl::json& args); //!< SimulationManager::triggerEvent(const std::string&, const nl::json&)
    py::dict& observation_space() const; //!< SimulationManager::observation_space
    py::dict& action_space() const; //!< SimulationManager::action_space
    py::dict& observation() const; //!< SimulationManager::observation
    py::dict& action() const; //!< SimulationManager::action
    std::map<std::string,bool>& dones() const; //!< SimulationManager::dones
    std::map<std::string,double>& scores() const; //!< SimulationManager::scores
    std::map<std::string,std::map<std::string,double>>& rewards() const; //!< SimulationManager::rewards
    std::map<std::string,std::map<std::string,double>>& totalRewards() const; //!< SimulationManager::totalRewards
    const std::map<std::string,std::weak_ptr<Agent>>& experts() const; //!< SimulationManager::experts
    const std::map<std::size_t,std::weak_ptr<Agent>>& agentsToAct() const; //!< SimulationManager::agentsToAct
    const std::map<std::size_t,std::weak_ptr<Agent>>& agentsActed() const; //!< SimulationManager::agentsActed
    std::int32_t worker_index() const; //!< SimulationManager::worker_index() const
    std::int32_t vector_index() const; //!< SimulationManager::vector_index() const
    std::int32_t episode_index() const; //!< SimulationManager::episode_index() const
    bool& manualDone(); //!< SimulationManager::manualDone
    std::uint64_t getAgentStepCount(const std::string& fullName_) const; //!< SimulationManager::getAgentStepCount(const std::string&) const
    std::uint64_t getAgentStepCount(const std::shared_ptr<Agent>& agent) const; //!< SimulationManager::getAgentStepCount(const std::shared_ptr<Agent>&) const
    void setManualDone(const bool& b); //!< SimulationManager::manualDone = b;
    void requestToKillAsset(const std::shared_ptr<Asset>& asset); //!< SimulationManager::requestToKillAsset(const std::shared_ptr<Asset>&)
    void addDependency(const SimPhase& phase,const std::shared_ptr<Entity>& predecessor, const std::shared_ptr<Entity>& successor); //!< SimulationManager::addDependency(const SimPhase&,const std::shared_ptr<Entity>&,const std::shared_ptr<Entity>&)
    void removeDependency(const SimPhase& phase,const std::shared_ptr<Entity>& predecessor, const std::shared_ptr<Entity>& successor); //!< SimulationManager::removeDependency(const SimPhase&,const std::shared_ptr<Entity>&,const std::shared_ptr<Entity>&)
    void addDependencyGenerator(std::function<void(const std::shared_ptr<Callback>&)> generator); //!< SimulationManager::addDependencyGenerator(std::function<void(const std::shared_ptr<Callback>&)>)
    /**
     * @brief SimulationManager::getAsset<T>(const std::string&) const
     */
    template<std::derived_from<PhysicalAsset> T=PhysicalAsset>
    std::weak_ptr<T> getAsset(const std::string& fullName_) const{
        return manager.lock()->getAsset<T>(fullName_);
    }
    /**
     * @brief SimulationManager::getAssets<T>() const
     */
    template<std::derived_from<PhysicalAsset> T=PhysicalAsset>
    auto getAssets() const{
        return manager.lock()->getAssets<T>();
    }
    /**
     * @brief SimulationManager::getAssets<T>(typename MapIterable<T,std::string,PhysicalAsset>::MatcherType) const
     */
    template<std::derived_from<PhysicalAsset> T=PhysicalAsset>
    auto getAssets(typename MapIterable<T,std::string,PhysicalAsset>::MatcherType matcher) const{
        return manager.lock()->getAssets<T>(matcher);
    }
    /**
     * @brief SimulationManager::getAgent<T>(const std::string&) const
     */
    template<std::derived_from<Agent> T=Agent>
    std::weak_ptr<T> getAgent(const std::string& fullName_) const{
        return manager.lock()->getAgent<T>(fullName_);
    }
    /**
     * @brief SimulationManager::getAgents<T>() const
     */
    template<std::derived_from<Agent> T=Agent>
    auto getAgents() const{
        return manager.lock()->getAgents<T>();
    }
    /**
     * @brief SimulationManager::getAgents<T>(typename MapIterable<T,std::string,Agent>::MatcherType) const
     */
    template<std::derived_from<Agent> T=Agent>
    auto getAgents(typename MapIterable<T,std::string,Agent>::MatcherType matcher) const{
        return manager.lock()->getAgents<T>(matcher);
    }
    /**
     * @brief SimulationManager::getController<T>(const std::string&) const
     */
    template<std::derived_from<Controller> T=Controller>
    std::weak_ptr<T> getController(const std::string& fullName_) const{
        return manager.lock()->getController<T>(fullName_);
    }
    /**
     * @brief SimulationManager::getControllers<T>() const
     */
    template<std::derived_from<Controller> T=Controller>
    auto getControllers() const{
        return manager.lock()->getControllers<T>();
    }
    /**
     * @brief SimulationManager::getControllers<T>(typename MapIterable<T,std::string,Controller>::MatcherType) const
     */
    template<std::derived_from<Controller> T=Controller>
    auto getControllers(typename MapIterable<T,std::string,Controller>::MatcherType matcher) const{
        return manager.lock()->getControllers<T>(matcher);
    }
    /**
     * @brief SimulationManager::getRuler<T>() const
     */
    template<std::derived_from<Ruler> T=Ruler>
    std::weak_ptr<T> getRuler() const{
        return manager.lock()->getRuler<T>();
    }
    /**
     * @brief SimulationManager::getViewer<T>() const
     */
    template<std::derived_from<Viewer> T=Viewer>
    std::weak_ptr<T> getViewer() const{
        return manager.lock()->getViewer<T>();
    }
    /**
     * @brief SimulationManager::getRewardGenerator<T>(const int&) const
     */
    template<std::derived_from<Reward> T=Reward>
    std::weak_ptr<T> getRewardGenerator(const int& idx) const{
        return manager.lock()->getRewardGenerator<T>(idx);
    }
    /**
     * @brief SimulationManager::getRewardGenerators<T>() const
     */
    template<std::derived_from<Reward> T=Reward>
    auto getRewardGenerators() const{
        return manager.lock()->getRewardGenerators<T>();
    }
    /**
     * @brief SimulationManager::getRewardGenerators<T>(typename VectorIterable<T,Reward>::MatcherType) const
     */
    template<std::derived_from<Reward> T=Reward>
    auto getRewardGenerators(typename VectorIterable<T,Reward>::MatcherType matcher) const{
        return manager.lock()->getRewardGenerators<T>(matcher);
    }
    /**
     * @brief SimulationManager::getCallback<T>(const std::string&) const
     */
    template<std::derived_from<Callback> T=Callback>
    std::weak_ptr<T> getCallback(const std::string& name_) const{
        return manager.lock()->getCallback<T>(name_);
    }
    /**
     * @brief SimulationManager::getCallbacks<T>() const
     */
    template<std::derived_from<Callback> T=Callback>
    auto getCallbacks() const{
        return manager.lock()->getCallbacks<T>();
    }
    /**
     * @brief SimulationManager::getCallbacks<T>(typename MapIterable<T,std::string,Callback>::MatcherType) const
     */
    template<std::derived_from<Callback> T=Callback>
    auto getCallbacks(typename MapIterable<T,std::string,Callback>::MatcherType matcher) const{
        return manager.lock()->getCallbacks<T>(matcher);
    }
    /**
     * @brief SimulationManager::getLogger<T>(const std::string&) const
     */
    template<std::derived_from<Callback> T=Callback>
    std::weak_ptr<T> getLogger(const std::string& name_) const{
        return manager.lock()->getLogger<T>(name_);
    }
    /**
     * @brief SimulationManager::getLoggers<T>() const
     */
    template<std::derived_from<Callback> T=Callback>
    auto getLoggers() const{
        return manager.lock()->getLoggers<T>();
    }
    /**
     * @brief SimulationManager::getLoggers<T>(typename MapIterable<T,std::string,Callback>::MatcherType) const
     */
    template<std::derived_from<Callback> T=Callback>
    auto getLoggers(typename MapIterable<T,std::string,Callback>::MatcherType matcher) const{
        return manager.lock()->getLoggers<T>(matcher);
    }
    nl::json getManagerConfig() const; //!< SimulationManager::getManagerConfig() const
    nl::json getFactoryModelConfig(bool withDefaultModels=true) const; //!< SimulationManager::getFactoryModelConfig(bool) const
};

/**
 * @class SimulationManagerAccessorForPhysicalAsset
 * @brief PhysicalAsset, Controller 用の SimulationManagerAccessor
 */
ASRC_DECLARE_DERIVED_CLASS_WITHOUT_TRAMPOLINE(SimulationManagerAccessorForPhysicalAsset,SimulationManagerAccessorBase)
    friend class SimulationManager;
    protected:
    using BaseType::BaseType;
    public:
    void triggerEvent(const std::string& name, const nl::json& args); //!< SimulationManager::triggerEvent(const std::string&, const nl::json&)
    void requestToKillAsset(const std::shared_ptr<Asset>& asset); //!< SimulationManager::requestToKillAsset(const std::shared_ptr<Asset>&)
    void addDependency(const SimPhase& phase,const std::shared_ptr<Entity>& predecessor, const std::shared_ptr<Entity>& successor); //!< SimulationManager::addDependency(const SimPhase&,const std::shared_ptr<Entity>&,const std::shared_ptr<Entity>&)
    void removeDependency(const SimPhase& phase,const std::shared_ptr<Entity>& predecessor, const std::shared_ptr<Entity>& successor); //!< SimulationManager::addDependency(const SimPhase&,const std::shared_ptr<Entity>&,const std::shared_ptr<Entity>&)
    void addDependencyGenerator(std::function<void(const std::shared_ptr<Asset>&)> generator); //!< SimulationManager::addDependencyGenerator(std::function<void(const std::shared_ptr<Asset>&)>)
    /**
     * @brief SimulationManager::getAsset<T>(const std::string&) const
     */
    template<std::derived_from<PhysicalAsset> T=PhysicalAsset>
    std::weak_ptr<T> getAsset(const std::string& fullName_) const{
        return manager.lock()->getAsset<T>(fullName_);
    }
    /**
     * @brief SimulationManager::getAssets<T>() const
     */
    template<std::derived_from<PhysicalAsset> T=PhysicalAsset>
    auto getAssets() const{
        return manager.lock()->getAssets<T>();
    }
    /**
     * @brief SimulationManager::getAssets<T>(typename MapIterable<T,std::string,PhysicalAsset>::MatcherType) const
     */
    template<std::derived_from<PhysicalAsset> T=PhysicalAsset>
    auto getAssets(typename MapIterable<T,std::string,PhysicalAsset>::MatcherType matcher) const{
        return manager.lock()->getAssets<T>(matcher);
    }
    /**
     * @brief SimulationManager::getAgent<T>(const std::string&) const
     */
    template<std::derived_from<Agent> T=Agent>
    std::weak_ptr<T> getAgent(const std::string& fullName_) const{
        return manager.lock()->getAgent<T>(fullName_);
    }
    /**
     * @brief SimulationManager::getAgents<T>() const
     */
    template<std::derived_from<Agent> T=Agent>
    auto getAgents() const{
        return manager.lock()->getAgents<T>();
    }
    /**
     * @brief SimulationManager::getAgents<T>(typename MapIterable<T,std::string,Agent>::MatcherType) const
     */
    template<std::derived_from<Agent> T=Agent>
    auto getAgents(typename MapIterable<T,std::string,Agent>::MatcherType matcher) const{
        return manager.lock()->getAgents<T>(matcher);
    }
    /**
     * @brief SimulationManager::getController<T>(const std::string&) const
     */
    template<std::derived_from<Controller> T=Controller>
    std::weak_ptr<T> getController(const std::string& fullName_) const{
        return manager.lock()->getController<T>(fullName_);
    }
    /**
     * @brief SimulationManager::getControllers<T>() const
     */
    template<std::derived_from<Controller> T=Controller>
    auto getControllers() const{
        return manager.lock()->getControllers<T>();
    }
    /**
     * @brief SimulationManager::getControllers<T>(typename MapIterable<T,std::string,Controller>::MatcherType) const
     */
    template<std::derived_from<Controller> T=Controller>
    auto getControllers(typename MapIterable<T,std::string,Controller>::MatcherType matcher) const{
        return manager.lock()->getControllers<T>(matcher);
    }
    /**
     * @brief SimulationManager::getRuler<T>() const
     */
    template<std::derived_from<Ruler> T=Ruler>
    std::weak_ptr<T> getRuler() const{
        return manager.lock()->getRuler<T>();
    }
    /**
     * @brief SimulationManager::createAgent<T>(const nl::json&,const std::string&,const std::map<std::string,std::shared_ptr<PhysicalAssetAccessor>>&, const FactoryHelperChain&, const std::shared_ptr<const Entity>&)
     * 
     * @details FactoryHelperChain は caller->getFactoryHelperChain() が自動的に渡される。
     */
    template<std::derived_from<Agent> T=Agent>
    std::weak_ptr<T> createAgent(const nl::json& agentConfig,const std::string& agentName,const std::map<std::string,std::shared_ptr<PhysicalAssetAccessor>>& parents, const std::shared_ptr<const Entity>&  caller){
        if(caller){
            return util::getShared<T,Agent>(manager.lock()->createAgent(agentConfig,agentName,parents,caller->getFactoryHelperChain(),caller));
        }else{
            return util::getShared<T,Agent>(manager.lock()->createAgent(agentConfig,agentName,parents,FactoryHelperChain(),caller));
        }
    }
    /**
     * @brief SimulationManager::requestInvitationToCommunicationBuffer(const std::string&,const std::shared_ptr<Asset>&)
     */
    bool requestInvitationToCommunicationBuffer(const std::string& bufferName,const std::shared_ptr<Asset>& asset);
    /**
     * @brief SimulationManager::generateCommunicationBuffer(const std::string&,const nl::json&,const nl::json&)
     */
    bool generateCommunicationBuffer(const std::string& name_,const nl::json& participants_,const nl::json& inviteOnRequest_);
};

/**
 * @class SimulationManagerAccessorForAgent
 * @brief Agent 用の SimulationManagerAccessor
 */
ASRC_DECLARE_DERIVED_CLASS_WITH_TRAMPOLINE(SimulationManagerAccessorForAgent,SimulationManagerAccessorBase)
    friend class SimulationManager;
    protected:
    using BaseType::BaseType;
    public:
    /**
     * @brief SimulationManager::requestInvitationToCommunicationBuffer(const std::string&,const std::shared_ptr<Asset>&)
     */
    virtual bool requestInvitationToCommunicationBuffer(const std::string& bufferName,const std::shared_ptr<Asset>& asset);
    /**
     * @brief SimulationManager::getRuler<T>() const
     */
    virtual std::weak_ptr<RulerAccessor> getRuler() const;
};
ASRC_DECLARE_DERIVED_TRAMPOLINE(SimulationManagerAccessorForAgent)
    virtual bool requestInvitationToCommunicationBuffer(const std::string& bufferName,const std::shared_ptr<Asset>& asset) override{
        PYBIND11_OVERRIDE(bool,Base,requestInvitationToCommunicationBuffer,bufferName,asset);
    }
    virtual std::weak_ptr<RulerAccessor> getRuler() const override{
        PYBIND11_OVERRIDE(std::weak_ptr<RulerAccessor>,Base,getRuler);
    }
};

void exportSimulationManager(py::module &m);

ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
