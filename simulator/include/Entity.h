/**
 * Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
 * @file
 * 
 * @brief シミュレーション登場物とそれを管理する基底クラスを定義する。
 * 
 * @details シミュレーション登場物を表す Entity 、 Entity を管理する EntityManager 、 これらへのアクセス制限を担う EntityAccessor 、 EntityManagerAccessor からなる。
 */
#pragma once
#include "Common.h"
#include <memory>
#include <random>
#include <set>
#include <pybind11/pybind11.h>
#include <nlohmann/json.hpp>
#include "Utility.h"
#include "JsonMutableFromPython.h"
#include "Factory.h"
#include "util/dependency_graph.h"
namespace py=pybind11;
namespace nl=nlohmann;

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)

enum class SimPhase;
class EntityManagerAccessor;
class EntityAccessor;

/**
 * @internal
 * @class EntityConstructionInfo
 * @brief [For internal use] Entity の生成に用いられた情報を保持するクラス。
 * 
 * @details EntityManager のシリアライズ時に使用される。ユーザによる利用は想定していない。
 */
ASRC_DECLARE_NON_POLYMORPHIC_DATA_CLASS(EntityConstructionInfo)
    bool removed;
    EntityIdentifier entityIdentifier;
    bool isManaged;
    FactoryHelperChain factoryHelperChain;
    nl::json config;
    void serialize(asrc::core::util::AvailableArchiveTypes& archive);
    template<asrc::core::traits::cereal_archive Archive>
    void serialize(Archive& archive){
        asrc::core::util::AvailableArchiveTypes wrapped(std::ref(archive));
        serialize(wrapped);
    }
    EntityConstructionInfo();
    EntityConstructionInfo(const std::shared_ptr<Entity>& entity,EntityIdentifier id);
};

/**
 * @class EntityManager
 * @brief Entity の生成・管理を行うクラス。
 * 
 * @details Factory のラッパーとして、生成した Entity の寿命もshared_ptrとして管理するクラス。
 *          シミュレーションの実行処理とは分離しておく。
 *          SimulationManager はこれを継承する。
 */
ASRC_DECLARE_BASE_REF_CLASS(EntityManager)
    friend class Entity;
    friend class EntityAccessor;
    protected:
    /**
     * @brief EntityManager はコンストラクタ外で追加の初期化処理(EntityManager::addInstance)が必要なため、protectedとする。
     *        インスタンスを生成する際には EntityManager::create() を用いること。
     */
    EntityManager(const std::int32_t& vector_index_);
    public:
    virtual ~EntityManager();

    /**
     * @brief publicコンストラクタに相当するもの。C++側ではこちらを呼ぶ。
     * 
     * @param [in] args コンストラクタが受け入れる任意の引数。
     */
    template<std::derived_from<EntityManager> T=EntityManager, typename... Args>
    static std::shared_ptr<T> create(Args&&... args){
        auto ret=BaseType::create<T>(std::forward<Args>(args)...);
        addInstance(ret);
        return ret;
    }

    /**
     * @brief Python用のtrampolineクラスのインスタンスを生成する。Python側に__init__を設けるときはこちらを用いる。
     * 
     * @param [in] args コンストラクタが受け入れる任意の引数。
     */
    template<std::derived_from<EntityManager> T=EntityManager, typename... Args>
    static std::shared_ptr<typename T::template TrampolineType<>> create_for_trampoline(Args&&... args){
        auto ret=BaseType::create_for_trampoline<T>(std::forward<Args>(args)...);
        addInstance(ret);
        return ret;
    }

    private:
    /**
     * @brief EntityManager インスタンスを追加する。
     */
    static void addInstance(const std::shared_ptr<EntityManager>& manager);

    /**
     * @brief 引数に与えたvector indexと一致する EntityManager インスタンスを返す。
     */
    static std::shared_ptr<EntityManager> getManagerInstance(std::int32_t vector_index);

    /**
     * @brief 引数に与えたUUIDと一致する EntityManager インスタンスを返す。
     */
    static std::shared_ptr<EntityManager> getManagerInstance(const boost::uuids::uuid& uuid);

    protected:
    /**
     * @brief 自身のUUIDを返す。
     */
    boost::uuids::uuid getUUID() const;

    /**
     * @brief 自身のUUIDを変更する。
     */
    void setUUID(boost::uuids::uuid new_uuid);

    /**
     * @brief 自身が管理している Entity のUUIDを返す。
     */
    boost::uuids::uuid getUUIDOfEntity(const std::shared_ptr<const Entity>& entity) const;

    public:
    /**
     * @brief このプロセスのworker indexを変更する。
     */
    static void setWorkerIndex(const std::int32_t& w);

    /**
     * @brief このプロセスのworker indexを返す。
     */
    static std::int32_t getWorkerIndex();

    /**
     * @brief このプロセスでまだ使用されていない最小の非負のvector indexを返す。
     */
    static std::int32_t getUnusedVectorIndex();

    /**
     * @brief 自身のvector indexを返す。
     */
    std::int32_t getVectorIndex() const;

    /**
     * @brief 自身のepisode indexを返す。
     */
    std::int32_t getEpisodeIndex() const;

    /**
     * @brief 次に生成するnon-episodicな Entity のentity indexを返す。
     * 
     * @param [in] preference 有効な値を与えた場合、使用済でなければこれのentity indexを返す。
     *                        与えなかった場合、未使用の非負値のうち最小のものを返す。
     * @param [in] increment  trueにした場合かつreferenceが指定されていない場合、未使用値を返すと同時にその値を使用済とする。
     */
    std::int32_t getNextEpisodeIndependentEntityIndex(const std::int32_t& preference=EntityIdentifier::invalid_entity_index,bool increment=true);

    /**
     * @brief 次に生成するepisodicな Entity のentity indexを返す。
     * 
     * @param [in] preference 有効な値を与えた場合、使用済でなければこれのentity indexを返す。
     *                        与えなかった場合、未使用の非負値のうち最小のものを返す。
     * @param [in] increment  trueにした場合かつreferenceが指定されていない場合、未使用値を返すと同時にその値を使用済とする。
     */
    std::int32_t getNextEntityIndex(const std::int32_t& preference=EntityIdentifier::invalid_entity_index,bool increment=true);

    /**
     * @brief 次に生成するnon-episodicな Entity の EntityIdentifier を返す。
     * 
     * @param [in] preference 有効な値を与えた場合、使用済でなければこれのentity indexを使用して生成する。
     *                        与えなかった場合、未使用の非負値のうち最小のものをentity indexとする。
     * @param [in] increment  trueにした場合かつreferenceが指定されていない場合、未使用のentity index値を返すと同時にその値を使用済とする。
     */
    EntityIdentifier getNextEpisodeIndependentEntityIdentifier(const EntityIdentifier& preference=EntityIdentifier(),bool increment=true);

    /**
     * @brief 次に生成するepisodicな Entity の EntityIdentifier を返す。
     * 
     * @param [in] preference 有効な値を与えた場合、使用済でなければこれのentity indexを使用して生成する。
     *                        与えなかった場合、未使用の非負値のうち最小のものをentity indexとする。
     * @param [in] increment  trueにした場合かつreferenceが指定されていない場合、未使用のentity index値を返すと同時にその値を使用済とする。
     */
    EntityIdentifier getNextEntityIdentifier(const EntityIdentifier& preference=EntityIdentifier(),bool increment=true);

    /**
     * @brief この EntityManager インスタンスを管理対象から取り除く。
     * 
     * @details getManagerInstance で取得不可能となり関連する Entity の from_json が使用不可能となる。
     */
    virtual void purge();

    ///@name model configの操作
    ///@{
    /**
     * @brief  Factory のdefault model以外の、このインスタンス固有のmodelとして追加されたmodelを全て削除する。
     */
    void clearModels();

    /**
     * @brief modelを追加する。
     * 
     * @param [in] baseName     追加先のグループ名(base name)
     * @param [in] modelName    modelの名称
     * @param [in] modelConfig_ configの値
     */
    void addModel(const std::string& baseName,const std::string& modelName,const nl::json& modelConfig_);

    /**
     * @brief modelをjson objectから読み込んで追加する。
     * 
     * @details jsonの形式はbase nameを第1階層、model nameを第2階層のキーとして、model configを第2階層の値に記述する。
     *          複数同時に追加することができる。
     * @code
     *  {
     *      "baseName":{
     *          "modelName":{
     *              // configを表すobject (dict)
     *          }
     *      }
     *  }
     * @endcode
     */
    void addModelsFromJson(const nl::json& j);

    /**
     * @brief modelをjsonファイルから読み込んで追加する。
     * 
     * @details jsonの形式はaddModelsFromJsonと同じ。
     * 
     * @param [in] filePath jsonファイルのパス
     */
    void addModelsFromJsonFile(const std::string& filePath);

    /**
     * @brief 自身が持つmodel configの一覧を返す。
     * 
     * @param [in] withDefaultModels trueの場合、 Factory のdefault modelも合わせて返す。falseの場合、自身の固有のmodelのみ。
     * 
     * @details jsonの形式はbase nameを第1階層、model nameを第2階層のキーとして、model configを第2階層の値に記述したものとなる。
     *          addModelsFromJson()と互換性がある。
     * @code
     *  {
     *      "baseName":{
     *          "modelName":{
     *              // configを表すobject (dict)
     *          }
     *      }
     *  }
     * @endcode
     */
    nl::json getModelConfigs(bool withDefaultModels) const;

    /**
     * @brief 現在のmodel configを、引数に与えたjsonをpatchとしてmerge patch処理を行うことで書き換える。
     * 
     * @details configの値としてnullを与えることでmodelの削除が可能。(merge patchの仕様による挙動)
     * 
     */
    void reconfigureModelConfigs(const nl::json& j);

    /**
     * @brief Factory::resolveModelConfig を呼び出し、[className, modelName, modelConfig, FactoryHelperChain]の組を返す。
     * 
     * @details 許容される引数は Factory::resolveModelConfig が受け入れるもの全て。
     */
    template<typename... Args>
    std::tuple<std::string,std::string,nl::json,FactoryHelperChain> resolveModelConfig(Args&&... args) const{
        return factory->resolveModelConfig(std::forward<Args>(args)...);
    }

    ///@}

    /**
     * @name Entity の生成に関するメンバ関数
     * @{
     * 
     * xxxImplに実体を実装し、xxxOverloaderで外部からの呼び出しを可能とする引数セットを定義し、xxxでテンプレート関数として外部に公開する。
     * 派生クラスではxxxImpl又はxxxOverloaderをオーバーライドしてよい。
     * 
     */
    public:
    /**
     * @brief Entity を生成する関数の実体その1
     * 
     * @param [in] isManaged        Entity の寿命を EntityManager 側で管理するかどうか。
     * @param [in] isEpisodic       Entity の寿命をエピソードに紐付けるかどうか。
     * @param [in] baseName         Entity のグループ名(base name)を指定する。省略不可。
     * @param [in] className        Entity のクラス名を指定する。""として省略するとモデルに紐付けられたクラスが使用される。
     * @param [in] modelName        Entity のモデル名を指定する。""として省略するとclassNameとmodelConfigから無名モデルとして生成される。
     * @param [in] modelConfig      Entity のmodel configを指定する。モデルに紐付けられたコンフィグがある場合、それに対するmerge patchとして働く。
     * @param [in] instanceConfig   Entity のinstance configを指定する。
     * @param [in] helperChain      生成時にクラス名、モデル名の解決に使用する FactoryHelperChain 。省略可。
     * @param [in] caller           生成元の Entity 。省略可。
     * 
     */
    virtual std::shared_ptr<Entity> createEntityImpl(bool isManaged, bool isEpisodic, const std::string& baseName,const std::string& className,const std::string& modelName,const nl::json& modelConfig,const nl::json& instanceConfig, const FactoryHelperChain& helperChain, const std::shared_ptr<const Entity>& caller);

    virtual std::shared_ptr<Entity> createUnmanagedEntityOverloader(bool isEpisodic, const std::string& baseName,const std::string& className,const std::string& modelName,const nl::json& modelConfig,const nl::json& instanceConfig, const FactoryHelperChain& helperChain=FactoryHelperChain());
    virtual std::shared_ptr<Entity> createUnmanagedEntityOverloader(bool isEpisodic, const std::string& baseName,const std::string& className,const std::string& modelName,const nl::json& modelConfig,const nl::json& instanceConfig, const std::shared_ptr<const Entity>& caller);
    virtual std::shared_ptr<Entity> createUnmanagedEntityOverloader(bool isEpisodic, const std::string& baseName,const std::string& modelName,const nl::json& instanceConfig, const FactoryHelperChain& helperChain=FactoryHelperChain());
    virtual std::shared_ptr<Entity> createUnmanagedEntityOverloader(bool isEpisodic, const std::string& baseName,const std::string& modelName,const nl::json& instanceConfig, const std::shared_ptr<const Entity>& caller);
    virtual std::shared_ptr<Entity> createEntityOverloader(bool isEpisodic, const std::string& baseName,const std::string& className,const std::string& modelName,const nl::json& modelConfig,const nl::json& instanceConfig, const FactoryHelperChain& helperChain=FactoryHelperChain());
    virtual std::shared_ptr<Entity> createEntityOverloader(bool isEpisodic, const std::string& baseName,const std::string& className,const std::string& modelName,const nl::json& modelConfig,const nl::json& instanceConfig, const std::shared_ptr<const Entity>& caller);
    virtual std::shared_ptr<Entity> createEntityOverloader(bool isEpisodic, const std::string& baseName,const std::string& modelName,const nl::json& instanceConfig, const FactoryHelperChain& helperChain=FactoryHelperChain());
    virtual std::shared_ptr<Entity> createEntityOverloader(bool isEpisodic, const std::string& baseName,const std::string& modelName,const nl::json& instanceConfig, const std::shared_ptr<const Entity>& caller);

    virtual std::shared_ptr<Entity> createUnmanagedEntityByClassNameOverloader(bool isEpisodic, const std::string& baseName,const std::string& className,const nl::json& modelConfig,const nl::json& instanceConfig, const FactoryHelperChain& helperChain=FactoryHelperChain());
    virtual std::shared_ptr<Entity> createUnmanagedEntityByClassNameOverloader(bool isEpisodic, const std::string& baseName,const std::string& className,const nl::json& modelConfig,const nl::json& instanceConfig, const std::shared_ptr<const Entity>& caller);
    virtual std::shared_ptr<Entity> createEntityByClassNameOverloader(bool isEpisodic, const std::string& baseName,const std::string& className,const nl::json& modelConfig,const nl::json& instanceConfig, const FactoryHelperChain& helperChain=FactoryHelperChain());
    virtual std::shared_ptr<Entity> createEntityByClassNameOverloader(bool isEpisodic, const std::string& baseName,const std::string& className,const nl::json& modelConfig,const nl::json& instanceConfig, const std::shared_ptr<const Entity>& caller);

    /**
     * @brief Entity を生成する関数の実体その2。ユーザが頻繁に指定することになる部分を一つのjsonにまとめて指定できるようにするもの。
     * 
     * @param [in] isManaged        Entity の寿命を EntityManager 側で管理するかどうか。
     * @param [in] j                生成オプションをobject(dict)で持ったjson。isEpisodic、baseName、className、modelName、modelConfig、instanceConfigを指定可能。
     * @param [in] helperChain      生成時にクラス名、モデル名の解決に使用する FactoryHelperChain 。省略可。
     * @param [in] caller           生成元の Entity 。省略可。
     * 
     */
    virtual std::shared_ptr<Entity> createEntityImpl(bool isManaged, const nl::json& j, const FactoryHelperChain& helperChain, const std::shared_ptr<const Entity>& caller);
    virtual std::shared_ptr<Entity> createUnmanagedEntityOverloader(const nl::json& j, const FactoryHelperChain& helperChain=FactoryHelperChain());
    virtual std::shared_ptr<Entity> createUnmanagedEntityOverloader(const nl::json& j, const std::shared_ptr<const Entity>& caller);
    virtual std::shared_ptr<Entity> createEntityOverloader(const nl::json& j, const FactoryHelperChain& helperChain=FactoryHelperChain());
    virtual std::shared_ptr<Entity> createEntityOverloader(const nl::json& j, const std::shared_ptr<const Entity>& caller);

    /**
     * @brief Entity を生成する関数の実体その3。jsonに EntityIdentifier やfull name等を含めて与えることで、
     *        既存の Entity が見つかった場合はそれを返し、見つからなかった場合は新たに生成する。
     * 
     * @param [in] isManaged        Entity の寿命を EntityManager 側で管理するかどうか。
     * @param [in] j                生成オプションをobject(dict)で持ったjson。
     * @param [in] helperChain      生成時にクラス名、モデル名の解決に使用する FactoryHelperChain 。省略可。
     * @param [in] caller           生成元の Entity 。省略可。
     * 
     * @details 既存の Entity を返すのは以下の3パターン。
     *          (1) Entity::from_json(j) で復元可能な場合
     *          (2) j["entityIdenfier"]と一致する EntityIdentifier を持つ Entity が存在した場合
     *          (3) j["entityFullName"]と一致するfull nameを持つ Entity が存在した場合
     * 
     */
    virtual std::shared_ptr<Entity> createOrGetEntityImpl(bool isManaged, const nl::json& j, const FactoryHelperChain& helperChain, const std::shared_ptr<const Entity>& caller);
    virtual std::shared_ptr<Entity> createOrGetUnmanagedEntityOverloader(const nl::json& j, const FactoryHelperChain& helperChain=FactoryHelperChain());
    virtual std::shared_ptr<Entity> createOrGetUnmanagedEntityOverloader(const nl::json& j, const std::shared_ptr<const Entity>& caller);
    virtual std::shared_ptr<Entity> createOrGetEntityOverloader(const nl::json& j, const FactoryHelperChain& helperChain=FactoryHelperChain());
    virtual std::shared_ptr<Entity> createOrGetEntityOverloader(const nl::json& j, const std::shared_ptr<const Entity>& caller);

    /**
     * @brief unmanagedな Entity を生成する際に外から実際に呼び出すテンプレート関数。Overloaderを呼び出し可能な任意の引数を使える。
     * 
     * @tparam T 生成した Entity をどのクラスとして受け取るか。 Entity 又はその派生クラスを指定できる。
     */
    template<std::derived_from<Entity> T=Entity, typename... Args>
    std::shared_ptr<T> createUnmanagedEntity(Args&&... args){
        return std::dynamic_pointer_cast<T>(createUnmanagedEntityOverloader(std::forward<Args>(args)...));
    }

    /**
     * @brief [For backward compatibility] unmanagedな Entity をクラス名とモデルコンフィグから生成する際に外から実際に呼び出すテンプレート関数。
     * 
     * @tparam T 生成したEntityをどのクラスとして受け取るか。 Entity 又はその派生クラスを指定できる。
     * 
     * @details createUnmanagedEntity() でも同様の Entity 生成が可能なので敢えて使用しなくてもよい。
     */
    template<std::derived_from<Entity> T=Entity, typename... Args>
    std::shared_ptr<T> createUnmanagedEntityByClassName(Args&&... args){
        return std::dynamic_pointer_cast<T>(createUnmanagedEntityByClassNameOverloader(std::forward<Args>(args)...));
    }

    /**
     * @brief unmanagedな Entity を生成する際に外から実際に呼び出すテンプレート関数。Overloaderを呼び出し可能な任意の引数を使える。
     * 
     * @tparam T 生成した Entity をどのクラスとして受け取るか。 Entity 又はその派生クラスを指定できる。
     */
    template<std::derived_from<Entity> T=Entity, typename... Args>
    std::shared_ptr<T> createOrGetUnmanagedEntity(Args&&... args){
        return std::dynamic_pointer_cast<T>(createOrGetUnmanagedEntityOverloader(std::forward<Args>(args)...));
    }

    /**
     * @brief managedな Entity を生成する際に外から実際に呼び出すテンプレート関数。Overloaderを呼び出し可能な任意の引数を使える。
     * 
     * @tparam T 生成した Entity をどのクラスとして受け取るか。 Entity 又はその派生クラスを指定できる。
     */
    template<std::derived_from<Entity> T=Entity, typename... Args>
    std::shared_ptr<T> createEntity(Args&&... args){
        return std::dynamic_pointer_cast<T>(createEntityOverloader(std::forward<Args>(args)...));
    }

    /**
     * @brief [For backward compatibility] managedな Entity をクラス名とモデルコンフィグから生成する際に外から実際に呼び出すテンプレート関数。
     * 
     * @tparam T 生成した Entity をどのクラスとして受け取るか。 Entity 又はその派生クラスを指定できる。
     * 
     * @note createUnmanagedEntity() でも同様の Entity 生成が可能なので敢えて使用しなくてもよい。
     */
    template<std::derived_from<Entity> T=Entity, typename... Args>
    std::shared_ptr<T> createEntityByClassName(Args&&... args){
        return std::dynamic_pointer_cast<T>(createEntityByClassNameOverloader(std::forward<Args>(args)...));
    }

    /**
     * @brief managedな Entity を生成する際に外から実際に呼び出すテンプレート関数。Overloaderを呼び出し可能な任意の引数を使える。
     * 
     * @tparam T 生成した Entity をどのクラスとして受け取るか。 Entity 又はその派生クラスを指定できる。
     */
    template<std::derived_from<Entity> T=Entity, typename... Args>
    std::shared_ptr<T> createOrGetEntity(Args&&... args){
        return std::dynamic_pointer_cast<T>(createOrGetEntityOverloader(std::forward<Args>(args)...));
    }
    ///@}

    // Entity 生成処理を派生クラスでカスタマイズするためのフック
    public:
    /**
     * @internal
     * @brief [For internal use] 新たな Entity を登録する。ユーザが手動で呼ぶことは想定していない。
     * 
     * @param [in] isManaged この EntityManager で管理するかどうか
     * @param [in] entity 登録対象の Entity
     * 
     */
    virtual void registerEntity(bool isManaged,const std::shared_ptr<Entity>& entity);
    /**
     * @internal
     * @brief [For internal use] entityに渡すべき EntityManagerAccessor を返す。
     */
    virtual std::shared_ptr<EntityManagerAccessor> createAccessorFor(const std::shared_ptr<const Entity>& entity);

    ///@name Entity の取得、削除
    ///@{
    private:
    void removeUUIDOfEntity(const EntityIdentifier& id); //!< [For internal use] Entity のUUID情報を削除する
    public:
    /**
     * @brief エピソードの終了処理を行う。
     * 
     * @param [in] increment episode indexをインクリメントするかどうか。
     */
    virtual void cleanupEpisode(bool increment=true);
    /**
     * @brief Entityを削除する。
     */
    virtual void removeEntity(const std::shared_ptr<Entity>& entity);
    /**
     * @internal
     * @brief [For internal use] Entityを削除する。
     */
    virtual void removeEntity(const std::weak_ptr<Entity>& entity,const EntityIdentifier& id,const std::string& name);
    /**
     * @brief 指定したEntityIdentifier を持つ Entity を返す。
     * 
     * @details
     *  - 見つからなかった場合はnullptrを返す。
     */
    std::shared_ptr<Entity> getEntityByIDImpl(const EntityIdentifier& id) const;
    /**
     * @brief 指定したfull nameを持つ Entity を返す。
     * 
     * @details
     *  - 見つからなかった場合はnullptrを返す。
     */
    std::shared_ptr<Entity> getEntityByNameImpl(const std::string& name) const;

    /**
     * @brief 指定した EntityIdentifier を持つ Entity をTにキャストして返す。
     * 
     * @details
     *  - 見つからなかった場合はnullptrを返す。
     *  - 変換不可能だった場合もnullptrを返す。
     */
    template<std::derived_from<Entity> T=Entity>
    std::shared_ptr<T> getEntityByID(const EntityIdentifier& id) const{
        return std::dynamic_pointer_cast<T>(getEntityByIDImpl(id));
    }
    /**
     * @brief 指定したfull nameを持つ Entity をTにキャストして返す。
     * 
     * @details
     *  - 見つからなかった場合はnullptrを返す。
     *  - 変換不可能だった場合もnullptrを返す。
     */
    template<std::derived_from<Entity> T=Entity>
    std::shared_ptr<T> getEntityByName(const std::string& name) const{
        return std::dynamic_pointer_cast<T>(getEntityByNameImpl(name));
    }
    public:

    /**
     * @internal
     * @brief [For internal use] 指定した EntityIdentifier を持つ Entity を返す。
     *        callerが取得権限を持っているかを判定し、持っていなければ例外を投げる。
     * 
     * @details
     *  - 見つからなかった場合はnullptrを返す。
     */
    std::shared_ptr<Entity> getEntityByIDImpl(const EntityIdentifier& id, const std::shared_ptr<const Entity>& caller) const;
    /**
     * @internal
     * @brief [For internal use] 指定したfull nameを持つ Entity を返す。
     *        callerが取得権限を持っているかを判定し、持っていなければ例外を投げる。
     * 
     * @details
     *  - 見つからなかった場合はnullptrを返す。
     */
    std::shared_ptr<Entity> getEntityByNameImpl(const std::string& name, const std::shared_ptr<const Entity>& caller) const;
    /**
     * @internal
     * @brief [For internal use] 指定した EntityIdentifier を持つ Entity をTにキャストして返す。
     *        callerが取得権限を持っているかを判定し、持っていなければ例外を投げる。
     * 
     * @details
     *  - 見つからなかった場合はnullptrを返す。
     *  - 変換不可能だった場合もnullptrを返す。
     */
    template<std::derived_from<Entity> T=Entity>
    std::shared_ptr<T> getEntityByID(const EntityIdentifier& id, const std::shared_ptr<const Entity>& caller) const{
        return std::dynamic_pointer_cast<T>(getEntityByIDImpl(id,caller));
    }
    /**
     * @internal
     * @brief [For internal use] 指定したfull nameを持つ Entity をTにキャストして返す。
     *        callerが取得権限を持っているかを判定し、持っていなければ例外を投げる。
     * 
     * @details
     *  - 見つからなかった場合はnullptrを返す。
     *  - 変換不可能だった場合もnullptrを返す。
     */
    template<std::derived_from<Entity> T=Entity>
    std::shared_ptr<T> getEntityByName(const std::string& name, const std::shared_ptr<const Entity>& caller) const{
        return std::dynamic_pointer_cast<T>(getEntityByNameImpl(name,caller));
    }
    /**
     * @brief 指定したuuidを持つ EntityAccessor を返す。
     */
    std::shared_ptr<EntityAccessor> getEntityAccessorImpl(const boost::uuids::uuid& uuid) const;
    /**
     * @brief 指定したuuidを持つ EntityAccessor をTにキャストして返す。
     */
    template<std::derived_from<EntityAccessor> T=EntityAccessor, typename... Args>
    std::shared_ptr<T> getEntityAccessor(const boost::uuids::uuid& uuid) const{
        auto ret=std::dynamic_pointer_cast<T>(getEntityAccessorImpl(uuid));
        if(ret){
            return ret;
        }else{
            throw std::runtime_error("EntityAccessor with uuidForAccessor="+boost::uuids::to_string(uuid)+" is not an instance of '"+py::type_id<T>()+"'.");
        }
    }
    /**
     * @brief 指定した EntityAccessor に対応する Entity を返す。
     */
    template<
        std::derived_from<Entity> T=Entity,
        std::derived_from<EntityAccessor> U=EntityAccessor,
        typename... Args
    >
    std::shared_ptr<T> getEntityFromAccessor(const std::shared_ptr<U>& accessor) const{
        try{
            if(accessor && !accessor->entity.expired()){
                auto ret=std::dynamic_pointer_cast<T>(accessor->entity.lock());
                if(ret){
                    return ret;
                }else{
                    throw std::runtime_error(
                        "Entity(id="+accessor->getEntityID().toString()
                        +", fullName="+accessor->getFullName()
                        +", baseName="+accessor->getFactoryBaseName()
                        +", className="+accessor->getFactoryClassName()
                        +", modelName="+accessor->getFactoryModelName()
                        +") is not an instance of '"+py::type_id<T>()+"'."
                    );
                }
            }else{
                throw std::runtime_error("The given accessor is not valid.");
            }
        }catch(std::exception& ex){
            throw std::runtime_error("EntityManager::getEntityFromAccessor() failed. [reason='"+std::string(ex.what())+"']");
        }
    }
    ///@}

    /**
     * @brief Entityが自身によって管理されているかどうかを返す。
     */
    bool isManaged(const std::shared_ptr<const Entity>& entity) const;

    /**
     * @internal
     * @brief [For internal use] 参照としてのシリアライズ
     * 
     * @details EntityManager の参照渡しは自身のUUIDを用いて実現する。
     */
    template<asrc::core::traits::output_archive Archive>
    static void save_as_reference_impl(Archive& ar, const std::shared_ptr<const PtrBaseType>& t){
        if(t){
            ar(::cereal::make_nvp("u",t->getUUID()));
        }else{
            throw std::runtime_error("EntityManager::save_as_reference_impl failed. [reason='The given pointer is nullptr.']");
        }
    }

    /**
     * @internal
     * @brief [For internal use] 参照としてのデシリアライズ
     * 
     * @details EntityManager の参照渡しは自身のUUIDを用いて実現する。
     */
    template<asrc::core::traits::input_archive Archive>
    static std::shared_ptr<PtrBaseType> load_as_reference_impl(Archive& ar){
        try{
            boost::uuids::uuid u;
            ar(::cereal::make_nvp("u",u));
            return EntityManager::getManagerInstance(u);
        }catch(std::exception& ex){
            throw std::runtime_error("EntityManager::load_as_reference_impl failed. [reason='"+std::string(ex.what())+"']");
        }
    }
    /** @name 内部状態のシリアライゼーション (Experimental)
     * 
     * EntityManager の内部状態のシリアライゼーションは、 serializeInternalState() を呼ぶことで行う。
     * その内部処理は次の5段階で実行され、各段階の挙動は派生クラスでカスタマイズ可能である。
     * 
     * (1) serializeBeforeEntityReconstructionInfo()
     *      Entity の再生成前に必要な処理(時刻やmodel configの復元等)を行う。
     * (2) serializeEntityReconstructionInfo()
     *      Entity の再生成を行う。 Entity::initialize() は呼ばれない。
     * (3) serializeAfterEntityReconstructionInfo
     *      Entity の再生成後かつ内部状態の復元前に必要な処理を行う。
     * (4) serializeEntityInternalStates()
     *      Entity の内部状態の復元を行う。ここでも Entity::initialize() は呼ばれない。
     * (5) serializeAfterEntityInternalStates()
     *      Entity の内部状態の復元後に可能となる処理を行う。
     * 
     */
    ///@{
    protected:
    bool isDuringDeserialization; //!< デシリアライズ中であるかどうかを表すフラグ
    std::vector<std::shared_ptr<Entity>> tmp_entities; //!< デシリアライズ時に一時的にインスタンスを保持しておくためのコンテナ
    public:
    /**
     * @brief 内部状態のシリアライズ/デシリアライズを行う。
     * 
     * @param [in,out] archive          入出力対象のArchive
     * @param [in] full                 全状態量を入出力するかどうか。falseの場合、時間変化しない定数等は省略される。ただし、実際に何が入出力されるかはEntity及び派生クラスでの実装依存。
     * @param [in] serializationConfig_ シリアライゼーションに関する挙動を制御するためのコンフィグを与えるための引数。現時点では未実装。
     */
    virtual void serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full, const nl::json& serializationConfig_);
    virtual void serializeBeforeEntityReconstructionInfo(asrc::core::util::AvailableArchiveTypes & archive, bool full, const nl::json& serializationConfig_);
    virtual void serializeEntityReconstructionInfo(asrc::core::util::AvailableArchiveTypes & archive, bool full, const nl::json& serializationConfig_);
    virtual void serializeAfterEntityReconstructionInfo(asrc::core::util::AvailableArchiveTypes & archive, bool full, const nl::json& serializationConfig_);
    virtual void serializeEntityInternalStates(asrc::core::util::AvailableArchiveTypes & archive, bool full, const nl::json& serializationConfig_);
    virtual void serializeAfterEntityInternalStates(asrc::core::util::AvailableArchiveTypes & archive, bool full, const nl::json& serializationConfig_);
    std::shared_ptr<Entity> reconstructEntity(const EntityConstructionInfo& info); //!< [For internal use]
    void reconstructEntities(const std::vector<EntityConstructionInfo>& infos); //!< [For internal use]
    void reconstructEntities(asrc::core::util::AvailableInputArchiveTypes & archive, const std::string& name="entityConstructionInfos"); //!< [For internal use]

    /**
     * @internal
     * @brief [For internal use]
     */
    template<class DataType>
    requires (
        traits::basic_json<DataType>
        || traits::string_like<DataType>
    )
    void reconstructEntities(const DataType& data){
        if constexpr (traits::string_like<DataType>){
            std::istringstream iss(data);
            cereal::PortableBinaryInputArchive archive(iss);
            ::asrc::core::util::AvailableInputArchiveTypes wrappedArchive(std::ref(archive));
            reconstructEntities(wrappedArchive);
        }else if constexpr (std::same_as<DataType,nl::ordered_json>){
            ::asrc::core::util::NLJSONInputArchive archive(data,true);
            ::asrc::core::util::AvailableInputArchiveTypes wrappedArchive(std::ref(archive));
            reconstructEntities(wrappedArchive);
        }else{
            nl::ordered_json oj=data;
            ::asrc::core::util::NLJSONInputArchive archive(oj,true);
            ::asrc::core::util::AvailableInputArchiveTypes wrappedArchive(std::ref(archive));
            reconstructEntities(wrappedArchive);
        }
    }

    EntityConstructionInfo getEntityConstructionInfo(const std::shared_ptr<Entity>& entity,EntityIdentifier id) const; //!< [For internal use]

    /**
     * @internal
     * @brief [For internal use]
     */
    void serializeInternalStateOfEntity(const std::vector<std::pair<EntityIdentifier,bool>>& indices,asrc::core::util::AvailableArchiveTypes & archive, const std::string& name="entityInternalStateSerializers");
    
    /**
     * @internal
     * @brief [For internal use]
     */
    template<class DataType>
    requires (
        traits::basic_json<DataType>
        || traits::string_like<DataType>
    )
    void loadInternalStateOfEntity(const std::vector<std::pair<EntityIdentifier,bool>>& indices,const DataType& data){
        if constexpr (traits::string_like<DataType>){
            std::istringstream iss(data);
            cereal::PortableBinaryInputArchive archive(iss);
            ::asrc::core::util::AvailableArchiveTypes wrappedArchive(std::ref(archive));
            serializeInternalStateOfEntity(indices,wrappedArchive);
        }else if constexpr (std::same_as<DataType,nl::ordered_json>){
            ::asrc::core::util::NLJSONInputArchive archive(data,true);
            ::asrc::core::util::AvailableArchiveTypes wrappedArchive(std::ref(archive));
            serializeInternalStateOfEntity(indices,wrappedArchive);
        }else{
            nl::ordered_json oj=data;
            ::asrc::core::util::NLJSONInputArchive archive(oj,true);
            ::asrc::core::util::AvailableArchiveTypes wrappedArchive(std::ref(archive));
            serializeInternalStateOfEntity(indices,wrappedArchive);
        }
    }
    ///@}
    protected:
    mutable std::recursive_mutex mtx;
    static std::shared_mutex static_mtx;
    std::shared_ptr<Factory> factory; //!< 自身専用のmodel configを保持する Factory インスタンス
    boost::uuids::uuid uuid; //!< EntityManager 自身のuuid
    static std::int32_t _worker_index;
    std::int32_t _vector_index;
    std::int32_t _episode_index;
    std::int32_t nextEpisodeIndependentEntityIndex;
    std::int32_t nextEntityIndex;
    static std::map<std::int32_t,std::weak_ptr<EntityManager>> managerInstances; //!< このプロセス中に存在する EntityManager インスタンスの一覧
    static std::map<std::int32_t,boost::uuids::uuid> uuidOfManagerInstances; //!< vector_indexからUUIDへのマッピング
    static std::map<boost::uuids::uuid,std::int32_t> uuidOfManagerInstancesR; //!< UUIDからvector_indexへのマッピング
    std::map<EntityIdentifier,std::shared_ptr<Entity>> entities; //!< 自身が管理している Entity の一覧
    mutable std::map<EntityIdentifier, std::weak_ptr<Entity>> unmanagedEntities; //!< 自身が管理していないが自身から生成されたunmanagedな Entity の一覧
    mutable std::map<std::string,EntityIdentifier> entityNames; //!< 自身から生成された Entity のfull nameの一覧
    mutable std::map<EntityIdentifier,boost::uuids::uuid> uuidOfEntity; //!< EntityIdentifier からUUIDへのマッピング
    mutable std::map<boost::uuids::uuid,EntityIdentifier> uuidOfEntityR; //!< UUIDから EntityIdentifier へのマッピング
    mutable std::map<EntityIdentifier,boost::uuids::uuid> uuidOfEntityForEntityAccessors; //!< EntityIdentifier から EntityAccessor のUUIDへのマッピング
    mutable std::map<boost::uuids::uuid,EntityIdentifier> uuidOfEntityForEntityAccessorsR; //!< EntityAccessor のUUIDから EntityIdentifier へのマッピング
    mutable std::map<EntityIdentifier,boost::uuids::uuid> uuidOfEntityForTracks; //!< EntityIdentifier から TrackBase 用のUUIDへのマッピング
    mutable std::map<boost::uuids::uuid,EntityIdentifier> uuidOfEntityForTracksR; //!< TrackBase 用のUUIDから EntityIdentifier へのマッピング
    mutable std::vector<EntityIdentifier> entityCreationOrder; //!< Entity の生成順を記録したもの
    mutable std::set<EntityIdentifier> createdEntitiesSinceLastSerialization; //!< 前回のシリアライゼーション実行時以降に新たに生成された Entity の一覧
    mutable std::set<EntityIdentifier> removedEntitiesSinceLastSerialization; //!< 前回のシリアライゼーション実行時以降に削除された Entity の一覧
};

/**
 * @class EntityManagerAccessor
 * @brief EntityManager とその派生クラスのメンバに対するアクセス制限を実現するためのAccessorクラス。
 * 
 * @details pybind11経由でPython側に公開するメンバは原則publicとする必要があるため、protected相当のアクセス制限を実現することが難しい。
 *          Accessorを介したアクセスに限定することで EntityManager インスタンス自身へのアクセスを防止し、
 *          メンバごとのアクセス可否を細かく制御できるようにする。
 * 
 */
ASRC_DECLARE_BASE_REF_CLASS(EntityManagerAccessor)
    friend class EntityManager;
    protected:
    EntityManagerAccessor(const std::shared_ptr<EntityManager>& manager_);
    EntityManagerAccessor(const std::shared_ptr<EntityManagerAccessor>& original_);
    public:
    virtual ~EntityManagerAccessor()=default;
    protected:
    std::weak_ptr<EntityManager> manager;
    public:
    /**
     * @brief 自身が有効なインスタンスを指しているかどうかを返す。
     */
    bool expired() const noexcept;
    /**
     * @brief 自身が指しているインスタンスが引数と同じかどうかを返す。
     */
    bool isSame(const std::shared_ptr<EntityManager>& manager_) const;
    /**
     * @brief 自身が指しているインスタンスが引数と同じかどうかを返す。
     */
    bool isSame(const std::shared_ptr<EntityManagerAccessor>& original_) const;
    /**
     * @brief 指定したuuidを持つ EntityAccessor を返す。
     */
    template<std::derived_from<EntityAccessor> T=Entity, typename... Args>
    std::shared_ptr<T> getEntityAccessor(const boost::uuids::uuid& uuid){
        return manager.lock()->getEntityAccessor<T>(uuid);
    }
};

ASRC_DECLARE_BASE_REF_TRAMPOLINE(EntityManagerAccessor)
    // No virtual function for the base class.
};

/**
 * @class Entity
 * @brief Factory により生成・管理されるオブジェクトの基底クラス。
 * 
 */
ASRC_DECLARE_BASE_REF_CLASS(Entity)
/**/
    private:
    friend class Factory;
    friend class EntityConstructionInfo;
    friend class EntityManager;
    friend class EntityAccessor;
    EntityIdentifier entityID;
    boost::uuids::uuid uuid; //!< Entity 自身を識別するUUID
    boost::uuids::uuid uuidForAccessor;//!< EntityAccessor へのアクセス権を管理するためのUUID
    boost::uuids::uuid uuidForTrack; //!< 同一性の識別のためのUUID
    FactoryHelperChain factoryHelperChain; //!< 自身が生成された際の FactoryHelperChain
    std::string factoryBaseName; //!< Factory におけるbaseNameが格納される。
    std::string factoryClassName; //!< Factory におけるクラス名が格納される。
    std::string factoryModelName; //!< Factoryにおけるモデル名が格納される。
    std::string fullName; //!< インスタンス名。instanceConfigで外から指定すること。生成後の変更は不可。getterでアクセスする。
    bool isMonitored; //!< EntityManager にshared_ptr又はweak_ptrとして保持されているか否か。falseの場合は EntityManager から取り除かれていることを意味する。
    std::shared_ptr<EntityAccessor> accessor; //!< 自身へのアクセス制御を行う EntityAccessor
    std::weak_ptr<EntityManager> entityManager; //!< Entity の生成用。privateメンバとして保持し、これを利用するメンバ関数をpublicで定義することで EntityManager の基幹機能へのアクセス制限を実現する。
    public:
    std::shared_ptr<EntityManagerAccessor> manager; //!< EntityManager の派生クラスのメンバへのアクセス用変数。派生クラスではこの変数を必要に応じダウンキャストして使用する。
    static const std::string baseName; //!< Factory におけるこのクラスのグループ名(base name)
    const bool isDummy;//!< ダミーとして生成されたか否か。コンストラクタ引数に両方null(None)を与えた際にTrueとなる。
    std::mt19937 randomGen; //!< 乱数生成器
    nl::json modelConfig; //!< 生成時に与えられたmodel config
    nl::json instanceConfig; //!< 生成時に与えられたinstance config
    std::map<SimPhase,std::uint64_t> firstTick; //!< 各phaseの初回処理時刻(tick)
    std::map<SimPhase,std::uint64_t> interval; //!< 各phaseの処理周期(tick)
    //constructors & destructor
    /**
     * @brief model configとinstance configの2引数をとるコンストラクタ。
     * 
     * @attention 両方にnullを入れた場合、ダミーとして扱うこととし、コンストラクタの処理をバイパスする等してエラー無くインスタンス生成ができるようにすること。
     * 
     * @attention コンストラクタ内では EntityManager にアクセスできないため、派生クラスでは可能な限り initialize() の方に初期化処理を記述することを推奨する。
     */
    Entity(const nl::json& modelConfig_,const nl::json& instanceConfig_);
    virtual ~Entity();
    private:
    void setFactoryCreationMetadata(const std::string& baseName, const std::string& className, const std::string& modelName, const FactoryHelperChain& helperChain); //!< [For internal use]
    void setEntityManager(const std::shared_ptr<EntityManager>& em); //!< [For internal use]
    void setEntityID(const EntityIdentifier& id); //!< [For internal use]
    void setFullName(const std::string& fullName); //!< [For internal use]
    boost::uuids::uuid getUUID() const;//!< Entity 自身を識別するUUIDを返す。
    protected:
    /**
     * @brief EntityManager から与えられた EntityManagerAccessor を自身のメンバに代入する。
     * 
     * @note virtualとしたのはC++側の派生クラスにおいて、都度ダウンキャストしないで済むような新たなキャスト済変数を追加することを許容するため。
     */
    virtual void setManagerAccessor(const std::shared_ptr<EntityManagerAccessor>& ema);
    public:
    /**
     * @brief 派生クラスの初期化処理のうち、自身を生成した親以外のEntityに依存しないものを記述する。
     * 
     * @details 自身を生成した EntityManager から、 EntityManager::registerEntity() の中で自動的に呼ばれる。
     */
    virtual void initialize();
    /**
     * @brief 子Entityを持つならばここで生成処理を記述する。
     * 
     * @details 自身を生成した EntityManager から、 EntityManager::registerEntity() の中で initialize() の直後に呼ばれる。
     */
    virtual void makeChildren();
    /**
     * @brief 派生クラスの初期化処理のうち、他の Entity に依存するものがあれば記述する。
     * 
     * @details SimulationManager::reset() において、全 Entity の生成後に呼ばれる。
     */
    virtual void validate();
    EntityIdentifier getEntityID() const; //!< 自身の EntityIdentifier を返す。
    boost::uuids::uuid getUUIDForAccessor() const; //!< EntityAccessor へのアクセス権を管理するためのUUIDを返す。
    boost::uuids::uuid getUUIDForTrack() const; //!< TrackBase における同一性の識別のためのUUIDを返す。
    bool isManaged() const; //!< EntityManager にshared_ptrとして保持されているか否か。falseの場合は他の場所でshared_ptrとして管理されていることを表す。
    bool isEpisodic() const; //!< 自身の寿命がエピソードに紐付けられているかどうかを返す。
    void seed(const unsigned int& seed_); //!< randomGen のシードを設定する。
    std::string getFactoryBaseName() const; //!< Factory におけるbaseNameを返す。
    std::string getFactoryClassName() const; //!< Factory におけるクラス名を返す。
    std::string getFactoryModelName() const; //!< Factory におけるモデル名を返す。
    FactoryHelperChain getFactoryHelperChain() const; //!< 自身が生成された際に Factory で使用された FactoryHelperChain を返す。
    std::string resolveClassName(const std::string& baseName, const std::string& className) const; //!< 自身の FactoryHelperChain で見える範囲でクラス名を検索し、その完全なクラス名を返す。
    std::string resolveModelName(const std::string& baseName, const std::string& modelName) const; //!< 自身の FactoryHelperChain で見える範囲でモデル名を検索し、その完全なモデル名を返す。
    std::filesystem::path resolveFilePath(const std::string& path) const; //!< 自身の FactoryHelperChain で見える範囲でファイルパスを解決する。
    std::filesystem::path resolveFilePath(const std::filesystem::path& path) const; //!< 自身の FactoryHelperChain で見える範囲でファイルパスを解決する。
    virtual std::string getFullName() const; //!< 自身のfull nameを返す。
    /**
     * @internal
     * @brief [For internal use] 自身へのアクセス制御を行う EntityAccessor を返す。
     *        派生クラスで必要に応じて独自の EntityAccessor を定義してここで生成する。
     * @attention これを外から直接呼び出した場合はメンバとして保持されないので、weak_ptrで受け取ることができない。
     *       原則として、テンプレート関数の Entity::getAccessor() の方を呼び出すことを推奨する。
     */
    virtual std::shared_ptr<EntityAccessor> getAccessorImpl();
    /**
     * @brief 自身へのアクセス制御を行う EntityAccessor を返す。
     * 
     * @details こちらで呼び出した場合はメンバとして保持している EntityAccessor インスタンスを返すので、weak_ptrで直接受け取ってもよい。
     */
    template<std::derived_from<EntityAccessor> T=EntityAccessor>
    std::shared_ptr<T> getAccessor(){
        if(!accessor){
            accessor=getAccessorImpl();
        }
        auto ret=std::dynamic_pointer_cast<T>(accessor);
        if(ret){
            return ret;
        }else{
            throw std::runtime_error(
                "Entity(id="+getEntityID().toString()
                +", fullName="+getFullName()
                +", baseName="+getFactoryBaseName()
                +", className="+getFactoryClassName()
                +", modelName="+getFactoryModelName()
                +") is not accessible via '"+py::type_id<T>()+"'."
            );
        }
    }
    virtual std::uint64_t getFirstTick(const SimPhase& phase) const; //!< phaseの初回処理時刻(tick)を返す。
    virtual std::uint64_t getInterval(const SimPhase& phase) const; //!< phaseの処理周期(tick)を返す。
    virtual std::uint64_t getNextTick(const SimPhase& phase,const std::uint64_t now); //!< now以降のphaseの次回処理時刻(tick)を返す。
    virtual bool isTickToRunPhaseFunc(const SimPhase& phase,const std::uint64_t now); //!< nowが自身にとってphaseを処理する時刻(tick)に該当するかどうかを返す。
    virtual bool isSame(const std::shared_ptr<Entity>&  other); //!< 自身とotherが同一かどうかを返す。

    /**
     * @internal
     * @brief [For internal use] 参照としてのシリアライズ
     * 
     * @details Entity の参照渡しは自身を生成した EntityManager と自身のUUIDの組を用いて実現する。
     */
    template<asrc::core::traits::output_archive Archive>
    static void save_as_reference_impl(Archive& ar, const std::shared_ptr<const PtrBaseType>& t){
        if(t){
            if(t->entityManager.expired()){
                throw std::runtime_error("Entity::save_as_reference_impl failed. [reason='The given Entity was not created by an EntityManager or the EntityManager is not alive.']");
            }else{
                auto em=t->entityManager.lock();
                ar(
                    ::cereal::make_nvp("m",em->getUUID()),
                    ::cereal::make_nvp("e",t->getUUID())
                );
            }
        }else{
            throw std::runtime_error("Entity::save_as_reference_impl failed. [reason='The given pointer is nullptr.']");
        }
    }
    /**
     * @internal
     * @brief [For internal use] 参照としてのデシリアライズ
     * 
     * @details Entity の参照渡しは自身を生成した EntityManager と自身のUUIDの組を用いて実現する。
     */
    template<asrc::core::traits::input_archive Archive>
    static std::shared_ptr<PtrBaseType> load_as_reference_impl(Archive& ar){
        try{
            boost::uuids::uuid m,e;
            ar(
                CEREAL_NVP(m),
                CEREAL_NVP(e)
            );
            auto entityManager = EntityManager::getManagerInstance(m);
            if(entityManager){
                EntityIdentifier id=entityManager->uuidOfEntityR.at(e);
                auto ret=entityManager->getEntityByID(id);
                if(!ret){
                    throw std::runtime_error("Entity with id="+id.toString()+" was not found.");
                }
                return ret;
            }else{
                throw std::runtime_error("EntityManager instance with uuid="+boost::uuids::to_string(m)+" was not found.");
            }
        }catch(std::exception& ex){
            throw std::runtime_error("Entity::load_as_reference_impl failed. [reason='"+std::string(ex.what())+"']");
        }
    }

    /**
     * @brief EntityManager::resolveModelConfig を呼び出し、[className, modelName, modelConfig, FactoryHelperChain]の組を返す。
     * 
     * @attention 許容される引数は Factory::resolveModelConfig が受け入れるもの全て。
     *       ただし、末尾にfactoryHelperChainとcallerの2つが自動的に付加されるのでそれらを除くこと。
     */
    template<typename... Args>
    std::tuple<std::string,std::string,nl::json,FactoryHelperChain> resolveModelConfig(const std::string& baseName,Args&&... args) const{
        if(entityManager.expired()){
            throw std::runtime_error("Entity::resolveModelConfig can be called iff the Entity was created by an EntityManager which is still alive.");
        }else{
            auto ret=entityManager.lock()->resolveModelConfig(baseName,std::forward<Args>(args)...,getFactoryHelperChain(),true);
            if(!Factory::checkPermissionForGet(this->shared_from_this(),baseName)){
                throw std::runtime_error("Entity::resolveModelConfig is not allowed for baseName='"+baseName+"' from '"+getFactoryBaseName()+"'.");
            }
            return ret;
        }
    }
    /**
     * @brief EntityManager::resolveModelConfig を呼び出し、[className, modelName, modelConfig, FactoryHelperChain]の組を返す。
     * 
     * @attention 許容される引数は Factory::resolveModelConfig が受け入れるもの全て。
     *       ただし、末尾にfactoryHelperChainとcallerの2つが自動的に付加されるのでそれらを除くこと。
     */
    std::tuple<std::string,std::string,nl::json,FactoryHelperChain> resolveModelConfig(const nl::json& j) const{
        if(entityManager.expired()){
            throw std::runtime_error("Entity::resolveModelConfig can be called iff the Entity was created by an EntityManager which is still alive.");
        }else{
            auto ret=entityManager.lock()->resolveModelConfig(j,getFactoryHelperChain(),true);
            std::string baseName=j.at("baseName");
            if(!Factory::checkPermissionForGet(this->shared_from_this(),baseName)){
                throw std::runtime_error("Entity::resolveModelConfig is not allowed for baseName='"+baseName+"' from '"+getFactoryBaseName()+"'.");
            }
            return ret;
        }
    }

    /**
     * @brief unmanagedな Entity を生成する。 EntityManager::createUnmanagedEntity を呼び出す。
     * 
     * @details
     *  - 自身が EntityManager から生成されていなければ利用不可。
     *  - 引数の末尾に自身がcallerとして付加されるので、権限の無い種類の Entity は生成できない。
     * 
     */
    template<std::derived_from<Entity> T=Entity, typename... Args>
    std::shared_ptr<T> createUnmanagedEntity(Args&&... args) const{
        if(entityManager.expired()){
            throw std::runtime_error("Entity::createUnmanagedEntity can be called iff the Entity was created by an EntityManager which is still alive.");
        }else{
            return entityManager.lock()->createUnmanagedEntity<T>(std::forward<Args>(args)..., this->shared_from_this());
        }
    }

    /**
     * @brief unmanagedな Entity を生成する。 EntityManager::createUnmanagedEntityByClassName を呼び出す。
     * 
     * @details
     *  - 自身が EntityManager から生成されていなければ利用不可。
     *  - 引数の末尾に自身がcallerとして付加されるので、権限の無い種類の Entity は生成できない。
     * 
     */
    template<std::derived_from<Entity> T=Entity, typename... Args>
    std::shared_ptr<T> createUnmanagedEntityByClassName(Args&&... args) const{
        if(entityManager.expired()){
            throw std::runtime_error("Entity::createUnmanagedEntityByClassName can be called iff the Entity was created by an EntityManager which is still alive.");
        }else{
            return entityManager.lock()->createUnmanagedEntityByClassName<T>(std::forward<Args>(args)..., this->shared_from_this());
        }
    }

    /**
     * @brief unmanagedな Entity を検索し、無ければ生成する。 EntityManager::createOrGetUnmanagedEntity を呼び出す。
     * 
     * @details
     *  - 自身が EntityManager から生成されていなければ利用不可。
     *  - 引数の末尾に自身がcallerとして付加されるので、権限の無い種類の Entity は生成できない。
     * 
     */
    template<std::derived_from<Entity> T=Entity, typename... Args>
    std::shared_ptr<T> createOrGetUnmanagedEntity(Args&&... args) const{
        if(entityManager.expired()){
            throw std::runtime_error("Entity::createOrGetUnmanagedEntity can be called iff the Entity was created by an EntityManager which is still alive.");
        }else{
            return entityManager.lock()->createOrGetUnmanagedEntity<T>(std::forward<Args>(args)..., this->shared_from_this());
        }
    }

    /**
     * @brief managedな Entity を生成する。 EntityManager::createManagedEntity を呼び出す。
     * 
     * @details
     *  - 自身が EntityManager から生成されていなければ利用不可。
     *  - 引数の末尾に自身がcallerとして付加されるので、権限の無い種類の Entity は生成できない。
     * 
     */
    template<std::derived_from<Entity> T=Entity, typename... Args>
    std::shared_ptr<T> createEntity(Args&&... args) const{
        if(entityManager.expired()){
            throw std::runtime_error("Entity::createEntity can be called iff the Entity was created by an EntityManager which is still alive.");
        }else{
            return entityManager.lock()->createEntity<T>(std::forward<Args>(args)..., this->shared_from_this());
        }
    }

    /**
     * @brief managedな Entity を生成する。 EntityManager::createManagedEntityByClassName を呼び出す。
     * 
     * @details
     *  - 自身が EntityManager から生成されていなければ利用不可。
     *  - 引数の末尾に自身がcallerとして付加されるので、権限の無い種類の Entity は生成できない。
     * 
     */
    template<std::derived_from<Entity> T=Entity, typename... Args>
    std::shared_ptr<T> createEntityByClassName(Args&&... args) const{
        if(entityManager.expired()){
            throw std::runtime_error("Entity::createEntityByClassName can be called iff the Entity was created by an EntityManager which is still alive.");
        }else{
            return entityManager.lock()->createEntityByClassName<T>(std::forward<Args>(args)..., this->shared_from_this());
        }
    }

    /**
     * @brief managedな Entity を検索し、無ければ生成する。 EntityManager::createOrGetManagedEntity を呼び出す。
     * 
     * @details
     *  - 自身が EntityManager から生成されていなければ利用不可。
     *  - 引数の末尾に自身がcallerとして付加されるので、権限の無い種類の Entity は生成できない。
     * 
     */
    template<std::derived_from<Entity> T=Entity, typename... Args>
    std::shared_ptr<T> createOrGetEntity(Args&&... args) const{
        if(entityManager.expired()){
            throw std::runtime_error("Entity::createOrGetEntity can be called iff the Entity was created by an EntityManager which is still alive.");
        }else{
            return entityManager.lock()->createOrGetEntity<T>(std::forward<Args>(args)..., this->shared_from_this());
        }
    }

    /**
     * @brief 指定した EntityIdentifier を持つ Entity をTにキャストして返す。
     * 
     * @details
     *  - 見つからなかった場合はnullptrを返す。
     *  - 変換不可能だった場合もnullptrを返す。
     *  - 自身が EntityManager から生成されていなければ利用不可。
     *  - 引数の末尾に自身がcallerとして付加されるので、権限の無い種類の Entity は生成できない。
     */
    template<std::derived_from<Entity> T=Entity>
    std::shared_ptr<T> getEntityByID(const EntityIdentifier& id) const{
        if(entityManager.expired()){
            throw std::runtime_error("Entity::getEntityByID can be called iff the Entity was created by an EntityManager which is still alive.");
        }else{
            return entityManager.lock()->getEntityByID<T>(id, this->shared_from_this());
        }
    }
    /**
     * @brief 指定したfull nameを持つ Entity をTにキャストして返す。
     * 
     * @details
     *  - 見つからなかった場合はnullptrを返す。
     *  - 変換不可能だった場合もnullptrを返す。
     *  - 自身が EntityManager から生成されていなければ利用不可。
     *  - 引数の末尾に自身がcallerとして付加されるので、権限の無い種類の Entity は生成できない。
     */
    template<std::derived_from<Entity> T=Entity>
    std::shared_ptr<T> getEntityByName(const std::string& name) const{
        if(entityManager.expired()){
            throw std::runtime_error("Entity::getEntityByName can be called iff the Entity was created by an EntityManager which is still alive.");
        }else{
            return entityManager.lock()->getEntityByName<T>(name, this->shared_from_this());
        }
    }
    /**
     * @brief 内部状態のシリアライゼーションを行う。 (Experimental)
     * 
     * @param [in,out] archive          入出力対象のArchive
     * @param [in] full                 全状態量を入出力するかどうか。falseの場合、時間変化しない定数等は省略される。ただし、実際に何が入出力されるかは Entity 及び派生クラスでの実装依存。
     * 
     * @attention 
     *  - 派生クラスでオーバーライドして実際の処理を記述すること。
     *  - 各 Entity のデシリアライズ順序はインスタンス生成が古い順であり各 Entity 側からは特定できないため、他の Entity の内部状態に依存する処理は順序が明らかなものを除き避けること。
     */
    virtual void serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full);
};

/**
 * @class EntityAccessor
 * @brief Entity とその派生クラスのメンバに対するアクセス制限を実現するためのAccessorクラス。
 * 
 * @details pybind11に公開するメンバは原則publicとする必要があるため、protected相当のアクセス制限を実現することが難しい。
 *          Accessorを介したアクセスに限定することで Entity インスタンス自身へのアクセスを防止し、
 *          メンバごとのアクセス可否を細かく制御できるようにする。
 * 
 */
ASRC_DECLARE_BASE_REF_CLASS(EntityAccessor)
    friend class EntityManager;
    public:
    EntityAccessor(const std::shared_ptr<Entity>& entity_);
    virtual ~EntityAccessor()=default;
    /**
     * @brief 自身が有効なインスタンスを指しているかどうかを返す。
     */
    bool expired() const noexcept;
    /**
     * @brief 自身が指しているインスタンスが引数と同じかどうかを返す。
     */
    bool isSame(const std::shared_ptr<Entity>& entity_) const;
    /**
     * @brief 自身が指しているインスタンスが引数と同じかどうかを返す。
     */
    bool isSame(const std::shared_ptr<EntityAccessor>& original_) const;
    EntityIdentifier getEntityID() const; //!< 自身が指しているインスタンスの EntityIdentifier を返す。
    boost::uuids::uuid getUUIDForAccessor() const; //!< EntityAccessor へのアクセス権を管理するためのUUIDを返す。
    boost::uuids::uuid getUUIDForTrack() const; //!< TrackBase における同一性の識別のためのUUIDを返す。
    std::string getFactoryBaseName() const; //!< Factory におけるbaseNameを返す。
    std::string getFactoryClassName() const; //!< Factory におけるクラス名を返す。
    std::string getFactoryModelName() const; //!< Factory におけるモデル名を返す。
    std::string getFullName() const; //!< 自身が指しているインスタンスのfull nameを返す。
    /**
     * @brief 自身が指しているインスタンスがクラスTのインスタンスかどうかを返す。
     */
    template<class T>
    bool isinstance(){
        return util::isinstance<T>(entity);
    }
    /**
     * @brief 自身が指しているインスタンスがclsのインスタンスかどうかを返す。
     */
    bool isinstancePY(const py::object& cls);
    /**
     * @internal
     * @brief [For internal use] 参照としてのシリアライズ
     * 
     * @details EntityAccessor の参照渡しは自身を生成した EntityManager と自身のUUIDの組を用いて実現する。
     */
    template<asrc::core::traits::output_archive Archive>
    static void save_as_reference_impl(Archive& ar, const std::shared_ptr<const PtrBaseType>& t){
        if(t){
            if(t->entity.expired()){
                throw std::runtime_error("EntityAccessor::save_as_reference_impl failed. [reason='The Entity is expired.']");
            }else{
                auto e=t->entity.lock();
                if(e->entityManager.expired()){
                    throw std::runtime_error("EntityAccessor::save_as_reference_impl failed. [reason='The EntityManager is expired.']");
                }
                auto em=e->entityManager.lock();
                ar(
                    ::cereal::make_nvp("m",em->getUUID()),
                    ::cereal::make_nvp("e",e->getUUIDForAccessor())
                );
            }
        }else{
            throw std::runtime_error("EntityAccessor::save_as_reference_impl failed. [reason='The given pointer is nullptr.']");
        }
    }
    /**
     * @internal
     * @brief [For internal use] 参照としてのデシリアライズ
     * 
     * @details EntityAccessor の参照渡しは自身を生成した EntityManager と自身のUUIDの組を用いて実現する。
     */
    template<asrc::core::traits::input_archive Archive>
    static std::shared_ptr<PtrBaseType> load_as_reference_impl(Archive& ar){
        try{
            boost::uuids::uuid m,e;
            ar(
                CEREAL_NVP(m),
                CEREAL_NVP(e)
            );
            auto entityManager = EntityManager::getManagerInstance(m);
            if(entityManager){
                auto ret=entityManager->getEntityAccessor(e);
                if(!ret){
                    throw std::runtime_error("Entity with uuidForAccessor="+boost::uuids::to_string(e)+" was not found.");
                }
                return ret;
            }else{
                throw std::runtime_error("EntityManager instance with uuid="+boost::uuids::to_string(m)+" was not found.");
            }
        }catch(std::exception& ex){
            throw std::runtime_error("EntityAccessor::load_as_reference_impl failed. [reason='"+std::string(ex.what())+"']");
        }
    }
    protected:
    std::weak_ptr<Entity> entity;
};

ASRC_DECLARE_BASE_REF_TRAMPOLINE(EntityManager)
    virtual void purge() override{
            PYBIND11_OVERRIDE(void,Base,purge);
    }
    virtual std::shared_ptr<Entity> createEntityImpl(bool isManaged, bool isEpisodic, const std::string& baseName,const std::string& className,const std::string& modelName,const nl::json& modelConfig,const nl::json& instanceConfig, const FactoryHelperChain& helperChain, const std::shared_ptr<const Entity>& caller) override{
        PYBIND11_OVERRIDE(std::shared_ptr<Entity>,Base,createEntityImpl,isManaged,isEpisodic,baseName,className,modelName,modelConfig,instanceConfig,helperChain,caller);
    }
    virtual std::shared_ptr<Entity> createUnmanagedEntityOverloader(bool isEpisodic, const std::string& baseName,const std::string& className,const std::string& modelName,const nl::json& modelConfig,const nl::json& instanceConfig, const FactoryHelperChain& helperChain=FactoryHelperChain()) override{
        PYBIND11_OVERRIDE(std::shared_ptr<Entity>,Base,createUnmanagedEntityOverloader,isEpisodic,baseName,className,modelName,modelConfig,instanceConfig,helperChain);
    }
    virtual std::shared_ptr<Entity> createUnmanagedEntityOverloader(bool isEpisodic, const std::string& baseName,const std::string& className,const std::string& modelName,const nl::json& modelConfig,const nl::json& instanceConfig, const std::shared_ptr<const Entity>& caller) override{
        PYBIND11_OVERRIDE(std::shared_ptr<Entity>,Base,createUnmanagedEntityOverloader,isEpisodic,baseName,className,modelName,modelConfig,instanceConfig,caller);
    }
    virtual std::shared_ptr<Entity> createUnmanagedEntityOverloader(bool isEpisodic, const std::string& baseName,const std::string& modelName,const nl::json& instanceConfig, const FactoryHelperChain& helperChain=FactoryHelperChain()) override{
        PYBIND11_OVERRIDE(std::shared_ptr<Entity>,Base,createUnmanagedEntityOverloader,isEpisodic,baseName,modelName,instanceConfig,helperChain);
    }
    virtual std::shared_ptr<Entity> createUnmanagedEntityOverloader(bool isEpisodic, const std::string& baseName,const std::string& modelName,const nl::json& instanceConfig, const std::shared_ptr<const Entity>& caller) override{
        PYBIND11_OVERRIDE(std::shared_ptr<Entity>,Base,createUnmanagedEntityOverloader,isEpisodic,baseName,modelName,instanceConfig,caller);
    }
    virtual std::shared_ptr<Entity> createUnmanagedEntityByClassNameOverloader(bool isEpisodic, const std::string& baseName,const std::string& className,const nl::json& modelConfig,const nl::json& instanceConfig, const FactoryHelperChain& helperChain=FactoryHelperChain()) override{
        PYBIND11_OVERRIDE(std::shared_ptr<Entity>,Base,createUnmanagedEntityByClassNameOverloader,isEpisodic,baseName,className,modelConfig,instanceConfig,helperChain);
    }
    virtual std::shared_ptr<Entity> createUnmanagedEntityByClassNameOverloader(bool isEpisodic, const std::string& baseName,const std::string& className,const nl::json& modelConfig,const nl::json& instanceConfig, const std::shared_ptr<const Entity>& caller) override{
        PYBIND11_OVERRIDE(std::shared_ptr<Entity>,Base,createUnmanagedEntityByClassNameOverloader,isEpisodic,baseName,className,modelConfig,instanceConfig,caller);
    }
    virtual std::shared_ptr<Entity> createEntityImpl(bool isManaged, const nl::json& j, const FactoryHelperChain& helperChain, const std::shared_ptr<const Entity>& caller) override{
        PYBIND11_OVERRIDE(std::shared_ptr<Entity>,Base,createEntityImpl,isManaged,j,helperChain,caller);
    }
    virtual std::shared_ptr<Entity> createUnmanagedEntityOverloader(const nl::json& j, const FactoryHelperChain& helperChain=FactoryHelperChain()) override{
        PYBIND11_OVERRIDE(std::shared_ptr<Entity>,Base,createUnmanagedEntityOverloader,j,helperChain);
    }
    virtual std::shared_ptr<Entity> createUnmanagedEntityOverloader(const nl::json& j, const std::shared_ptr<const Entity>& caller) override{
        PYBIND11_OVERRIDE(std::shared_ptr<Entity>,Base,createUnmanagedEntityOverloader,j,caller);
    }
    virtual std::shared_ptr<Entity> createOrGetEntityImpl(bool isManaged, const nl::json& j, const FactoryHelperChain& helperChain, const std::shared_ptr<const Entity>& caller) override{
        PYBIND11_OVERRIDE(std::shared_ptr<Entity>,Base,createOrGetEntityImpl,isManaged,j,helperChain,caller);
    }
    virtual std::shared_ptr<Entity> createOrGetUnmanagedEntityOverloader(const nl::json& j, const FactoryHelperChain& helperChain=FactoryHelperChain()) override{
        PYBIND11_OVERRIDE(std::shared_ptr<Entity>,Base,createOrGetUnmanagedEntityOverloader,j,helperChain);
    }
    virtual std::shared_ptr<Entity> createOrGetUnmanagedEntityOverloader(const nl::json& j, const std::shared_ptr<const Entity>& caller) override{
        PYBIND11_OVERRIDE(std::shared_ptr<Entity>,Base,createOrGetUnmanagedEntityOverloader,j,caller);
    }
    virtual std::shared_ptr<Entity> createEntityOverloader(bool isEpisodic, const std::string& baseName,const std::string& className,const std::string& modelName,const nl::json& modelConfig,const nl::json& instanceConfig, const FactoryHelperChain& helperChain=FactoryHelperChain()) override{
        PYBIND11_OVERRIDE(std::shared_ptr<Entity>,Base,createEntityOverloader,isEpisodic,baseName,className,modelName,modelConfig,instanceConfig,helperChain);
    }
    virtual std::shared_ptr<Entity> createEntityOverloader(bool isEpisodic, const std::string& baseName,const std::string& className,const std::string& modelName,const nl::json& modelConfig,const nl::json& instanceConfig, const std::shared_ptr<const Entity>& caller) override{
        PYBIND11_OVERRIDE(std::shared_ptr<Entity>,Base,createEntityOverloader,isEpisodic,baseName,className,modelName,modelConfig,instanceConfig,caller);
    }
    virtual std::shared_ptr<Entity> createEntityOverloader(bool isEpisodic, const std::string& baseName,const std::string& modelName,const nl::json& instanceConfig, const FactoryHelperChain& helperChain=FactoryHelperChain()) override{
        PYBIND11_OVERRIDE(std::shared_ptr<Entity>,Base,createEntityOverloader,isEpisodic,baseName,modelName,instanceConfig,helperChain);
    }
    virtual std::shared_ptr<Entity> createEntityOverloader(bool isEpisodic, const std::string& baseName,const std::string& modelName,const nl::json& instanceConfig, const std::shared_ptr<const Entity>& caller) override{
        PYBIND11_OVERRIDE(std::shared_ptr<Entity>,Base,createEntityOverloader,isEpisodic,baseName,modelName,instanceConfig,caller);
    }
    virtual std::shared_ptr<Entity> createEntityByClassNameOverloader(bool isEpisodic, const std::string& baseName,const std::string& className,const nl::json& modelConfig,const nl::json& instanceConfig, const FactoryHelperChain& helperChain=FactoryHelperChain()) override{
        PYBIND11_OVERRIDE(std::shared_ptr<Entity>,Base,createEntityByClassNameOverloader,isEpisodic,baseName,className,modelConfig,instanceConfig,helperChain);
    }
    virtual std::shared_ptr<Entity> createEntityByClassNameOverloader(bool isEpisodic, const std::string& baseName,const std::string& className,const nl::json& modelConfig,const nl::json& instanceConfig, const std::shared_ptr<const Entity>& caller) override{
        PYBIND11_OVERRIDE(std::shared_ptr<Entity>,Base,createEntityByClassNameOverloader,isEpisodic,baseName,className,modelConfig,instanceConfig,caller);
    }
    virtual std::shared_ptr<Entity> createEntityOverloader(const nl::json& j, const FactoryHelperChain& helperChain=FactoryHelperChain()) override{
        PYBIND11_OVERRIDE(std::shared_ptr<Entity>,Base,createEntityOverloader,j,helperChain);
    }
    virtual std::shared_ptr<Entity> createEntityOverloader(const nl::json& j, const std::shared_ptr<const Entity>& caller) override{
        PYBIND11_OVERRIDE(std::shared_ptr<Entity>,Base,createEntityOverloader,j,caller);
    }
    virtual std::shared_ptr<Entity> createOrGetEntityOverloader(const nl::json& j, const FactoryHelperChain& helperChain=FactoryHelperChain()) override{
        PYBIND11_OVERRIDE(std::shared_ptr<Entity>,Base,createOrGetEntityOverloader,j,helperChain);
    }
    virtual std::shared_ptr<Entity> createOrGetEntityOverloader(const nl::json& j, const std::shared_ptr<const Entity>& caller) override{
        PYBIND11_OVERRIDE(std::shared_ptr<Entity>,Base,createOrGetEntityOverloader,j,caller);
    }
    virtual void registerEntity(bool isManaged,const std::shared_ptr<Entity>& entity) override{
        PYBIND11_OVERRIDE(void,Base,registerEntity,isManaged,entity);
    }
    virtual std::shared_ptr<EntityManagerAccessor> createAccessorFor(const std::shared_ptr<const Entity>& entity) override{
        PYBIND11_OVERRIDE(std::shared_ptr<EntityManagerAccessor>,Base,createAccessorFor,entity);
    }
    virtual void cleanupEpisode(bool increment=true) override{
        PYBIND11_OVERRIDE(void,Base,cleanupEpisode,increment);
    }
    virtual void removeEntity(const std::shared_ptr<Entity>& entity) override{
        PYBIND11_OVERRIDE(void,Base,removeEntity,entity);
    }
    virtual void removeEntity(const std::weak_ptr<Entity>& entity,const EntityIdentifier& id,const std::string& name) override{
        PYBIND11_OVERRIDE(void,Base,removeEntity,entity,id,name);
    }
    virtual void serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full, const nl::json& serializationConfig_) override{
        PYBIND11_OVERRIDE(void,Base,serializeInternalState,archive,full,serializationConfig_);
    }
    virtual void serializeBeforeEntityReconstructionInfo(asrc::core::util::AvailableArchiveTypes & archive, bool full, const nl::json& serializationConfig_) override{
        PYBIND11_OVERRIDE(void,Base,serializeBeforeEntityReconstructionInfo,archive,full,serializationConfig_);
    }
    virtual void serializeEntityReconstructionInfo(asrc::core::util::AvailableArchiveTypes & archive, bool full, const nl::json& serializationConfig_) override{
        PYBIND11_OVERRIDE(void,Base,serializeEntityReconstructionInfo,archive,full,serializationConfig_);
    }
    virtual void serializeAfterEntityReconstructionInfo(asrc::core::util::AvailableArchiveTypes & archive, bool full, const nl::json& serializationConfig_) override{
        PYBIND11_OVERRIDE(void,Base,serializeAfterEntityReconstructionInfo,archive,full,serializationConfig_);
    }
    virtual void serializeEntityInternalStates(asrc::core::util::AvailableArchiveTypes & archive, bool full, const nl::json& serializationConfig_) override{
        PYBIND11_OVERRIDE(void,Base,serializeEntityInternalStates,archive,full,serializationConfig_);
    }
    virtual void serializeAfterEntityInternalStates(asrc::core::util::AvailableArchiveTypes & archive, bool full, const nl::json& serializationConfig_) override{
        PYBIND11_OVERRIDE(void,Base,serializeAfterEntityInternalStates,archive,full,serializationConfig_);
    }
};

ASRC_DECLARE_BASE_REF_TRAMPOLINE(Entity)
    virtual std::string getFullName() const override{
        PYBIND11_OVERRIDE(std::string,Base,getFullName);
    }
    virtual std::shared_ptr<EntityAccessor> getAccessorImpl() override{
        PYBIND11_OVERRIDE(std::shared_ptr<EntityAccessor>,Base,getAccessorImpl);
    }
    virtual void initialize() override{
        PYBIND11_OVERRIDE(void,Base,initialize);
    }
    virtual void makeChildren() override{
        PYBIND11_OVERRIDE(void,Base,makeChildren);
    }
    virtual void validate() override{
        PYBIND11_OVERRIDE(void,Base,validate);
    }
    virtual std::uint64_t getFirstTick(const SimPhase& phase) const override{
        PYBIND11_OVERRIDE(std::uint64_t,Base,getFirstTick,phase);
    }
    virtual std::uint64_t getInterval(const SimPhase& phase) const override{
        PYBIND11_OVERRIDE(std::uint64_t,Base,getInterval,phase);
    }
    virtual std::uint64_t getNextTick(const SimPhase& phase,const std::uint64_t now) override{
        PYBIND11_OVERRIDE(std::uint64_t,Base,getNextTick,phase,now);
    }
    virtual bool isTickToRunPhaseFunc(const SimPhase& phase,const std::uint64_t now) override{
        PYBIND11_OVERRIDE(bool,Base,isTickToRunPhaseFunc,phase,now);
    }
    virtual bool isSame(const std::shared_ptr<Entity>&  other) override{
        PYBIND11_OVERRIDE(bool,Base,isSame,other);
    }
    virtual void serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full) override{
        PYBIND11_OVERRIDE(void,Base,serializeInternalState,archive,full);
    }
};

ASRC_DECLARE_BASE_REF_TRAMPOLINE(EntityAccessor)
    // No virtual function for the base class.
};

void exportEntity(py::module &m);

/**
 * @brief Entity の派生クラスをPython側へ公開する際に、必須コンストラクタのdefまでまとめて行うための関数テンプレート。
 * 
 * @details Entity の派生クラスは、2つのnl::json(modelConfig,instanceConfig)を引数に取るコンストラクタが存在し、
 *          かつそれらがPython側へ公開されることを必須としている。
 *          この関数テンプレートでは、その2引数コンストラクタをPython側に公開する。
 * 
 * @tparam T             公開対象のクラス
 * @tparam Parent        公開時の親となるPythonオブジェクトのクラス。引数からの推論を基本とする。
 * @tparam ...Extras     公開時のオプションのクラスで、py::class_に与えるものと同じ。引数からの推論を基本とする。
 * 
 * @param [in] m         公開時の親となるPythonオブジェクト。通常はpy::module_やpy::class_を与える。
 * @param [in] className Python側から見た公開時のクラス名。
 * @param [in] ...extras 公開時のオプション。py::class_に与えるものと同じ。
 * 
 * @return Python側へ公開されたクラスオブジェクト。
 * 
 */
template<std::derived_from<Entity> T,typename Parent, typename ... Extras>
auto expose_entity_subclass(Parent& m, const char* className, Extras&& ... extras){
    return expose_common_class<T>(m,className,std::forward<Extras>(extras)...)
    .def(::asrc::core::py_init<const nl::json&,const nl::json&>())
    ;
}

template<class T=Entity>
struct PYBIND11_EXPORT EntityIdentifierGetter{
    EntityIdentifier operator()(const std::shared_ptr<const T>& entity) const{
        if(entity){
            return entity->getEntityID();
        }else{
            return EntityIdentifier();
        }
    }
};

template<class T=Entity>
struct PYBIND11_EXPORT EntityStringifier{
    std::string operator()(const std::shared_ptr<const T>& entity) const{
        if(entity){
            return entity->getFullName();
        }else{
            return "nullptr";
        }
    }
};

template<class Instance=Entity>
using EntityDependencyGraph=DependencyGraph<Instance,EntityIdentifier,EntityIdentifierGetter<Instance>,EntityStringifier<Instance>>;

ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)

ASRC_PYBIND11_MAKE_OPAQUE(std::vector<std::shared_ptr<::asrc::core::Entity>>);
ASRC_PYBIND11_MAKE_OPAQUE(std::vector<std::shared_ptr<::asrc::core::EntityConstructionInfo>>);
ASRC_PYBIND11_MAKE_OPAQUE(std::map<std::int32_t,std::weak_ptr<::asrc::core::EntityManager>>);
ASRC_PYBIND11_MAKE_OPAQUE(std::map<::asrc::core::EntityIdentifier,std::shared_ptr<::asrc::core::Entity>>);
ASRC_PYBIND11_MAKE_OPAQUE(std::map<::asrc::core::EntityIdentifier, std::weak_ptr<::asrc::core::Entity>>);
ASRC_PYBIND11_MAKE_OPAQUE(std::map<asrc::core::SimPhase,std::uint64_t>);
