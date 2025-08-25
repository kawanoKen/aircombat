/**
 * Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
 * @file
 * 
 * @brief Entityの生成を行うFactoryクラス
 * 
 */
#pragma once
#include "Common.h"
#include "EntityIdentifier.h"
#include "Utility.h"
#include <iostream>
#include <memory>
#include <deque>
#include <set>
#include <mutex>
#include <shared_mutex>
#include <filesystem>
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <nlohmann/json.hpp>
#include <cereal/types/deque.hpp>

namespace py=pybind11;
namespace nl=nlohmann;

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)

class Entity;
class EntityManager;
class FactoryHelper;
class FactoryHelperChain;

/**
 * @class Factory
 * @brief Entityの生成を行うクラス。
 */
class PYBIND11_EXPORT Factory: public std::enable_shared_from_this<Factory>{//Only Entity
    friend class Entity;
    friend class EntityManager;
    friend class FactoryHelper;
    friend class FactoryHelperChain;
    Factory();
    template<class F>
    struct make_shared_enabler:F{
        template<typename... Args>
        explicit make_shared_enabler(Args&&... args):F(std::forward<Args>(args)...){}
    };
    mutable std::recursive_mutex mtx;
    static std::shared_mutex static_mtx;
    public:
    typedef std::function<std::shared_ptr<Entity>(const nl::json&,const nl::json&)> creatorType;
    ~Factory()=default;
    static std::shared_ptr<Factory> create();
    static void clear();
    void reset();

    /**
     * @brief C++クラスを追加する。
     * 
     * @param [in] baseName  追加先のグループ名(base name)
     * @param [in] className クラス名
     * @param [in] fn        コンストラクタを呼び出す関数オブジェクト(const nl::json&型のmodelConfigとinstanceConfigを引数にとる)
     */
    static void addClass(const std::string& baseName,const std::string& className,creatorType fn);

    /**
     * @brief Pythonクラスを追加する。
     * 
     * @param [in] baseName  追加先のグループ名(base name)
     * @param [in] className クラス名
     * @param [in] clsObj    Pythonクラスオブジェクト(const nl::json&型のmodelConfigとinstanceConfigを引数にとる__init__を持つ)
     */
    static void addPythonClass(const std::string& baseName,const std::string& className,py::object clsObj);

    /**
     * @brief staticメンバとして共有されるmodelを追加する。
     * 
     * @param [in] baseName     追加先のグループ名(base name)
     * @param [in] modelName    modelの名称
     * @param [in] modelConfig_ configの値
     * 
     * @details
     *   config_ の形式は、 doc/src/format/Factory_model_config.json のとおり。
     * @include{strip} doc/src/format/Factory_model_config.json
     */
    static void addDefaultModel(const std::string& baseName,const std::string& modelName,const nl::json& modelConfig_);

    /**
     * @brief staticメンバとして共有されるmodelをjson objectから読み込んで追加する。
     * 
     * @details jsonの形式はbase nameを第1階層、model nameを第2階層のキーとして、model configを第2階層の値に記述する。
     *          複数同時に追加することができる。
     * 
     *          config_ の形式は、 doc/src/format/Factory_model_config.json のとおり。
     * @include{strip} doc/src/format/Factory_add_model_config.json
     */
    static void addDefaultModelsFromJson(const nl::json& j);

    /**
     * @brief staticメンバとして共有されるmodelをjsonファイルから読み込んで追加する。
     * 
     * @details jsonの形式はaddDefaultModelsFromJsonと同様に、 doc/src/format/Factory_model_config.json のとおり。
     * @include{strip} doc/src/format/Factory_add_model_config.json
     * 
     * @param [in] filePath jsonファイルのパス
     */
    static void addDefaultModelsFromJsonFile(const std::string& filePath);

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
     * @brief baseNameグループにclassNameという名前のクラスが存在するかどうかを返す。
     * 
     * @attention この関数では完全一致のみ判定し、パッケージ名の省略等には対応しない。
     */
    static bool hasClass(const std::string& baseName,const std::string& className);

    /**
     * @brief staticメンバとして保持している共有モデルのbaseNameグループにmodelNameという名前のモデルが存在するかどうかを返す。
     * 
     * @attention この関数では完全一致のみ判定し、パッケージ名の省略等には対応しない。
     */
    static bool hasDefaultModel(const std::string& baseName,const std::string& modelName);

    /**
     * @brief Factoryインスタンス固有のモデルのbaseNameグループにmodelNameという名前のモデルが存在するかどうかを返す。
     * 
     * @attention この関数では完全一致のみ判定し、パッケージ名の省略等には対応しない。
     */
    bool hasModel(const std::string& baseName,const std::string& modelName) const;

    /**
     * @brief baseName グループから modelName という名前のモデルを探してそのmodel configを返す。
     * 
     * @param [in] withDefaultModels trueの場合、staticメンバとして保持している共有モデルも合わせて探す。falseの場合、Factoryインスタンス固有の固有のmodelのみ。
     * 
     * @details 見つからなかった場合はnull jsonを返す。
     */
    nl::json getModelConfig(const std::string& baseName,const std::string& modelName,bool withDefaultModels=true) const;

    /**
     * @brief 自身が持つmodel configの一覧を返す。
     * 
     * @param [in] withDefaultModels trueの場合、staticメンバとして保持している共有モデルも合わせて返す。falseの場合、Factoryインスタンス固有の固有のmodelのみ。
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
    nl::json getModelConfigs(bool withDefaultModels=true) const;

    /**
     * @brief 引数で指定したクラス、モデルに対応する、完全な形での[className, modelName, modelConfig, FactoryHelperChain]の組を返す。
     * 
     * @details パッケージ名が省略されている場合等に、適切なパッケージのクラス、モデルを探し出す機能もこの関数が担う。
     * 
     * @param [in] baseName          グループ名
     * @param [in] className         クラス名。与えた場合、モデルが持つクラス名指定を上書きする。
     * @param [in] modelName         モデル名。省略した場合、classNameとmodelConfigから新たな無名モデルを生成する。
     * @param [in] modelConfig       model config(のパッチ)。検索で見つかったmodel configにmerge_patchによりマージされる。
     * @param [in] helperChain       パッケージ名を検索する際の FactoryHelperChain
     * @param [in] withDefaultModels trueの場合、staticメンバとして保持している共有モデルも検索対象とする。falseの場合、Factoryインスタンス固有の固有のmodelのみ。
     * 
     */
    std::tuple<std::string,std::string,nl::json,FactoryHelperChain> resolveModelConfig(const std::string& baseName,const std::string& className,const std::string& modelName,const nl::json& modelConfig,const FactoryHelperChain& helperChain,bool withDefaultModels=true) const;

    /**
     * @brief 引数で指定したクラス、モデルに対応する、完全な形での[className, modelName, modelConfig, FactoryHelperChain]の組を返す。
     * 
     * @details パッケージ名が省略されている場合等に、適切なパッケージのクラス、モデルを探し出す機能もこの関数が担う。
     * 
     * @param [in] baseName          グループ名
     * @param [in] modelName         モデル名。省略した場合、classNameとmodelConfigから新たな無名モデルを生成する。
     * @param [in] helperChain       パッケージ名を検索する際の FactoryHelperChain
     * @param [in] withDefaultModels trueの場合、staticメンバとして保持している共有モデルも検索対象とする。falseの場合、Factoryインスタンス固有の固有のmodelのみ。
     * 
     */
    std::tuple<std::string,std::string,nl::json,FactoryHelperChain> resolveModelConfig(const std::string& baseName,const std::string& modelName,const FactoryHelperChain& helperChain,bool withDefaultModels=true) const;

    /**
     * @brief 引数で指定したクラス、モデルに対応する、完全な形での[className, modelName, modelConfig, FactoryHelperChain]の組を返す。
     * 
     * @details パッケージ名が省略されている場合等に、適切なパッケージのクラス、モデルを探し出す機能もこの関数が担う。
     * 
     * @param [in] baseName          グループ名
     * @param [in] j                 文字列を与えた場合はモデル名として解釈し、object(dict)を与えた場合は"className","modelName","modelConfig"の3つのキーの値を抽出して用いる。
     * @param [in] helperChain       パッケージ名を検索する際の FactoryHelperChain
     * @param [in] withDefaultModels trueの場合、staticメンバとして保持している共有モデルも検索対象とする。falseの場合、Factoryインスタンス固有の固有のmodelのみ。
     * 
     */
    std::tuple<std::string,std::string,nl::json,FactoryHelperChain> resolveModelConfig(const std::string& baseName,const nl::json& j,const FactoryHelperChain& helperChain,bool withDefaultModels=true) const;

    /**
     * @brief 引数で指定したクラス、モデルに対応する、完全な形での[className, modelName, modelConfig, FactoryHelperChain]の組を返す。
     * 
     * @details パッケージ名が省略されている場合等に、適切なパッケージのクラス、モデルを探し出す機能もこの関数が担う。
     * 
     * @param [in] j                 文字列を与えた場合はモデル名として解釈し、object(dict)を与えた場合は"baseName","className","modelName","modelConfig"の3つのキーの値を抽出して用いる。
     * @param [in] helperChain       パッケージ名を検索する際の FactoryHelperChain
     * @param [in] withDefaultModels trueの場合、staticメンバとして保持している共有モデルも検索対象とする。falseの場合、Factoryインスタンス固有の固有のmodelのみ。
     * 
     */
    std::tuple<std::string,std::string,nl::json,FactoryHelperChain> resolveModelConfig(const nl::json& j,const FactoryHelperChain& helperChain,bool withDefaultModels=true) const;

    /**
     * @brief 引数で指定したクラス、モデルに対応する、完全な形での[className, modelName, modelConfig, FactoryHelperChain]の組を返す。
     * 
     * @details パッケージ名が省略されている場合等に、適切なパッケージのクラス、モデルを探し出す機能もこの関数が担う。
     * 
     * @param [in] baseName          グループ名
     * @param [in] className         クラス名。与えた場合、モデルが持つクラス名指定を上書きする。
     * @param [in] modelName         モデル名。省略した場合、classNameとmodelConfigから新たな無名モデルを生成する。
     * @param [in] modelConfig       model config(のパッチ)。検索で見つかったmodel configにmerge_patchによりマージされる。
     * @param [in] withDefaultModels trueの場合、staticメンバとして保持している共有モデルも検索対象とする。falseの場合、Factoryインスタンス固有の固有のmodelのみ。
     * 
     */
    std::tuple<std::string,std::string,nl::json,FactoryHelperChain> resolveModelConfig(const std::string& baseName,const std::string& className,const std::string& modelName,const nl::json& modelConfig,bool withDefaultModels=true) const;

    /**
     * @brief 引数で指定したクラス、モデルに対応する、完全な形での[className, modelName, modelConfig, FactoryHelperChain]の組を返す。
     * 
     * @details パッケージ名が省略されている場合等に、適切なパッケージのクラス、モデルを探し出す機能もこの関数が担う。
     * 
     * @param [in] baseName          グループ名
     * @param [in] modelName         モデル名。省略した場合、classNameとmodelConfigから新たな無名モデルを生成する。
     * @param [in] withDefaultModels trueの場合、staticメンバとして保持している共有モデルも検索対象とする。falseの場合、Factoryインスタンス固有の固有のmodelのみ。
     * 
     */
    std::tuple<std::string,std::string,nl::json,FactoryHelperChain> resolveModelConfig(const std::string& baseName,const std::string& modelName,bool withDefaultModels=true) const;

    /**
     * @brief 引数で指定したクラス、モデルに対応する、完全な形での[className, modelName, modelConfig, FactoryHelperChain]の組を返す。
     * 
     * @details パッケージ名が省略されている場合等に、適切なパッケージのクラス、モデルを探し出す機能もこの関数が担う。
     * 
     * @param [in] baseName          グループ名
     * @param [in] j                 文字列を与えた場合はモデル名として解釈し、object(dict)を与えた場合は"className","modelName","modelConfig"の3つのキーの値を抽出して用いる。
     * @param [in] withDefaultModels trueの場合、staticメンバとして保持している共有モデルも検索対象とする。falseの場合、Factoryインスタンス固有の固有のmodelのみ。
     * 
     */
    std::tuple<std::string,std::string,nl::json,FactoryHelperChain> resolveModelConfig(const std::string& baseName,const nl::json& j,bool withDefaultModels=true) const;

    /**
     * @brief 引数で指定したクラス、モデルに対応する、完全な形での[className, modelName, modelConfig, FactoryHelperChain]の組を返す。
     * 
     * @details パッケージ名が省略されている場合等に、適切なパッケージのクラス、モデルを探し出す機能もこの関数が担う。
     * 
     * @param [in] j                 文字列を与えた場合はモデル名として解釈し、object(dict)を与えた場合は"baseName","className","modelName","modelConfig"の3つのキーの値を抽出して用いる。
     * @param [in] withDefaultModels trueの場合、staticメンバとして保持している共有モデルも検索対象とする。falseの場合、Factoryインスタンス固有の固有のmodelのみ。
     * 
     */
    std::tuple<std::string,std::string,nl::json,FactoryHelperChain> resolveModelConfig(const nl::json& j,bool withDefaultModels=true) const;
    protected:
    std::tuple<std::string,std::string,nl::json,FactoryHelperChain> resolveModelConfigSub(const std::string& baseName,const std::string& className,const std::string& modelName,const nl::json& modelConfig,const FactoryHelperChain& prefixChain,const FactoryHelperChain& originalChain,bool withDefaultModels=true) const;
    std::tuple<std::string,std::string,nl::json,FactoryHelperChain> resolveModelConfigSub(const std::string& baseName,const std::string& modelName,const FactoryHelperChain& prefixChain,const FactoryHelperChain& originalChain,bool withDefaultModels=true) const;
    std::tuple<std::string,std::string,nl::json,FactoryHelperChain> resolveModelConfigSub(const std::string& baseName,const nl::json& j,const FactoryHelperChain& prefixChain,const FactoryHelperChain& originalChain,bool withDefaultModels=true) const;
    public:

    /**
     * @brief 現在のこのインスタンス固有のmodel configを、引数に与えたjsonをpatchとしてmerge patch処理を行うことで書き換える。
     * 
     * @details configの値としてnullを与えることでmodelの削除が可能。(merge patchの仕様による挙動)
     * 
     */
    void reconfigureModelConfigs(const nl::json& j);

    /**
     * @brief baseNameグループのclassNameという名前のクラスのコンストラクタを返す。
     */
    static creatorType getCreator(const std::string& baseName,const std::string& className);

    //Entityの生成
    //xxxImplに実体を実装し、xxxOverloaderで外部から呼び出し可能とする引数セットを定義し、xxxでテンプレート関数として外部に公開
    protected:
    /**
     * @internal
     * @brief [For internal use] Entity を生成する関数の実体その1。
     * 
     * @param [in] baseName         Entity のグループ名(base name)を指定する。省略不可。
     * @param [in] className        Entity のクラス名を指定する。""として省略するとモデルに紐付けられたクラスが使用される。
     * @param [in] modelName        Entity のモデル名を指定する。""として省略するとclassNameとmodelConfigから無名モデルとして生成される。
     * @param [in] modelConfig      Entity のmodel configを指定する。モデルに紐付けられたコンフィグがある場合、それに対するmerge patchとして働く。
     * @param [in] instanceConfig   Entity のinstance configを指定する。
     * @param [in] helperChain      生成時にクラス名、モデル名の解決に使用する FactoryHelperChain 。省略可。
     * @param [in] caller           生成元の Entity 。省略可。
     */
    std::shared_ptr<Entity> createImpl(const std::string& baseName,const std::string& className,const std::string& modelName,const nl::json& modelConfig,const nl::json& instanceConfig, const FactoryHelperChain& helperChain, const std::shared_ptr<const Entity>& caller);

    /**
     * @brief Entity を生成する関数のオーバーロード定義の一つ。
     * 
     * @param [in] baseName         Entity のグループ名(base name)を指定する。省略不可。
     * @param [in] className        Entity のクラス名を指定する。""として省略するとモデルに紐付けられたクラスが使用される。
     * @param [in] modelName        Entity のモデル名を指定する。""として省略するとclassNameとmodelConfigから無名モデルとして生成される。
     * @param [in] modelConfig      Entity のmodel configを指定する。モデルに紐付けられたコンフィグがある場合、それに対するmerge patchとして働く。
     * @param [in] instanceConfig   Entity のinstance configを指定する。
     * 
     * @details helperChain は空、 caller は nullptr となる。
     */
    std::shared_ptr<Entity> createOverloader(const std::string& baseName,const std::string& className,const std::string& modelName,const nl::json& modelConfig,const nl::json& instanceConfig);
    
    /**
     * @brief Entity を生成する関数のオーバーロード定義の一つ。
     * 
     * @param [in] baseName         Entity のグループ名(base name)を指定する。省略不可。
     * @param [in] className        Entity のクラス名を指定する。""として省略するとモデルに紐付けられたクラスが使用される。
     * @param [in] modelName        Entity のモデル名を指定する。""として省略するとclassNameとmodelConfigから無名モデルとして生成される。
     * @param [in] modelConfig      Entity のmodel configを指定する。モデルに紐付けられたコンフィグがある場合、それに対するmerge patchとして働く。
     * @param [in] instanceConfig   Entity のinstance configを指定する。
     * @param [in] helperChain      生成時にクラス名、モデル名の解決に使用する FactoryHelperChain 。省略可。
     * 
     * @details caller は nullptr となる。
     */
    std::shared_ptr<Entity> createOverloader(const std::string& baseName,const std::string& className,const std::string& modelName,const nl::json& modelConfig,const nl::json& instanceConfig, const FactoryHelperChain& helperChain);

    /**
     * @brief Entity を生成する関数のオーバーロード定義の一つ。
     * 
     * @param [in] baseName         Entity のグループ名(base name)を指定する。省略不可。
     * @param [in] className        Entity のクラス名を指定する。""として省略するとモデルに紐付けられたクラスが使用される。
     * @param [in] modelName        Entity のモデル名を指定する。""として省略するとclassNameとmodelConfigから無名モデルとして生成される。
     * @param [in] modelConfig      Entity のmodel configを指定する。モデルに紐付けられたコンフィグがある場合、それに対するmerge patchとして働く。
     * @param [in] instanceConfig   Entity のinstance configを指定する。
     * @param [in] caller           生成元の Entity 。省略可。
     * 
     * @details helperChain は caller->getHelperChain() となる。
     */
    std::shared_ptr<Entity> createOverloader(const std::string& baseName,const std::string& className,const std::string& modelName,const nl::json& modelConfig,const nl::json& instanceConfig, const std::shared_ptr<const Entity>& caller);

    /**
     * @brief Entity を生成する関数のオーバーロード定義の一つ。
     * 
     * @param [in] baseName         Entity のグループ名(base name)を指定する。省略不可。
     * @param [in] modelName        Entity のモデル名を指定する。""として省略するとclassNameとmodelConfigから無名モデルとして生成される。
     * @param [in] instanceConfig   Entity のinstance configを指定する。
     * 
     * @details className は ""、 modelConfig は""、helperChain は空、 caller は nullptr となる。
     */
    std::shared_ptr<Entity> createOverloader(const std::string& baseName,const std::string& modelName,const nl::json& instanceConfig);

    /**
     * @brief Entity を生成する関数のオーバーロード定義の一つ。
     * 
     * @param [in] baseName         Entity のグループ名(base name)を指定する。省略不可。
     * @param [in] modelName        Entity のモデル名を指定する。""として省略するとclassNameとmodelConfigから無名モデルとして生成される。
     * @param [in] instanceConfig   Entity のinstance configを指定する。
     * @param [in] helperChain      生成時にクラス名、モデル名の解決に使用する FactoryHelperChain 。省略可。
     * 
     * @details className は ""、 modelConfig は""、 caller は nullptr となる。
     */
    std::shared_ptr<Entity> createOverloader(const std::string& baseName,const std::string& modelName,const nl::json& instanceConfig, const FactoryHelperChain& helperChain);

    /**
     * @brief Entity を生成する関数のオーバーロード定義の一つ。
     * 
     * @param [in] baseName         Entity のグループ名(base name)を指定する。省略不可。
     * @param [in] modelName        Entity のモデル名を指定する。""として省略するとclassNameとmodelConfigから無名モデルとして生成される。
     * @param [in] instanceConfig   Entity のinstance configを指定する。
     * @param [in] caller           生成元の Entity 。省略可。
     * 
     * @details className は ""、 modelConfig は""、  helperChain は caller->getHelperChain() となる。
     */
    std::shared_ptr<Entity> createOverloader(const std::string& baseName,const std::string& modelName,const nl::json& instanceConfig, const std::shared_ptr<const Entity>& caller);

    /**
     * @brief モデル名を用いずに Entity を生成する関数のオーバーロード定義の一つ。
     * 
     * @param [in] baseName         Entity のグループ名(base name)を指定する。省略不可。
     * @param [in] className        Entity のクラス名を指定する。""として省略するとモデルに紐付けられたクラスが使用される。
     * @param [in] modelConfig      Entity のmodel configを指定する。このオーバーロードではmerge対象が存在しないのでここで与えたものがそのままmodel configとなる。
     * @param [in] instanceConfig   Entity のinstance configを指定する。
     * 
     * @details helperChain は空、 caller は nullptr となる。
     */
    std::shared_ptr<Entity> createByClassNameOverloader(const std::string& baseName,const std::string& className,const nl::json& modelConfig,const nl::json& instanceConfig);

    /**
     * @brief モデル名を用いずに Entity を生成する関数のオーバーロード定義の一つ。
     * 
     * @param [in] baseName         Entity のグループ名(base name)を指定する。省略不可。
     * @param [in] className        Entity のクラス名を指定する。""として省略するとモデルに紐付けられたクラスが使用される。
     * @param [in] modelConfig      Entity のmodel configを指定する。このオーバーロードではmerge対象が存在しないのでここで与えたものがそのままmodel configとなる。
     * @param [in] instanceConfig   Entity のinstance configを指定する。
     * @param [in] helperChain      生成時にクラス名、モデル名の解決に使用する FactoryHelperChain 。省略可。
     * 
     * @details caller は nullptr となる。
     */
    std::shared_ptr<Entity> createByClassNameOverloader(const std::string& baseName,const std::string& className,const nl::json& modelConfig,const nl::json& instanceConfig, const FactoryHelperChain& helperChain);

    /**
     * @brief モデル名を用いずに Entity を生成する関数のオーバーロード定義の一つ。
     * 
     * @param [in] baseName         Entity のグループ名(base name)を指定する。省略不可。
     * @param [in] className        Entity のクラス名を指定する。""として省略するとモデルに紐付けられたクラスが使用される。
     * @param [in] modelConfig      Entity のmodel configを指定する。このオーバーロードではmerge対象が存在しないのでここで与えたものがそのままmodel configとなる。
     * @param [in] instanceConfig   Entity のinstance configを指定する。
     * @param [in] caller           生成元の Entity 。省略可。
     * 
     * @details helperChain は caller->getHelperChain() となる。
     */
    std::shared_ptr<Entity> createByClassNameOverloader(const std::string& baseName,const std::string& className,const nl::json& modelConfig,const nl::json& instanceConfig, const std::shared_ptr<const Entity>& caller);

    /**
     * @internal
     * @brief [For internal use] Entity を生成する関数の実体その2。
     * 
     * @param [in] j object(dict)で与える。
     * @param [in] helperChain      生成時にクラス名、モデル名の解決に使用する FactoryHelperChain 。省略可。
     * @param [in] caller           生成元の Entity 。省略可。
     * 
     * @details
     *      引数のjsonに含めるべき値は以下の通り。
     * 
     *      - baseName                  : Entity のグループ名(base name)を指定する。省略不可。
     *      - className 又は class      : Entity のクラス名を指定する。""として省略するとモデルに紐付けられたクラスが使用される。
     *      - modelName 又は model      : Entity のモデル名を指定する。""として省略するとclassNameとmodelConfigから無名モデルとして生成される。
     *      - modelConfig 又は config   : Entity のmodel configを指定する。モデルに紐付けられたコンフィグがある場合、それに対するmerge patchとして働く。
     *      - instanceConfig            : Entity のinstance configを指定する。省略不可。
     */
    std::shared_ptr<Entity> createImpl(const nl::json& j, const FactoryHelperChain& helperChain, const std::shared_ptr<const Entity>& caller);

    /**
     * @brief モデル名を用いずに Entity を生成する関数のオーバーロード定義の一つ。
     * 
     * @param [in] j object(dict)で与える。
     * 
     * @details helperChain は空、 caller は nullptr となる。
     */
    std::shared_ptr<Entity> createOverloader(const nl::json& j);

    /**
     * @brief モデル名を用いずに Entity を生成する関数のオーバーロード定義の一つ。
     * 
     * @param [in] j object(dict)で与える。
     * @param [in] helperChain      生成時にクラス名、モデル名の解決に使用する FactoryHelperChain 。省略可。
     * 
     * @details caller は nullptr となる。
     * 
     * @details
     *      引数のjsonに含めるべき値は以下の通り。
     * 
     *      - baseName                  : Entity のグループ名(base name)を指定する。省略不可。
     *      - className 又は class      : Entity のクラス名を指定する。""として省略するとモデルに紐付けられたクラスが使用される。
     *      - modelName 又は model      : Entity のモデル名を指定する。""として省略するとclassNameとmodelConfigから無名モデルとして生成される。
     *      - modelConfig 又は config   : Entity のmodel configを指定する。モデルに紐付けられたコンフィグがある場合、それに対するmerge patchとして働く。
     *      - instanceConfig            : Entity のinstance configを指定する。省略不可。
     */
    std::shared_ptr<Entity> createOverloader(const nl::json& j, const FactoryHelperChain& helperChain);
    
    /**
     * @brief モデル名を用いずに Entity を生成する関数のオーバーロード定義の一つ。
     * 
     * @param [in] j object(dict)で与える。
     * @param [in] caller           生成元の Entity 。省略可。
     * 
     * @details helperChain は caller->getHelperChain() となる。
     * 
     * @details
     *      引数のjsonに含めるべき値は以下の通り。
     * 
     *      - baseName                  : Entity のグループ名(base name)を指定する。省略不可。
     *      - className 又は class      : Entity のクラス名を指定する。""として省略するとモデルに紐付けられたクラスが使用される。
     *      - modelName 又は model      : Entity のモデル名を指定する。""として省略するとclassNameとmodelConfigから無名モデルとして生成される。
     *      - modelConfig 又は config   : Entity のmodel configを指定する。モデルに紐付けられたコンフィグがある場合、それに対するmerge patchとして働く。
     *      - instanceConfig            : Entity のinstance configを指定する。省略不可。
     */
    std::shared_ptr<Entity> createOverloader(const nl::json& j, const std::shared_ptr<const Entity>& caller);
    public:
    /**
     * @brief Entity を生成する際に外から実際に呼び出すテンプレート関数。createOverloaderを呼び出し可能な任意の引数を使える。
     * 
     * @tparam T 生成した Entity をどのクラスとして受け取るか。 Entity 又はその派生クラスを指定できる。
     */
    template<std::derived_from<Entity> T=Entity, typename... Args>
    requires ( sizeof...(Args) > 0 ) // Factoryインスタンスのcreateを除外する
    std::shared_ptr<T> create(Args&&... args){
        return std::dynamic_pointer_cast<T>(createOverloader(std::forward<Args>(args)...));
    }
    /**
     * @brief モデル名を用いずに Entity を生成する際に外から実際に呼び出すテンプレート関数。createByClassNameOverloaderを呼び出し可能な任意の引数を使える。
     * 
     * @tparam T 生成した Entity をどのクラスとして受け取るか。 Entity 又はその派生クラスを指定できる。
     */
    template<std::derived_from<Entity> T=Entity, typename... Args>
    std::shared_ptr<T> createByClassName(Args&&... args){
        return std::dynamic_pointer_cast<T>(createByClassNameOverloader(std::forward<Args>(args)...));
    }

    public:
    /**
     * @brief C++側での継承関係チェック
     * @details derivedBaseNameグループにあるderivedClassNameという名前のクラスがBaseの派生クラスであるかどうかを判定する。
     */
    template<class Base>
    static bool issubclassof(const std::string& derivedBaseName,const std::string& derivedClassName){
        auto dummy=Factory::create()->createByClassName<Entity>(derivedBaseName,derivedClassName,nl::json(nullptr),nl::json(nullptr));
        std::shared_ptr<Base> t=std::dynamic_pointer_cast<Base>(dummy);
        return (bool)t;
    }

    /**
     * @brief Python側クラスオブジェクトによる継承関係チェック
     * @details derivedBaseNameグループにあるderivedClassNameという名前のクラスがbaseClsの派生クラスであるかどうかを判定する。
     */
    static bool issubclassof(py::object baseCls,const std::string& derivedBaseName,const std::string& derivedClassName);

    /**
     * @brief [For internal use] 登録済の全クラスのダミーインスタンスをmodelConfigとinstanceConfigをnullとして生成するテストを実行する。
     */
    static void dummyCreationTest();

    //baseNameの管理
    private:
    static std::map<std::string, std::string> baseNameTree; //!< [For internal use] baseNameの階層構造
    static std::map<std::string, std::set<std::string>> permissionsForCreate; //!< [For internal use] 各baseNameをキーとして、そのインスタンスを生成することができるbaseNameのset
    static std::map<std::string, std::set<std::string>> permissionsForGet; //!< [For internal use] 各baseNameをキーとして、そのインスタンスを取得することができるbaseNameのset
    static std::set<std::string> mergeParentPermission(const std::set<std::string>& child, const std::set<std::string>& parent); //!< [For internal use] childのpermissionにparentのpsermissionをマージする。
    /**
     * @brief [For internal use] callerBaseNameがtargetBaseNameのインスタンスの生成又は取得をすることができるかどうかを判定する実体。
     */
    static bool checkPermissionImpl(const std::string& callerBaseName, const std::string& targetBaseName,const std::map<std::string, std::set<std::string>>& permissions);
    public:
    /**
     * @brief [For internal use] 新たなグループをbaseNameという名前で追加する。
     * 
     * @param [in] baseName            グループ名
     * @param [in] parentBaseName      親となるグループ名
     * @param [in] permissionForCreate このグループのインスタンスを生成してよいbaseNameのset。
     * @param [in] permissionForGet    このグループのインスタンスを生成してよいbaseNameのset。
     * 
     * @details
     *      permissionには、baseNameを直接指定する以外にも、"AnyOthers"による全許可や、"!"による除外が可能である。
     */
    static void addBaseName(const std::string& baseName, const std::string& parentBaseName, const std::set<std::string>& permissionForCreate, const std::set<std::string>& permissionForGet);
    static bool hasBaseName(const std::string& baseName); //!< baseNameという名前のグループが存在するかどうかを返す。
    /**
     * @brief callerがtargetBaseNameのインスタンスを生成することができるかどうかを返す。
     */
    static bool checkPermissionForCreate(const std::shared_ptr<const Entity>& caller, const std::string& targetBaseName);
    /**
     * @brief callerがtargetBaseNameのインスタンスを取得することができるかどうかを返す。
     */
    static bool checkPermissionForGet(const std::shared_ptr<const Entity>& caller, const std::string& targetBaseName);

    //Helperの管理
    protected:
    static std::pair<std::string,std::string> parseName(const std::string& fullName); //!< [For internal use] ".区切りの文字列を先頭の"."より前と後の二つに分割する。
    private:
    static void addHelper(std::shared_ptr<FactoryHelper> helper,const std::string& packageName); //!< [For internal use] パッケージ名packageNameに対応するFactoryHelperとしてhelperを登録する。
    static std::shared_ptr<FactoryHelper> getHelperFromPackageName(const std::string& packageName); //!< [For internal use] パッケージ名packageNameに対応するFactoryHelperを返す。
    static std::shared_ptr<FactoryHelper> getHelperFromClassOrModelName(const std::string& fullName); //!< [For internal use] 指定したクラス名又はモデル名が属するパッケージに対応するFactoryHelperを返す。
    static std::map<std::string,std::map<std::string,creatorType>> creators; //!< 各クラスのコンストラクタ
    static std::map<std::string,std::map<std::string,nl::json>> defaultModelConfigs; //!< 共有モデル
    std::map<std::string,std::map<std::string,nl::json>> modelConfigs; //!< インスタンス固有のモデル
    static std::map<std::string,std::shared_ptr<FactoryHelper>> helpers; //!< 各パッケージ名に対応するFactoryHelper
    static const bool acceptOverwriteDefaults; //!< trueの場合、staticメンバとして登録された共有モデルを上書き可能となる。
    static const bool noErrorForDuplication; //!< trueの場合、クラスやモデルのの重複登録時にエラーを出さない。
    static const bool acceptPackageNameOmission; //!< trueの場合、 Entity の生成・取得時のクラス名、モデル名の指定においてパッケージ名の省略が可能となる。

    public:
    void serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full); //!< 内部状態のシリアライゼーション
    private:
    mutable std::map<std::string,std::set<std::string>> createdModelConfigsSinceLastSerialization; //!< 前回のシリアライズ実行時以降に追加されたモデル
    mutable std::map<std::string,std::set<std::string>> removedModelConfigsSinceLastSerialization; //!< 前回のシリアライズ実行時以降に削除されたモデル
};

/**
 * @brief Factoryにクラスを登録する。factoryHelperという名前でFactoryHelperインスタンスにアクセスできる場所で呼ぶこと。
 * 
 * @param [in] baseName  グループ名 (base name)
 * @param [in] className クラス名 (Entity の派生クラスを指定すること。)
 */
#define FACTORY_ADD_CLASS(baseName,className) factoryHelper->addClass(#baseName,#className,&className::create<typename className::Type,const nl::json&,const nl::json&>);

/**
 * @brief Factoryにクラスを登録する。factoryHelperという名前でFactoryHelperインスタンスにアクセスできる場所で呼ぶこと。
 * 
 * @param [in] baseName  グループ名 (base name)
 * @param [in] className クラス名 (Entity の派生クラスを指定すること。)
 * @param [in] registerClassName Factoryに登録する際の名称。
 */
#define FACTORY_ADD_CLASS_NAME(baseName,className,registerClassName) factoryHelper->addClass(#baseName,registerClassName,&className::create<typename className::Type,const nl::json&,const nl::json&>);

/**
 * @brief Factoryに、classNameを基底クラスとするグループを登録する。
 * 
 * @param [in] className 新たなグループの基底クラスとなるクラスの名前(Entity の派生クラスを指定すること。)
 * @param [in] parentBaseName      親となるグループ名
 * @param [in] permissionForCreate このグループのインスタンスを生成してよいbaseNameのset。
 * @param [in] permissionForGet    このグループのインスタンスを生成してよいbaseNameのset。
 * 
 * @details classNameのメンバ変数 static const std::string baseName に新たなグループ名を設定しておくこと。
 */
#define FACTORY_ADD_BASE(className,...)\
assert(&className::baseName != &className::BaseType::baseName);\
Factory::addBaseName(\
    className::baseName,\
    className::BaseType::baseName,\
    __VA_ARGS__\
);\

/**
 * @internal
 * @class FactoryHelper
 * @brief [For internal use] パッケージ(プラグイン)ごとに生成され、クラス名やモデル名の検索を助けるクラス。
 */
ASRC_DECLARE_BASE_REF_CLASS(FactoryHelper)
    mutable std::recursive_mutex mtx;
    public:
    template<asrc::core::traits::output_archive Archive>
    static void save_as_reference_impl(Archive& ar, const std::shared_ptr<const PtrBaseType>& t){
        if(t){
            if(!t->isValid){
                throw std::runtime_error("FactoryHelper::save_as_reference_impl failed. [reason='Invalid FactoryHelper']");
            }
            ar(::cereal::make_nvp("packageName",t->packageName));
        }else{
            throw std::runtime_error("FactoryHelper::save_as_reference_impl failed. [reason='The given pointer is nullptr.']");
        }
    }
    template<asrc::core::traits::input_archive Archive>
    static std::shared_ptr<PtrBaseType> load_as_reference_impl(Archive& ar){
        try{
            std::string packageName;
            ar(::cereal::make_nvp("packageName",packageName));
            auto ret=Factory::getHelperFromPackageName(packageName);
            if(ret){
                return ret;
            }
            throw std::runtime_error("No package named '"+packageName+"' found.");
        }catch(std::exception& ex){
            throw std::runtime_error("EntityAccessor::load_as_reference_impl failed. [reason='"+std::string(ex.what())+"']");
        }
    }
    protected:
    friend class Factory;
    friend class FactoryHelperChain;
    bool isValid;
    std::string moduleName;
    std::string packageName;
    std::filesystem::path packagePath;
    std::set<std::string> aliases;
    std::string prefix;
    std::map<std::string, std::shared_ptr<FactoryHelper>> dependingModulePrefixes;

    std::map<std::string,std::map<std::string,Factory::creatorType>> creators;
    std::map<std::string,std::map<std::string,nl::json>> defaultModelConfigs;
    std::map<std::string,std::shared_ptr<FactoryHelper>> subHelpers;
    public:
    FactoryHelper(const std::string& _moduleName);
    ~FactoryHelper();
    void validate(const std::string& _packageName, const std::string& _packagePath);
    void addAliasPackageName(const std::string& _parent);
    bool hasAlias(const std::string& _packageName) const;
    void addDependency(const std::string& name, std::shared_ptr<FactoryHelper> helper);
    std::shared_ptr<FactoryHelper> createSubHelper(const std::string& name);
    std::pair<std::string,bool> resolveClassName(const std::string& baseName, const std::string& className) const;
    std::pair<std::string,bool> resolveModelName(const std::string& baseName, const std::string& modelName, std::shared_ptr<const Factory> factory=nullptr) const;
    void addClass(const std::string& baseName,const std::string& className,Factory::creatorType fn);
    void addPythonClass(const std::string& baseName,const std::string& className,py::object clsObj);
    void addDefaultModel(const std::string& baseName,const std::string& modelName,const nl::json& modelConfig_);
    void addDefaultModelsFromJson(const nl::json& j);
    void addDefaultModelsFromJsonFile(const std::string& filePath);
    std::filesystem::path getPackagePath() const;
};

ASRC_DECLARE_BASE_REF_TRAMPOLINE(FactoryHelper)
};

/**
 * @internal
 * @class FactoryHelperChain
 * @brief [For internal use] Entity インスタンスごとに生成され、クラス名やモデル名の検索を助けるクラス。
 */
class PYBIND11_EXPORT FactoryHelperChain{
    std::deque<std::shared_ptr<FactoryHelper>> helpers;
    public:
    FactoryHelperChain();
    FactoryHelperChain(std::shared_ptr<FactoryHelper> helper);
    FactoryHelperChain(const nl::json& chain_j);
    FactoryHelperChain(const std::vector<FactoryHelperChain>& chains);
    FactoryHelperChain(py::tuple args);
    template<typename... Sources>
    FactoryHelperChain(Sources... others){
        std::vector<FactoryHelperChain> others_v = {FactoryHelperChain(others)...};
        for(FactoryHelperChain& other: others_v){
            for(std::shared_ptr<FactoryHelper>& helper: other.helpers){
                helpers.push_back(helper);
            }
        }
    }
    ~FactoryHelperChain();
    void push_front(std::shared_ptr<FactoryHelper> helper);
    void push_back(std::shared_ptr<FactoryHelper> helper);
    std::shared_ptr<FactoryHelper> front();
    void pop_front();
    std::string resolveClassName(const std::string& baseName, const std::string& className) const;
    std::string resolveModelName(const std::string& baseName, const std::string& modelName, std::shared_ptr<const Factory> factory=nullptr) const;
    std::filesystem::path resolveFilePath(const std::string& path) const;
    std::filesystem::path resolveFilePath(const std::filesystem::path& path) const;
    template<asrc::core::traits::cereal_archive Archive>
    void serialize(Archive & archive){
        archive(helpers);
    }
};
#define ASRC_INTERNAL_CEREAL_PROLOGUE_EPILOGUE_DISABLER_FOR_FactoryHelperChain(Archive) \
inline void prologue(Archive & archive, const FactoryHelperChain& value){} \
inline void epilogue(Archive & archive, const FactoryHelperChain& value){} \

ASRC_INTERNAL_CEREAL_PROLOGUE_EPILOGUE_DISABLER_FOR_FactoryHelperChain(::asrc::core::util::NLJSONOutputArchive)
ASRC_INTERNAL_CEREAL_PROLOGUE_EPILOGUE_DISABLER_FOR_FactoryHelperChain(::asrc::core::util::NLJSONInputArchive)
ASRC_INTERNAL_CEREAL_PROLOGUE_EPILOGUE_DISABLER_FOR_FactoryHelperChain(::cereal::JSONOutputArchive)
ASRC_INTERNAL_CEREAL_PROLOGUE_EPILOGUE_DISABLER_FOR_FactoryHelperChain(::cereal::JSONInputArchive)
ASRC_INTERNAL_CEREAL_PROLOGUE_EPILOGUE_DISABLER_FOR_FactoryHelperChain(::cereal::XMLOutputArchive)
ASRC_INTERNAL_CEREAL_PROLOGUE_EPILOGUE_DISABLER_FOR_FactoryHelperChain(::cereal::XMLInputArchive)
#undef ASRC_INTERNAL_CEREAL_PROLOGUE_EPILOGUE_DISABLER_FOR_FactoryHelperChain


void exportFactory(py::module &m);

ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
