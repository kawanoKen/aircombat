/**
 * Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
 * @file
 * 
 * @brief つねに値型としてシリアライゼーションを行うタイプの基底クラス
 */
#pragma once
#include "util/macros/common_macros.h"
#include <map>
#include <memory>
#include <shared_mutex>
#include <pybind11/pybind11.h>
#include <nlohmann/json.hpp>
#include "traits/traits.h"
#include "util/serialization/serialization.h"
#include "PythonableBase.h"
namespace py=pybind11;
namespace nl=nlohmann;

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)

/**
 * @brief つねに値型としてシリアライゼーションを行うタイプのうち、多態的な振る舞いが必要な場合に用いる基底クラス
 * 
 * @details
 *      cerealを用いて nl::basic_json やバイナリデータとの相互変換を可能とするものである。
 * 
 *      ユーザ定義の基底クラスをPtrBaseTypeとしたとき、以下のような手順により使用する。
 *      1. PtrBaseTypeを、PolymorphicSerializableAsValue<PtrBaseType> を継承して宣言・定義する。
 * 
 *          このとき、後述するマクロを用いる等により、要件を満たす形で実装すること。
 * 
 *      2. PtrBaseTypeで以下のメンバ関数をオーバーライドして実際のシリアライズ処理を記述する。
 *          1. virtual void polymorphic_assign(const PtrBaseType& other)
 * 
 *              Archiveから読み込まれたPtrBaseType(又はその派生クラス)の値から自身の値を復元する。
 *              ポインタや参照でなく値型変数として使用している変数に対してin-placeな復元を行う際に、
 *              Archive中のデータが自身と異なる型だった場合にダウンキャスト・アップキャストに相当する柔軟な復元処理を可能とするために設けた。
 *              適宜dynamic_castによって引数の型を判定しつつ復元処理を行うこと。
 * 
 *          2. virtual void serialize_impl(asrc::core::traits::AvailableArchiveTypes & archive)
 * 
 *              archiveへの入出力処理を記述する。引数は、使用可能なArchive型の std::reference_wrapper の std::variant として渡される。
 *              operator()がないので、 archive(args...) の代わりに以下のヘルパーマクロ及び関数の使用を推奨する。いずれも入力・出力共通のインターフェースである。
 *              こだわりがなければ、 pybind11::object とそれ以外を区別しなくて済む[3]を使用することを推奨する。
 *              - [1] operator_call(archive,args...)
 * 
 *                  archiveの中身にstd::visitでアクセスしてそのoperator()(args...)を呼び出す。
 *                  pybind11::objectでなく通常のC++型を入出力する場合はこれを用いても問題ない。
 *              - [2] operator_func(func,archive,args...)
 * 
 *                  cerealのArchive又はAvailableArchiveTypesの非const参照とargs...を引数に取る関数funcを、archiveの中身にstd::visitでアクセスして呼び出す。
 *                  入出力処理をカスタマイズしたい場合に用いる。
 *              - [3] ASRC_SERIALIZE_NVP(archive,args...) 
 * 
 *                  asrc::core::util::serialize_with_type_infoをfuncとして[2]を呼び出すマクロ。
 *                  pybind11::objectを型情報とともに読み書きすることができる。もし[1]を用いて pybind11::object を出力すると型情報無しとなるためload時に別途与えなければならない。
 *              
 *              親クラスの関数を呼び出すため場合は BaseType::serialize_impl(archive);のように serialize_impl を直接呼ぶこと。
 * 
 *              cerealの標準の形式である archive(cereal::base_class<BaseType>(this)) を std::visit で呼び出さないこと。
 * 
 *      3. 全てのPtrBaseType及び派生クラスについて、シリアライズ実行前に以下のいずれかの関数が実行されるようにすること。
 *          - C++版派生クラスの場合: PtrBaseType::addDerivedCppClass<DerivedClass>();
 *          - Python版派生クラスの場合: PtrBaseType.addDerivedPythonClass(DerivedClass)
 * 
 *      4. 全てのPtrBaseType及び派生クラスについて、デフォルトコンストラクタが存在すること。
 *          デフォルトコンストラクタはprotectedであってもよいが、Python側からはinit可能となることに留意すること。
 * 
 *      5. PtrBaseTypeを継承する場合は、派生クラス用マクロを使用することを推奨する。
 * 
 *      6. Python側への公開は、Pythonモジュールとしてインポートされた際のエントリポイントから呼び出される場所で、 asrc::core::expose_common_class を呼び出す。
 *          例えば以下のようになる。
 *          @code{.cpp}
 *          ::asrc::core::expose_common_class<YourClass>(module,"YourClass")
 *          .def(::asrc::core::py_init<ArgTypes...>())
 *          .def(&other_function_bindings)
 *          ...
 *          ;
 *          @endcode
 * 
 *  ユーザによるカスタマイゼーションポイントは上記2.の2箇所を想定しているが、PtrBaseTypeにはそれ以外の共通要件が多く存在するため、
 *  ユーザは以下のマクロを使用してPtrBaseTypeを宣言・実装することを推奨する。
 * 
 *  ただし、クラステンプレートをPtrBaseTypeとして使用したい場合はこれらのマクロを使用できないため、同等の宣言及び実装をユーザ自身が行う必要がある。
 *  @par 基底クラス用マクロ
 *      1. ASRC_DECLARE_BASE_DATA_CLASS(className,...)
 * 
 *          ヘッダファイルに記述することでclassNameという名称でPtrBaseTypeを宣言する。
 *          クラス宣言の途中で切れるため、ユーザ定義メンバはこのマクロの次の行以降にそのまま記述すればよい。
 *          マクロ終了時点でのアクセス指定子はpublicである。
 *          最後にクラスを閉じる"};"を忘れないこと。
 *          マクロの可変引数部分には実験的に、他に継承したいインターフェースクラス(純粋仮想関数のみを持つクラス)を記述できるようにしている。
 * 
 *      2. ASRC_DEFINE_BASE_DATA_CLASS(className)
 * 
 *          ソースファイル(.cpp)に記述することでclassNameという名称でPtrBaseTypeの非テンプレート部分の実体を定義する。
 * 
 *      3. ASRC_DECLARE_BASE_DATA_TRAMPOLINE(className,...)
 * 
 *          ヘッダファイルのASRC_DECLARE_BASE_DATA_CLASSの後に記述することで、Python側でclassNameを継承するための中継用クラス("trampoline"クラス)が宣言される。
 *          クラス宣言の途中で切れるため、ユーザ定義の仮想メンバ関数がある場合はこのマクロの次の行以降にそのまま記述すればよい。
 *          マクロ終了時点でのアクセス指定子はpublicである。
 *          最後にクラスを閉じる"};"を忘れないこと。
 * 
 *  @par 派生クラス用マクロ(基底クラスがPolymorphicSerializableAsReferenceと共通)
 *      1. ASRC_DECLARE_DERIVED_CLASS_WITH_TRAMPOLINE(className,baseName,...)
 * 
 *          ヘッダファイルに記述することで、baseNameの派生クラスをclassNameという名称で宣言する。
 * 
 *          この派生クラスで新たにPython側での継承を想定する仮想メンバ関数を追加する場合にはこのマクロを用いる。
 *          クラス宣言の途中で切れるため、ユーザ定義メンバはこのマクロの次の行以降にそのまま記述すればよい。
 * 
 *          マクロ終了時点でのアクセス指定子はpublicである。
 *          最後にクラスを閉じる"};"を忘れないこと。
 * 
 *          マクロの可変引数部分には実験的に、他に継承したいインターフェースクラス(純粋仮想関数のみを持つクラス)を記述できるようにしている。
 * 
 *      2. ASRC_DECLARE_DERIVED_CLASS_WITHOUT_TRAMPOLINE(className,baseName)
 * 
 *          ヘッダファイルに記述することで、baseNameの派生クラスをclassNameという名称で宣言する。
 * 
 *          新たな仮想メンバ関数を追加しない場合にはこのマクロを用いる。
 *          クラス宣言の途中で切れるため、ユーザ定義メンバはこのマクロの次の行以降にそのまま記述すればよい。
 * 
 *          マクロ終了時点でのアクセス指定子はpublicである。
 *          最後にクラスを閉じる"};"を忘れないこと。
 * 
 *      3. ASRC_DECLARE_DERIVED_TRAMPOLINE(className,...)
 * 
 *          ヘッダファイルのASRC_DECLARE_DERIVED_CLASS_WITH_TRAMPOLINEの後に記述することで、Python側でclassNameを継承するための中継用クラス("trampoline"クラス)が宣言される。
 * 
 *          クラス宣言の途中で切れるため、ユーザ定義の仮想メンバ関数がある場合はこのマクロの次の行以降にそのまま記述すればよい。
 * 
 *          マクロ終了時点でのアクセス指定子はpublicである。
 *          最後にクラスを閉じる"};"を忘れないこと。
 * 
 *          クラス内クラスの場合もこのマクロを使用してASRC_DECLARE_DERIVED_TRAMPOLINE(outer::inner)のように記述できる。

 *  @par PtrBaseTypeの要件
 *      1. 以下の4つのstaticメンバ変数を持つこと。privateメンバとすることを推奨する。
 *          - static std::shared_mutex static_mtx;
 *          - static std::map<std::string,SaverType> savers;
 *          - static std::map<std::string,LoaderType> loaders;
 *          - static std::unordered_set<std::string> demangledNameCache;
 *          なお、PolymorphicSerializableAsValue自身がこれらのメンバを持たない理由は、テンプレートクラスであるせいで共有ライブラリ間での共有に難があるからである。
 *              1. ヘッダではextern template classとしておきソースファイルでインスタンス化する。
 *              2. 非テンプレートとなるPtrBaseType側のstaticメンバとして宣言し、ソースファイルで定義する。
 *          
 *          の2種類の対策を検討した結果、前者はexplicit instantiationに関するnamespaceの制約が強いため、後者を採用した。
 * 
 *  @par PtrBaseType及び派生クラスの要件
 *      1. 以下のエイリアステンプレートが定義されていること。
 *          - using Type = ... ; // 自分自身を表す。
 *          - using BaseType = ... ; // 自身の直接の親クラスを表す。
 *          - template<class T=Type> using TrampolineType = ... ; // 自身に対応する"trampoline"クラスを表す。
 *          - template<class T=Type> using PyClassInfo = ... ; // 自身をPython側に公開する際に使用されるヘルパークラスを表す。
 * 
 *      2. デフォルトコンストラクタを持つこと。
*/
template<class Base>
class PYBIND11_EXPORT PolymorphicSerializableAsValue : public PolymorphicPythonableBase<Base>{
    public:
    using PtrBaseType = Base; //!< ユーザ定義の、実質的な基底クラス。
    using Type = PolymorphicSerializableAsValue<PtrBaseType>; //!< 自分自身の型を表すエイリアス。
    using BaseType = PolymorphicPythonableBase<PtrBaseType>; //!< 直接の親クラスを表すエイリアス。
    using SaverType=std::function<void(::asrc::core::util::AvailableArchiveTypes&, PtrBaseType*)>;
    using LoaderType=std::function<void(::asrc::core::util::AvailableArchiveTypes&, std::shared_ptr<PtrBaseType>&)>;

    /**
     * @brief PtrBaseType又はその派生クラスのオブジェクトから自身への代入を行う関数。
     * 
     * @details
     *      必要に応じてダウンキャストをtry〜catchで囲んでメンバ変数の代入を試みていくことを想定。
     * 
     * @attention 派生クラスでオーバーライドすること。
     */
    virtual void polymorphic_assign(const PtrBaseType& other){
        // Override this function in derived class.
        throw std::runtime_error("You must override 'virtual void polymorphic_assign(const PtrBaseType&)' at least in "+cereal::util::demangle(typeid(PtrBaseType).name())+".");
    }
    /**
     * @brief 自身の値を表す情報を保存する処理の実体。
     * 
     * @attention 派生クラスでオーバーライドすること。
     */
    virtual void serialize_impl(::asrc::core::util::AvailableArchiveTypes& archive){
        // Override this function in derived class.
        throw std::runtime_error("You must override 'template<class Archive> void serialize_impl(AvailableArchiveTypes &)' at least in "+cereal::util::demangle(typeid(PtrBaseType).name())+".");
    }
    /**
     * @brief 自身の値を表す情報を nlohmann::json として返す。
     * 
     * @note nlohmann::adl_serializer 経由で↓の serialize を呼ぶ。
     */
    template<::asrc::core::traits::basic_json BasicJsonType=nl::ordered_json>
    BasicJsonType to_json() const{
        return *this;
    }
    /**
     * @brief 自身の値を表す nlohmann::json から復元する。
     * 
     * @note nlohmann::adl_serializer 経由で↓の serialize を呼ぶ。
     */
    template<::asrc::core::traits::basic_json BasicJsonType=nl::ordered_json>
    void load_from_json(const BasicJsonType& j){
        j.get_to(*this);
    }
    /**
     * @brief PtrBaseType又はその派生クラスの値を表す nlohmann::json から std::shared_ptr<T>として復元する。
     * 
     * @note nlohmann::adl_serializer 経由で↓の serialize を呼ぶ。
     */
    template<std::derived_from<PtrBaseType> T=PtrBaseType, ::asrc::core::traits::basic_json BasicJsonType=nl::ordered_json>
    static std::shared_ptr<T> from_json(const BasicJsonType& j){
        std::shared_ptr<T> ret;
        j.get_to(ret);
        return std::move(ret);
    }
    /**
     * @brief 自身の値を表す情報をバイナリデータとして返す。
     */
    std::string to_binary() const{
        std::stringstream oss;
        {
            cereal::PortableBinaryOutputArchive ar(oss);
            ar(*this);
        }
        return oss.str();
    }
    /**
     * @brief 自身の値を表すバイナリデータから復元する。
     */
    void load_from_binary(const std::string& str){
        std::istringstream iss(str);
        cereal::PortableBinaryInputArchive ar(iss);
        ar(*this);
    }
    /**
     * @brief PtrBaseType又はその派生クラスの値を表すバイナリデータから std::shared_ptr<T>として復元する。
     */
    template<std::derived_from<PtrBaseType> T=PtrBaseType>
    static std::shared_ptr<T> from_binary(const std::string& str){
        std::shared_ptr<T> ret;
        {
            std::istringstream iss(str);
            cereal::PortableBinaryInputArchive ar(iss);
            serialize_as_value(ar,ret);
        }
        return std::move(ret);
    }
    /**
     * @brief 自身の値を表す情報を保存する。
     */
    template<asrc::core::traits::cereal_archive Archive>
    requires (
        std::constructible_from<::asrc::core::util::AvailableArchiveTypes,std::reference_wrapper<Archive>>
    )
    void serialize(Archive & archive){
        if constexpr (asrc::core::traits::output_archive<Archive>){
            std::string demangledName = this->getDemangledName();
            if constexpr (::asrc::core::traits::same_archive_as<Archive,::asrc::core::NLJSONOutputArchive>){
                archive( CEREAL_NVP_("polymorphic_name",demangledName) );
            }else{
                // output archiveではpolymorphic_idがconst char*をキーとしたmapで管理されているため、
                // registerPolymorphicTypeに一時変数のdemangledName.c_str()を渡すと上手く動かない(同じ文字列でもアドレスが異なるため。)
                // この対策として、PyTypeInfoのstaticメンバとしてdemangledNameを保持しておくことでポインタを固定する。
                auto it=PtrBaseType::demangledNameCache.find(demangledName);
                if(it==PtrBaseType::demangledNameCache.end()){
                    it=PtrBaseType::demangledNameCache.insert(demangledName).first;
                }
                std::uint32_t nameid = archive.registerPolymorphicType((*it).c_str());
                archive( CEREAL_NVP_("polymorphic_id", nameid) );
                if( nameid & cereal::detail::msb_32bit ){
                    archive( CEREAL_NVP_("polymorphic_name", demangledName) );
                }
            }
            ::asrc::core::util::AvailableArchiveTypes wrapped(std::ref(archive));
            callSaver(demangledName,wrapped,static_cast<PtrBaseType*>(this));
        }else{
            std::string demangledName;
            if constexpr (::asrc::core::traits::same_archive_as<Archive,::asrc::core::NLJSONInputArchive>){
                const nl::ordered_json& j=archive.getCurrentNode();
                if(j.contains("polymorphic_name")){
                    demangledName=j.at("polymorphic_name");
                }else{
                    demangledName=this->getDemangledName();
                }
            }else{
                std::uint32_t nameid;
                archive( CEREAL_NVP_("polymorphic_id", nameid) );
                if( nameid & cereal::detail::msb_32bit ){
                    archive( CEREAL_NVP_("polymorphic_name", demangledName) );
                    archive.registerPolymorphicName(nameid, demangledName);
                }else{
                    demangledName = archive.getPolymorphicName(nameid);
                }
            }
            std::shared_ptr<PtrBaseType> sptr(std::shared_ptr<PtrBaseType>(),static_cast<PtrBaseType*>(this)); //no-deleter with alias constructor
            ::asrc::core::util::AvailableArchiveTypes wrapped(std::ref(archive));
            callLoader(demangledName,wrapped,sptr);
        }
    }
    /**
     * @brief std::shared_ptr<T> の値を表す情報を保存する。
     */
    template<
        asrc::core::traits::cereal_archive Archive,
        std::derived_from<PtrBaseType> T=PtrBaseType
    >
    static void serialize_as_value(Archive& archive, std::shared_ptr<T>& t){
        if constexpr (asrc::core::traits::output_archive<Archive>){
            if(t){
                // polymorphism handling
                std::string demangledName = t->getDemangledName();
                if constexpr (::asrc::core::traits::same_archive_as<Archive,::asrc::core::NLJSONOutputArchive>){
                    archive( CEREAL_NVP_("polymorphic_name",demangledName) );
                }else{
                    // output archiveではpolymorphic_idがconst char*をキーとしたmapで管理されているため、
                    // registerPolymorphicTypeに一時変数のdemangledName.c_str()を渡すと上手く動かない(同じ文字列でもアドレスが異なるため。)
                    // この対策として、PyTypeInfoのstaticメンバとしてdemangledNameを保持しておくことでポインタを固定する。
                    auto it=PtrBaseType::demangledNameCache.find(demangledName);
                    if(it==PtrBaseType::demangledNameCache.end()){
                        it=PtrBaseType::demangledNameCache.insert(demangledName).first;
                    }
                    std::uint32_t nameid = archive.registerPolymorphicType((*it).c_str());
                    archive( CEREAL_NVP_("polymorphic_id", nameid) );
                    if( nameid & cereal::detail::msb_32bit ){
                        archive( CEREAL_NVP_("polymorphic_name", demangledName) );
                    }
                }
                ::asrc::core::util::AvailableArchiveTypes wrapped(std::ref(archive));
                callSaver(demangledName,wrapped,t.get());
            }else{
                // nullptr
                if constexpr (::asrc::core::traits::same_archive_as<Archive,::asrc::core::NLJSONOutputArchive>){
                    std::string demangledName=cereal::util::demangle(typeid(std::nullptr_t).name());
                    archive( CEREAL_NVP_("polymorphic_name",demangledName) );
                }else{
                    std::uint32_t nameid = 0;
                    archive( CEREAL_NVP_("polymorphic_id", nameid) );
                }
            }
        }else{
            std::string demangledName;
            if constexpr (::asrc::core::traits::same_archive_as<Archive,::asrc::core::NLJSONInputArchive>){
                const nl::ordered_json& j=archive.getCurrentNode();
                if(j.contains("polymorphic_name")){
                    demangledName=j.at("polymorphic_name");
                }else{
                    if(t){
                        demangledName=t->getDemangledName();
                    }else{
                        demangledName=cereal::util::demangle(typeid(PtrBaseType).name());
                    }
                }
            }else{
                std::uint32_t nameid;
                archive( CEREAL_NVP_("polymorphic_id", nameid) );
                if(nameid==0){
                    // nullptr
                    demangledName=cereal::util::demangle(typeid(std::nullptr_t).name());
                }else if( nameid & cereal::detail::msb_32bit ){
                    archive( CEREAL_NVP_("polymorphic_name", demangledName) );
                    archive.registerPolymorphicName(nameid, demangledName);
                }else{
                    demangledName = archive.getPolymorphicName(nameid);
                }
            }
            if(demangledName != cereal::util::demangle(typeid(std::nullptr_t).name())){
                std::shared_ptr<PtrBaseType> empty;
                ::asrc::core::util::AvailableArchiveTypes wrapped(std::ref(archive));
                callLoader(demangledName,wrapped,empty);
                if constexpr (std::same_as<T,PtrBaseType>){
                    t=empty;
                }else{
                    if(empty){
                        t=std::dynamic_pointer_cast<T>(empty);
                        if(!t){
                            throw cereal::Exception("The loaded object is not an instance of '"+cereal::util::demangle(typeid(T).name())+"'.");
                        }
                    }else{
                        t=nullptr;
                    }
                }
            }else{
                t=nullptr;
            }
        }
    }
    /**
     * @internal
     * @brief [For internal use] nameで指定された型に対応する保存処理を呼び出す。
     */
    static void callSaver(const std::string& name, ::asrc::core::util::AvailableArchiveTypes& archive, PtrBaseType* src){
        if(PtrBaseType::savers.find(name)==PtrBaseType::savers.end()){
            throw std::runtime_error("'"+name+"' is not registered as a derived class of '"+cereal::util::demangle(typeid(PtrBaseType).name())+"'.");
        }else{
            PtrBaseType::savers.at(name)(archive,src);
        }
    }
    /**
     * @internal
     * @brief [For internal use] nameで指定された型に対応する復元処理を呼び出す。
     */
    static void callLoader(const std::string& name, ::asrc::core::util::AvailableArchiveTypes& archive, std::shared_ptr<PtrBaseType>& dst){
        if(PtrBaseType::loaders.find(name)==PtrBaseType::loaders.end()){
            throw std::runtime_error("'"+name+"' is not registered as a derived class of '"+cereal::util::demangle(typeid(PtrBaseType).name())+"'.");
        }
        PtrBaseType::loaders.at(name)(archive,dst);
    }
    /**
     * @brief PtrBase又はこれを継承したC++クラスに対応する保存・復元処理を登録する。
     * 
     * @tparam T 派生クラス
     * 
     * @attention 派生クラスを定義した場合、エントリポイントの中で必ず実行されるようにすること。
     *          その派生クラスをPython側に公開するタイミングで実行するのがよいと思われる。
     *          実例は Track.cpp にある。
     */
    template<std::derived_from<PtrBaseType> T=PtrBaseType>
    static void addDerivedCppClass(){
        std::lock_guard<std::shared_mutex> lock(PtrBaseType::static_mtx);
        std::string demangledName = cereal::util::demangle(typeid(T).name());
        if(PtrBaseType::savers.find(demangledName)==PtrBaseType::savers.end()){
            PtrBaseType::savers[demangledName]=&saverForCpp<T>;
        }
        if(PtrBaseType::loaders.find(demangledName)==PtrBaseType::loaders.end()){
            PtrBaseType::loaders[demangledName]=&loaderForCpp<T>;
        }
    }
    /**
     * @internal
     * @brief [For internal use] T型の保存処理
     */
    template<std::derived_from<PtrBaseType> T=PtrBaseType>
    static void saverForCpp(::asrc::core::util::AvailableArchiveTypes& archive, PtrBaseType* src){
        std::string demangledName = cereal::util::demangle(typeid(T).name());
        src->serialize_impl(archive);
    }
    /**
     * @internal
     * @brief [For internal use] T型の復元処理
     */
    template<std::derived_from<PtrBaseType> T=PtrBaseType>
    static void loaderForCpp(::asrc::core::util::AvailableArchiveTypes& archive, std::shared_ptr<PtrBaseType>& dst){
        std::string demangledName = cereal::util::demangle(typeid(T).name());
        if(dst){
            if(typeid(*dst.get())==typeid(T)){
                // 同じ型なら直接serialize_implを呼ぶ
                dst->serialize_impl(archive);
            }else{
                // 違う型なら代入演算子に委ねる
                T ret;
                ret.serialize_impl(archive);
                dst->polymorphic_assign(std::move(ret));
            }
        }else{
            std::shared_ptr<T> ret=T::template create<T>();
            ret->serialize_impl(archive);
            dst=std::move(ret);
        }
    }
    /**
     * @brief PtrBase又はこれを継承したPythonクラスに対応する保存・復元処理を登録する。
     * 
     * @param [in] clsObj 派生クラスオブジェクト
     * 
     * @attention 派生クラスを定義した場合、その定義直後に実行することを推奨する。
     */
    static void addDerivedPythonClass(pybind11::type clsObj){
        SaverType saver=[clsObj](::asrc::core::util::AvailableArchiveTypes& archive, PtrBaseType* src){
            pybind11::cast(src,pybind11::return_value_policy::reference).attr("serialize_impl")(pybind11::cast(archive,pybind11::return_value_policy::reference));
        };
        LoaderType loader=[clsObj](::asrc::core::util::AvailableArchiveTypes& archive, std::shared_ptr<PtrBaseType>& dst){
            if(dst){
                pybind11::object pyDst=pybind11::cast(dst,pybind11::return_value_policy::reference);
                if(clsObj.is(pybind11::type::of(pyDst))){
                    // 同じ型なら直接serialize_implを呼ぶ
                    pyDst.attr("serialize_impl")(pybind11::cast(archive,pybind11::return_value_policy::reference));
                }else{
                    // 違う型なら代入演算子に委ねる
                    pybind11::object obj = clsObj();
                    obj.attr("serialize_impl")(pybind11::cast(archive,pybind11::return_value_policy::reference));
                    pyDst.attr("polymorphic_assign")(obj);
                }
            }else{
                // Pythonオブジェクトを生成した後C++オブジェクトとしてのみの管理を行うとPython側で定義した派生クラスの情報が抜け落ちる。
                // そのため、Pythonオブジェクトをshared_ptrとしてPtrBaseTypeへのalias constructorでその寿命を紐付ける。
                std::shared_ptr<pybind11::object> obj = std::make_shared<pybind11::object>(clsObj());
                auto ret=std::shared_ptr<PtrBaseType>(obj,obj->cast<std::shared_ptr<PtrBaseType>>().get());
                obj->attr("serialize_impl")(pybind11::cast(archive,pybind11::return_value_policy::reference));
                dst=std::move(ret);
            }
        };
        std::lock_guard<std::shared_mutex> lock(PtrBaseType::static_mtx);
        std::string name=pybind11::repr(clsObj);
        if(PtrBaseType::savers.find(name)==PtrBaseType::savers.end()){
            PtrBaseType::savers[name]=saver;
        }
        if(PtrBaseType::loaders.find(name)==PtrBaseType::loaders.end()){
            PtrBaseType::loaders[name]=loader;
        }
    }
};

/**
 * @brief つねに値型としてシリアライゼーションを行うタイプのうち、多態的な振る舞いが不要な場合に用いる基底クラス
 * 
 */
template<class Base>
class PYBIND11_EXPORT NonPolymorphicSerializableAsValue : public NonPolymorphicPythonableBase<Base>{
    public:
    using NonPolymorphicPythonableBase<Base>::PyClassInfo;
};

/**
 * @brief Tが PolymorphicSerializableAsValue の派生クラスかどうかを判定する
 */
template<class T>
concept polymorphic_serializable_as_value = polymorphic_pythonable<T> && std::derived_from<T,PolymorphicSerializableAsValue<typename T::PtrBaseType>>;
/**
 * @brief Tが NonPolymorphicSerializableAsValue の派生クラスかどうかを判定する
 */
template<class T>
concept non_polymorphic_serializable_as_value = non_polymorphic_pythonable<T> && std::derived_from<T,NonPolymorphicSerializableAsValue<T>>;

template<polymorphic_serializable_as_value type_,class... options>
requires (
    pybind11::detail::all_of<pybind11::detail::is_strict_base_of<options, type_>...>::value
)
class PyClassInfoBase<type_,options...>{
    public:
    using PyClassType=pybind11::class_<type_,typename type_::template TrampolineType<>,std::shared_ptr<type_>,options...>;
    template <typename... Extra>
    static PyClassType class_(pybind11::handle scope, const char *name, const Extra &...extra) {
        return std::move(PyClassType(scope, name, extra...));
    }
};

template<polymorphic_serializable_as_value T>
struct def_common_serialization<T>{
    static constexpr bool op_enable_if_hook = true; // これによりcls.defのオーバーロードが実体化され、以下のexecuteを呼べるようになる

    template <typename Class, typename... Extra>
    requires ( std::same_as<T,typename Class::type> )
    void execute(Class &cl, const Extra &...extra) const{
        if constexpr (std::same_as<T,typename T::PtrBaseType>){
            cl.def("polymorphic_assign",&T::polymorphic_assign)
            .def("getDemangledName",&T::getDemangledName)
            .def("to_json",&T::template to_json<nl::ordered_json>)
            .def("load_from_json",&T::template load_from_json<nl::ordered_json>)
            .def_static("from_json",[](const nl::ordered_json& obj){return T::from_json(obj);})
            .def("to_binary",[](const T& v){return pybind11::bytes(v.to_binary());})
            .def("load_from_binary",[](T& v,const pybind11::bytes& str){return v.load_from_binary(str);})
            .def_static("from_binary",[](const pybind11::bytes& str){return T::from_binary(str);})
            .def("serialize_impl",&T::serialize_impl)
            .def("save",&::asrc::core::util::save_func_exposed_to_python<T>)
            .def("load",&::asrc::core::util::load_func_exposed_to_python<T>)
            .def_static("static_load",[](::asrc::core::util::AvailableInputArchiveTypes& archive){
                return ::asrc::core::util::static_load_func_exposed_to_python<
                    std::shared_ptr<typename T::PtrBaseType>
                >(archive);
            })
            .def_property_readonly_static("_allow_cereal_serialization_in_cpp",[](pybind11::object /* self */){return true;})
            ;
        }
        cl
        .def(pybind11::pickle(
            //TODO C++側のアドレスは一致するがPythonのobject idが変わるのでisが使えなくなる。Unpicklerでpersistent_loadを用いる必要がある。
            [](const T &t){//__getstate__
                return pybind11::bytes(t.to_binary());
            },
            [](const pybind11::bytes& data){//__setstate__
                return T::template from_binary<T>(data);
            }
        ))
        .def_static("addDerivedPythonClass",&T::addDerivedPythonClass)
        ;
    }
};

template<non_polymorphic_serializable_as_value T>
struct def_common_serialization<T>{
    static constexpr bool op_enable_if_hook = true; // これによりcls.defのオーバーロードが実体化され、以下のexecuteを呼べるようになる

    template <typename Class, typename... Extra>
    requires ( std::same_as<T,typename Class::type> )
    void execute(Class &cl, const Extra &...extra) const{
        cl
        .def("getDemangledName",&T::getDemangledName)
        .def("to_json",&T::template to_json<nl::ordered_json>)
        .def("load_from_json",&T::template load_from_json<nl::ordered_json>)
        .def_static("from_json",&T::template from_json<nl::ordered_json>)
        .def("to_binary",[](const T& v){return pybind11::bytes(v.to_binary());})
        .def("load_from_binary",[](T& v,const pybind11::bytes& str){return v.load_from_binary(str);})
        .def_static("from_binary",[](const pybind11::bytes& str){return T::from_binary(str);})
        .def("save",&::asrc::core::util::save_func_exposed_to_python<T>)
        .def("load",&::asrc::core::util::load_func_exposed_to_python<T>)
        .def_static("static_load",&::asrc::core::util::static_load_func_exposed_to_python<T>)
        .def_property_readonly_static("_allow_cereal_serialization_in_cpp",[](pybind11::object /* self */){return true;})
        .def(pybind11::pickle(
            //TODO C++側のアドレスは一致するがPythonのobject idが変わるのでisが使えなくなる。Unpicklerでpersistent_loadを用いる必要がある。
            [](const T &t){//__getstate__
                return pybind11::bytes(t.to_binary());
            },
            [](const pybind11::bytes& data){//__setstate__
                return T::from_binary(data);
            }
        ))
        ;
    }
};

ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)

namespace cereal{
    /**
     * @brief PolymorphicSerializableAsValue の保存処理を呼び出すcereal用関数
     */
    template <class Archive, ::asrc::core::polymorphic_serializable_as_value T>
    inline void CEREAL_SAVE_FUNCTION_NAME( Archive & ar, std::shared_ptr<T> & ptr )
    {
        ::asrc::core::traits::get_true_value_type_t<std::shared_ptr<T>>::serialize_as_value(ar,ptr);
    }
    /**
     * @brief PolymorphicSerializableAsValue の復元処理を呼び出すcereal用関数
     */
    template <class Archive, ::asrc::core::polymorphic_serializable_as_value T>
    inline void CEREAL_LOAD_FUNCTION_NAME( Archive & ar, std::shared_ptr<T> & ptr )
    {
        ::asrc::core::traits::get_true_value_type_t<std::shared_ptr<T>>::serialize_as_value(ar,ptr);
    }
    /**
     * @brief NonPolymorphicSerializableAsValue の保存処理を呼び出すcereal用関数
     */
    template <class Archive, ::asrc::core::non_polymorphic_serializable_as_value T>
    inline void CEREAL_SAVE_FUNCTION_NAME( Archive & ar, std::shared_ptr<T> const & ptr )
    {
        ar(CEREAL_NVP_("valid",bool(ptr)));
        if(ptr){
            ar(CEREAL_NVP_("value",*ptr));
        }
    }
    /**
     * @brief NonPolymorphicSerializableAsValue の復元処理を呼び出すcereal用関数
     */
    template <class Archive, ::asrc::core::non_polymorphic_serializable_as_value T>
    inline void CEREAL_LOAD_FUNCTION_NAME( Archive & ar, std::shared_ptr<T> & ptr )
    {
        bool valid;
        ar(CEREAL_NVP_("valid",valid));
        if(valid){
            if(ptr){
                ar(CEREAL_NVP_("value",*ptr));
            }else{
                ptr=std::make_shared<T>();
                ar(CEREAL_NVP_("value",*ptr));
            }
        }
    }
}

/**
 * @brief つねに値型としてシリアライゼーションを行う基底クラスの共通部分の定義を行うマクロ。
 * ソースファイル側で呼ぶ。
 * 
 * @param className クラス名
 */
#define ASRC_DEFINE_BASE_DATA_CLASS(className)\
std::shared_mutex className::static_mtx;\
std::map<std::string, className::SaverType> className::savers;\
std::map<std::string, className::LoaderType> className::loaders;\
std::unordered_set<std::string> className::demangledNameCache;\

#ifndef __DOXYGEN__
/**
 * @brief つねに値型としてシリアライゼーションを行う基底クラスを共通部分とともに宣言するマクロ。
 * ヘッダファイル側で呼ぶ。
 * 
 * @param className クラス名
 * @param ... PolymorphicSerializableAsValue 以外に継承元のインターフェースクラスがある場合それらのクラス名を書く
 */
#define ASRC_DECLARE_BASE_DATA_CLASS(className,...)\
class className;\
template<class Base>\
class className##Wrap;\
class PYBIND11_EXPORT className: public ::asrc::core::PolymorphicSerializableAsValue<className> __VA_OPT__(, ASRC_INTERNAL_INHERIT_INTERFACE_CLASS(__VA_ARGS__)){\
    friend class ::asrc::core::PolymorphicSerializableAsValue<className>;\
    pybind11::handle _internal_handle;\
    private:\
    static std::shared_mutex static_mtx;\
    static std::map<std::string,SaverType> savers;\
    static std::map<std::string,LoaderType> loaders;\
    static std::unordered_set<std::string> demangledNameCache;\
    public:\
    using Type = className;\
    using BaseType = ::asrc::core::PolymorphicSerializableAsValue<className>;\
    template<class T=Type> using TrampolineType=className##Wrap<T>;\
    template<class T=Type> using PyClassInfo=::asrc::core::PyClassInfoBase<T __VA_OPT__(, ASRC_INTERNAL_ENUMERATE_INTERFACE_CLASS(__VA_ARGS__))>;\

/**
 * つねに値型としてシリアライゼーションを行う基底クラスのtrampolineクラスを共通部分とともに宣言するマクロ。
 * ヘッダファイル側で呼ぶ。
 * 
 * @param className クラス名
 * @param ... PolymorphicSerializableAsValue 以外に継承元のインターフェースクラスがある場合それらのクラス名を書く
 */
#define ASRC_DECLARE_BASE_DATA_TRAMPOLINE(className,...)\
template<class Base=className>\
class className##Wrap: public virtual Base __VA_OPT__(, ASRC_INTERNAL_INHERIT_INTERFACE_TRAMPOLINE(__VA_ARGS__)){\
    public:\
    ASRC_INTERNAL_USING_INTERFACE_TRAMPOLINE_CONSTRUCTORS(__VA_ARGS__)\
    using Base::Base;\
    virtual void polymorphic_assign(const typename Base::PtrBaseType& other) override{\
        PYBIND11_OVERRIDE(void,Base,polymorphic_assign,other);\
    }\
    virtual void serialize_impl(::asrc::core::util::AvailableArchiveTypes& archive) override{\
        PYBIND11_OVERRIDE(void,Base,serialize_impl,archive);\
    }\
    virtual std::string getDemangledName() const override{\
        return pybind11::repr(pybind11::type::of(pybind11::cast(this,pybind11::return_value_policy::reference)));\
    }\

/**
 * @brief つねに値型としてシリアライゼーションを行う非ポリモーフィックなクラスを共通部分とともに宣言するマクロ。
 * ヘッダファイル側で呼ぶ。
 * 
 * @param className クラス名
 * @param ... NonPolymorphicSerializableAsValue 以外に継承元のインターフェースクラスがある場合それらのクラス名を書く
 */
#define ASRC_DECLARE_NON_POLYMORPHIC_DATA_CLASS(className,...)\
class PYBIND11_EXPORT className: __VA_OPT__(ASRC_INTERNAL_INHERIT_CLASS(__VA_ARGS__) , ) public ::asrc::core::NonPolymorphicSerializableAsValue<className>{\
    public:\
    template<class T=className> using PyClassInfo=::asrc::core::PyClassInfoBase<T __VA_OPT__(, ASRC_INTERNAL_ENUMERATE_INTERFACE_CLASS(__VA_ARGS__))>;\
    template<::asrc::core::traits::basic_json BasicJsonType=nl::ordered_json>\
    BasicJsonType to_json() const{\
        return *this;\
    }\
    template<::asrc::core::traits::basic_json BasicJsonType=nl::ordered_json>\
    void load_from_json(const BasicJsonType& j){\
        j.get_to(*this);\
    }\
    template<::asrc::core::traits::basic_json BasicJsonType=nl::ordered_json>\
    static className from_json(const BasicJsonType& j){\
        className ret;\
        j.get_to(ret);\
        return std::move(ret);\
    }\
    std::string to_binary() const{\
        std::stringstream oss;\
        {\
            cereal::PortableBinaryOutputArchive ar(oss);\
            ar(*this);\
        }\
        return oss.str();\
    }\
    void load_from_binary(const std::string& str){\
        std::istringstream iss(str);\
        cereal::PortableBinaryInputArchive ar(iss);\
        ar(*this);\
    }\
    static className from_binary(const std::string& str){\
        className ret;\
        {\
            std::istringstream iss(str);\
            cereal::PortableBinaryInputArchive ar(iss);\
            ar(ret);\
        }\
        return std::move(ret);\
    }\

#else
//
// Doxygen使用時の__VA_OPT__不使用版
//

/**
 * @brief ASRC_DECLARE_BASE_DATA_CLASS の可変引数がない場合の実体
 * 
 * @details Doxygenは__VA_OPT__に対応していないため、可変引数の有無で分岐するマクロとしてドキュメント化している。
 */
#define ASRC_DECLARE_BASE_DATA_CLASS_IMPL_NO_VA_ARG(className)\
class className;\
template<class Base>\
class className##Wrap;\
class PYBIND11_EXPORT className: public ::asrc::core::PolymorphicSerializableAsValue<className>{\
    friend class ::asrc::core::PolymorphicSerializableAsValue<className>;\
    pybind11::handle _internal_handle;\
    private:\
    static std::shared_mutex static_mtx;\
    static std::map<std::string,SaverType> savers;\
    static std::map<std::string,LoaderType> loaders;\
    static std::unordered_set<std::string> demangledNameCache;\
    public:\
    using Type = className;\
    using BaseType = ::asrc::core::PolymorphicSerializableAsValue<className>;\
    template<class T=Type> using TrampolineType=className##Wrap<T>;\
    template<class T=Type> using PyClassInfo=::asrc::core::PyClassInfoBase<T>;

/**
 * @brief ASRC_DECLARE_BASE_DATA_CLASS の可変引数がある場合の実体
 * 
 * @details Doxygenは__VA_OPT__に対応していないため、可変引数の有無で分岐するマクロとしてドキュメント化している。
 */
#define ASRC_DECLARE_BASE_DATA_CLASS_IMPL_VA_ARGS(className,...)\
class className;\
template<class Base>\
class className##Wrap;\
class PYBIND11_EXPORT className: public ::asrc::core::PolymorphicSerializableAsValue<className> , ASRC_INTERNAL_INHERIT_INTERFACE_CLASS(__VA_ARGS__){\
    friend class ::asrc::core::PolymorphicSerializableAsValue<className>;\
    pybind11::handle _internal_handle;\
    private:\
    static std::shared_mutex static_mtx;\
    static std::map<std::string,SaverType> savers;\
    static std::map<std::string,LoaderType> loaders;\
    static std::unordered_set<std::string> demangledNameCache;\
    public:\
    using Type = className;\
    using BaseType = ::asrc::core::PolymorphicSerializableAsValue<className>;\
    template<class T=Type> using TrampolineType=className##Wrap<T>;\
    template<class T=Type> using PyClassInfo=::asrc::core::PyClassInfoBase<T, ASRC_INTERNAL_ENUMERATE_INTERFACE_CLASS(__VA_ARGS__)>;

/**
 * @brief つねに値型としてシリアライゼーションを行う基底クラスの共通部分の定義を行うマクロ。
 * ソースファイル側で呼ぶ。
 * 
 * @param className クラス名
 */
#define ASRC_DECLARE_BASE_DATA_CLASS(...) ASRC_INTERNAL_DEFINE_VA_MACRO(ASRC_DECLARE_BASE_DATA_CLASS,1,__VA_ARGS__)

/**
 * @brief ASRC_DECLARE_BASE_DATA_TRAMPOLINE の可変引数がない場合の実体
 * 
 * @details Doxygenは__VA_OPT__に対応していないため、可変引数の有無で分岐するマクロとしてドキュメント化している。
 */
#define ASRC_DECLARE_BASE_DATA_TRAMPOLINE_IMPL_NO_VA_ARG(className)\
template<class Base=className>\
class className##Wrap: public virtual Base{\
    public:\
    using Base::Base;\
    virtual void polymorphic_assign(const typename Base::PtrBaseType& other) override{\
        PYBIND11_OVERRIDE(void,Base,polymorphic_assign,other);\
    }\
    virtual void serialize_impl(::asrc::core::util::AvailableArchiveTypes& archive) override{\
        PYBIND11_OVERRIDE(void,Base,serialize_impl,archive);\
    }\
    virtual std::string getDemangledName() const override{\
        return pybind11::repr(pybind11::type::of(pybind11::cast(this,pybind11::return_value_policy::reference)));\
    }

/**
 * @brief ASRC_DECLARE_BASE_DATA_TRAMPOLINE の可変引数がある場合の実体
 * 
 * @details Doxygenは__VA_OPT__に対応していないため、可変引数の有無で分岐するマクロとしてドキュメント化している。
 */
#define ASRC_DECLARE_BASE_DATA_TRAMPOLINE_IMPL_VA_ARGS(className,...)\
template<class Base=className>\
class className##Wrap: public virtual Base , ASRC_INTERNAL_INHERIT_INTERFACE_TRAMPOLINE(__VA_ARGS__){\
    public:\
    ASRC_INTERNAL_USING_INTERFACE_TRAMPOLINE_CONSTRUCTORS(__VA_ARGS__)\
    using Base::Base;\
    virtual void polymorphic_assign(const typename Base::PtrBaseType& other) override{\
        PYBIND11_OVERRIDE(void,Base,polymorphic_assign,other);\
    }\
    virtual void serialize_impl(::asrc::core::util::AvailableArchiveTypes& archive) override{\
        PYBIND11_OVERRIDE(void,Base,serialize_impl,archive);\
    }\
    virtual std::string getDemangledName() const override{\
        return pybind11::repr(pybind11::type::of(pybind11::cast(this,pybind11::return_value_policy::reference)));\
    }

/**
 * つねに値型としてシリアライゼーションを行う基底クラスのtrampolineクラスを共通部分とともに宣言するマクロ。
 * ヘッダファイル側で呼ぶ。
 * 
 * @param className クラス名
 * @param ... PolymorphicSerializableAsValue 以外に継承元のインターフェースクラスがある場合それらのクラス名を書く
 */
#define ASRC_DECLARE_BASE_DATA_TRAMPOLINE(...) ASRC_INTERNAL_DEFINE_VA_MACRO(ASRC_DECLARE_BASE_DATA_TRAMPOLINE,1,__VA_ARGS__)

/**
 * @brief ASRC_DECLARE_NON_POLYMORPHIC_DATA_CLASS の可変引数がない場合の実体
 * 
 * @details Doxygenは__VA_OPT__に対応していないため、可変引数の有無で分岐するマクロとしてドキュメント化している。
 */
#define ASRC_DECLARE_NON_POLYMORPHIC_DATA_CLASS_IMPL_NO_VA_ARG(className)\
class PYBIND11_EXPORT className: public ::asrc::core::NonPolymorphicSerializableAsValue<className>{\
    public:\
    template<class T=className> using PyClassInfo=::asrc::core::PyClassInfoBase<T>;\
    template<::asrc::core::traits::basic_json BasicJsonType=nl::ordered_json>\
    BasicJsonType to_json() const{\
        return *this;\
    }\
    template<::asrc::core::traits::basic_json BasicJsonType=nl::ordered_json>\
    void load_from_json(const BasicJsonType& j){\
        j.get_to(*this);\
    }\
    template<::asrc::core::traits::basic_json BasicJsonType=nl::ordered_json>\
    static className from_json(const BasicJsonType& j){\
        className ret;\
        j.get_to(ret);\
        return std::move(ret);\
    }\
    std::string to_binary() const{\
        std::stringstream oss;\
        {\
            cereal::PortableBinaryOutputArchive ar(oss);\
            ar(*this);\
        }\
        return oss.str();\
    }\
    void load_from_binary(const std::string& str){\
        std::istringstream iss(str);\
        cereal::PortableBinaryInputArchive ar(iss);\
        ar(*this);\
    }\
    static className from_binary(const std::string& str){\
        className ret;\
        {\
            std::istringstream iss(str);\
            cereal::PortableBinaryInputArchive ar(iss);\
            ar(ret);\
        }\
        return std::move(ret);\
    }

/**
 * @brief ASRC_DECLARE_NON_POLYMORPHIC_DATA_CLASS の可変引数がある場合の実体
 * 
 * @details Doxygenは__VA_OPT__に対応していないため、可変引数の有無で分岐するマクロとしてドキュメント化している。
 */
#define ASRC_DECLARE_NON_POLYMORPHIC_DATA_CLASS_IMPL_VA_ARGS(className,...)\
class PYBIND11_EXPORT className: ASRC_INTERNAL_INHERIT_CLASS(__VA_ARGS__) , public ::asrc::core::NonPolymorphicSerializableAsValue<className>{\
    public:\
    template<class T=className> using PyClassInfo=::asrc::core::PyClassInfoBase<T, ASRC_INTERNAL_ENUMERATE_INTERFACE_CLASS(__VA_ARGS__)>;\
    template<::asrc::core::traits::basic_json BasicJsonType=nl::ordered_json>\
    BasicJsonType to_json() const{\
        return *this;\
    }\
    template<::asrc::core::traits::basic_json BasicJsonType=nl::ordered_json>\
    void load_from_json(const BasicJsonType& j){\
        j.get_to(*this);\
    }\
    template<::asrc::core::traits::basic_json BasicJsonType=nl::ordered_json>\
    static className from_json(const BasicJsonType& j){\
        className ret;\
        j.get_to(ret);\
        return std::move(ret);\
    }\
    std::string to_binary() const{\
        std::stringstream oss;\
        {\
            cereal::PortableBinaryOutputArchive ar(oss);\
            ar(*this);\
        }\
        return oss.str();\
    }\
    void load_from_binary(const std::string& str){\
        std::istringstream iss(str);\
        cereal::PortableBinaryInputArchive ar(iss);\
        ar(*this);\
    }\
    static className from_binary(const std::string& str){\
        className ret;\
        {\
            std::istringstream iss(str);\
            cereal::PortableBinaryInputArchive ar(iss);\
            ar(ret);\
        }\
        return std::move(ret);\
    }

/**
 * @brief つねに値型としてシリアライゼーションを行う非ポリモーフィックなクラスを共通部分とともに宣言するマクロ。
 * ヘッダファイル側で呼ぶ。
 * 
 * @param className クラス名
 * @param ... NonPolymorphicSerializableAsValue 以外に継承元のインターフェースクラスがある場合それらのクラス名を書く
 */
#define ASRC_DECLARE_NON_POLYMORPHIC_DATA_CLASS(...) ASRC_INTERNAL_DEFINE_VA_MACRO(ASRC_DECLARE_NON_POLYMORPHIC_DATA_CLASS,1,__VA_ARGS__)

#endif
//
