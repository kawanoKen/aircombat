/**
 * Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
 * @file
 * 
 * @brief つねに既存インスタンスへの参照型としてシリアライゼーションを行うタイプの基底クラス
 */
#pragma once
#include "util/macros/common_macros.h"
#include <memory>
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
 * @brief つねに既存インスタンスへの参照型としてシリアライゼーションを行うタイプの基底クラス。
 * 
 * @details
 *      enabled_shaed_from_thisの継承によりthisポインタからshared_ptrを取得可能にしつつ、
 *      cerealを用いてnl::basic_jsonやバイナリとの相互変換を可能とするものである。
 * 
 *      ユーザ定義の基底クラスをPtrBaseTypeとしたとき、以下のような手順により使用する。
 *      1. PtrBaseTypeを、PolymorphicSerializableAsReference<PtrBaseType>を継承して宣言・定義する。
 * 
 *          このとき、後述するマクロを用いる等により、要件を満たす形で実装すること。
 * 
 *      2. PtrBaseTypeで以下のメンバ関数をオーバーライドして実際のシリアライズ処理を記述する。
 * 
 *          オーバーライドは省略可能だが、デフォルトでは生ポインタのアドレス値そのものを入出力する危険な形であるため、できる限りオーバーライドすることを推奨する。
 *          特に、このクラスではshared_ptrのインスタンス管理は一切行わないため、他のクラスで適切に管理されていることを確認して使用すること。
 *          1. template<class Archive> static void save_as_reference_impl(Archive& archive, const std::shared_ptr<const PtrBaseType>& t)
 * 
 *              Archiveへの出力処理を記述する。
 * 
 *          2. template<class Archive> static std::shared_ptr<PtrBaseType> load_as_reference_impl(Archive& archive)
 * 
 *              Archiveからの読み込み処理を記述する。
 * 
 *      3. PtrBaseTypeを継承する場合は、派生クラス用マクロを使用することを推奨する。
 * 
 *      4. Python側への公開は、Pythonモジュールとしてインポートされた際のエントリポイントから呼び出される場所で、 asrc::core::expose_common_class を呼び出す。
 *          例えば以下のようになる。
 *          @code{.cpp}
 *          ::asrc::core::expose_common_class<YourClass>(module,"YourClass")
 *          .def(::asrc::core::py_init<ArgTypes...>())
 *          .def(&other_function_bindings)
 *          ...
 *          ;
 *          @endcode

 *  ユーザによるカスタマイゼーションポイントは上記2.の2箇所を想定しているが、PtrBaseTypeにはそれ以外の共通要件が多く存在するため、
 *  ユーザは以下のマクロを使用してPtrBaseTypeを宣言・実装することを推奨する。
 *  
 *  ただし、クラステンプレートをPtrBaseTypeとして使用したい場合はこれらのマクロを使用できないため、同等の宣言及び実装をユーザ自身が行う必要がある。
 *  @par 基底クラス用マクロ
 *      1. ASRC_DECLARE_BASE_REF_CLASS(className,...)
 * 
 *          ヘッダファイルに記述することでclassNameという名称でPtrBaseTypeを宣言する。
 * 
 *          クラス宣言の途中で切れるため、ユーザ定義メンバはこのマクロの次の行以降にそのまま記述すればよい。
 * 
 *          マクロ終了時点でのアクセス指定子はpublicである。
 *          最後にクラスを閉じる"};"を忘れないこと。
 * 
 *          マクロの可変引数部分には実験的に、他に継承したいインターフェースクラス(純粋仮想関数のみを持つクラス)を記述できるようにしている。
 * 
 *      2. ASRC_DECLARE_BASE_REF_TRAMPOLINE(className,...)
 * 
 *          ヘッダファイルのASRC_DECLARE_BASE_REF_CLASSの後に記述することで、Python側でclassNameを継承するための中継用クラス("trampoline"クラス)が宣言される。
 * 
 *          クラス宣言の途中で切れるため、ユーザ定義の仮想メンバ関数がある場合はこのマクロの次の行以降にそのまま記述すればよい。
 * 
 *          マクロ終了時点でのアクセス指定子はpublicである。
 *          最後にクラスを閉じる"};"を忘れないこと。
 * 
 *  @par 派生クラス用マクロ(基底クラスが PolymorphicSerializableAsValue と共通)
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
 *          ヘッダファイルの ASRC_DECLARE_DERIVED_CLASS_WITH_TRAMPOLINE の後に記述することで、Python側でclassNameを継承するための中継用クラス("trampoline"クラス)が宣言される。
 * 
 *          クラス宣言の途中で切れるため、ユーザ定義の仮想メンバ関数がある場合はこのマクロの次の行以降にそのまま記述すればよい。
 * 
 *          マクロ終了時点でのアクセス指定子はpublicである。
 *          最後にクラスを閉じる"};"を忘れないこと。
 * 
 *          クラス内クラスの場合もこのマクロを使用して ASRC_DECLARE_DERIVED_TRAMPOLINE(outer::inner) のように記述できる。

 *  @par PtrBaseType及び派生クラスの要件
 *      1. 以下のエイリアステンプレートが定義されていること。
 *          - using Type = ... ; // 自分自身を表す。
 *          - using BaseType = ... ; // 自身の直接の親クラスを表す。
 *          - template<class T=Type> using TrampolineType = ... ; // 自身に対応する"trampoline"クラスを表す。
 *          - template<class T=Type> using PyClassInfo = ... ; // 自身をPython側に公開する際に使用されるヘルパークラスを表す。
 */
template<class Base>
class PYBIND11_EXPORT PolymorphicSerializableAsReference: public PolymorphicPythonableBase<Base>, public std::enable_shared_from_this<Base>{
    
    public:
    using PtrBaseType = Base; //!< ユーザ定義の、実質的な基底クラスを表す。
    using Type = PolymorphicSerializableAsReference<PtrBaseType>; //!< 自分自身を表す。
    using BaseType = PolymorphicPythonableBase<PtrBaseType>; //!< 自身の直接の親クラスを表す。

    /**
     * @brief std::shared_ptr<PtrBaseType> への参照を表す情報を保存する処理の実体。
     * 
     * @attention この基底クラスでは生ポインタのアドレスをそのまま使用しているため、
     *          派生クラスではなるべくオーバーライドしてより安全な方法をとることを推奨する。
     */
    template<asrc::core::traits::output_archive Archive>
    static void save_as_reference_impl(Archive& ar, const std::shared_ptr<const PtrBaseType>& t){
        // Override this function in derived class.
        intptr_t value=reinterpret_cast<intptr_t>(t.get());
        ar(CEREAL_NVP(value));
    }

    /**
     * std::shared_ptr<PtrBaseType> への参照を表す情報から復元する処理の実体。
     * 
     * @attention この基底クラスでは生ポインタのアドレスをそのまま使用しているため、
     *          派生クラスではなるべくオーバーライドしてより安全な方法をとることを推奨する。
     */
    template<asrc::core::traits::input_archive Archive>
    static std::shared_ptr<PtrBaseType> load_as_reference_impl(Archive& ar){
        // Override this function in derived class.
        intptr_t value;
        ar(value);
        PtrBaseType* raw=reinterpret_cast<PtrBaseType*>(value);
        assert(raw);
        return raw->shared_from_this();
    }

    /**
     * @brief std::shared_ptr<PtrBaseType> への参照を表す情報を保存する。
     */
    template<
        asrc::core::traits::output_archive Archive,
        std::derived_from<PtrBaseType> T=PtrBaseType
    >
    static void save_as_reference(Archive& ar, const std::shared_ptr<std::add_const_t<T>>& t){
        ar(::cereal::make_nvp("valid",bool(t)));
        if(t){
            PtrBaseType::save_as_reference_impl(ar,t);
        }
    }
    /**
     * std::shared_ptr<PtrBaseType> への参照を表す情報から復元する。
     */
    template<
        asrc::core::traits::input_archive Archive,
        std::derived_from<PtrBaseType> T=PtrBaseType
    >
    static void load_as_reference(Archive& ar, std::shared_ptr<T>& t){
        bool valid;
        ar(CEREAL_NVP(valid));
        if(valid){
            t = std::dynamic_pointer_cast<T>(PtrBaseType::load_as_reference_impl(ar));
        }else{
            t = nullptr;
        }
    }

    /**
     * @brief 自身への参照を表す情報を nlohmann::json として返す。
     * 
     * @note nlohmann::adl_serializer 経由で↑の save_as_reference を呼ぶ。
     */
    template<::asrc::core::traits::basic_json BasicJsonType=nl::ordered_json>
    BasicJsonType to_json() const{
        return this->shared_from_this(); // nlohmann::adl_serializer経由で↑のsave_as_referenceを呼ぶ。
    }
    /**
     * @brief std::shared_ptr<PtrBaseType> への参照を表す nlohmann::json から復元する。
     * 
     * @note nlohmann::adl_serializer 経由で↑の load_as_reference を呼ぶ。
     */
    template<std::derived_from<PtrBaseType> T=PtrBaseType, ::asrc::core::traits::basic_json BasicJsonType=nl::ordered_json>
    static std::shared_ptr<T> from_json(const BasicJsonType& j){
        return j;  // nlohmann::adl_serializer経由で↑のload_as_referenceを呼ぶ。
    }
    /**
     * @brief std::shared_ptr<PtrBaseType> への参照を表す nlohmann::json から復元し、 std::weak_ptr として返す。
     * 
     * @note nlohmann::adl_serializer 経由で↑の load_as_reference を呼ぶ。
     */
    template<std::derived_from<PtrBaseType> T=PtrBaseType, ::asrc::core::traits::basic_json BasicJsonType=nl::ordered_json>
    static std::weak_ptr<T> from_json_weakref(const BasicJsonType& j){
        return j; // nlohmann::adl_serializer経由で↑のload_as_referenceを呼ぶ。
    }
    /**
     * @brief 自身への参照を表す情報をバイナリデータとして返す。
     */
    std::string to_binary() const{
        std::stringstream oss;
        {
            cereal::PortableBinaryOutputArchive ar(oss);
            ar(this->shared_from_this());
        }
        return oss.str();
    }
    /**
     * @brief std::shared_ptr<PtrBaseType> への参照を表すバイナリデータから復元する。
     */
    template<std::derived_from<PtrBaseType> T=PtrBaseType>
    static std::shared_ptr<T> from_binary(const std::string& str){
        std::shared_ptr<T> ret;
        {
            std::istringstream iss(str);
            cereal::PortableBinaryInputArchive ar(iss);
            ar(ret);
        }
        return ret;
    }
    /**
     * @brief 自std::shared_ptr<PtrBaseType> への参照を表すバイナリデータから復元し、 std::weak_ptr として返す。
     */
    template<std::derived_from<PtrBaseType> T=PtrBaseType>
    static std::weak_ptr<T> from_binary_weakref(const std::string& str){
        std::weak_ptr<T> ret;
        {
            std::istringstream iss(str);
            cereal::PortableBinaryInputArchive ar(iss);
            ar(ret);
        }
        return ret;
    }
};

/**
 * @brief Tが PolymorphicSerializableAsReference の派生クラスかどうかを判定する
 */
template<class T>
concept polymorphic_serializable_as_reference = polymorphic_pythonable<T> && std::derived_from<T,PolymorphicSerializableAsReference<typename T::PtrBaseType>>;

template<polymorphic_serializable_as_reference type_,class... options>
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

/**
 * @brief 派生クラスのweak_ptrから基底クラスのweak_ptrへのimplicit conversionを追加する。
 * 
 * @details weak_ref同士は継承関係にないため、各派生クラスに対応するコンストラクタを基底クラス側に個別に用意する必要がある。
 */
template<polymorphic_serializable_as_reference T,polymorphic_serializable_as_reference U=typename T::BaseType>
void setup_implicitly_convertible_chain_from_weak_to_weak(){
    auto cls=pybind11::type::of<std::weak_ptr<U>>().template cast<pybind11::class_<std::weak_ptr<U>>>();
    cls.def(py_init<const std::weak_ptr<T>&>());
    pybind11::implicitly_convertible<std::weak_ptr<T>, std::weak_ptr<U>>();
    if constexpr (!std::same_as<U,typename T::PtrBaseType>){
        setup_implicitly_convertible_chain_from_weak_to_weak<T,typename U::BaseType>();
    }
}
/**
 * @brief 派生クラスのshared_ptrから基底クラスのweak_ptrへのimplicit conversionを追加する。
 */
template<polymorphic_serializable_as_reference T,polymorphic_serializable_as_reference U=typename T::BaseType>
void setup_implicitly_convertible_chain_from_shared_to_weak(){
    pybind11::implicitly_convertible<std::shared_ptr<T>, std::weak_ptr<U>>();
    if constexpr (!std::same_as<U,typename T::PtrBaseType>){
        setup_implicitly_convertible_chain_from_shared_to_weak<T,typename U::BaseType>();
    }
}

template<polymorphic_serializable_as_reference T>
struct weak_ptr_exposer<T> {
    template<typename Parent, typename ... Extras>
    static void expose(Parent& m, const char* className, Extras&& ... extras){
        using WeakType = std::weak_ptr<T>;
        std::string weakTypeName="weak_ptr<"+std::string(className)+">";
        pybind11::class_<WeakType>(m,weakTypeName.c_str(),std::forward<Extras>(extras)...)
        .def(py_init<const std::shared_ptr<T>&>())
        .def(py_init<const std::weak_ptr<T>&>())
        .def("__call__",[](WeakType& v) -> pybind11::object {
            if(v.expired()){
                return pybind11::none();
            }else{
                return pybind11::cast(v.lock());
            }
        })
        .def("use_count",&WeakType::use_count)
        .def("to_json",[](const WeakType& v){
            nl::ordered_json ret=v;
            return std::move(ret);
        })
        .def_static("from_json",[](const nl::ordered_json& obj){return T::template from_json_weakref<T>(obj);})
        .def("to_binary",[](const WeakType& v){
            std::stringstream oss;
            {
                cereal::PortableBinaryOutputArchive ar(oss);
                ar(v);
            }
            return pybind11::bytes(oss.str());
        })
        .def_static("from_binary",[](const pybind11::bytes& str){return T::template from_binary_weakref<T>(str);})
        .def("save",&::asrc::core::util::save_func_exposed_to_python<WeakType>)
        .def("load",&::asrc::core::util::load_func_exposed_to_python<WeakType>)
        .def_static("static_load",&::asrc::core::util::static_load_func_exposed_to_python<WeakType>)
        .def_property_readonly_static("_allow_cereal_serialization_in_cpp",[](pybind11::object /* self */){return true;})
        .def(pybind11::pickle(
            //TODO C++側のアドレスは一致するがPythonのobject idが変わるのでisが使えなくなる。Unpicklerでpersistent_loadを用いる必要がある。
            [](const T &t){//__getstate__
                return pybind11::bytes(t.to_binary());
            },
            [](const pybind11::bytes& data){//__setstate__
                return T::template from_binary<T>(data);
            }
        ))
        ;
        if constexpr (!std::same_as<T,typename T::PtrBaseType>){
            setup_implicitly_convertible_chain_from_weak_to_weak<T,typename T::BaseType>();
        }
    }
};

template<polymorphic_serializable_as_reference T>
struct def_common_serialization<T>{
    static constexpr bool op_enable_if_hook = true; // これによりcls.defのオーバーロードが実体化され、以下のexecuteを呼べるようになる

    template <typename Class, typename... Extra>
    requires ( std::same_as<T,typename Class::type> )
    void execute(Class &cl, const Extra &...extra) const{
        using SharedType = std::shared_ptr<T>;
        using WeakType = std::weak_ptr<T>;
        if constexpr (std::same_as<T,typename T::PtrBaseType>){
            cl
            .def("getDemangledName",&T::getDemangledName)
            .def("to_json",&T::template to_json<nl::ordered_json>)
            .def("to_binary",[](const T& v){return pybind11::bytes(v.to_binary());})
            .def("save",[&](const T& v, ::asrc::core::util::AvailableOutputArchiveTypes& archive){
                ::asrc::core::util::save_func_exposed_to_python(v.shared_from_this(),archive);
            })
            .def_property_readonly_static("_allow_cereal_serialization_in_cpp",[](pybind11::object /* self */){return true;})
            ;
        }
        cl
        .def("as_weak",[](T& t){return WeakType(std::dynamic_pointer_cast<T>(t.shared_from_this()));})
        .def_static("from_json",[](const nl::ordered_json& obj){return T::template from_json<T>(obj);})
        .def_static("from_json_weakref",[](const nl::ordered_json& obj){return T::template from_json_weakref<T>(obj);})
        .def_static("from_binary",[](const pybind11::bytes& str){return T::template from_binary<T>(str);})
        .def_static("from_binary_weakref",[](const pybind11::bytes& str){return T::template from_binary_weakref<T>(str);})
        .def_static("static_load",&::asrc::core::util::static_load_func_exposed_to_python<SharedType>)
        .def_static("static_load_weakref",&::asrc::core::util::static_load_func_exposed_to_python<WeakType>)
        .def(pybind11::pickle(
            //TODO C++側のアドレスは一致するがPythonのobject idが変わるのでisが使えなくなる。Unpicklerでpersistent_loadを用いる必要がある。
            [](const T &t){//__getstate__
                return pybind11::bytes(t.to_binary());
            },
            [](const pybind11::bytes& data){//__setstate__
                return T::template from_binary<T>(data);
            }
        ))
        ;
        if constexpr (!std::same_as<T,typename T::PtrBaseType>){
            setup_implicitly_convertible_chain_from_shared_to_weak<T,typename T::BaseType>();
        }
    }
};

ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)

#ifndef __DOXYGEN__
/**
 * @brief つねに既存インスタンスへの参照型としてシリアライゼーションを行う基底クラスを共通部分とともに宣言するマクロ。
 * ヘッダファイル側で呼ぶ。
 * 
 * @param className クラス名
 * @param ... PolymorphicSerializableAsReference 以外に継承元のインターフェースクラスがある場合それらのクラス名を書く
 */
#define ASRC_DECLARE_BASE_REF_CLASS(className,...)\
class className;\
template<class Base>\
class className##Wrap;\
class PYBIND11_EXPORT className: public ::asrc::core::PolymorphicSerializableAsReference<className> __VA_OPT__(, ASRC_INTERNAL_INHERIT_INTERFACE_CLASS(__VA_ARGS__)){\
    friend class ::asrc::core::PolymorphicSerializableAsReference<className>;\
    public:\
    using Type = className;\
    using BaseType = ::asrc::core::PolymorphicSerializableAsReference<className>;\
    template<class T=Type> using TrampolineType=className##Wrap<T>;\
    template<class T=Type> using PyClassInfo=::asrc::core::PyClassInfoBase<T __VA_OPT__(, ASRC_INTERNAL_ENUMERATE_INTERFACE_CLASS(__VA_ARGS__))>;\

/**
 * @brief つねに既存インスタンスへの参照型としてシリアライゼーションを行う基底クラスのtrampolineクラスを共通部分とともに宣言するマクロ。
 * ヘッダファイル側で呼ぶ。
 * 
 * @param className クラス名
 * @param ... PolymorphicSerializableAsReference 以外に継承元のインターフェースクラスがある場合それらのクラス名を書く
 */
#define ASRC_DECLARE_BASE_REF_TRAMPOLINE(className,...)\
template<class Base=className>\
class className##Wrap: public virtual Base __VA_OPT__(, ASRC_INTERNAL_INHERIT_INTERFACE_TRAMPOLINE(__VA_ARGS__)){\
    public:\
    ASRC_INTERNAL_USING_INTERFACE_TRAMPOLINE_CONSTRUCTORS(__VA_ARGS__)\
    using Base::Base;\
    virtual std::string getDemangledName() const override{\
        return pybind11::repr(pybind11::type::of(pybind11::cast(this->shared_from_this())));\
    }\

#else
//
// Doxygen使用時の__VA_OPT__不使用版
//

/**
 * @brief ASRC_DECLARE_BASE_REF_CLASS の可変引数がない場合の実体
 * 
 * @details Doxygenは__VA_OPT__に対応していないため、可変引数の有無で分岐するマクロとしてドキュメント化している。
 */
#define ASRC_DECLARE_BASE_REF_CLASS_IMPL_NO_VA_ARG(className)\
class className;\
template<class Base>\
class className##Wrap;\
class PYBIND11_EXPORT className: public ::asrc::core::PolymorphicSerializableAsReference<className>{\
    friend class ::asrc::core::PolymorphicSerializableAsReference<className>;\
    public:\
    using Type = className;\
    using BaseType = ::asrc::core::PolymorphicSerializableAsReference<className>;\
    template<class T=Type> using TrampolineType=className##Wrap<T>;\
    template<class T=Type> using PyClassInfo=::asrc::core::PyClassInfoBase<T>;

/**
 * @brief ASRC_DECLARE_BASE_REF_CLASS の可変引数がある場合の実体
 * 
 * @details Doxygenは__VA_OPT__に対応していないため、可変引数の有無で分岐するマクロとしてドキュメント化している。
 */
#define ASRC_DECLARE_BASE_REF_CLASS_IMPL_VA_ARGS(className,...)\
class className;\
template<class Base>\
class className##Wrap;\
class PYBIND11_EXPORT className: public ::asrc::core::PolymorphicSerializableAsReference<className> , ASRC_INTERNAL_INHERIT_INTERFACE_CLASS(__VA_ARGS__){\
    friend class ::asrc::core::PolymorphicSerializableAsReference<className>;\
    public:\
    using Type = className;\
    using BaseType = ::asrc::core::PolymorphicSerializableAsReference<className>;\
    template<class T=Type> using TrampolineType=className##Wrap<T>;\
    template<class T=Type> using PyClassInfo=::asrc::core::PyClassInfoBase<T, ASRC_INTERNAL_ENUMERATE_INTERFACE_CLASS(__VA_ARGS__)>;

/**
 * @brief つねに既存インスタンスへの参照型としてシリアライゼーションを行う基底クラスを共通部分とともに宣言するマクロ。
 * ヘッダファイル側で呼ぶ。
 * 
 * @param className クラス名
 * @param ... PolymorphicSerializableAsReference 以外に継承元のインターフェースクラスがある場合それらのクラス名を書く
 */
#define ASRC_DECLARE_BASE_REF_CLASS(...) ASRC_INTERNAL_DEFINE_VA_MACRO(ASRC_DECLARE_BASE_REF_CLASS,1,__VA_ARGS__)

/**
 * @brief ASRC_DECLARE_BASE_REF_TRAMPOLINE の可変引数がない場合の実体
 * 
 * @details Doxygenは__VA_OPT__に対応していないため、可変引数の有無で分岐するマクロとしてドキュメント化している。
 */
#define ASRC_DECLARE_BASE_REF_TRAMPOLINE_IMPL_NO_VA_ARG(className)\
template<class Base=className>\
class className##Wrap: public virtual Base{\
    public:\
    using Base::Base;\
    virtual std::string getDemangledName() const override{\
        return pybind11::repr(pybind11::type::of(pybind11::cast(this->shared_from_this())));\
    }

/**
 * @brief ASRC_DECLARE_BASE_REF_TRAMPOLINE の可変引数がある場合の実体
 * 
 * @details Doxygenは__VA_OPT__に対応していないため、可変引数の有無で分岐するマクロとしてドキュメント化している。
 */
#define ASRC_DECLARE_BASE_REF_TRAMPOLINE_IMPL_VA_ARGS(className,...)\
template<class Base=className>\
class className##Wrap: public virtual Base , ASRC_INTERNAL_INHERIT_INTERFACE_TRAMPOLINE(__VA_ARGS__){\
    public:\
    ASRC_INTERNAL_USING_INTERFACE_TRAMPOLINE_CONSTRUCTORS(__VA_ARGS__)\
    using Base::Base;\
    virtual std::string getDemangledName() const override{\
        return pybind11::repr(pybind11::type::of(pybind11::cast(this->shared_from_this())));\
    }

/**
 * @brief つねに既存インスタンスへの参照型としてシリアライゼーションを行う基底クラスのtrampolineクラスを共通部分とともに宣言するマクロ。
 * ヘッダファイル側で呼ぶ。
 * 
 * @param className クラス名
 * @param ... PolymorphicSerializableAsReference 以外に継承元のインターフェースクラスがある場合それらのクラス名を書く
 */
#define ASRC_DECLARE_BASE_REF_TRAMPOLINE(...) ASRC_INTERNAL_DEFINE_VA_MACRO(ASRC_DECLARE_BASE_REF_TRAMPOLINE,1,__VA_ARGS__)

#endif
//