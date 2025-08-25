/**
 * Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
 * @file
 * 
 * @brief Python側に公開可能とするための基底クラス群を使用するためのヘッダ
 */
#pragma once
#include "PythonableBase.h"
#include "SerializableAsValue.h"
#include "SerializableAsReference.h"
#include "stl_bind.h"

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)

template<class T>
requires ( std::is_enum_v<T> )
struct def_common_serialization<T>{
    static constexpr bool op_enable_if_hook = true; // これによりcls.defのオーバーロードが実体化され、以下のexecuteを呼べるようになる

    template <typename Class, typename... Extra>
    void execute(Class &cl, const Extra &...extra) const{
        cl
        .def("to_json",[](T& v){return ::asrc::core::util::enumToJson<T,nl::ordered_json>(v);})
        .def("load_from_json",[](T& v,const nl::ordered_json& j){j.get_to(v);})
        .def_static("from_json",[](const nl::ordered_json& j){return ::asrc::core::util::jsonToEnum<T>(j);})
        .def("to_binary",[](T& v){return ::asrc::core::util::enumToBinary(v);})
        .def("load_from_binary",[](T& v,const std::string& j){v=::asrc::core::util::binaryToEnum<T>(j);})
        .def_static("from_binary",[](const std::string& j){return ::asrc::core::util::binaryToEnum<T>(j);})
        .def("save",&::asrc::core::util::save_func_exposed_to_python<T>)
        .def("load",&::asrc::core::util::load_func_exposed_to_python<T>)
        .def_static("static_load",&::asrc::core::util::static_load_func_exposed_to_python<T>)
        .def_property_readonly_static("_allow_cereal_serialization_in_cpp",[](pybind11::object /* self */){return true;})
        ;
    }
};

/**
 * @brief enum classを数値,文字列, nl::basic_json のいずれからでもインスタンス化可能な形でPython側に公開する。
 *
 * @details
 * @code{.cpp}
 * ::asrc::core::expose_enum_value_helper(
 *     ::asrc::core::expose_enum_class<EnumType>(module,"EnumType")
 *     ,"value1"
 *     ,"value2"
 *     ,"value3"
 * );
 * @endcode
 * のように用いる。
 */
template<typename T,typename Parent, typename ... Extras>
requires ( std::is_enum_v<T> )
pybind11::enum_<T> expose_enum_class(Parent& m, const char* className, Extras&& ... extras){
    auto ret=pybind11::enum_<T>(m,className,std::forward<Extras>(extras)...);
    ret
    .def(pybind11::init(&::asrc::core::util::strToEnum<T>))
    .def(pybind11::init(&::asrc::core::util::jsonToEnum<T>))
    .def(def_common_serialization<T>())
    ;
    return std::move(ret);
    
}

/**
 * @brief enum classを数値,文字列, nl::basic_json のいずれからでもインスタンス化可能な形でPython側に公開する。
 *
 * @details
 * @code{.cpp}
 * ::asrc::core::expose_enum_value_helper(
 *     ::asrc::core::expose_enum_class<EnumType>(module,"EnumType")
 *     ,"value1"
 *     ,"value2"
 *     ,"value3"
 * );
 * @endcode
 * のように用いる。
 */
template<typename PyEnumType,
    std::enable_if_t<std::is_same_v<PyEnumType,pybind11::enum_<typename PyEnumType::Base::type>>,std::nullptr_t> = nullptr
>
PyEnumType&& expose_enum_value_helper(PyEnumType&& enumObj, const std::string& name){
    enumObj.value(name.c_str(),magic_enum::enum_cast<typename PyEnumType::Base::type>(name).value());
    return std::forward<PyEnumType>(enumObj);
}
/**
 * @brief enum classを数値,文字列, nl::basic_json のいずれからでもインスタンス化可能な形でPython側に公開する。
 *
 * @details
 * @code{.cpp}
 * ::asrc::core::expose_enum_value_helper(
 *     ::asrc::core::expose_enum_class<EnumType>(module,"EnumType")
 *     ,"value1"
 *     ,"value2"
 *     ,"value3"
 * );
 * @endcode
 * のように用いる。
 */
template<typename PyEnumType, typename... Args,
    std::enable_if_t<std::is_same_v<PyEnumType,pybind11::enum_<typename PyEnumType::Base::type>>,std::nullptr_t> = nullptr
>
PyEnumType&& expose_enum_value_helper(PyEnumType&&  enumObj, const std::string& name, Args... args){
    expose_enum_value_helper(
        expose_enum_value_helper(std::forward<PyEnumType>(enumObj),name),
        std::forward<Args>(args)...
    );
    return std::forward<PyEnumType>(enumObj);
}

ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)

//
// 派生クラスの宣言・定義・公開に関するマクロ
//

/** 
 * @brief ASRC_DECLARE_BASE_{REF|DATA}_CLASS を使って宣言されたクラスの派生クラスを宣言するマクロその1。
 * 
 * @details この派生クラスで新たなvirtualメンバ関数を宣言せず、基底クラスのpure virtual関数の実体をこのクラスで実装することもない場合にこちらを用いる。
 *
 * @param className クラス名
 * @param baseName 継承元のクラス名
 */
#define ASRC_DECLARE_DERIVED_CLASS_WITHOUT_TRAMPOLINE(className,baseName)\
class PYBIND11_EXPORT className:public baseName{\
    public:\
    using Type = className;\
    using BaseType = baseName;\
    template<class T=Type> using TrampolineType=BaseType::TrampolineType<T>;\
    template<class T=Type> using PyClassInfo=::asrc::core::PyClassInfoBase<T,typename T::BaseType>;

#ifndef __DOXYGEN__
/** 
 * @brief ASRC_DECLARE_BASE_{REF|DATA}_CLASS を使って宣言されたクラスの派生クラスを宣言するマクロその2。
 * 
 * @details この派生クラスで新たにvirtualメンバ関数を宣言するか、基底クラスのpure virtual関数の実体をこのクラスで実装する場合にこちらを用いる。
 *          別途 ASRC_DECLARE_DERIVED_TRAMPOLINE を呼ぶ必要がある。
 *
 * @param className クラス名
 * @param baseName 継承元のクラス名
 * @param ... baseName 以外に継承元のインターフェースクラスがある場合それらのクラス名を書く
 */
#define ASRC_DECLARE_DERIVED_CLASS_WITH_TRAMPOLINE(className,baseName,...)\
class className;\
template<class Base>\
class className##Wrap;\
class PYBIND11_EXPORT className: public baseName __VA_OPT__(, ASRC_INTERNAL_INHERIT_INTERFACE_CLASS(__VA_ARGS__)){\
    public:\
    using Type = className;\
    using BaseType = baseName;\
    template<class T=Type> using TrampolineType=className##Wrap<T>;\
    template<class T=Type> using PyClassInfo=::asrc::core::PyClassInfoBase<T,typename T::BaseType __VA_OPT__(,) ASRC_INTERNAL_ENUMERATE_INTERFACE_CLASS(__VA_ARGS__)>;

/**
 * @brief ASRC_DECLARE_DERIVED_CLASS_WITH_TRAMPOLINE で宣言した派生クラスに対応する"trampoline class"を宣言するマクロ。
 * 
 * @details テンプレートクラスであるため、ヘッダのみで完結する。
 * 
 * @param className クラス名
 * @param ... baseName 以外に継承元のインターフェースクラスがある場合それらのクラス名を書く
 */
#define ASRC_DECLARE_DERIVED_TRAMPOLINE(className,...)\
template<class Base=className>\
class className##Wrap: public className::BaseType::TrampolineType<Base> __VA_OPT__(, ASRC_INTERNAL_INHERIT_INTERFACE_TRAMPOLINE(__VA_ARGS__)){\
    public:\
    using BaseTrampoline = className::BaseType::TrampolineType<Base>;\
    __VA_OPT__(ASRC_INTERNAL_USING_INTERFACE_TRAMPOLINE_CONSTRUCTORS(__VA_ARGS__))\
    using BaseTrampoline::BaseTrampoline;

#else
//
// Doxygen使用時の__VA_OPT__不使用版
//

/**
 * @brief ASRC_DECLARE_DERIVED_CLASS_WITH_TRAMPOLINE の可変引数がない場合の実体
 * 
 * @details Doxygenは__VA_OPT__に対応していないため、可変引数の有無で分岐するマクロとしてドキュメント化している。
 */
#define ASRC_DECLARE_DERIVED_CLASS_WITH_TRAMPOLINE_IMPL_NO_VA_ARG(className,baseName)\
class className;\
template<class Base>\
class className##Wrap;\
class PYBIND11_EXPORT className: public baseName{\
    public:\
    using Type = className;\
    using BaseType = baseName;\
    template<class T=Type> using TrampolineType=className##Wrap<T>;\
    template<class T=Type> using PyClassInfo=::asrc::core::PyClassInfoBase<T,typename T::BaseType>;

/**
 * @brief ASRC_DECLARE_DERIVED_CLASS_WITH_TRAMPOLINE の可変引数がある場合の実体
 * 
 * @details Doxygenは__VA_OPT__に対応していないため、可変引数の有無で分岐するマクロとしてドキュメント化している。
 */
#define ASRC_DECLARE_DERIVED_CLASS_WITH_TRAMPOLINE_IMPL_VA_ARGS(className,baseName,...)\
class className;\
template<class Base>\
class className##Wrap;\
class PYBIND11_EXPORT className: public baseName , ASRC_INTERNAL_INHERIT_INTERFACE_CLASS(__VA_ARGS__){\
    public:\
    using Type = className;\
    using BaseType = baseName;\
    template<class T=Type> using TrampolineType=className##Wrap<T>;\
    template<class T=Type> using PyClassInfo=::asrc::core::PyClassInfoBase<T,typename T::BaseType , ASRC_INTERNAL_ENUMERATE_INTERFACE_CLASS(__VA_ARGS__)>;

/** 
 * @brief ASRC_DECLARE_BASE_{REF|DATA}_CLASS を使って宣言されたクラスの派生クラスを宣言するマクロその2。
 * 
 * @details この派生クラスで新たにvirtualメンバ関数を宣言するか、基底クラスのpure virtual関数の実体をこのクラスで実装する場合にこちらを用いる。
 *          別途 ASRC_DECLARE_DERIVED_TRAMPOLINE を呼ぶ必要がある。
 *
 * @param className クラス名
 * @param baseName 継承元のクラス名
 * @param ... baseName 以外に継承元のインターフェースクラスがある場合それらのクラス名を書く
 */
#define ASRC_DECLARE_DERIVED_CLASS_WITH_TRAMPOLINE(...) ASRC_INTERNAL_DEFINE_VA_MACRO(ASRC_DECLARE_DERIVED_CLASS_WITH_TRAMPOLINE,2,__VA_ARGS__)

/**
 * @brief ASRC_DECLARE_DERIVED_TRAMPOLINEの可変引数がない場合の実体
 * 
 * @details Doxygenは__VA_OPT__に対応していないため、可変引数の有無で分岐するマクロとしてドキュメント化している。
 */
#define ASRC_DECLARE_DERIVED_TRAMPOLINE_IMPL_NO_VA_ARG(className)\
template<class Base=className>\
class className##Wrap: public className::BaseType::TrampolineType<Base>{\
    public:\
    using BaseTrampoline = className::BaseType::TrampolineType<Base>;\
    using BaseTrampoline::BaseTrampoline;

/**
 * @brief ASRC_DECLARE_DERIVED_TRAMPOLINEの可変引数がある場合の実体
 * 
 * @details Doxygenは__VA_OPT__に対応していないため、可変引数の有無で分岐するマクロとしてドキュメント化している。
 */
    #define ASRC_DECLARE_DERIVED_TRAMPOLINE_IMPL_VA_ARGS(className,...)\
template<class Base=className>\
class className##Wrap: public className::BaseType::TrampolineType<Base> , ASRC_INTERNAL_INHERIT_INTERFACE_TRAMPOLINE(__VA_ARGS__){\
    public:\
    using BaseTrampoline = className::BaseType::TrampolineType<Base>;\
    ASRC_INTERNAL_USING_INTERFACE_TRAMPOLINE_CONSTRUCTORS(__VA_ARGS__)\
    using BaseTrampoline::BaseTrampoline;

/**
 * @brief ASRC_DECLARE_DERIVED_CLASS_WITH_TRAMPOLINE で宣言した派生クラスに対応する"trampoline class"を宣言するマクロ。
 * 
 * @details テンプレートクラスであるため、ヘッダのみで完結する。
 * 
 * @param className クラス名
 * @param ... baseName 以外に継承元のインターフェースクラスがある場合それらのクラス名を書く
 */
#define ASRC_DECLARE_DERIVED_TRAMPOLINE(...) ASRC_INTERNAL_DEFINE_VA_MACRO(ASRC_DECLARE_DERIVED_TRAMPOLINE,1,__VA_ARGS__)

#endif

//
// インターフェースクラスの宣言・定義・公開に関するマクロ
//

#ifndef __DOXYGEN__
#define ASRC_INTERNAL_ENUMERATE_INTERFACE_CLASS_IMPL(className) className
#define ASRC_INTERNAL_ENUMERATE_INTERFACE_CLASS(...) \
ASRC_INTERNAL_EXPAND(ASRC_INTERNAL_FOREACH(ASRC_INTERNAL_ENUMERATE_INTERFACE_CLASS_IMPL, ASRC_INTERNAL_COMMA __VA_OPT__(,) __VA_ARGS__))
#define ASRC_INTERNAL_INHERIT_INTERFACE_CLASS_IMPL(className) public virtual className
#define ASRC_INTERNAL_INHERIT_INTERFACE_CLASS(...) \
ASRC_INTERNAL_EXPAND(ASRC_INTERNAL_FOREACH(ASRC_INTERNAL_INHERIT_INTERFACE_CLASS_IMPL, ASRC_INTERNAL_COMMA __VA_OPT__(,) __VA_ARGS__))
#define ASRC_INTERNAL_INHERIT_INTERFACE_TRAMPOLINE_IMPL(className) public virtual className::TrampolineType<Base>
#define ASRC_INTERNAL_INHERIT_INTERFACE_TRAMPOLINE(...) \
ASRC_INTERNAL_EXPAND(ASRC_INTERNAL_FOREACH(ASRC_INTERNAL_INHERIT_INTERFACE_TRAMPOLINE_IMPL, ASRC_INTERNAL_COMMA __VA_OPT__(,) __VA_ARGS__))
#define ASRC_INTERNAL_USING_INTERFACE_TRAMPOLINE_CONSTRUCTORS_IMPL(className) using className##Wrap<Base>::className##Wrap;
#define ASRC_INTERNAL_USING_INTERFACE_TRAMPOLINE_CONSTRUCTORS(...) \
ASRC_INTERNAL_EXPAND(ASRC_INTERNAL_FOREACH(ASRC_INTERNAL_USING_INTERFACE_TRAMPOLINE_CONSTRUCTORS_IMPL, ASRC_INTERNAL_SEMICOLON __VA_OPT__(,) __VA_ARGS__))
#define ASRC_INTERNAL_INHERIT_CLASS_IMPL(className) public className
#define ASRC_INTERNAL_INHERIT_CLASS(...) \
ASRC_INTERNAL_EXPAND(ASRC_INTERNAL_FOREACH(ASRC_INTERNAL_INHERIT_CLASS_IMPL, ASRC_INTERNAL_COMMA __VA_OPT__(,) __VA_ARGS__))
#else
//
// Doxygen使用時の__VA_OPT__不使用版
//
#define ASRC_INTERNAL_ENUMERATE_INTERFACE_CLASS_IMPL(className) className
#define ASRC_INTERNAL_ENUMERATE_INTERFACE_CLASS(...) \
ASRC_INTERNAL_EXPAND(ASRC_INTERNAL_FOREACH(ASRC_INTERNAL_ENUMERATE_INTERFACE_CLASS_IMPL, ASRC_INTERNAL_COMMA , __VA_ARGS__))
#define ASRC_INTERNAL_INHERIT_INTERFACE_CLASS_IMPL(className) public virtual className
#define ASRC_INTERNAL_INHERIT_INTERFACE_CLASS(...) \
ASRC_INTERNAL_EXPAND(ASRC_INTERNAL_FOREACH(ASRC_INTERNAL_INHERIT_INTERFACE_CLASS_IMPL, ASRC_INTERNAL_COMMA , __VA_ARGS__))
#define ASRC_INTERNAL_INHERIT_INTERFACE_TRAMPOLINE_IMPL(className) public virtual className::TrampolineType<Base>
#define ASRC_INTERNAL_INHERIT_INTERFACE_TRAMPOLINE(...) \
ASRC_INTERNAL_EXPAND(ASRC_INTERNAL_FOREACH(ASRC_INTERNAL_INHERIT_INTERFACE_TRAMPOLINE_IMPL, ASRC_INTERNAL_COMMA , __VA_ARGS__))
#define ASRC_INTERNAL_USING_INTERFACE_TRAMPOLINE_CONSTRUCTORS_IMPL(className) using className##Wrap<Base>::className##Wrap;
#define ASRC_INTERNAL_USING_INTERFACE_TRAMPOLINE_CONSTRUCTORS(...) \
ASRC_INTERNAL_EXPAND(ASRC_INTERNAL_FOREACH(ASRC_INTERNAL_USING_INTERFACE_TRAMPOLINE_CONSTRUCTORS_IMPL, ASRC_INTERNAL_SEMICOLON , __VA_ARGS__))
#define ASRC_INTERNAL_INHERIT_CLASS_IMPL(className) public className
#define ASRC_INTERNAL_INHERIT_CLASS(...) \
ASRC_INTERNAL_EXPAND(ASRC_INTERNAL_FOREACH(ASRC_INTERNAL_INHERIT_CLASS_IMPL, ASRC_INTERNAL_COMMA , __VA_ARGS__))
#endif

#ifndef __DOXYGEN__
/**
 * @brief インターフェースクラスを宣言するマクロ。
 * 
 * @details
 *      これを用いて宣言すると、asrc::core::expose_interface_class 関数によりPython側でもこのインターフェースクラスを公開できるようになる。
 *      あくまでC++ベースのインターフェースクラスなので、メンバ変数を持たず、コンストラクタも引数なしとする。
 *      インターフェースの階層化は可能であり、マクロの２つめ以降の引数に親となるインターフェースクラスを列挙すればよい。
 * 
 * @param className クラス名
 * @param ... PolymorphicSerializableAsReference 以外に継承元のインターフェースクラスがある場合それらのクラス名を書く
 */
#define ASRC_DECLARE_INTERFACE_CLASS(className,...)\
class className;\
template<class Base>\
class className##Wrap;\
class PYBIND11_EXPORT className __VA_OPT__(: ASRC_INTERNAL_INHERIT_INTERFACE_CLASS(__VA_ARGS__)){\
    public:\
    virtual ~className()=default;\
    using Type = className;\
    template<class T=Type> using TrampolineType=className##Wrap<T>;\
    template<class T=Type> using PyClassInfo=::asrc::core::PyClassInfoBase<T,typename T::TrampolineType<>,std::shared_ptr<T> __VA_OPT__(,) ASRC_INTERNAL_ENUMERATE_INTERFACE_CLASS(__VA_ARGS__)>;\

/**
 * @brief ASRC_DECLARE_INTERFACE_CLASS で宣言したインターフェースクラスに対応する"trampoline class"を宣言するマクロ。
 * 
 * @details テンプレートクラスであるため、ヘッダのみで完結する。
 * 
 * @param className クラス名
 * @param ... 継承元のインターフェースクラスがある場合それらのクラス名を書く
 */
#define ASRC_DECLARE_INTERFACE_TRAMPOLINE(className,...)\
template<class Base=className>\
class className##Wrap: public virtual Base __VA_OPT__(, ASRC_INTERNAL_INHERIT_INTERFACE_TRAMPOLINE(__VA_ARGS__)){\
    public:\
    __VA_OPT__(ASRC_INTERNAL_USING_INTERFACE_TRAMPOLINE_CONSTRUCTORS(__VA_ARGS__))\
    using Base::Base;\

#else
//
// Doxygen使用時の__VA_OPT__不使用版
//

/**
 * @brief ASRC_DECLARE_INTERFACE_CLASS の可変引数がない場合の実体
 * 
 * @details Doxygenは__VA_OPT__に対応していないため、可変引数の有無で分岐するマクロとしてドキュメント化している。
 */
#define ASRC_DECLARE_INTERFACE_CLASS_IMPL_NO_VA_ARG(className)\
class className;\
template<class Base>\
class className##Wrap;\
class PYBIND11_EXPORT className{\
    public:\
    virtual ~className()=default;\
    using Type = className;\
    template<class T=Type> using TrampolineType=className##Wrap<T>;\
    template<class T=Type> using PyClassInfo=::asrc::core::PyClassInfoBase<T,typename T::TrampolineType<>,std::shared_ptr<T>>;

/**
 * @brief ASRC_DECLARE_INTERFACE_CLASS の可変引数がある場合の実体
 * 
 * @details Doxygenは__VA_OPT__に対応していないため、可変引数の有無で分岐するマクロとしてドキュメント化している。
 */
#define ASRC_DECLARE_INTERFACE_CLASS_IMPL_VA_ARGS(className,...)\
class className;\
template<class Base>\
class className##Wrap;\
class PYBIND11_EXPORT className : ASRC_INTERNAL_INHERIT_INTERFACE_CLASS(__VA_ARGS__){\
    public:\
    virtual ~className()=default;\
    using Type = className;\
    template<class T=Type> using TrampolineType=className##Wrap<T>;\
    template<class T=Type> using PyClassInfo=::asrc::core::PyClassInfoBase<T,typename T::TrampolineType<>,std::shared_ptr<T>, ASRC_INTERNAL_ENUMERATE_INTERFACE_CLASS(__VA_ARGS__)>;

/**
 * @brief インターフェースクラスを宣言するマクロ。
 * 
 * @details
 *      これを用いて宣言すると、asrc::core::expose_interface_class 関数によりPython側でもこのインターフェースクラスを公開できるようになる。
 *      あくまでC++ベースのインターフェースクラスなので、メンバ変数を持たず、コンストラクタも引数なしとする。
 *      インターフェースの階層化は可能であり、マクロの２つめ以降の引数に親となるインターフェースクラスを列挙すればよい。
 * 
 * @param ... 自身のクラス名と、継承元のインターフェースクラスがある場合それらのクラス名を続けて書く
 */
#define ASRC_DECLARE_INTERFACE_CLASS(...) ASRC_INTERNAL_DEFINE_VA_MACRO(ASRC_DECLARE_INTERFACE_CLASS,1,__VA_ARGS__)

/**
 * @brief ASRC_DECLARE_INTERFACE_TRAMPOLINE の可変引数がない場合の実体
 * 
 * @details Doxygenは__VA_OPT__に対応していないため、可変引数の有無で分岐するマクロとしてドキュメント化している。
 */
#define ASRC_DECLARE_INTERFACE_TRAMPOLINE_IMPL_NO_VA_ARG(className)\
template<class Base=className>\
class className##Wrap: public virtual Base{\
    public:\
    using Base::Base;

/**
 * @brief ASRC_DECLARE_INTERFACE_TRAMPOLINE の可変引数がある場合の実体
 * 
 * @details Doxygenは__VA_OPT__に対応していないため、可変引数の有無で分岐するマクロとしてドキュメント化している。
 */
#define ASRC_DECLARE_INTERFACE_TRAMPOLINE_IMPL_VA_ARGS(className,...)\
template<class Base=className>\
class className##Wrap: public virtual Base , ASRC_INTERNAL_INHERIT_INTERFACE_TRAMPOLINE(__VA_ARGS__){\
    public:\
    using Base::Base;

/**
 * @brief ASRC_DECLARE_INTERFACE_CLASS で宣言したインターフェースクラスに対応する"trampoline class"を宣言するマクロ。
 * 
 * @details テンプレートクラスであるため、ヘッダのみで完結する。
 * 
 * @param className クラス名
 * @param ... 継承元のインターフェースクラスがある場合それらのクラス名を書く
 */
#define ASRC_DECLARE_INTERFACE_TRAMPOLINE(...) ASRC_INTERNAL_DEFINE_VA_MACRO(ASRC_DECLARE_INTERFACE_TRAMPOLINE,1,__VA_ARGS__)

#endif

//
// Python側に公開したクラスのattributeの定義を容易にするためのマクロ群。
//
#ifndef __DOXYGEN__

/**
 * @brief クラスのメンバ関数をPython側に公開するマクロ。関数名の入力ミスを避けたい場合に使える。
 * 
 * @param CLASS クラス名
 * @param func メンバ関数名
 * @param ... py::return_value_policy や py::keep_alive 等の py::class_::def に渡す追加の引数
 */
#define ASRC_DEF_FUNC(CLASS,func,...) .def(#func,&CLASS::func __VA_OPT__(,) __VA_ARGS__)

/**
 * @brief クラスのメンバ関数をGIL取得なしでPython側に公開するマクロ。関数名の入力ミスを避けたい場合に使える。
 * 
 * @param CLASS クラス名
 * @param func メンバ関数名
 * @param ... py::return_value_policy や py::keep_alive 等の py::class_::def に渡す追加の引数
 */
#define ASRC_DEF_FUNC_NO_GIL(CLASS,func,...) .def(#func,&CLASS::func,pybind11::call_guard<pybind11::gil_scoped_release>() __VA_OPT__(,) __VA_ARGS__)

/**
 * @brief クラスのstaticメンバ関数をPython側に公開するマクロ。関数名の入力ミスを避けたい場合に使える。
 * 
 * @param CLASS クラス名
 * @param func メンバ関数名
 * @param ... py::return_value_policy や py::keep_alive 等の py::class_::def に渡す追加の引数
 */
#define ASRC_DEF_STATIC_FUNC(CLASS,func,...) .def_static(#func,&CLASS::func __VA_OPT__(,) __VA_ARGS__)

/**
 * @brief クラスのstaticメンバ関数をGIL取得なしでPython側に公開するマクロ。関数名の入力ミスを避けたい場合に使える。
 * 
 * @param CLASS クラス名
 * @param func メンバ関数名
 * @param ... py::return_value_policy や py::keep_alive 等の py::class_::def に渡す追加の引数
 */
#define ASRC_DEF_STATIC_FUNC_NO_GIL(CLASS,func,...) .def_static(#func,&CLASS::func,pybind11::call_guard<pybind11::gil_scoped_release>() __VA_OPT__(,) __VA_ARGS__)

/**
 * @brief クラスのメンバ変数を読み書き可能な形でPython側に公開するマクロ。変数名の入力ミスを避けたい場合に使える。
 * 
 * @param CLASS クラス名
 * @param field メンバ変数名
 * @param ... py::return_value_policy や py::keep_alive 等の py::class_::def に渡す追加の引数
 */
#define ASRC_DEF_READWRITE(CLASS,field,...) .def_readwrite(#field,&CLASS::field __VA_OPT__(,) __VA_ARGS__)

/**
 * @brief クラスのメンバ変数を読み取り専用な形でPython側に公開するマクロ。変数名の入力ミスを避けたい場合に使える。
 * 
 * @param CLASS クラス名
 * @param field メンバ変数名
 * @param ... py::return_value_policy や py::keep_alive 等の py::class_::def に渡す追加の引数
 */
#define ASRC_DEF_READONLY(CLASS,field,...) .def_readonly(#field,&CLASS::field __VA_OPT__(,) __VA_ARGS__)

#else

#define ASRC_DEF_FUNC_IMPL_NO_VA_ARG(CLASS,func) .def(#func,&CLASS::func)
#define ASRC_DEF_FUNC_IMPL_VA_ARGS(CLASS,func,...) .def(#func,&CLASS::func, __VA_ARGS__)

/**
 * @brief クラスのメンバ関数をPython側に公開するマクロ。関数名の入力ミスを避けたい場合に使える。
 * 
 * @param CLASS クラス名
 * @param func メンバ関数名
 * @param ... py::return_value_policy や py::keep_alive 等の py::class_::def に渡す追加の引数
 */
#define ASRC_DEF_FUNC(...) ASRC_INTERNAL_DEFINE_VA_MACRO(ASRC_DEF_FUNC,2,__VA_ARGS__)

#define ASRC_DEF_FUNC_NO_GIL_IMPL_NO_VA_ARG(CLASS,func) .def(#func,&CLASS::func,pybind11::call_guard<pybind11::gil_scoped_release>())
#define ASRC_DEF_FUNC_NO_GIL_IMPL_VA_ARGS(CLASS,func,...) .def(#func,&CLASS::func,pybind11::call_guard<pybind11::gil_scoped_release>(), __VA_ARGS__)

/**
 * @brief クラスのメンバ関数をGIL取得なしでPython側に公開するマクロ。関数名の入力ミスを避けたい場合に使える。
 * 
 * @param CLASS クラス名
 * @param func メンバ関数名
 * @param ... py::return_value_policy や py::keep_alive 等の py::class_::def に渡す追加の引数
 */
#define ASRC_DEF_FUNC_NO_GIL(...) ASRC_INTERNAL_DEFINE_VA_MACRO(ASRC_DEF_FUNC_NO_GIL,2,__VA_ARGS__)

#define ASRC_DEF_STATIC_FUNC_IMPL_NO_VA_ARG(CLASS,func) .def_static(#func,&CLASS::func)
#define ASRC_DEF_STATIC_FUNC_IMPL_VA_ARGS(CLASS,func,...) .def_static(#func,&CLASS::func, __VA_ARGS__)

/**
 * @brief クラスのstaticメンバ関数をPython側に公開するマクロ。関数名の入力ミスを避けたい場合に使える。
 * 
 * @param CLASS クラス名
 * @param func メンバ関数名
 * @param ... py::return_value_policy や py::keep_alive 等の py::class_::def に渡す追加の引数
 */
#define ASRC_DEF_STATIC_FUNC(...) ASRC_INTERNAL_DEFINE_VA_MACRO(ASRC_DEF_STATIC_FUNC,2,__VA_ARGS__)

#define ASRC_DEF_STATIC_FUNC_NO_GIL_IMPL_NO_VA_ARG(CLASS,func) .def_static(#func,&CLASS::func,pybind11::call_guard<pybind11::gil_scoped_release>())
#define ASRC_DEF_STATIC_FUNC_NO_GIL_IMPL_VA_ARGS(CLASS,func,...) .def_static(#func,&CLASS::func,pybind11::call_guard<pybind11::gil_scoped_release>() __VA_OPT__(,) __VA_ARGS__)

/**
 * @brief クラスのstaticメンバ関数をGIL取得なしでPython側に公開するマクロ。関数名の入力ミスを避けたい場合に使える。
 * 
 * @param CLASS クラス名
 * @param func メンバ関数名
 * @param ... py::return_value_policy や py::keep_alive 等の py::class_::def に渡す追加の引数
 */
#define ASRC_DEF_STATIC_FUNC_NO_GIL(...) ASRC_INTERNAL_DEFINE_VA_MACRO(ASRC_DEF_STATIC_FUNC_NO_GIL,2,__VA_ARGS__)

#define ASRC_DEF_READWRITE_IMPL_NO_VA_ARG(CLASS,field) .def_readwrite(#field,&CLASS::field)
#define ASRC_DEF_READWRITE_IMPL_VA_ARGS(CLASS,field,...) .def_readwrite(#field,&CLASS::field, __VA_ARGS__)

/**
 * @brief クラスのメンバ変数を読み書き可能な形でPython側に公開するマクロ。変数名の入力ミスを避けたい場合に使える。
 * 
 * @param CLASS クラス名
 * @param field メンバ変数名
 * @param ... py::return_value_policy や py::keep_alive 等の py::class_::def に渡す追加の引数
 */
#define ASRC_DEF_READWRITE(...) ASRC_INTERNAL_DEFINE_VA_MACRO( ASRC_DEF_READWRITE,2,__VA_ARGS__)

#define ASRC_DEF_READONLY_IMPL_NO_VA_ARG(CLASS,field) .def_readonly(#field,&CLASS::field)
#define ASRC_DEF_READONLY_IMPL_VA_ARGS(CLASS,field,...) .def_readonly(#field,&CLASS::field, __VA_ARGS__)

/**
 * @brief クラスのメンバ変数を読み取り専用な形でPython側に公開するマクロ。変数名の入力ミスを避けたい場合に使える。
 * 
 * @param CLASS クラス名
 * @param field メンバ変数名
 * @param ... py::return_value_policy や py::keep_alive 等の py::class_::def に渡す追加の引数
 */
#define ASRC_DEF_READONLY(...) ASRC_INTERNAL_DEFINE_VA_MACRO(ASRC_DEF_READONLY,2,__VA_ARGS__)

#endif

// ==== macros below are for backward compatibility ====

/// @sa ASRC_DECLARE_BASE_REF_CLASS(...)
#define DECLARE_BASE_CLASS(...) ASRC_DECLARE_BASE_REF_CLASS(__VA_ARGS__)

/// @sa ASRC_DECLARE_BASE_REF_TRAMPOLINE(...)
#define DECLARE_BASE_TRAMPOLINE(...) ASRC_DECLARE_BASE_REF_TRAMPOLINE(__VA_ARGS__)

/// @sa ASRC_DECLARE_DERIVED_CLASS_WITH_TRAMPOLINE(...)
#define DECLARE_CLASS_WITH_TRAMPOLINE(...) ASRC_DECLARE_DERIVED_CLASS_WITH_TRAMPOLINE(__VA_ARGS__)

/// @sa ASRC_DECLARE_DERIVED_CLASS_WITHOUT_TRAMPOLINE(className,baseName)
#define DECLARE_CLASS_WITHOUT_TRAMPOLINE(className,baseName) ASRC_DECLARE_DERIVED_CLASS_WITHOUT_TRAMPOLINE(className,baseName)

/// @sa ASRC_DECLARE_DERIVED_TRAMPOLINE(...)
#define DECLARE_TRAMPOLINE(...) ASRC_DECLARE_DERIVED_TRAMPOLINE(__VA_ARGS__)

#ifndef __DOXYGEN__

/// @sa ASRC_DECLARE_TRAMPOLINE(outerClassName::className,...)
#define DECLARE_INNER_CLASS_TRAMPOLINE(outerClassName,className,...) ASRC_DECLARE_TRAMPOLINE(outerClassName::className __VA_OPT__(,) __VA_ARGS__)

#else

#define DECLARE_INNER_CLASS_TRAMPOLINE_IMPL_NO_VA_ARG(outerClassName,className) ASRC_DECLARE_TRAMPOLINE(outerClassName::className)
#define DECLARE_INNER_CLASS_TRAMPOLINE_IMPL_VA_ARGS(outerClassName,className,...) ASRC_DECLARE_TRAMPOLINE(outerClassName::className , __VA_ARGS__)

/// @sa ASRC_DECLARE_TRAMPOLINE(outerClassName::className,...)
#define DECLARE_INNER_CLASS_TRAMPOLINE(...) ASRC_INTERNAL_DEFINE_VA_MACRO(DECLARE_INNER_CLASS_TRAMPOLINE,2,__VA_ARGS__)
#endif

/// @sa ::asrc::core::expose_common_class
#define EXPOSE_CLASS_WITHOUT_INIT_IMPL(parentObject,classIdentifier,className,localizer)\
::asrc::core::expose_common_class<classIdentifier>(parentObject,ASRC_INTERNAL_PY_CLASS_NAME(className,localizer))

/// @sa ::asrc::core::expose_entity_subclass
#define EXPOSE_CLASS_IMPL(parentObject,classIdentifier,className,localizer)\
EXPOSE_CLASS_WITHOUT_INIT_IMPL(parentObject,classIdentifier,className,localizer)\
.def(::asrc::core::py_init<const nl::json&,const nl::json&>())\

/// @sa ::asrc::core::expose_common_class<className>(m)
#define EXPOSE_BASE_CLASS_WITHOUT_INIT(className) EXPOSE_CLASS_WITHOUT_INIT_IMPL(m,className,className,ASRC_INTERNAL_NOP)

/// @sa ::asrc::core::expose_common_class<className>(m,pybind11::module_local(true))
#define EXPOSE_LOCAL_BASE_CLASS_WITHOUT_INIT(className) EXPOSE_CLASS_WITHOUT_INIT_IMPL(m,className,className,ASRC_INTERNAL_PY_LOCALIZER)

/// @sa ::asrc::core::expose_entity_subclass<className>(m)
#define EXPOSE_BASE_CLASS(className) EXPOSE_CLASS_IMPL(m,className,className,ASRC_INTERNAL_NOP)

/// @sa ::asrc::core::expose_entity_subclass<className>(m,pybind11::module_local(true))
#define EXPOSE_LOCAL_BASE_CLASS(className) EXPOSE_CLASS_IMPL(m,className,className,ASRC_INTERNAL_PY_LOCALIZER)

/// @sa ::asrc::core::expose_common_class<baseClassName::className>(baseClassObject,className)
#define EXPOSE_BASE_INNER_CLASS_WITHOUT_INIT(baseClassObject,baseClassName,className) EXPOSE_CLASS_WITHOUT_INIT_IMPL(baseClassObject,baseClassName::className,className,ASRC_INTERNAL_NOP)

/// @sa ::asrc::core::expose_common_class<baseClassName::className>(baseClassObject,className,pybind11::module_local(true))
#define EXPOSE_LOCAL_BASE_INNER_CLASS_WITHOUT_INIT(baseClassObject,baseClassName,className) EXPOSE_CLASS_WITHOUT_INIT_IMPL(baseClassObject,baseClassName::className,className,ASRC_INTERNAL_PY_LOCALIZER)

/// @sa ::asrc::core::expose_entity_subclass<baseClassName::className>(baseClassObject,className)
#define EXPOSE_BASE_INNER_CLASS(baseClassObject,baseClassName,className) EXPOSE_CLASS_IMPL(baseClassObject,baseClassName::className,className,ASRC_INTERNAL_NOP)

/// @sa ::asrc::core::expose_entity_subclass<baseClassName::className>(baseClassObject,className,pybind11::module_local(true))
#define EXPOSE_LOCAL_BASE_INNER_CLASS(baseClassObject,baseClassName,className) EXPOSE_CLASS_IMPL(baseClassObject,baseClassName::className,className,ASRC_INTERNAL_PY_LOCALIZER)

/// @sa ::asrc::core::expose_common_class<className>(m)
#define EXPOSE_CLASS_WITHOUT_INIT(className) EXPOSE_CLASS_WITHOUT_INIT_IMPL(m,className,className,ASRC_INTERNAL_NOP)

/// @sa ::asrc::core::expose_common_class<className>(m,pybind11::module_local(true))
#define EXPOSE_LOCAL_CLASS_WITHOUT_INIT(className) EXPOSE_CLASS_WITHOUT_INIT_IMPL(m,className,className,ASRC_INTERNAL_PY_LOCALIZER)

/// @sa ::asrc::core::expose_entity_subclass<className>(m)
#define EXPOSE_CLASS(className) EXPOSE_CLASS_IMPL(m,className,className,ASRC_INTERNAL_NOP)

/// @sa ::asrc::core::expose_entity_subclass<className>(m,pybind11::module_local(true))
#define EXPOSE_LOCAL_CLASS(className) EXPOSE_CLASS_IMPL(m,className,className,ASRC_INTERNAL_PY_LOCALIZER)

/// @sa ::asrc::core::expose_common_class<baseClassName::className>(baseClassObject,className)
#define EXPOSE_INNER_CLASS_WITHOUT_INIT(baseClassObject,baseClassName,className) EXPOSE_CLASS_WITHOUT_INIT_IMPL(baseClassObject,baseClassName::className,className,ASRC_INTERNAL_NOP)

/// @sa ::asrc::core::expose_common_class<baseClassName::className>(baseClassObject,className,pybind11::module_local(true))
#define EXPOSE_LOCAL_INNER_CLASS_WITHOUT_INIT(baseClassObject,baseClassName,className) EXPOSE_CLASS_WITHOUT_INIT_IMPL(baseClassObject,baseClassName::className,className,ASRC_INTERNAL_PY_LOCALIZER)

/// @sa ::asrc::core::expose_entity_subclass<baseClassName::className>(baseClassObject,className)
#define EXPOSE_INNER_CLASS(baseClassObject,baseClassName,className) EXPOSE_CLASS_IMPL(baseClassObject,baseClassName::className,className,ASRC_INTERNAL_NOP)

/// @sa ::asrc::core::expose_entity_subclass<baseClassName::className>(baseClassObject,className,pybind11::module_local(true))
#define EXPOSE_LOCAL_INNER_CLASS(baseClassObject,baseClassName,className) EXPOSE_CLASS_IMPL(baseClassObject,baseClassName::className,className,ASRC_INTERNAL_PY_LOCALIZER)

#define DEF_FUNC(...) ASRC_DEF_FUNC(__VA_ARGS__)
#define DEF_FUNC_NO_GIL(...) ASRC_DEF_FUNC_NO_GIL(__VA_ARGS__)
#define DEF_STATIC_FUNC(...) ASRC_DEF_STATIC_FUNC(__VA_ARGS__)
#define DEF_STATIC_FUNC_NO_GIL(...) ASRC_DEF_STATIC_FUNC_NO_GIL(__VA_ARGS__)
#define DEF_READWRITE(...) ASRC_DEF_READWRITE(__VA_ARGS__)
#define DEF_READONLY(...) ASRC_DEF_READONLY(__VA_ARGS__)

#define BIND_VECTOR_NAME(value_type,name,local) :asrc::core::bind_stl_container_name<std::vector<value_type>>(m,name,::pybind11::module_local(local))
#define BIND_VECTOR(value_type,local) :asrc::core::bind_stl_container<std::vector<value_type>>(m,::pybind11::module_local(local))
#define BIND_MAP_NAME(key_type,value_type,name,local) ::asrc::core::bind_stl_container_name<std::map<key_type,value_type>>(m,name,::pybind11::module_local(local))
#define BIND_MAP(key_type,value_type,local) ::asrc::core::bind_stl_container<std::map<key_type,value_type>>(m,::pybind11::module_local(local))
#define DEF_PICKLE_VIA_JSON(CLASS)\
.def(pybind11::pickle( \
    [](const CLASS & p){\
        nl::ordered_json j=p;\
        return pybind11::make_tuple(j);\
    },\
    [](pybind11::tuple t){\
        nl::ordered_json j=t[0];\
        return CLASS(j);\
    }\
))

// ==== macros above are for backward compatibility ====
