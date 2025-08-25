/** Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
 * @file
 * 
 * @brief あるクラスTをPython側に公開したときの__init__の振る舞いをカスタマイズするためのフック
 * 
 * @details
 * 
 * @par 使用方法
 * 
 * asrc::core::py_init_XXX_implを特殊化し、そのメンバ関数executeに__init__の処理内容を記述する。
 * 記述方法はpybind11/detail/init.hを参考にされたい。
 * 
 * Python側への公開時にはpybind11::initの代わりに、cls.def(asrc::core::py_init<...>(...));
 * のようにdefの引数として与える。このとき、defの追加引数としてpybind11::args等を加えることも可能である。
 */
#pragma once
#include "util/macros/common_macros.h"
#include <concepts>
#include <pybind11/pybind11.h>

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)

template<typename T, typename... Args>
struct py_init_constructor_impl {
    template <typename Class, typename... Extra>
    requires ( std::same_as<T,typename Class::type> )
    void execute(Class &cl, const Extra &...extra) const{
        cl.def(pybind11::init<Args...>(),extra...);
    }
};
template<typename... Args>
struct py_init_constructor {
    static constexpr bool op_enable_if_hook = true; //!< これによりcls.defのオーバーロードが実体化され、以下のexecuteを呼べるようになる

    template <typename Class, typename... Extra>
    void execute(Class &cl, const Extra &...extra) const{
        py_init_constructor_impl<typename Class::type,Args...>().execute(cl,extra...);
    }
};
template<typename... Args>
py_init_constructor<Args...> py_init() {
    return {};
}
template<typename T, typename... Args>
struct py_init_alias_constructor_impl {
    template <typename Class, typename... Extra>
    requires ( std::same_as<T,typename Class::type> )
    void execute(Class &cl, const Extra &...extra) const{
        cl.def(pybind11::init_alias<Args...>(),extra...);
    }
};
template<typename... Args>
struct py_init_alias_constructor {
    static constexpr bool op_enable_if_hook = true; // これによりcls.defのオーバーロードが実体化され、以下のexecuteを呼べるようになる

    template <typename Class, typename... Extra>
    void execute(Class &cl, const Extra &...extra) const{
        py_init_alias_constructor_impl<typename Class::type,Args...>().execute(cl,extra...);
    }
};
template<typename... Args>
py_init_alias_constructor<Args...> py_init_alias() {
    return {};
}
template <typename T,
          typename CFunc,
          typename AFunc = pybind11::detail::void_type (*)(),
          typename = pybind11::detail::function_signature_t<CFunc>,
          typename = pybind11::detail::function_signature_t<AFunc>>
struct py_init_factory_impl;

template <typename T, typename Func, typename Return, typename... Args>
struct py_init_factory_impl<T, Func, pybind11::detail::void_type (*)(), Return(Args...)> {
    std::remove_reference_t<Func> class_factory;
    py_init_factory_impl(Func &&f) : class_factory(std::forward<Func>(f)) {}

    template <typename Class, typename... Extra>
    requires ( std::same_as<T,typename Class::type> )
    void execute(Class &cl, const Extra &...extra) && {
        cl.def(pybind11::init(std::move(class_factory)),extra...);
    }
};

template <typename T,
          typename CFunc,
          typename AFunc,
          typename CReturn,
          typename... CArgs,
          typename AReturn,
          typename... AArgs>
struct py_init_factory_impl<T, CFunc, AFunc, CReturn(CArgs...), AReturn(AArgs...)> {
    std::remove_reference_t<CFunc> class_factory;
    std::remove_reference_t<AFunc> alias_factory;
    py_init_factory_impl(CFunc &&c, AFunc &&a)
        : class_factory(std::forward<CFunc>(c)), alias_factory(std::forward<AFunc>(a)) {}

    template <typename Class, typename... Extra>
    requires ( std::same_as<T,typename Class::type> )
    void execute(Class &cl, const Extra &...extra) && {
        cl.def(pybind11::init(std::move(class_factory),std::move(alias_factory)),extra...);
    }
};

template <typename CFunc,
          typename AFunc = pybind11::detail::void_type (*)(),
          typename = pybind11::detail::function_signature_t<CFunc>,
          typename = pybind11::detail::function_signature_t<AFunc>>
struct py_init_factory;

template <typename Func, typename Return, typename... Args>
struct py_init_factory<Func, pybind11::detail::void_type (*)(), Return(Args...)> {
    static constexpr bool op_enable_if_hook = true; // これによりcls.defのオーバーロードが実体化され、以下のexecuteを呼べるようになる

    mutable std::remove_reference_t<Func> class_factory;
    py_init_factory(Func &&f) : class_factory(std::forward<Func>(f)) {}

    template <typename Class, typename... Extra>
    void execute(Class &cl, const Extra &...extra) const{
        py_init_factory_impl<typename Class::type,Func>(std::move(class_factory)).execute(cl,extra...);
    }
};

template <typename CFunc,
          typename AFunc,
          typename CReturn,
          typename... CArgs,
          typename AReturn,
          typename... AArgs>
struct py_init_factory<CFunc, AFunc, CReturn(CArgs...), AReturn(AArgs...)> {
    static constexpr bool op_enable_if_hook = true; // これによりcls.defのオーバーロードが実体化され、以下のexecuteを呼べるようになる

    mutable std::remove_reference_t<CFunc> class_factory;
    mutable std::remove_reference_t<AFunc> alias_factory;
    py_init_factory(CFunc &&c, AFunc &&a)
        : class_factory(std::forward<CFunc>(c)), alias_factory(std::forward<AFunc>(a)) {}

    template <typename Class, typename... Extra>
    void execute(Class &cl, const Extra &...extra) const{
        py_init_factory_impl<typename Class::type, CFunc, AFunc>(std::move(class_factory),std::move(alias_factory)).execute(cl,extra...);
    }
};

template <typename Func, typename Ret = py_init_factory<Func>>
Ret py_init(Func &&f) {
    return {std::forward<Func>(f)};
}
template <typename CFunc, typename AFunc, typename Ret = py_init_factory<CFunc, AFunc>>
Ret py_init(CFunc &&c, AFunc &&a) {
    return {std::forward<CFunc>(c), std::forward<AFunc>(a)};
}

ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
