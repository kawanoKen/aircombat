// Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#pragma once
#include <stack>
#include <memory>
#include <variant>
#include <pybind11/pybind11.h>
#include <cereal/cereal.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/external/base64.hpp>
#include <boost/iostreams/stream.hpp>
#include "../../util/macros/common_macros.h"
#include "../../traits/traits.h"
#include "NLJSONArchive.h"
#include "pyobject.h"

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)
ASRC_INLINE_NAMESPACE_BEGIN(util)
struct PYBIND11_EXPORT DummyOutputArchive{};
struct PYBIND11_EXPORT DummyInputArchive{};

//
// 使用してよいArchiveをstd::variantとしてまとめたもの。
// これを引数に取り、後述のヘルパー関数を用いることでシリアライズ用関数を非テンプレート化することができる。
//
using AvailableOutputArchiveTypes = std::variant<
    DummyOutputArchive, // Dummy candidate for making this variant default constructible.
    std::reference_wrapper<asrc::core::util::NLJSONOutputArchive>,
    std::reference_wrapper<cereal::PortableBinaryOutputArchive>,
    std::reference_wrapper<cereal::JSONOutputArchive>,
    std::reference_wrapper<cereal::XMLOutputArchive>
>;
using AvailableInputArchiveTypes = std::variant<
    DummyInputArchive, // Dummy candidate for making this variant default constructible.
    std::reference_wrapper<asrc::core::util::NLJSONInputArchive>,
    std::reference_wrapper<cereal::PortableBinaryInputArchive>,
    std::reference_wrapper<cereal::JSONInputArchive>,
    std::reference_wrapper<cereal::XMLInputArchive>
>;
using AvailableArchiveTypes = std::variant<
    DummyOutputArchive, // Dummy candidate for making this variant default constructible.
    DummyInputArchive, // Dummy candidate for making this variant default constructible.
    std::reference_wrapper<asrc::core::util::NLJSONOutputArchive>,
    std::reference_wrapper<cereal::PortableBinaryOutputArchive>,
    std::reference_wrapper<cereal::JSONOutputArchive>,
    std::reference_wrapper<cereal::XMLOutputArchive>,
    std::reference_wrapper<asrc::core::util::NLJSONInputArchive>,
    std::reference_wrapper<cereal::PortableBinaryInputArchive>,
    std::reference_wrapper<cereal::JSONInputArchive>,
    std::reference_wrapper<cereal::XMLInputArchive>
>;

//
// Archiveの種類を判定する関数
//
template<class VariantArchive>
requires (
    std::same_as<VariantArchive,AvailableOutputArchiveTypes>
    || std::same_as<VariantArchive,AvailableInputArchiveTypes>
    || std::same_as<VariantArchive,AvailableArchiveTypes>
)
bool isValidArchive(VariantArchive& archive){
    return std::visit([](auto& ar) {
        return (
            !std::is_same_v<std::remove_cv_t<std::remove_reference_t<decltype(ar)>>,::asrc::core::util::DummyOutputArchive>
            && !std::is_same_v<std::remove_cv_t<std::remove_reference_t<decltype(ar)>>,::asrc::core::util::DummyInputArchive>
        );
    },archive);
}
template<traits::cereal_archive Archive>
constexpr bool isValidArchive(Archive& archive){
    return true;
}
template<class VariantArchive>
requires (
    std::same_as<VariantArchive,AvailableOutputArchiveTypes>
    || std::same_as<VariantArchive,AvailableInputArchiveTypes>
    || std::same_as<VariantArchive,AvailableArchiveTypes>
)
bool isOutputArchive(VariantArchive& archive){
    return std::visit([](auto& ar) {
        if constexpr(
            std::is_same_v<std::remove_cv_t<std::remove_reference_t<decltype(ar)>>,::asrc::core::util::DummyOutputArchive>
            || std::is_same_v<std::remove_cv_t<std::remove_reference_t<decltype(ar)>>,::asrc::core::util::DummyInputArchive>
        ){
            return false;
        }else{
            return (
                traits::output_archive<std::remove_reference_t<decltype(ar.get())>>
            );
        }
    },archive);
}
template<traits::cereal_archive Archive>
constexpr bool isOutputArchive(Archive& archive){
    return traits::output_archive<Archive>;
}
template<class VariantArchive>
requires (
    std::same_as<VariantArchive,AvailableOutputArchiveTypes>
    || std::same_as<VariantArchive,AvailableInputArchiveTypes>
    || std::same_as<VariantArchive,AvailableArchiveTypes>
)
bool isInputArchive(VariantArchive& archive){
    return std::visit([](auto& ar) {
        if constexpr(
            std::is_same_v<std::remove_cv_t<std::remove_reference_t<decltype(ar)>>,::asrc::core::util::DummyOutputArchive>
            || std::is_same_v<std::remove_cv_t<std::remove_reference_t<decltype(ar)>>,::asrc::core::util::DummyInputArchive>
        ){
            return false;
        }else{
            return (
                traits::input_archive<std::remove_reference_t<decltype(ar.get())>>
            );
        }
    },archive);
}
template<traits::cereal_archive Archive>
constexpr bool isInputArchive(Archive& archive){
    return traits::input_archive<Archive>;
}

//
// AvailableArchiveTypesを引数に取る任意の関数オブジェクトをArchiveのoperator()で呼び出せるようにするためのラッパー
//
struct PYBIND11_EXPORT VariantArchiveSaverWrapper{
    std::function<void(AvailableOutputArchiveTypes&)> saver;
    VariantArchiveSaverWrapper(const std::function<void(AvailableOutputArchiveTypes&)>& f):saver(f){}
    template<class Archive>
    void save(Archive & archive) const{
        AvailableOutputArchiveTypes wrapped(std::ref(archive));
        saver(wrapped);
    }
};
struct PYBIND11_EXPORT VariantArchiveLoaderWrapper{
    std::function<void(AvailableInputArchiveTypes&)> loader;
    VariantArchiveLoaderWrapper(const std::function<void(AvailableInputArchiveTypes&)>& f):loader(f){}
    template<class Archive>
    void load(Archive & archive){
        AvailableInputArchiveTypes wrapped(std::ref(archive));
        loader(wrapped);
    }
};
struct PYBIND11_EXPORT VariantArchiveSerializerWrapper{
    std::function<void(AvailableArchiveTypes&)> serializer;
    VariantArchiveSerializerWrapper(const std::function<void(AvailableArchiveTypes&)>& f):serializer(f){}
    template<class Archive>
    void serialize(Archive & archive){
        AvailableArchiveTypes wrapped(std::ref(archive));
        serializer(wrapped);
    }
};

//
// AvailableArchiveTypesの中身のoperator()にvisitするためのラッパー
// call_operator(archive, args...)のように使用する。
// call_operatorの第1引数には特定のArchiveの参照を直接与えても良い。
//
template<class ... Args>
struct ArchiveOperatorWrapper{
    std::tuple<Args&&...> args_tuple;

    ArchiveOperatorWrapper(Args&& ... args):args_tuple(std::forward_as_tuple(std::forward<Args>(args)...)){}

    template<class Archive>
    void operator()(Archive&& archive){
        if constexpr(
            traits::output_archive<std::remove_reference_t<Archive>>
            || traits::input_archive<std::remove_reference_t<Archive>>
        ){
            std::apply(
                [&](auto&& ... args){
                    archive(std::forward<decltype(args)>(args)...);
                },
                args_tuple
            );
        }else if constexpr(
            !std::is_same_v<std::remove_cv_t<std::remove_reference_t<Archive>>,::asrc::core::util::DummyOutputArchive>
            && !std::is_same_v<std::remove_cv_t<std::remove_reference_t<Archive>>,::asrc::core::util::DummyInputArchive>
        ){
            std::apply(
                [&](auto&& ... args){
                    archive.get()(std::forward<decltype(args)>(args)...);
                },
                args_tuple
            );
        }else{
            throw cereal::Exception("The given archive object is not an instance of available types.");
        }
    }
};
template<typename ... Args>
inline decltype(auto) call_operator(::asrc::core::util::AvailableOutputArchiveTypes& archive, Args&& ... args){
    return std::visit(ArchiveOperatorWrapper<Args...>{std::forward<Args>(args)...}, archive);
}
template<typename ... Args>
inline decltype(auto) call_operator(::asrc::core::util::AvailableInputArchiveTypes& archive, Args&& ... args){
    return std::visit(ArchiveOperatorWrapper<Args...>{std::forward<Args>(args)...}, archive);
}
template<typename ... Args>
inline decltype(auto) call_operator(::asrc::core::util::AvailableArchiveTypes& archive, Args&& ... args){
    return std::visit(ArchiveOperatorWrapper<Args...>{std::forward<Args>(args)...}, archive);
}
template<traits::output_archive Archive, typename ... Args>
inline decltype(auto) call_operator(Archive& archive, Args&& ... args){
    return ArchiveOperatorWrapper<Args...>{std::forward<Args>(args)...}(std::forward<Archive>(archive));
}
template<traits::input_archive Archive, typename ... Args>
inline decltype(auto) call_operator(Archive& archive, Args&& ... args){
    return ArchiveOperatorWrapper<Args...>{std::forward<Args>(args)...}(std::forward<Archive>(archive));
}

//
// AvailableArchiveTypesの中身を引数に取る任意の関数funcにvisitするためのラッパー
// call_func(func, archive, args...)のように使用する。funcの第1引数がArchiveとなり、第2引数以降はargs...で指定する。
// archiveにはvariant以外にも特定のArchiveの参照を直接与えても良い。
//
template<class Func, class ... Args>
struct ArchiveFuncWrapper{
    Func func;
    std::tuple<Args&&...> args_tuple;

    ArchiveFuncWrapper(Func f, Args&& ... args):func(std::forward<Func>(f)),args_tuple(std::forward_as_tuple(std::forward<Args>(args)...)){}

    template<class Archive>
    decltype(auto) operator()(Archive&& archive){
        if constexpr(
            traits::output_archive<std::remove_reference_t<Archive>>
            || traits::input_archive<std::remove_reference_t<Archive>>
        ){
            std::apply(
                [&](auto&& ... args){
                    return func(std::forward<Archive>(archive), std::forward<decltype(args)>(args)...);
                },
                args_tuple
            );
        }else if constexpr(
            !std::is_same_v<std::remove_cv_t<std::remove_reference_t<Archive>>,::asrc::core::util::DummyOutputArchive>
            && !std::is_same_v<std::remove_cv_t<std::remove_reference_t<Archive>>,::asrc::core::util::DummyInputArchive>
        ){
            std::apply(
                [&](auto&& ... args){
                    return func(archive.get(), std::forward<decltype(args)>(args)...);
                },
                args_tuple
            );
        }else{
            throw cereal::Exception("The given archive object is not an instance of available types.");
        }
    }
};
template<typename Func, typename ... Args>
inline decltype(auto) call_func(Func&& func, ::asrc::core::util::AvailableOutputArchiveTypes& archive, Args&& ... args){
    return std::visit(ArchiveFuncWrapper<Func,Args...>{std::forward<Func>(func),std::forward<Args>(args)...}, archive);
}
template<typename Func, typename ... Args>
inline decltype(auto) call_func(Func&& func, ::asrc::core::util::AvailableInputArchiveTypes& archive, Args&& ... args){
    return std::visit(ArchiveFuncWrapper<Func,Args...>{std::forward<Func>(func),std::forward<Args>(args)...}, archive);
}
template<typename Func, typename ... Args>
inline decltype(auto) call_func(Func&& func, ::asrc::core::util::AvailableArchiveTypes& archive, Args&& ... args){
    return std::visit(ArchiveFuncWrapper<Func,Args...>{std::forward<Func>(func),std::forward<Args>(args)...}, archive);
}
template<typename Func, traits::cereal_archive Archive, typename ... Args>
inline decltype(auto) call_func(Func&& func, Archive& archive, Args&& ... args){
    return ArchiveFuncWrapper<Func,Args...>{std::forward<Func>(func),std::forward<Args>(args)...}(archive);
}

//
// call_funcを呼び出すためのマクロ
// call_funcの第1引数は関数オブジェクトになるため、関数テンプレートを直接与えることができない。
// このマクロは第1引数で与えた関数名をラップするジェネリックラムダを生成するので、
// ユーザ目線では関数名を直接与えてcall_funcを呼び出すようなコードが書けるようになる。
//
#define ASRC_INTERNAL_FORWARD_VARIABLE(v) std::forward<decltype((v))>(v)
#define ASRC_INTERNAL_GENERIC_LAMBDA(function_name) \
[&](auto&& ... args){ \
    return function_name(std::forward<decltype(args)>(args)...); \
}

#define ASRC_CALL_ARCHIVE_FUNC(function_name, archive_name, ...) \
    ::asrc::core::util::call_func( \
        ASRC_INTERNAL_GENERIC_LAMBDA(function_name) \
        , ASRC_INTERNAL_FOREACH(ASRC_INTERNAL_FORWARD_VARIABLE, ASRC_INTERNAL_COMMA, archive_name __VA_OPT__(,) __VA_ARGS__) \
    ) \

//
// 変数名と同じ名前をつけたNameValuePairを作りつつ、call_funcでserialize_with_type_infoに転送するマクロ
// py::objectは型情報を付与し、それ以外はそのままの形でarchiveのoperator()に転送される。
// ユーザ目線では多くの場合はこのマクロを使用すれば様々な型の変数を同じ記述方法で入出力できるようになる。
//
#define ASRC_SERIALIZE_FORWARD_AS_NVP(variable) ::cereal::make_nvp(#variable,std::forward<decltype((variable))>(variable))
#define ASRC_SERIALIZE_NVP(archive_variable, ...) \
    do { \
        ::asrc::core::util::call_func( \
            ASRC_INTERNAL_GENERIC_LAMBDA(::asrc::core::util::serialize_with_type_info) \
            , archive_variable \
            __VA_OPT__(, ASRC_INTERNAL_FOREACH(ASRC_SERIALIZE_FORWARD_AS_NVP, ASRC_INTERNAL_COMMA , __VA_ARGS__))\
        ); \
    } while (false);


//
// Python側にsave,load,serialize,static_loadの4種類のシリアライズ用メンバを公開するためのヘルパー関数
// シリアライズ対象型をテンプレートパラメータで指定して、m.def("func",&func<T>)のようなかたちでバインドできる。
//
template<class ValueType>
void save_func_exposed_to_python(const ValueType& v, ::asrc::core::util::AvailableOutputArchiveTypes& archive){
    call_operator(archive,v);
}
template<class ValueType>
void load_func_exposed_to_python(ValueType& v, ::asrc::core::util::AvailableInputArchiveTypes& archive){
    call_operator(archive,v);
}
template<class ValueType>
void serialize_func_exposed_to_python(ValueType& v, ::asrc::core::util::AvailableArchiveTypes& archive){
    call_operator(archive,v);
}
template<class ValueType>
ValueType static_load_func_exposed_to_python(::asrc::core::util::AvailableInputArchiveTypes& archive){
    ValueType ret;
    call_operator(archive,ret);
    return std::move(ret);
}

void exportSerializer(pybind11::module &m);

ASRC_NAMESPACE_END(util)
ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
