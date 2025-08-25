// Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
//
// ある型TがクラステンプレートTemplateを継承しているかどうかを判定するメタ関数
// 非型テンプレートパラメータには対応していない。
//
#pragma once
#include <type_traits>
#include <concepts>
#include "../util/macros/common_macros.h"
#include "pointer_like.h"

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)
ASRC_NAMESPACE_BEGIN(traits)

template<typename T, template<typename...> typename Template>
struct is_derived_from_template {
    private:
    using Type=get_true_value_type_t<T>;
    template<typename ... Args>
    static std::true_type test(const Template<Args...>*);
    static std::false_type test(...);

    public:
    static constexpr bool value = decltype(test(std::declval<Type*>()))::value;
};

template<typename T, template<typename...> typename Template>
concept derived_from_template = is_derived_from_template<T,Template>::value;

ASRC_NAMESPACE_END(traits)
ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
