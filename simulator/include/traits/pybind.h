// Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
//
// pybind11の利用に関するtraits
//
#pragma once
#include <concepts>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include "../util/macros/common_macros.h"

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)
ASRC_NAMESPACE_BEGIN(traits)

template<typename T>
using is_pyobject = ::pybind11::detail::is_pyobject<T>;
template<typename T>
concept pyobject =is_pyobject<T>::value;

template<typename T>
using is_pyhandle = std::is_base_of<::pybind11::handle,T>;
template<typename T>
concept pyhandle =is_pyhandle<T>::value;

template<typename T>
struct is_pyaccessor : std::false_type {};
template<class Policy>
struct is_pyaccessor<::pybind11::detail::accessor<Policy>> : std::true_type {};
template<typename T>
concept pyaccessor =is_pyaccessor<T>::value;


template<pyobject Object>
using is_sequence_like_pyobject = std::disjunction<
    std::is_same<Object,::pybind11::sequence>
    ,std::is_same<Object,::pybind11::list>
    ,std::is_same<Object,::pybind11::anyset>
    ,std::is_same<Object,::pybind11::tuple>
    ,std::is_same<Object,::pybind11::args>
>;
template<class Object>
concept sequence_like_pyobject = pyobject<Object> && is_sequence_like_pyobject<Object>::value;

template<pyobject Object>
using is_set_like_pyobject = std::disjunction<
    std::is_same<Object,::pybind11::anyset>
    ,std::is_same<Object,::pybind11::set>
    ,std::is_same<Object,::pybind11::frozenset>
>;
template<class Object>
concept set_like_pyobject = pyobject<Object> && is_set_like_pyobject<Object>::value;

template<pyobject Object>
using is_mapping_like_pyobject = std::disjunction<
    std::is_same<Object,::pybind11::dict>
    ,std::is_same<Object,::pybind11::kwargs>
>;
template<class Object>
concept mapping_like_pyobject = pyobject<Object> && is_mapping_like_pyobject<Object>::value;


template<class Type>
concept pybind11_bind_vector_allowed = requires {
    ::pybind11::bind_vector<Type>(::pybind11::handle(),"");
};
template<class Type>
concept pybind11_bind_map_allowed = requires {
    ::pybind11::bind_map<Type>(::pybind11::handle(),"");
};

ASRC_NAMESPACE_END(traits)
ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)

