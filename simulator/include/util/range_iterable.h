// Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
//std::ranges::rangeをPython側でイテラブルとして扱えるようにする

#pragma once
#include <cereal/details/util.hpp>
#include <ranges>
#include <pybind11/pybind11.h>
#include "macros/common_macros.h"

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)
ASRC_INLINE_NAMESPACE_BEGIN(util)

template<std::ranges::range RangeType>
struct PYBIND11_EXPORT RangeIterableForPy{
	// rangeをPython側に公開するためのラッパー
	using StoredRangeType = std::conditional_t<std::is_lvalue_reference_v<RangeType>,RangeType,std::decay_t<RangeType>>;
	using IteratorType=decltype(std::declval<StoredRangeType>().begin()) ;
	RangeIterableForPy(RangeType&& r):
	 range(std::forward<RangeType>(r)),
	 it(range.begin()),
	 first(true)
	{
	}
	auto next(){
        if(first){
            first=false;
        }else{
            ++it;
        }
        if(it==range.end()){
            throw pybind11::stop_iteration();
        }
		return *it;
	}

	private:
	StoredRangeType range;
	IteratorType it;
    bool first;
};

template<std::ranges::range RangeType>
inline RangeIterableForPy<RangeType> makeRangeIterableForPy(RangeType&& r){
	return RangeIterableForPy<RangeType>(std::forward<RangeType>(r));
}

template<std::ranges::range RangeType, typename ... Extras>
pybind11::class_<RangeIterableForPy<RangeType>> expose_range_name(pybind11::module &m,const std::string& className, Extras&& ... extras){
	try{
		return pybind11::class_<RangeIterableForPy<RangeType>>(m,className.c_str(),std::forward<Extras>(extras)...)
		.def("__iter__",
			[](RangeIterableForPy<RangeType>& it)-> RangeIterableForPy<RangeType>& {return it;}
			,pybind11::return_value_policy::reference_internal
		)
		.def("__next__",[](RangeIterableForPy<RangeType>& v){
			return v.next();
		},pybind11::return_value_policy::reference_internal)
		;
    }catch(std::exception& ex){
        if(pybind11::detail::get_type_info(typeid(RangeIterableForPy<RangeType>))){
            // already bound
            auto ret=pybind11::reinterpret_borrow<pybind11::class_<RangeIterableForPy<RangeType>>>(pybind11::detail::get_type_handle(typeid(RangeIterableForPy<RangeType>),true));
            assert(ret);
            return std::move(ret);
        }else{
            throw ex;
        }
	}
}
template<std::ranges::range RangeType, typename ... Extras>
pybind11::class_<RangeIterableForPy<RangeType>> expose_range(pybind11::module &m, Extras&& ... extras){
	return expose_range_name<RangeType,Extras...>(m,cereal::util::demangle(typeid(RangeType).name()),std::forward<Extras>(extras)...);
}

ASRC_NAMESPACE_END(util)
ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
