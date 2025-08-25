// Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
//shared_ptrを要素に持つmapとvectorについて条件に合致したもののみを抽出しつつweak_ptrとして走査するイテレータ
#pragma once
#include "Common.h"
#include <memory>
#include <iterator>
#include <vector>
#include <map>
#include <functional>
#include <pybind11/pybind11.h>
namespace py=pybind11;

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)
ASRC_INLINE_NAMESPACE_BEGIN(util)

template<class ReturnType,class KeyType,class ValueType,
    std::enable_if_t<std::is_base_of_v<ValueType,ReturnType>,std::nullptr_t> = nullptr
>
class PYBIND11_EXPORT MapIterator:public std::iterator<std::forward_iterator_tag,std::weak_ptr<ReturnType>>{
    public:
    typedef const typename std::map<KeyType,std::shared_ptr<ValueType>> ContainerType;
    typedef typename ContainerType::const_iterator ItType;
    typedef typename std::shared_ptr<const ReturnType> ConstSharedType;
    typedef typename std::weak_ptr<ReturnType> WeakType;
    typedef typename std::function<bool(ConstSharedType)> MatcherType;
    MapIterator(ItType it_,MatcherType matcher_,ItType end_):it(it_),matcher(matcher_),end(end_){}
    MapIterator(const MapIterator<ReturnType,KeyType,ValueType>& other):it(other.it),matcher(other.matcher),end(other.end){}
    ~MapIterator(){}
    bool operator==(const MapIterator<ReturnType,KeyType,ValueType>& other){
        return it==other.it;
    }
    bool operator!=(const MapIterator<ReturnType,KeyType,ValueType>& other){
        return it!=other.it;
    }
    const WeakType operator*() const{
        if constexpr (std::is_same_v<ReturnType,ValueType>){
            return (*it).second;
        }else{
            return std::dynamic_pointer_cast<ReturnType>((*it).second);
        }
    }
    WeakType operator*(){
        if constexpr (std::is_same_v<ReturnType,ValueType>){
            return (*it).second;
        }else{
            return std::dynamic_pointer_cast<ReturnType>((*it).second);
        }
    }
    const WeakType operator->() const{
        if constexpr (std::is_same_v<ReturnType,ValueType>){
            return it->second;
        }else{
            return std::dynamic_pointer_cast<ReturnType>(it->second);
        }
    }
    WeakType operator->(){
        if constexpr (std::is_same_v<ReturnType,ValueType>){
            return it->second;
        }else{
            return std::dynamic_pointer_cast<ReturnType>(it->second);
        }
    }
    MapIterator<ReturnType,KeyType,ValueType>& operator++(){
        while(it!=end){
            ++it;
            if constexpr (std::is_same_v<ReturnType,ValueType>){
                if(it==end || matcher(it->second)){
                    return *this;
                }
            }else{
                if(it==end || (util::isinstance<ReturnType>(it->second) && matcher(std::dynamic_pointer_cast<ReturnType>(it->second)))){
                    return *this;
                }
            }
        }
        return *this;
    }
    MapIterator<ReturnType,KeyType,ValueType> operator++(int){
        while(it!=end){
            ++it;
            if constexpr (std::is_same_v<ReturnType,ValueType>){
                if(it==end || matcher(it->second)){
                    return *this;
                }
            }else{
                if(it==end || (util::isinstance<ReturnType>(it->second) && matcher(std::dynamic_pointer_cast<ReturnType>(it->second)))){
                    return *this;
                }
            }
        }
        return *this;
    }
    private:
    MatcherType matcher;
    ItType it,end;
};
template<class ReturnType,class KeyType,class ValueType,
    std::enable_if_t<std::is_base_of_v<ValueType,ReturnType>,std::nullptr_t> = nullptr
>
class MapIterable{
    public:
    typedef const typename std::map<KeyType,std::shared_ptr<ValueType>> ContainerType;
    typedef MapIterator<ReturnType,KeyType,ValueType> iterator;
    typedef typename std::shared_ptr<const ReturnType> ConstSharedType;
    typedef typename std::weak_ptr<ReturnType> WeakType;
    typedef typename std::function<bool(ConstSharedType)> MatcherType;
    MapIterable(ContainerType& container_,MatcherType matcher_=[](ConstSharedType v){return true;}):container(container_),matcher(matcher_),it(begin()){}
    ~MapIterable(){}
    MapIterable<ReturnType,KeyType,ValueType>& iter(){
        it=begin();
        first=true;
        return *this;
    }
    WeakType next(){
        if(first){
            first=false;
        }else{
            ++it;
        }
        if(it==end()){
            throw pybind11::stop_iteration();
        }
        return *it;
    }
    iterator begin(){
        auto ret=container.begin();
        while(ret!=container.end()){
            if constexpr (std::is_same_v<ReturnType,ValueType>){
                if(matcher(ret->second)){
                    return iterator(ret,matcher,container.end());
                }
            }else{
                if(util::isinstance<ReturnType>(ret->second) && matcher(std::dynamic_pointer_cast<ReturnType>(ret->second))){
                    return iterator(ret,matcher,container.end());
                }
            }
            ++ret;
        }
        return iterator(ret,matcher,container.end());
    }
    iterator end(){
        return iterator(container.end(),matcher,container.end());
    }
    private:
    MatcherType matcher;
    ContainerType& container;
    iterator it;
    bool first=true;
};

template<class ReturnType,class ValueType,
    std::enable_if_t<std::is_base_of_v<ValueType,ReturnType>,std::nullptr_t> = nullptr
>
class VectorIterator:public std::iterator<std::forward_iterator_tag,std::weak_ptr<ReturnType>>{
    public:
    typedef const typename std::vector<std::shared_ptr<ValueType>> ContainerType;
    typedef typename ContainerType::const_iterator ItType;
    typedef typename std::shared_ptr<const ReturnType> ConstSharedType;
    typedef typename std::weak_ptr<ReturnType> WeakType;
    typedef typename std::function<bool(ConstSharedType)> MatcherType;
    VectorIterator(ItType it_,MatcherType matcher_,ItType end_):it(it_),matcher(matcher_),end(end_){}
    VectorIterator(const VectorIterator<ReturnType,ValueType>& other):it(other.it),matcher(other.matcher),end(other.end){}
    ~VectorIterator(){}
    bool operator==(const VectorIterator<ReturnType,ValueType>& other){
        return it==other.it;
    }
    bool operator!=(const VectorIterator<ReturnType,ValueType>& other){
        return it!=other.it;
    }
    const WeakType operator*() const{
        if constexpr (std::is_same_v<ReturnType,ValueType>){
            return *it;
        }else{
            return std::dynamic_pointer_cast<ReturnType>(*it);
        }
    }
    const WeakType operator->() const{
        if constexpr (std::is_same_v<ReturnType,ValueType>){
            return it->operator();
        }else{
            return std::dynamic_pointer_cast<ReturnType>(it->operator());
        }
    }
    VectorIterator<ReturnType,ValueType>& operator++(){
        while(it!=end){
            ++it;
            if constexpr (std::is_same_v<ReturnType,ValueType>){
                if(it==end || matcher(*it)){
                    return *this;
                }
            }else{
                if(it==end || (util::isinstance<ReturnType>(*it) && matcher(std::dynamic_pointer_cast<ReturnType>(*it)))){
                    return *this;
                }
            }
        }
        return *this;
    }
    VectorIterator<ReturnType,ValueType> operator++(int){
        while(it!=end){
            ++it;
            if constexpr (std::is_same_v<ReturnType,ValueType>){
                if(it==end || matcher(*it)){
                    return *this;
                }
            }else{
                if(it==end || (util::isinstance<ReturnType>(*it) && matcher(std::dynamic_pointer_cast<ReturnType>(*it)))){
                    return *this;
                }
            }
        }
        return *this;
    }
    private:
    MatcherType matcher;
    ItType it,end;
};
template<class ReturnType,class ValueType,
    std::enable_if_t<std::is_base_of_v<ValueType,ReturnType>,std::nullptr_t> = nullptr
>
class VectorIterable{
    public:
    typedef const typename std::vector<std::shared_ptr<ValueType>> ContainerType;
    typedef VectorIterator<ReturnType,ValueType> iterator;
    //typedef const VectorIterator<ReturnType,ValueType> const_iterator;
    typedef typename std::shared_ptr<const ReturnType> ConstSharedType;
    typedef typename std::weak_ptr<ReturnType> WeakType;
    typedef typename std::function<bool(ConstSharedType)> MatcherType;
    VectorIterable(ContainerType& container_,MatcherType matcher_=[](ConstSharedType v){return true;}):container(container_),matcher(matcher_),it(begin()){}
    ~VectorIterable(){}
    VectorIterable<ReturnType,ValueType>& iter(){
        //std::cout<<"iter()"<<std::endl;
        it=begin();
        first=true;
        return *this;
    }
    WeakType next(){
        //std::cout<<"next()"<<std::endl;
        if(first){
            first=false;
        }else{
            ++it;
        }
        if(it==end()){
            throw pybind11::stop_iteration();
        }
        return *it;
    }
    iterator begin(){
        auto ret=container.begin();
        while(ret!=container.end()){
            if constexpr (std::is_same_v<ReturnType,ValueType>){
                if(matcher(*ret)){
                    return iterator(ret,matcher,container.end());
                }
            }else{
                if(util::isinstance<ReturnType>(*ret) && matcher(std::dynamic_pointer_cast<ReturnType>(*ret))){
                    return iterator(ret,matcher,container.end());
                }
            }
            ++ret;
        }
        return iterator(ret,matcher,container.end());
    }
    iterator end(){
        return iterator(container.end(),matcher,container.end());
    }
    private:
    MatcherType matcher;
    ContainerType& container;
    iterator it;
    bool first=true;
};

ASRC_NAMESPACE_END(util)
ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
