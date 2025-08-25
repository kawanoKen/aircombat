// Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#pragma once
#include "Common.h"
#include <memory>
#include <pybind11/pybind11.h>
#include <nlohmann/json.hpp>
#include "traits/json.h"
namespace py=pybind11;
namespace nl=nlohmann;

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)
ASRC_INLINE_NAMESPACE_BEGIN(util)

//C++側で保持しているnl::basic_jsonをPython側から変更可能な形でアクセス可能にする。
//get()または__call__でプリミティブ型への変換に対応する。
template<traits::basic_json BasicJsonType>
class PYBIND11_EXPORT BasicJsonIterator{
    public://private:
    typename BasicJsonType::iterator it;
    using difference_type=typename BasicJsonType::iterator::difference_type;
    public:
    BasicJsonIterator(typename BasicJsonType::iterator it_):it(it_){}
    ~BasicJsonIterator()=default;
    bool operator==(const BasicJsonIterator& other) const{
        return it==other.it;
    }
    bool operator!=(const BasicJsonIterator& other) const{
        return it!=other.it;
    }
    bool operator<(const BasicJsonIterator& other) const{
        return it<other.it;
    }
    bool operator<=(const BasicJsonIterator& other) const{
        return it<=other.it;
    }
    bool operator>(const BasicJsonIterator& other) const{
        return it>other.it;
    }
    bool operator>=(const BasicJsonIterator& other) const{
        return it>=other.it;
    }
    BasicJsonIterator& operator+=(difference_type i){
        it+=i;
        return *this;
    }
    BasicJsonIterator& operator-=(difference_type i){
        it-=i;
        return *this;
    }
    BasicJsonIterator operator+(difference_type i) const{
        return BasicJsonIterator(it+i);
    }
    friend BasicJsonIterator operator+(difference_type i, const BasicJsonIterator& it){
        return it+i;
    }
    BasicJsonIterator operator-(difference_type i) const{
        return BasicJsonIterator(it-i);
    }
    difference_type operator-(const BasicJsonIterator& other) const{
        return it-other.it;
    }
    BasicJsonType& operator[](difference_type n) const{
        return it[n];
    }
    std::string key(){
        return it.key();
    }
    BasicJsonType& value(){
        return it.value();
    }
};
template<traits::basic_json BasicJsonType>
class PYBIND11_EXPORT BasicJsonReverseIterator{
    friend class BasicJsonMutableFromPython;
    public://private:
    typename BasicJsonType::reverse_iterator it;
    using difference_type=typename BasicJsonType::reverse_iterator::difference_type;
    public:
    BasicJsonReverseIterator(typename BasicJsonType::reverse_iterator it_):it(it_){}
    ~BasicJsonReverseIterator()=default;
    bool operator==(const BasicJsonReverseIterator& other) const{
        return it==other.it;
    }
    bool operator!=(const BasicJsonReverseIterator& other) const{
        return it!=other.it;
    }
    bool operator<(const BasicJsonReverseIterator& other) const{
        return it<other.it;
    }
    bool operator<=(const BasicJsonReverseIterator& other) const{
        return it<=other.it;
    }
    bool operator>(const BasicJsonReverseIterator& other) const{
        return it>other.it;
    }
    bool operator>=(const BasicJsonReverseIterator& other) const{
        return it>=other.it;
    }
    BasicJsonReverseIterator& operator+=(difference_type i){
        it+=i;
        return *this;
    }
    BasicJsonReverseIterator& operator-=(difference_type i){
        it-=i;
        return *this;
    }
    BasicJsonReverseIterator operator+(difference_type i) const{
        return BasicJsonReverseIterator(it+i);
    }
    friend BasicJsonReverseIterator operator+(difference_type i, const BasicJsonReverseIterator& it){
        return it+i;
    }
    BasicJsonReverseIterator operator-(difference_type i) const{
        return BasicJsonReverseIterator(it-i);
    }
    difference_type operator-(const BasicJsonReverseIterator& other) const{
        return it-other.it;
    }
    BasicJsonType& operator[](difference_type n) const{
        return it[n];
    }
    std::string key(){
        return it.key();
    }
    BasicJsonType& value(){
        return it.value();
    }
};
template<traits::basic_json BasicJsonType>
class PYBIND11_EXPORT BasicJsonKeyIterable{
    public:
    BasicJsonKeyIterable(BasicJsonType& j_):j(j_){
        it=j.begin();
        first=true;
    }
    ~BasicJsonKeyIterable()=default;
    std::string next(){
        if(first){
            first=false;
        }else{
            ++it;
        }
        if(it==j.end()){
            throw pybind11::stop_iteration();
        }
        return it.key();
    }
    private:
    BasicJsonType& j;
    bool first;
    typename BasicJsonType::iterator it;
};
template<traits::basic_json BasicJsonType>
class PYBIND11_EXPORT BasicJsonValueIterable{
    public:
    BasicJsonValueIterable(BasicJsonType& j_):j(j_){
        it=j.begin();
        first=true;
    }
    ~BasicJsonValueIterable()=default;
    BasicJsonType& next(){
        if(first){
            first=false;
        }else{
            ++it;
        }
        if(it==j.end()){
            throw pybind11::stop_iteration();
        }
        return it.value();
    }
    private:
    BasicJsonType& j;
    bool first;
    typename BasicJsonType::iterator it;
};
template<traits::basic_json BasicJsonType>
class PYBIND11_EXPORT BasicJsonItemIterable{
    public:
    BasicJsonItemIterable(BasicJsonType& j_):j(j_){
        it=j.begin();
        first=true;
    }
    ~BasicJsonItemIterable()=default;
    std::pair<std::string,BasicJsonType&> next(){
        if(first){
            first=false;
        }else{
            ++it;
        }
        if(it==j.end()){
            throw pybind11::stop_iteration();
        }
        return {it.key(),it.value()};
    }
    private:
    BasicJsonType& j;
    bool first;
    typename BasicJsonType::iterator it;
};
template<traits::basic_json BasicJsonType>
class PYBIND11_EXPORT BasicJsonConstKeyIterable{
    public:
    BasicJsonConstKeyIterable(const BasicJsonType& j_):j(j_){
        it=j.begin();
        first=true;
    }
    ~BasicJsonConstKeyIterable()=default;
    std::string next(){
        if(first){
            first=false;
        }else{
            ++it;
        }
        if(it==j.end()){
            throw pybind11::stop_iteration();
        }
        return it.key();
    }
    private:
    const BasicJsonType& j;
    bool first;
    typename BasicJsonType::const_iterator it;
};
template<traits::basic_json BasicJsonType>
class PYBIND11_EXPORT BasicJsonConstValueIterable{
    public:
    BasicJsonConstValueIterable(const BasicJsonType& j_):j(j_){
        it=j.begin();
        first=true;
    }
    ~BasicJsonConstValueIterable()=default;
    const BasicJsonType& next(){
        if(first){
            first=false;
        }else{
            ++it;
        }
        if(it==j.end()){
            throw pybind11::stop_iteration();
        }
        return it.value();
    }
    private:
    const BasicJsonType& j;
    bool first;
    typename BasicJsonType::const_iterator it;
};
template<traits::basic_json BasicJsonType>
class PYBIND11_EXPORT BasicJsonConstItemIterable{
    public:
    BasicJsonConstItemIterable(const BasicJsonType& j_):j(j_){
        it=j.begin();
        first=true;
    }
    ~BasicJsonConstItemIterable()=default;
    std::pair<std::string,const BasicJsonType&> next(){
        if(first){
            first=false;
        }else{
            ++it;
        }
        if(it==j.end()){
            throw pybind11::stop_iteration();
        }
        return {it.key(),it.value()};
    }
    private:
    const BasicJsonType& j;
    bool first;
    typename BasicJsonType::const_iterator it;
};

using JsonIterator=BasicJsonIterator<nl::json>;
using JsonReverseIterator=BasicJsonReverseIterator<nl::json>;
using JsonKeyIterable=BasicJsonKeyIterable<nl::json>;
using JsonValueIterable=BasicJsonValueIterable<nl::json>;
using JsonItemIterable=BasicJsonItemIterable<nl::json>;
using JsonConstKeyIterable=BasicJsonConstKeyIterable<nl::json>;
using JsonConstValueIterable=BasicJsonConstValueIterable<nl::json>;
using JsonConstItemIterable=BasicJsonConstItemIterable<nl::json>;

using OrderedJsonIterator=BasicJsonIterator<nl::ordered_json>;
using OrderedJsonReverseIterator=BasicJsonReverseIterator<nl::ordered_json>;
using OrderedJsonKeyIterable=BasicJsonKeyIterable<nl::ordered_json>;
using OrderedJsonValueIterable=BasicJsonValueIterable<nl::ordered_json>;
using OrderedJsonItemIterable=BasicJsonItemIterable<nl::ordered_json>;
using OrderedJsonConstKeyIterable=BasicJsonConstKeyIterable<nl::ordered_json>;
using OrderedJsonConstValueIterable=BasicJsonConstValueIterable<nl::ordered_json>;
using OrderedJsonConstItemIterable=BasicJsonConstItemIterable<nl::ordered_json>;

void exportJsonMutableFromPython(py::module &m);

ASRC_NAMESPACE_END(util)
ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
