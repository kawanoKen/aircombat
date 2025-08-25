/**
 * Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
 * @file
 * 
 * @brief 雑多なユーティリティ関数を集めたファイル
 */
#pragma once
#include "Common.h"
#include <memory>
#include <pybind11/pybind11.h>
#include <nlohmann/json.hpp>
#include "traits/traits.h"
#include "util/serialization/serialization.h"
#include "Pythonable.h"
namespace py=pybind11;
namespace nl=nlohmann;

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)
ASRC_INLINE_NAMESPACE_BEGIN(util)

/**
 * @brief fがDerivedのインスタンスかどうかを返す。
 */
template<class Derived,class Base>
bool isinstance(const std::shared_ptr<Base>& f){
    std::shared_ptr<Derived> t=std::dynamic_pointer_cast<Derived>(f);
    return (bool)t;
}
/**
 * @brief fがDerivedのインスタンスかどうかを返す。
 */
template<class Derived,class Base>
bool isinstance(const std::shared_ptr<const Base>& f){
    std::shared_ptr<const Derived> t=std::dynamic_pointer_cast<const Derived>(f);
    return (bool)t;
}
/**
 * @brief fがDerivedのインスタンスかどうかを返す。
 */
template<class Derived,class Base>
bool isinstance(const std::weak_ptr<Base>& f){
    std::shared_ptr<Derived> t=std::dynamic_pointer_cast<Derived>(f.lock());
    return (bool)t;
}
/**
 * @brief fがDerivedのインスタンスかどうかを返す。
 */
template<class Derived,class Base>
bool isinstance(const std::weak_ptr<const Base>& f){
    std::shared_ptr<const Derived> t=std::dynamic_pointer_cast<const Derived>(f.lock());
    return (bool)t;
}
/**
 * @brief srcをstd::shared_ptr<Dst>にキャストして返す。
 */
template<class Dst,class Src>
std::shared_ptr<Dst> getShared(const std::shared_ptr<Src>& src){
    if constexpr (std::is_same_v<Dst,Src>){
        return src;
    }else{
        return std::dynamic_pointer_cast<Dst>(src);
    }
}
/**
 * @brief srcをstd::shared_ptr<Dst>にキャストして返す。
 */
template<class Dst,class Src>
std::shared_ptr<Dst> getShared(const std::weak_ptr<Src>& src){
    if constexpr (std::is_same_v<Dst,Src>){
        return src.lock();
    }else{
        return std::dynamic_pointer_cast<Dst>(src.lock());
    }
}
/**
 * @brief srcをstd::shared_ptr<Dst>にキャストして返す。
 */
template<class Src>
std::shared_ptr<Src> getShared(const std::shared_ptr<Src>& src){
    return src;
}
/**
 * @brief srcをstd::shared_ptr<Dst>にキャストして返す。
 */
template<class Src>
std::shared_ptr<Src> getShared(const std::weak_ptr<Src>& src){
    return src.lock();
}

template<class Scalar>
nl::json getValueFromDistributionImpl(const std::function<Scalar(Scalar,Scalar)>& sampler, const nl::json& first,const nl::json& second){
    if(first.is_number() && second.is_number()){
        return sampler(first.get<Scalar>(),second.get<Scalar>());
    }else if(first.is_array() && second.is_array()){
        if(first.size()==second.size()){
            std::size_t size=first.size();
            nl::json::array_t ret(size);
            for(std::size_t i=0;i<size;++i){
                ret[i]=getValueFromDistributionImpl<Scalar>(sampler,first.at(i),second.at(i));
            }
            return std::move(ret);
        }else{
            throw std::runtime_error("getValueFromJsonR failed. Distribution parameters need to have the same size.");
        }
    }else if(first.is_object() && second.is_object()){
        nl::json::object_t ret;
        for(auto && [key, first_val] : first.items()){
            if(second.contains(key)){
                ret[key]=getValueFromDistributionImpl<Scalar>(sampler,first_val,second.at(key));
            }else{
                throw std::runtime_error("getValueFromJsonR failed. Distribution parameters need to have the same keys when they are json object.");
            }
        }
        return std::move(ret);
    }else{
        throw std::runtime_error("getValueFromJsonR failed. Distribution parameters need to be number, array, or object to use distribution.");
    }
}

/**
 * @brief 引数として与えたjsonから値をサンプリングして抽出する。
 * 
 * @param [in] j   抽出元となるjson
 * @param [in] gen 乱数生成器
 * 
 * @details
 *      - j が object(dict)の場合
 * 
 *          "type"キーの値によって分岐する。省略時は"direct"とみなす。
 * 
 *          - "direct"の場合
 *              - j["type"]=="direct" かつ j が "value" キーを持っている場合
 * 
 *                  j["value"]を返す。
 *              - それ以外の場合
 * 
 *                  objectの各キーの値を再帰的に getValueFromJsonR を呼び出した結果に置き換えて返す。
 * 
 *          - "normal"の場合
 * 
 *              j["mean"]を平均、j["stddev"]を標準偏差とした正規分布からサンプリングして返す。
 *              平均と標準偏差はスカラーに限らず、形が一致していれば任意のテンソルを与えることができる。
 * 
 *          - "uniform"の場合
 * 
 *              j["low"]を下限、j["high"]を上限とした一様分布でサンプリングして返す。
 * 
 *              j["dtype"]を"float"か"int"で指定(省略時は"float")し、実数と整数のどちらでサンプリングするかを選択可能。
 * 
 *              上限と下限はスカラーに限らず、形が一致していれば任意のテンソルを与えることができる。
 * 
 *          - "choice"の場合
 * 
 *              j["weights"]で与えた重みにしたがって、j["candidates"]で与えた選択肢から一つを選んで返す。
 * 
 *      - j が array(list)の場合
 * 
 *          arrayの各要素を再帰的に getValueFromJsonR を呼び出した結果に置き換えて返す。
 *          
 *      - j がそれ以外(数値や文字列)の場合
 * 
 *          j をそのまま返す。
 */
template<class URBG>
nl::json getValueFromJsonR(const nl::json& j, URBG& gen){
    if(j.is_object()){
        std::string type;
        if(j.contains("type")){
            type=j.at("type");
        }else{
            type="direct";
        }
        if(type=="normal"){
            return getValueFromDistributionImpl<double>(
                [&gen](double mean,double stddev){
                    std::normal_distribution<double> dist(mean,stddev);
                    return dist(gen);
                },
                j.at("mean"),j.at("stddev")
            );
        }else if(type=="uniform"){
            std::string dtype;
            if(j.contains("dtype")){
                dtype=j.at("dtype");
            }else{
                dtype="float";
            }
            if(dtype=="int"){
                return getValueFromDistributionImpl<std::int64_t>(
                    [&gen](std::int64_t low,std::int64_t high){
                        std::uniform_int_distribution<std::int64_t> dist(low,high);
                        return dist(gen);
                    },
                    j.at("low"),j.at("high")
                );
            }else{
                return getValueFromDistributionImpl<double>(
                    [&gen](double low,double high){
                        std::uniform_real_distribution<double> dist(low,high);
                        return dist(gen);
                    },
                    j.at("low"),j.at("high")
                );
            }
        }else if(type=="choice"){
            nl::json weights=j.at("weights");
            nl::json candidates=j.at("candidates");
            std::discrete_distribution<std::size_t> dist(weights.begin(),weights.end());
            return getValueFromJsonR(candidates[dist(gen)],gen);
        }else{//direct
            if(j.contains("type") && j.contains("value")){
                return j.at("value");
            }else{
                nl::json ret=nl::json::object();
                for(auto& e:j.items()){
                    ret[e.key()]=getValueFromJsonR(e.value(),gen);
                }
                return std::move(ret);
            }
        }
    }else if(j.is_array()){
        nl::json ret=nl::json::array();
        for(auto& e:j){
            ret.push_back(getValueFromJsonR(e,gen));
        }
        return std::move(ret);
    }else{//direct
        return j;
    }
}

/**
 * @brief 引数として与えたjsonから値をサンプリングして抽出する。
 * 
 * @details 新たな乱数生成器を用いて getValueFromJsonR を呼び出す。
 * 
 * @sa getValueFromJsonR
 */
nl::json PYBIND11_EXPORT getValueFromJson(const nl::json& j);

/**
 * @brief 引数として与えたjsonの指定したキーから値をサンプリングして抽出する。
 * 
 * @details getValueFromJsonR(j[ptr],gen) を呼び出す。
 * 
 * @sa getValueFromJsonR
 */
template<class URBG>
nl::json getValueFromJsonKR(const nl::json &j,const nl::json::json_pointer& ptr,URBG& gen){
    if(j.contains(ptr)){
        return getValueFromJsonR(j.at(ptr),gen);
    }else{
        throw std::runtime_error("getValueFromJsonKR() failed. The given key '"+ptr.to_string()+"' was not found in j="+j.dump());
    }
}

/**
 * @brief 引数として与えたjsonの指定したキーから値をサンプリングして抽出する。
 * 
 * @details getValueFromJsonR(j[key],gen) を呼び出す。
 * 
 * @sa getValueFromJsonR
 */
template<class URBG>
nl::json getValueFromJsonKR(const nl::json &j,const std::string& key,URBG& gen){
    if(j.contains(key)){
        return getValueFromJsonR(j.at(key),gen);
    }else{
        if(key.empty() || key.at(0)=='/'){
            return getValueFromJsonKR(j,nl::json::json_pointer(key),gen);
        }else{
            throw py::key_error("getValueFromJsonKR() failed. The given key '"+key+"' was not found in j="+j.dump());
        }
    }
}

/**
 * @brief 引数として与えたjsonの指定したキーから値をサンプリングして抽出する。
 * 
 * @details 新たな乱数生成器を用いて getValueFromJsonR(j[ptr],gen) を呼び出す。
 * 
 * @sa getValueFromJsonR
 */

nl::json PYBIND11_EXPORT getValueFromJsonK(const nl::json &j,const nl::json::json_pointer& ptr);

/**
 * @brief 引数として与えたjsonの指定したキーから値をサンプリングして抽出する。
 * 
 * @details 新たな乱数生成器を用いて getValueFromJsonR(j[key],gen) を呼び出す。
 * 
 * @sa getValueFromJsonR
 */
nl::json PYBIND11_EXPORT getValueFromJsonK(const nl::json &j,const std::string& key);

/**
 * @brief 引数として与えたjsonの指定したキーから値をサンプリングして抽出する。(非存在時のデフォルト値付き)
 * 
 * @details キーが存在すれば getValueFromJsonR(j[ptr],gen) を呼び出す。
 *          存在しなければ defaultValue を返す。
 * 
 * @tparam ValueType 戻り値の型。 defaultValue からある程度自動推論されるが、場合によっては手動で指定が必要。
 * 
 * @sa getValueFromJsonR
 */
template<typename ValueType,class URBG,bool ShowExcept=true>
ValueType getValueFromJsonKRD(const nl::json &j,const nl::json::json_pointer& ptr,URBG& gen,const ValueType& defaultValue){
    if(j.contains(ptr)){
        return getValueFromJsonR(j.at(ptr),gen);
    }else{
        return defaultValue;
    }
}

/**
 * @brief 引数として与えたjsonの指定したキーから値をサンプリングして抽出する。(非存在時のデフォルト値付き)
 * 
 * @details キーが存在すれば getValueFromJsonR(j[key],gen) を呼び出す。
 *          存在しなければ defaultValue を返す。
 * 
 * @tparam ValueType 戻り値の型。 defaultValue からある程度自動推論されるが、場合によっては手動で指定が必要。
 * 
 * @sa getValueFromJsonR
 */
template<typename ValueType,class URBG,bool ShowExcept=true>
ValueType getValueFromJsonKRD(const nl::json &j,const std::string& key,URBG& gen,const ValueType& defaultValue){
    if(j.contains(key)){
        return getValueFromJsonR(j.at(key),gen);
    }else{
        if(key.empty() || key.at(0)=='/'){
            return getValueFromJsonKRD(j,nl::json::json_pointer(key),gen,defaultValue);
        }else{
            return defaultValue;
        }
    }
}

/**
 * @brief 引数として与えたjsonの指定したキーから値をサンプリングして抽出する。(非存在時のデフォルト値付き)
 * 
 * @details キーが存在すれば新たな乱数生成器を用いて getValueFromJsonR(j[ptr],gen) を呼び出す。
 *          存在しなければ defaultValue を返す。
 * 
 * @tparam ValueType 戻り値の型。 defaultValue からある程度自動推論されるが、場合によっては手動で指定が必要。
 * 
 * @sa getValueFromJsonR
 */
template<typename ValueType,bool ShowExcept=true>
ValueType getValueFromJsonKD(const nl::json &j,const nl::json::json_pointer& ptr,const ValueType& defaultValue){
    std::mt19937 gen;gen=std::mt19937(std::random_device()());
    if(j.contains(ptr)){
        return getValueFromJsonR(j.at(ptr),gen);
    }else{
        return defaultValue;
    }
}

/**
 * @brief 引数として与えたjsonの指定したキーから値をサンプリングして抽出する。(非存在時のデフォルト値付き)
 * 
 * @details キーが存在すれば新たな乱数生成器を用いて getValueFromJsonR(j[key],gen) を呼び出す。
 *          存在しなければ defaultValue を返す。
 * 
 * @tparam ValueType 戻り値の型。 defaultValue からある程度自動推論されるが、場合によっては手動で指定が必要。
 * 
 * @sa getValueFromJsonR
 */
template<typename ValueType,bool ShowExcept=true>
ValueType getValueFromJsonKD(const nl::json &j,const std::string& key,const ValueType& defaultValue){
    std::mt19937 gen;gen=std::mt19937(std::random_device()());
    if(j.contains(key)){
        return getValueFromJsonR(j.at(key),gen);
    }else{
        if(key.empty() || key.at(0)=='/'){
            return getValueFromJsonKRD(j,nl::json::json_pointer(key),gen,defaultValue);
        }else{
            return defaultValue;
        }
    }
}

/**
 * @brief base に patch をmerge_patch した結果を返す。baseは変更されない。
 * 
 * @details nl::jsonのmerge_patchはin-placeなメンバ関数のため、out-of-placeなmerge patchをここで実装するもの。
 */
nl::json PYBIND11_EXPORT merge_patch(const nl::json& base,const nl::json& patch);

void exportUtility(py::module &m);

ASRC_NAMESPACE_END(util)
ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
