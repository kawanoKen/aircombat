// Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#include "Utility.h"

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)
ASRC_INLINE_NAMESPACE_BEGIN(util)

    nl::json getValueFromJson(const nl::json& j){
        std::mt19937 gen;gen=std::mt19937(std::random_device()());
        return getValueFromJsonR(j,gen);
    }
    nl::json getValueFromJsonK(const nl::json &j,const nl::json::json_pointer& ptr){
        std::mt19937 gen;gen=std::mt19937(std::random_device()());
        if(j.contains(ptr)){
            return getValueFromJsonR(j.at(ptr),gen);
        }else{
            throw std::runtime_error("getValueFromJsonK() failed. The given key '"+ptr.to_string()+"' was not found in j="+j.dump());
        }
    }
    nl::json getValueFromJsonK(const nl::json &j,const std::string& key){
        std::mt19937 gen;gen=std::mt19937(std::random_device()());
        if(j.contains(key)){
            return getValueFromJsonR(j.at(key),gen);
        }else{
            if(key.empty() || key.at(0)=='/'){
                return getValueFromJsonKR(j,nl::json::json_pointer(key),gen);
            }else{
                throw py::key_error("getValueFromJsonK() failed. The given key '"+key+"' was not found in j="+j.dump());
            }
        }
    }

    nl::json merge_patch(const nl::json& base,const nl::json& patch){
        nl::json ret=base;
        ret.merge_patch(patch);
        return ret;
    }
    
void exportUtility(py::module &m){
    m.def("getValueFromJson",[](const nl::json& j)->py::object {
        return util::getValueFromJson(j);
    });
    m.def("getValueFromJsonKD",[](const nl::json&j,const std::string& key,const py::object& defaultValue)->py::object {
        return util::getValueFromJsonKD<nl::json,false>(j,key,defaultValue);
    });
    m.def("getValueFromJsonKRD",[](const nl::json&j,const std::string& key,std::mt19937& gen,const py::object& defaultValue)->py::object {
        return util::getValueFromJsonKRD<nl::json,std::mt19937,false>(j,key,gen,defaultValue);
    });
    m.def("getValueFromJsonK",[](const nl::json&j,const std::string& key)->py::object {
        return util::getValueFromJsonK(j,key);
    });
    m.def("getValueFromJsonKR",[](const nl::json&j,const std::string& key,std::mt19937& gen)->py::object {
        return util::getValueFromJsonKR<std::mt19937>(j,key,gen);
    });
    m.def("merge_patch",&util::merge_patch);
}

ASRC_NAMESPACE_END(util)
ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
