// Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#include "JsonMutableFromPython.h"
#include "Utility.h"
#include "Units.h"

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)
ASRC_INLINE_NAMESPACE_BEGIN(util)

template<traits::basic_json BasicJsonType>
void exportJsonMutableFromPythonSub(py::module &m,const std::string& prefix, const std::string& name)
{
    using namespace pybind11::literals;
    using BJI=BasicJsonIterator<BasicJsonType>;
    using BJRI=BasicJsonReverseIterator<BasicJsonType>;
    using BJKI=BasicJsonKeyIterable<BasicJsonType>;
    using BJVI=BasicJsonValueIterable<BasicJsonType>;
    using BJII=BasicJsonItemIterable<BasicJsonType>;
    using BJCKI=BasicJsonConstKeyIterable<BasicJsonType>;
    using BJCVI=BasicJsonConstValueIterable<BasicJsonType>;
    using BJCII=BasicJsonConstItemIterable<BasicJsonType>;

    py::class_<BJI>(m,(prefix+"Iterator").c_str())
    .def(py::self == py::self)
    .def(py::self != py::self)
    .def(py::self < py::self)
    .def(py::self > py::self)
    .def(py::self <= py::self)
    .def(py::self >= py::self)
    .def(py::self += typename BJI::difference_type())
    .def(py::self -= typename BJI::difference_type())
    .def(py::self + typename BJI::difference_type())
    .def(typename BJI::difference_type() + py::self)
    .def(py::self - typename BJI::difference_type())
    .def(py::self - py::self)
    .def("__getitem__",&BJI::operator[])
    DEF_FUNC(BJI,key)
    DEF_FUNC(BJI,value)
    ;
    py::class_<BJRI>(m,(prefix+"ReverseIterator").c_str())
    .def(py::self == py::self)
    .def(py::self != py::self)
    .def(py::self < py::self)
    .def(py::self > py::self)
    .def(py::self <= py::self)
    .def(py::self >= py::self)
    .def(py::self += typename BJRI::difference_type())
    .def(py::self -= typename BJRI::difference_type())
    .def(py::self + typename BJRI::difference_type())
    .def(typename BJRI::difference_type() + py::self)
    .def(py::self - typename BJRI::difference_type())
    .def(py::self - py::self)
    .def("__getitem__",&BJRI::operator[])
    DEF_FUNC(BJRI,key)
    DEF_FUNC(BJRI,value)
    ;
    py::class_<BJKI>(m,(prefix+"KeyIterable").c_str())
    .def("__iter__",[](BJKI& it)-> BJKI& {return it;})
    .def("__next__",&BJKI::next,py::return_value_policy::reference_internal)
    ;
    py::class_<BJVI>(m,(prefix+"ValueIterable").c_str())
    .def("__iter__",[](BJVI& it)-> BJVI& {return it;})
    .def("__next__",&BJVI::next,py::return_value_policy::reference_internal)
    ;
    py::class_<BJII>(m,(prefix+"ItemIterable").c_str())
    .def("__iter__",[](BJII& it)-> BJII& {return it;})
    .def("__next__",&BJII::next,py::return_value_policy::reference_internal)
    ;
    py::class_<BJCKI>(m,(prefix+"ConstKeyIterable").c_str())
    .def("__iter__",[](BJCKI& it)-> BJCKI& {return it;})
    .def("__next__",&BJCKI::next,py::return_value_policy::reference_internal)
    ;
    py::class_<BJCVI>(m,(prefix+"ConstValueIterable").c_str())
    .def("__iter__",[](BJCVI& it)-> BJCVI& {return it;})
    .def("__next__",&BJCVI::next,py::return_value_policy::reference_internal)
    ;
    py::class_<BJCII>(m,(prefix+"ConstItemIterable").c_str())
    .def("__iter__",[](BJCII& it)-> BJCII& {return it;})
    .def("__next__",&BJCII::next,py::return_value_policy::reference_internal)
    ;
    py::class_<BasicJsonType>(m,name.c_str())
    .def(py::init<py::object>())
    .def("dump",[](BasicJsonType& j,
                    const int indent = -1,
                    const char indent_char = ' ',
                    const bool ensure_ascii = false){
        return j.dump(indent,indent_char,ensure_ascii);
    })
    .def("type",[](const BasicJsonType& j){return j.type();})
    .def("type_name",[](const BasicJsonType& j){return j.type_name();})
    .def("is_primitive",[](const BasicJsonType& j){return j.is_primitive();})
    .def("is_structured",[](const BasicJsonType& j){return j.is_structured();})
    .def("is_null",[](const BasicJsonType& j){return j.is_null();})
    .def("is_boolean",[](const BasicJsonType& j){return j.is_boolean();})
    .def("is_number",[](const BasicJsonType& j){return j.is_number();})
    .def("is_number_integer",[](const BasicJsonType& j){return j.is_number_integer();})
    .def("is_number_unsigned",[](const BasicJsonType& j){return j.is_number_unsigned();})
    .def("is_number_float",[](const BasicJsonType& j){return j.is_number_float();})
    .def("is_object",[](const BasicJsonType& j){return j.is_object();})
    .def("is_array",[](const BasicJsonType& j){return j.is_array();})
    .def("is_string",[](const BasicJsonType& j){return j.is_string();})
    .def("is_binary",[](const BasicJsonType& j){return j.is_binary();})
    .def("is_discarded",[](const BasicJsonType& j){return j.is_discarded();})
    .def("get",[](const BasicJsonType& j)->py::object {return j.template get<py::object>();})
    .def("get",[](const BasicJsonType& j,const typename BasicJsonType::object_t::key_type& key,const py::object& defaultValue)->py::object {
        // BasicJsonTypeの本来のgetは__call__にも割り当てているのであまり使用されない。
        // object(dict)の場合にPython dictと同じgetメソッドを使用できるようにしておく。
        if(!j.is_object()){
            throw py::type_error("nljson.get(str, Any) cannot be called when it represents a non-object json.");
        }
        if(j.contains(key)){
            return j.at(key);
        }else if((key.empty() || key.at(0)=='/') && j.contains(typename BasicJsonType::json_pointer(key))){
            return j.at(typename BasicJsonType::json_pointer(key));
        }else{
            return defaultValue;
        }
    },"key"_a,"defaultValue"_a=py::none())
    .def("__call__",[](const BasicJsonType& j)->py::object {return j.template get<py::object>();})
    .def("get_to",[](const BasicJsonType& j,py::object& v)->py::object {
     	j.get_to(v);
        return v;
    })
    .def("__bool__",[](const BasicJsonType& j)->py::bool_ {
        if(j.is_null()){
            return false;
        }else if(j.is_boolean()){
            return j.template get<bool>();
        }else if(j.is_number_integer()){
            return py::bool_(py::int_(j.template get<typename BasicJsonType::number_integer_t>()));
        }else if(j.is_number_unsigned()){
            return py::bool_(py::int_(j.template get<typename BasicJsonType::number_unsigned_t>()));
        }else if(j.is_number_float()){
            return py::bool_(py::float_(j.template get<typename BasicJsonType::number_float_t>()));
        }else if(j.is_string()){
            return py::bool_(py::str(j.template get<typename BasicJsonType::string_t>()));
        }else if(j.is_array()){
            return j.size()>0;
        }else if(j.is_object()){
            return j.size()>0;
        }
        throw std::runtime_error("Could not convert a json to bool. json="+j.dump());
    })
    .def("__int__",[](const BasicJsonType& j)->py::int_ {
        if(j.is_boolean()){
            return py::int_(py::bool_(j.template get<bool>()));
        }else if(j.is_number_integer()){
            return py::int_(j.template get<typename BasicJsonType::number_integer_t>());
        }else if(j.is_number_unsigned()){
            return py::int_(j.template get<typename BasicJsonType::number_unsigned_t>());
        }else if(j.is_number_float()){
            return py::int_(py::float_(j.template get<typename BasicJsonType::number_float_t>()));
        }else if(j.is_string()){
            return py::int_(py::str(j.template get<typename BasicJsonType::string_t>()));
        }
        throw std::runtime_error("Could not convert a json to int. json="+j.dump());
    })
    .def("__float__",[](const BasicJsonType& j){
        if(j.is_boolean()){
            return py::float_(py::bool_(j.template get<bool>()));
        }else if(j.is_number_integer()){
            return py::float_(py::int_(j.template get<typename BasicJsonType::number_integer_t>()));
        }else if(j.is_number_unsigned()){
            return py::float_(py::int_(j.template get<typename BasicJsonType::number_unsigned_t>()));
        }else if(j.is_number_float()){
            return py::float_(j.template get<typename BasicJsonType::number_float_t>());
        }else if(j.is_string()){
            return py::float_(py::str(j.template get<typename BasicJsonType::string_t>()));
        }
        throw std::runtime_error("Could not convert a json to int. json="+j.dump());
    })
    .def("__str__",[](const BasicJsonType& j)->py::str {
        return j.dump();
    })
    .def("at",[](BasicJsonType& j,typename BasicJsonType::size_type idx){
        if(!j.is_array()){
            throw py::type_error("nljson.at(int) cannot be called when it represents a non-array json.");
        }
        return j.at(idx);
    })
    .def("at",[](const BasicJsonType& j,typename BasicJsonType::size_type idx){
        if(!j.is_array()){
            throw py::type_error("nljson.at(int) cannot be called when it represents a non-array json.");
        }
        return j.at(idx);
    })
    .def("at",[](BasicJsonType& j,const typename BasicJsonType::object_t::key_type& key){
        if(!j.is_object()){
            throw py::type_error("nljson.at(str) cannot be called when it represents a non-object json.");
        }
        if(j.contains(key)){
            return j.at(key);
        }else if(key.empty() || key.at(0)=='/'){
            typename BasicJsonType::json_pointer ptr(key);
            if(j.contains(ptr)){
                return j.at(ptr);
            }
        }
        throw py::key_error(key);
    })
    .def("at",[](const BasicJsonType& j,const typename BasicJsonType::object_t::key_type& key){
        if(!j.is_object()){
            throw py::type_error("nljson.at(str) cannot be called when it represents a non-object json.");
        }
        if(j.contains(key)){
            return j.at(key);
        }else if(key.empty() || key.at(0)=='/'){
            typename BasicJsonType::json_pointer ptr(key);
            if(j.contains(ptr)){
                return j.at(ptr);
            }
        }
        throw py::key_error(key);
    })
    .def("at_p",[](BasicJsonType& j,const typename BasicJsonType::object_t::key_type& key){
        if(!j.is_object()){
            throw py::type_error("nljson.at_p(str) cannot be called when it represents a non-object json.");
        }
        if(key.empty() || key.at(0)=='/'){
            typename BasicJsonType::json_pointer ptr(key);
            if(j.contains(ptr)){
                return j.at(ptr);
            }
        }else if(j.contains(key)){
            return j.at(key);
        }
        throw py::key_error(key);
    })
    .def("at_p",[](const BasicJsonType& j,const typename BasicJsonType::object_t::key_type& key){
        if(!j.is_object()){
            throw py::type_error("nljson.at_p(str) cannot be called when it represents a non-object json.");
        }
        if(key.empty() || key.at(0)=='/'){
            typename BasicJsonType::json_pointer ptr(key);
            if(j.contains(ptr)){
                return j.at(ptr);
            }
        }else if(j.contains(key)){
            return j.at(key);
        }
        throw py::key_error(key);
    })
    .def("__getitem__",[](BasicJsonType& j,typename BasicJsonType::size_type idx) -> BasicJsonType& {
        if(!j.is_array()){
            throw py::type_error("nljson[int] cannot be called when it represents a non-array json.");
        }
        return j.at(idx);
    },py::keep_alive<0,1>(),py::return_value_policy::reference)
    .def("__getitem__",[](BasicJsonType& j,const typename BasicJsonType::object_t::key_type& key) -> BasicJsonType& {
        if(!j.is_object()){
            throw py::type_error("nljson[str] cannot be called when it represents a non-object json.");
        }
        if(j.contains(key)){
            return j.at(key);
        }else if(key.empty() || key.at(0)=='/'){
            typename BasicJsonType::json_pointer ptr(key);
            if(j.contains(ptr)){
                return j.at(ptr);
            }
        }
        throw py::key_error(key);
    },py::keep_alive<0,1>(),py::return_value_policy::reference)
    .def("__getitem__",[](const BasicJsonType& j,typename BasicJsonType::size_type idx) -> BasicJsonType {
        if(!j.is_array()){
            throw py::type_error("nljson[int] cannot be called when it represents a non-array json.");
        }
        return j.at(idx);
    })
    .def("__getitem__",[](const BasicJsonType& j,const typename BasicJsonType::object_t::key_type& key) -> BasicJsonType {
        if(!j.is_object()){
            throw py::type_error("nljson[str] cannot be called when it represents a non-object json.");
        }
        if(j.contains(key)){
            return j.at(key);
        }else if(key.empty() || key.at(0)=='/'){
            typename BasicJsonType::json_pointer ptr(key);
            if(j.contains(ptr)){
                return j.at(ptr);
            }
        }
        throw py::key_error(key);
    })
    .def("__setitem__",[](BasicJsonType& j,typename BasicJsonType::size_type idx,py::object value){
        if(!j.is_array()){
            throw py::type_error("nljson[int] cannot be called when it represents a non-array json.");
        }
        j[idx]=value;
    })
    .def("__setitem__",[](BasicJsonType& j,const typename BasicJsonType::object_t::key_type& key,py::object value){
        if(!j.is_object()){
            throw py::type_error("nljson[str] cannot be called when it represents a non-object json.");
        }
        j[key]=value;
    })
    .def("value",[](const BasicJsonType& j,const typename BasicJsonType::object_t::key_type& key,py::object default_value){
        if(!j.is_object()){
            throw py::type_error("nljson.value(str, Any) cannot be called when it represents a non-object json.");
        }
        return j.value(key,default_value);
    })
    .def("front",[](const BasicJsonType& j){
        return j.front();
    })
    .def("back",[](const BasicJsonType& j){
        return j.back();
    })
    .def("erase",[](BasicJsonType& j,BJI pos)->BJI {
        return j.erase(pos.it);
    },py::keep_alive<0,1>())
    .def("erase",[](BasicJsonType& j,BJI first,BJI last)->BJI {
        return j.erase(first.it,last.it);
    })
    .def("erase",[](BasicJsonType& j,const typename BasicJsonType::object_t::key_type& key)->std::size_t {
        return j.erase(key);
    })
    .def("erase",[](BasicJsonType& j,const typename BasicJsonType::size_type idx){
        j.erase(idx);
    })
    .def("find",[](BasicJsonType& j,const typename BasicJsonType::object_t::key_type& key)->BJI {
        if(!j.is_object()){
            throw py::type_error("nljson.find(str) cannot be called when it represents a non-object json.");
        }
        return j.find(key);
    },py::keep_alive<0,1>())
    .def("count",[](const BasicJsonType& j,const typename BasicJsonType::object_t::key_type& key){
        if(!j.is_object()){
            throw py::type_error("nljson.count(str) cannot be called when it represents a non-object json.");
        }
        return j.count(key);
    })
    .def("contains",[](const BasicJsonType& j,const typename BasicJsonType::object_t::key_type& key){
        if(!j.is_object()){
            throw py::type_error("nljson.contains(str) cannot be called when it represents a non-object json.");
        }
        return j.contains(key) || ((key.empty() || key.at(0)=='/') && j.contains(typename BasicJsonType::json_pointer(key)));
    })
    .def("contains_p",[](const BasicJsonType& j,const typename BasicJsonType::object_t::key_type& ptr){
        if(!j.is_object()){
            throw py::type_error("nljson.contains_p(str) cannot be called when it represents a non-object json.");
        }
        return ((ptr.empty() || ptr.at(0)=='/') && j.contains(typename BasicJsonType::json_pointer(ptr))) || j.contains(ptr);
    })
    .def("__contains__",[](BasicJsonType& j,const typename BasicJsonType::object_t::key_type& key)->bool {
        if(!j.is_object()){
            throw py::type_error("nljson.__contains__(str) cannot be called when it represents a non-object json.");
        }
        return j.contains(key) || ((key.empty() || key.at(0)=='/') && j.contains(typename BasicJsonType::json_pointer(key)));
    })
    .def("begin",[](BasicJsonType& j)->BJI {
        return j.begin();
    },py::keep_alive<0,1>())
    .def("end",[](BasicJsonType& j)->BJI {
        return j.end();
    },py::keep_alive<0,1>())
    .def("rbegin",[](BasicJsonType& j)->BJRI {
        return j.rbegin();
    },py::keep_alive<0,1>())
    .def("rend",[](BasicJsonType& j)->BJRI {
        return j.rend();
    },py::keep_alive<0,1>())
    .def("__iter__",[](BasicJsonType& j){
        assert(j.is_object() || j.is_array());
        if(j.is_object()){
            return py::cast(BJKI(j));
        }else{
            return py::cast(BJVI(j));
        }
    },py::keep_alive<0,1>())
    .def("__iter__",[](const BasicJsonType& j){
        assert(j.is_object() || j.is_array());
        if(j.is_object()){
            return py::cast(BJCKI(j));
        }else{
            return py::cast(BJCVI(j));
        }
    },py::keep_alive<0,1>())
    .def("keys",[](BasicJsonType& j)->BJKI {
        assert(j.is_object());
        return BJKI(j);
    },py::keep_alive<0,1>())
    .def("keys",[](const BasicJsonType& j)->BJCKI {
        assert(j.is_object());
        return BJCKI(j);
    },py::keep_alive<0,1>())
    .def("values",[](BasicJsonType& j)->BJVI {
        assert(j.is_object());
        return BJVI(j);
    },py::keep_alive<0,1>())
    .def("values",[](const BasicJsonType& j)->BJCVI {
        assert(j.is_object());
        return BJCVI(j);
    },py::keep_alive<0,1>())
    .def("items",[](BasicJsonType& j)->BJII {
        assert(j.is_object());
        return BJII(j);
    },py::keep_alive<0,1>())
    .def("items",[](const BasicJsonType& j)->BJCII {
        assert(j.is_object());
        return BJCII(j);
    },py::keep_alive<0,1>())
    .def("empty",[](const BasicJsonType& j){
        return j.empty();
    })
    .def("size",[](const BasicJsonType& j){
        return j.size();
    })
    .def("__len__",[](const BasicJsonType& j){
        return j.size();
    })
    .def("max_size",[](const BasicJsonType& j){
        return j.max_size();
    })
    .def("clear",[](BasicJsonType& j){
        return j.clear();
    })
    .def("push_back",[](BasicJsonType& j,py::object val){
        return j.push_back(val);
    })
    .def("append",[](BasicJsonType& j,py::object val){
        return j.push_back(val);
    })
    .def(py::self += py::object())
    .def("insert",[](BasicJsonType& j, BJI pos, py::object val)->BJI {
        return j.insert(pos.it,val);
    },py::keep_alive<0,1>())
    .def("insert",[](BasicJsonType& j, BJI pos, typename BasicJsonType::size_type cnt, py::object val)->BJI {
        return j.insert(pos.it,cnt,val);
    },py::keep_alive<0,1>())
    .def("insert",[](BasicJsonType& j, BJI pos, BJI first, BJI last)->BJI {
        return j.insert(pos.it,first.it,last.it);
    },py::keep_alive<0,1>())
    .def("insert",[](BasicJsonType& j, BJI first, BJI last){
        j.insert(first.it,last.it);
    })
    .def("update",[](BasicJsonType& j,py::object j_){
        j.update(j_);
    })
    .def("update",[](BasicJsonType& j,BJI first, BJI last){
        j.update(first.it,last.it);
    })
    .def("swap",[](BasicJsonType& j,BasicJsonType& other){
        j.swap(other);
    })
    .def("flatten",[](const BasicJsonType& j){
        return j.flatten();
    })
    .def("unflatten",[](const BasicJsonType& j){
        return j.unflatten();
    })
    .def("patch",[](const BasicJsonType& j,const py::object& json_patch){
        return j.patch(json_patch);
    })
    .def_static("diff",[](const py::object& source,const py::object& target,const std::string& path=""){
        return BasicJsonType::diff(source,target,path);
    })
    .def("merge_patch",[](BasicJsonType& j,const py::object& apply_patch){
        j.merge_patch(apply_patch);
    })
    .def(py::pickle(
        [](const BasicJsonType& j){
            auto b=BasicJsonType::to_cbor(j);
            return py::bytes(reinterpret_cast<const char*>(b.data()),b.size());
        },
        [](const py::bytes& b){
            char* buffer=PYBIND11_BYTES_AS_STRING(b.ptr());
            std::size_t length=py::len(b);
            return BasicJsonType::from_cbor(buffer,buffer+length);
        }
    ))
    ;
    m.def("swap",[](BasicJsonType& left,BasicJsonType& right){
        swap(left,right);
    });

    py::implicitly_convertible<py::object,BasicJsonType>();
}

void exportJsonMutableFromPython(py::module &m){
    exportJsonMutableFromPythonSub<nl::json>(m,"Json","nljson");
    exportJsonMutableFromPythonSub<nl::ordered_json>(m,"OrderedJson","nlorderedjson");
}

ASRC_NAMESPACE_END(util)
ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
