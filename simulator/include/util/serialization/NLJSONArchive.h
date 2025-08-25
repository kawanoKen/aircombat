// Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
//
// nl::ordered_jsonをストレージとするcerealのOutput/InputArchive
// rapidjsonを用いるcereal標準のArchiveをベースとしているが、シリアライズ後のjson値に互換性はない。
//
#pragma once
#include <nlohmann/json.hpp>
#include <cereal/cereal.hpp>
#include <cereal/external/base64.hpp>
#include "../../util/macros/common_macros.h"
#include "../../traits/is_stl_container.h"
#include "../../traits/pointer_like.h"
#include "../../traits/is_string.h"
#include "../../traits/cereal.h"
#include "../../traits/json.h"
namespace nl=nlohmann;

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)
ASRC_INLINE_NAMESPACE_BEGIN(util)

class NLJSONOutputArchive : public cereal::OutputArchive<NLJSONOutputArchive>, public cereal::traits::TextArchive{
    // nl::ordered_json variant for cereal::JSONOutputArchive
    enum class NodeType { StartObject, InObject, StartArray, InArray };
    public:
    NLJSONOutputArchive(nl::ordered_json& j_, bool discardRootNameValuePair_=false) :
        cereal::OutputArchive<NLJSONOutputArchive>(this),
        discardRootNameValuePair(discardRootNameValuePair_),
        rootRef(j_),
        itsNextName(nullptr)
    {
        rootRef=nl::ordered_json();
        itsNodeRefStack.push(&rootRef);
        itsNameCounter.push(0);
        if(discardRootNameValuePair){
            itsNodeStack.push(NodeType::InObject);
        }else{
            itsNodeStack.push(NodeType::StartObject);
        }
    }
    ~NLJSONOutputArchive()=default;

    void saveBinaryValue( const void * data, size_t size, const char * name = nullptr ){
        setNextName( name );
        writeName();

        auto base64string = cereal::base64::encode( reinterpret_cast<const unsigned char *>( data ), size );
        saveValue( base64string );
    };

    void startNode(){
        writeName();
        itsNodeStack.push(NodeType::StartObject);
        itsNameCounter.push(0);
    }
    void finishNode(){
        nl::ordered_json* currentNode = itsNodeRefStack.top();
        if(discardRootNameValuePair && itsNodeStack.size()==1){
            return;
        }
        switch(itsNodeStack.top()){
            case NodeType::StartArray:
                (*currentNode)=nl::ordered_json::array();
                // fall through
            case NodeType::InArray:
                itsNodeRefStack.pop();
                break;
            case NodeType::StartObject:
                (*currentNode)=nl::ordered_json::object();
                // fall through
            case NodeType::InObject:
                itsNodeRefStack.pop();
                break;
        }
        itsNodeStack.pop();
        itsNameCounter.pop();
    }
    void setNextName( const char * name ){
        itsNextName = name;
    }
    template<typename T>
    void saveValue(const T& t){
        nl::ordered_json* currentNode = itsNodeRefStack.top();
        (*currentNode)=t;
        if(discardRootNameValuePair && itsNodeRefStack.size()==1){
            return;
        }
        itsNodeRefStack.pop();
    }
    void writeName(){
        NodeType const & nodeType = itsNodeStack.top();
        nl::ordered_json* currentNode = itsNodeRefStack.top();
        if(nodeType == NodeType::StartArray){
            (*currentNode)=nl::ordered_json::array();
            itsNodeStack.top() = NodeType::InArray;
        }else if(nodeType == NodeType::StartObject){
            (*currentNode)=nl::ordered_json::object();
            itsNodeStack.top() = NodeType::InObject;
        }
        if(discardRootNameValuePair && currentNode->is_null()){
            return;
        }
        if(currentNode->is_object()){
            if(itsNextName == nullptr){
                std::string name = "value" + std::to_string( itsNameCounter.top()++ );
                (*currentNode)[name]=nl::ordered_json();
                itsNodeRefStack.push(&currentNode->at(name));
            }else{
                (*currentNode)[itsNextName]=nl::ordered_json();
                itsNodeRefStack.push(&currentNode->at(itsNextName));
                itsNextName = nullptr;
            }
        }else if(currentNode->is_array()){
            currentNode->push_back(nl::ordered_json());
            itsNodeRefStack.push(&currentNode->at(currentNode->size()-1));
        }else{
            throw cereal::Exception("Current node is neither array nor object");
        }
    }
    void makeArray(){
        itsNodeStack.top() = NodeType::StartArray;
    }
    const nl::ordered_json& getCurrentNode(){
        // for debug purpose only
        return *itsNodeRefStack.top();
    }
    private:
    bool discardRootNameValuePair;
    nl::ordered_json& rootRef; //root node reference

    const char* itsNextName;
    std::stack<std::uint32_t> itsNameCounter;
    std::stack<nl::ordered_json*> itsNodeRefStack;
    std::stack<NodeType> itsNodeStack;
};
class NLJSONInputArchive : public cereal::InputArchive<NLJSONInputArchive>, public cereal::traits::TextArchive{
    // nl::ordered_json variant for cereal::JSONInputArchive
    public:
    NLJSONInputArchive(const nl::ordered_json& j, bool discardRootNameValuePair_=false) :
        cereal::InputArchive<NLJSONInputArchive>(this),
        discardRootNameValuePair(discardRootNameValuePair_),
        rootRef(j),
        itsNextName(nullptr)
    {
        itsNameCounter.push(0);
        itsIndexCounter.push(0);
        itsNodeRefStack.push(&rootRef);
    }
    ~NLJSONInputArchive()=default;

    void loadBinaryValue( void * data, size_t size, const char * name = nullptr ){
        itsNextName = name;

        std::string encoded;
        loadValue( encoded );
        auto decoded = cereal::base64::decode( encoded );

        if( size != decoded.size() )
          throw cereal::Exception("Decoded binary data size does not match specified size");

        std::memcpy( data, decoded.data(), decoded.size() );
        itsNextName = nullptr;
    };

    void startNode(){
        const nl::ordered_json* currentNode = itsNodeRefStack.top();
        if(discardRootNameValuePair && itsNodeRefStack.size()==1){
            itsNodeRefStack.push(currentNode);
            itsNextName = nullptr;
            itsNameCounter.push(0);
            itsIndexCounter.push(0);
            return;
        }
        if(currentNode->is_object()){
            if(itsNextName == nullptr){
                std::string name = "value" + std::to_string( itsNameCounter.top()++ );
                itsNodeRefStack.push(&currentNode->at(name));
            }else{
                itsNodeRefStack.push(&currentNode->at(itsNextName));
                itsNextName = nullptr;
            }
        }else{
            itsNodeRefStack.push(&currentNode->at(itsIndexCounter.top()++));
        }
        itsNameCounter.push(0);
        itsIndexCounter.push(0);
    }

    void finishNode(){
        itsNodeRefStack.pop();
        itsNameCounter.pop();
        itsIndexCounter.pop();
    }

    void setNextName( const char * name ){
        itsNextName = name;
    }

    template<typename T>
    void loadValue(T& t){
        const nl::ordered_json* currentNode = itsNodeRefStack.top();
        if(discardRootNameValuePair && itsNodeRefStack.size()==1){
            currentNode->get_to(t);
        }else{
            if(currentNode->is_object()){
                if(itsNextName == nullptr){
                    std::string name = "value" + std::to_string( itsNameCounter.top()++ );
                    currentNode->at(name).get_to(t);
                }else{
                    currentNode->at(itsNextName).get_to(t);
                    itsNextName = nullptr;
                }
            }else if(currentNode->is_array()){
                currentNode->at(itsIndexCounter.top()++).get_to(t);
            }else{
                currentNode->get_to(t);
            }
        }
    }
    const nl::ordered_json& getCurrentNode(){
        // 今いる階層のノードを返す。
        return *itsNodeRefStack.top();
    }
    const nl::ordered_json& getNextNode(){
        // 次に読み込まれるノードを返す。 (for internal use)
        const nl::ordered_json* currentNode = itsNodeRefStack.top();
        if(discardRootNameValuePair && itsNodeRefStack.size()==1){
            return *itsNodeRefStack.top();
        }else{
            if(currentNode->is_object()){
                if(itsNextName == nullptr){
                    std::string name = "value" + std::to_string( itsNameCounter.top()++ );
                    return currentNode->at(name);
                }else{
                    return currentNode->at(itsNextName);
                }
            }else if(currentNode->is_array()){
                return currentNode->at(itsIndexCounter.top());
            }else{
                return *currentNode;
            }
        }
    }
    bool contains(const typename nl::ordered_json::object_t::key_type& key){
        const nl::ordered_json* currentNode = itsNodeRefStack.top();
        if(currentNode->is_object()){
            return currentNode->contains(key);
        }else{
            return false;
        }
    }

    //! Loads the size for a SizeTag
    void loadSize(cereal::size_type & size){
        const nl::ordered_json* currentNode = itsNodeRefStack.top();
        if(currentNode->is_array()){
            size = currentNode->size();
        }else{
            throw std::runtime_error("cereal::SizeTag can only be loaded from a json array.");
        }
    }

    void checkNull(){
        // explicitなnullptr_tは値がnullかどうかをチェックする
        const nl::ordered_json* currentNode = itsNodeRefStack.top();
        const nl::ordered_json* nodeToBeChecked;
        if(discardRootNameValuePair && itsNodeRefStack.size()==1){
            nodeToBeChecked = currentNode;
        }else{
            if(currentNode->is_object()){
                if(itsNextName == nullptr){
                    std::string name = "value" + std::to_string( itsNameCounter.top()++ );
                    nodeToBeChecked = &(currentNode->at(name));
                }else{
                    nodeToBeChecked = &(currentNode->at(itsNextName));
                    itsNextName = nullptr;
                }
            }else if(currentNode->is_array()){
                nodeToBeChecked = &(currentNode->at(itsIndexCounter.top()++));
            }else{
                nodeToBeChecked = currentNode;
            }
        }
        if(!nodeToBeChecked->is_null()){
            throw std::runtime_error("non-null value encountered at explicit std::nullptr_t loading.");
        }
    }

    private:
    bool discardRootNameValuePair;
    const nl::ordered_json& rootRef; //root node reference

    const char * itsNextName;               //!< Next name set by NVP
    std::stack<std::uint32_t> itsNameCounter;
    std::stack<std::uint32_t> itsIndexCounter;
    std::stack<const nl::ordered_json*> itsNodeRefStack;
};

// ######################################################################
// NLJSONArchive prologue and epilogue functions
// ######################################################################

// ######################################################################
//! Prologue for NVPs for JSON archives
/*! NVPs do not start or finish nodes - they just set up the names */
template <class T>
inline void prologue( NLJSONOutputArchive &, cereal::NameValuePair<T> const & )
{ }

//! Prologue for NVPs for JSON archives
template <class T>
inline void prologue( NLJSONInputArchive &, cereal::NameValuePair<T> const & )
{ }

// ######################################################################
//! Epilogue for NVPs for JSON archives
/*! NVPs do not start or finish nodes - they just set up the names */
template <class T>
inline void epilogue( NLJSONOutputArchive &, cereal::NameValuePair<T> const & )
{ }

//! Epilogue for NVPs for JSON archives
/*! NVPs do not start or finish nodes - they just set up the names */
template <class T>
inline void epilogue( NLJSONInputArchive &, cereal::NameValuePair<T> const & )
{ }

// ######################################################################
//! Prologue for deferred data for JSON archives
/*! Do nothing for the defer wrapper */
template <class T>
inline void prologue( NLJSONOutputArchive &, cereal::DeferredData<T> const & )
{ }

//! Prologue for deferred data for JSON archives
template <class T>
inline void prologue( NLJSONInputArchive &, cereal::DeferredData<T> const & )
{ }

// ######################################################################
//! Epilogue for deferred for JSON archives
/*! NVPs do not start or finish nodes - they just set up the names */
template <class T>
inline void epilogue( NLJSONOutputArchive &, cereal::DeferredData<T> const & )
{ }

//! Epilogue for deferred for JSON archives
/*! Do nothing for the defer wrapper */
template <class T>
inline void epilogue( NLJSONInputArchive &, cereal::DeferredData<T> const & )
{ }

// ######################################################################
//! Prologue for SizeTags for JSON archives
/*! SizeTags are strictly ignored for JSON, they just indicate
    that the current node should be made into an array */
template <class T>
inline void prologue( NLJSONOutputArchive & ar, cereal::SizeTag<T> const & )
{
    ar.makeArray();
}

//! Prologue for SizeTags for JSON archives
template <class T>
inline void prologue( NLJSONInputArchive &, cereal::SizeTag<T> const & )
{ }

// ######################################################################
//! Epilogue for SizeTags for JSON archives
/*! SizeTags are strictly ignored for JSON */
template <class T>
inline void epilogue( NLJSONOutputArchive &, cereal::SizeTag<T> const & )
{ }

//! Epilogue for SizeTags for JSON archives
template <class T>
inline void epilogue( NLJSONInputArchive &, cereal::SizeTag<T> const & )
{ }

// ######################################################################
//! Prologue for all other types for JSON archives (except minimal types)
/*! Starts a new node, named either automatically or by some NVP,
    that may be given data by the type about to be archived

    Minimal types do not start or finish nodes */
template<class T, std::enable_if_t<std::conjunction_v<
    std::negation<std::is_arithmetic<T>>,
    std::negation<traits::is_string_like<T>>,
    std::negation<traits::is_basic_json<T>>,
    std::negation<cereal::traits::has_minimal_base_class_serialization<T, cereal::traits::has_minimal_output_serialization, NLJSONOutputArchive>>,
    std::negation<cereal::traits::has_minimal_output_serialization<T, NLJSONOutputArchive>>
>,std::nullptr_t> = nullptr>
inline void prologue( NLJSONOutputArchive & ar, const T& )
{
    ar.startNode();
}

//! Prologue for all other types for JSON archives
template<class T, std::enable_if_t<std::conjunction_v<
    std::negation<std::is_arithmetic<T>>,
    std::negation<traits::is_string_like<T>>,
    std::negation<traits::is_basic_json<T>>,
    std::negation<cereal::traits::has_minimal_base_class_serialization<T, cereal::traits::has_minimal_input_serialization, NLJSONInputArchive>>,
    std::negation<cereal::traits::has_minimal_input_serialization<T, NLJSONInputArchive>>
>,std::nullptr_t> = nullptr>
inline void prologue( NLJSONInputArchive & ar, const T& )
{
    ar.startNode();
}

// ######################################################################
//! Epilogue for all other types other for JSON archives (except minimal types)
/*! Finishes the node created in the prologue

    Minimal types do not start or finish nodes */
template<class T, std::enable_if_t<std::conjunction_v<
    std::negation<std::is_arithmetic<T>>,
    std::negation<traits::is_string_like<T>>,
    std::negation<traits::is_basic_json<T>>,
    std::negation<cereal::traits::has_minimal_base_class_serialization<T, cereal::traits::has_minimal_output_serialization, NLJSONOutputArchive>>,
    std::negation<cereal::traits::has_minimal_output_serialization<T, NLJSONOutputArchive>>
>,std::nullptr_t> = nullptr>
inline void epilogue( NLJSONOutputArchive & ar, const T& )
{
    ar.finishNode();
}

//! Epilogue for all other types other for JSON archives
template<class T, std::enable_if_t<std::conjunction_v<
    std::negation<std::is_arithmetic<T>>,
    std::negation<traits::is_string_like<T>>,
    std::negation<traits::is_basic_json<T>>,
    std::negation<cereal::traits::has_minimal_base_class_serialization<T, cereal::traits::has_minimal_input_serialization, NLJSONInputArchive>>,
    std::negation<cereal::traits::has_minimal_input_serialization<T, NLJSONInputArchive>>
>,std::nullptr_t> = nullptr>
inline void epilogue( NLJSONInputArchive & ar, const T& )
{
    ar.finishNode();
}

// ######################################################################
//! Prologue for nullptr for JSON archives
inline void prologue( NLJSONOutputArchive & ar, const std::nullptr_t & )
{
    ar.writeName();
}

//! Prologue for nullptr for JSON archives
inline void prologue( NLJSONInputArchive &, const std::nullptr_t & )
{ }

// ######################################################################
//! Epilogue for nullptr for JSON archives
inline void epilogue( NLJSONOutputArchive &, const std::nullptr_t & )
{ }

//! Epilogue for nullptr for JSON archives
inline void epilogue( NLJSONInputArchive &, const std::nullptr_t & )
{ }

// ######################################################################
//! Prologue for arithmetic types for JSON archives
template <class T, cereal::traits::EnableIf<std::is_arithmetic<T>::value> = cereal::traits::sfinae>
inline void prologue( NLJSONOutputArchive & ar, const T& )
{
    ar.writeName();
}

//! Prologue for arithmetic types for JSON archives
template <class T, cereal::traits::EnableIf<std::is_arithmetic<T>::value> = cereal::traits::sfinae>
inline void prologue( NLJSONInputArchive &, const T& )
{ }

// ######################################################################
//! Epilogue for arithmetic types for JSON archives
template <class T, cereal::traits::EnableIf<std::is_arithmetic<T>::value> = cereal::traits::sfinae>
inline void epilogue( NLJSONOutputArchive &, const T& )
{ }

//! Epilogue for arithmetic types for JSON archives
template <class T, cereal::traits::EnableIf<std::is_arithmetic<T>::value> = cereal::traits::sfinae>
inline void epilogue( NLJSONInputArchive &, const T& )
{ }

// ######################################################################
//! Prologue for strings for JSON archives
template <traits::std_basic_string T>
inline void prologue(NLJSONOutputArchive & ar, const T&)
{
    ar.writeName();
}

//! Prologue for strings for JSON archives
template <traits::std_basic_string T>
inline void prologue(NLJSONInputArchive &, const T&)
{ }

//! Prologue for string compatible char arrays for JSON archives
template <traits::string_literal T>
inline void prologue(NLJSONOutputArchive & ar, const T&)
{
    ar.writeName();
}

//! Prologue for string compatible char arrays for JSON archives
template <traits::string_literal T>
inline void prologue(NLJSONInputArchive &, const T&)
{ }

// ######################################################################
//! Epilogue for strings for JSON archives
template <traits::std_basic_string T>
inline void epilogue(NLJSONOutputArchive &, const T&)
{ }

//! Epilogue for strings for JSON archives
template <traits::std_basic_string T>
inline void epilogue(NLJSONInputArchive &, const T&)
{ }

//! Epilogue for string compatible char arrays for JSON archives
template <traits::string_literal T>
inline void epilogue(NLJSONOutputArchive &, const T&)
{ }

//! Epilogue for string compatible char arrays for JSON archives
template <traits::string_literal T>
inline void epilogue(NLJSONInputArchive &, const T&)
{ }

// ######################################################################
//! Prologue for nl::basic_json for JSON archives
template <traits::basic_json T>
inline void prologue( NLJSONOutputArchive & ar, const T& )
{
    ar.writeName();
}

//! Prologue for nl::basic_json for JSON archives
template <traits::basic_json T>
inline void prologue( NLJSONInputArchive &, const T& )
{ }

// ######################################################################
//! Epilogue for nl::basic_json for JSON archives
template <traits::basic_json T>
inline void epilogue( NLJSONOutputArchive &, const T& )
{ }

//! Epilogue for nl::basic_json for JSON archives
template <traits::basic_json T>
inline void epilogue( NLJSONInputArchive &, const T& )
{ }

// ######################################################################
// Common NLJSONArchive serialization functions
// ######################################################################
//! Serializing NVP types to JSON
template <class T>
inline void CEREAL_SAVE_FUNCTION_NAME( NLJSONOutputArchive & ar, cereal::NameValuePair<T> const & t )
{
    ar.setNextName( t.name );
    ar( t.value );
}

template <class T>
inline void CEREAL_LOAD_FUNCTION_NAME( NLJSONInputArchive & ar, cereal::NameValuePair<T> & t )
{
    ar.setNextName( t.name );
    ar( t.value );
}

//! Saving for nullptr to JSON
inline void CEREAL_SAVE_FUNCTION_NAME(NLJSONOutputArchive & ar, const std::nullptr_t & t)
{
    ar.saveValue( t );
}

//! Loading arithmetic from JSON
inline void CEREAL_LOAD_FUNCTION_NAME(NLJSONInputArchive & ar, std::nullptr_t & t)
{
    ar.checkNull();
}

//! Saving for arithmetic to JSON
template <class T, cereal::traits::EnableIf<std::is_arithmetic<T>::value> = cereal::traits::sfinae>
inline void CEREAL_SAVE_FUNCTION_NAME(NLJSONOutputArchive & ar, const T& t)
{
    ar.saveValue( t );
}

//! Loading arithmetic from JSON
template <class T, cereal::traits::EnableIf<std::is_arithmetic<T>::value> = cereal::traits::sfinae>
inline void CEREAL_LOAD_FUNCTION_NAME(NLJSONInputArchive & ar, T& t)
{
    ar.loadValue( t );
}

//! saving string to JSON
template <traits::std_basic_string T>
inline void CEREAL_SAVE_FUNCTION_NAME(NLJSONOutputArchive & ar, const T& t)
{
    ar.saveValue( t );
}

//! loading string from JSON
template <traits::std_basic_string T>
inline void CEREAL_LOAD_FUNCTION_NAME(NLJSONInputArchive & ar, T& t)
{
    ar.loadValue( t );
}

// serializing string compatible char pointers
// These functions disable C style array serialization in "cereal/types/commons.h".
template <traits::string_literal T>
inline void CEREAL_SAVE_FUNCTION_NAME(NLJSONOutputArchive & ar, const T& t)
{
    // 文字列リテラルはここに落ちる。nl::ordered_jsonを初期化子リストで初期化するときに文字列定数をリテラルで直接書いた場合等が該当する。
    using CharT = traits::get_true_value_type_t<std::decay_t<T>>;
    CEREAL_SAVE_FUNCTION_NAME(ar, std::basic_string(t));
}
template <traits::string_literal T>
inline void CEREAL_LOAD_FUNCTION_NAME(NLJSONInputArchive & ar, T& t)
{
    // 非constのchar[N]型変数に読み込みたいという需要はあまり無いと思われるが、
    // 一応用意しておく。
    using CharT = traits::get_true_value_type_t<std::decay_t<T>>;
    std::basic_string<CharT> str;
    CEREAL_LOAD_FUNCTION_NAME(ar, str);
    int i=0;
    for(auto& v: t){
        v=str[i];
        ++i;
    }
}

//! Saving for nl::basic_json to JSON
template <traits::basic_json T>
inline void CEREAL_SAVE_FUNCTION_NAME(NLJSONOutputArchive & ar, const T& t)
{
    ar.saveValue( t );
}

//! Loading nl::basic_json from JSON
template <traits::basic_json T>
inline void CEREAL_LOAD_FUNCTION_NAME(NLJSONInputArchive & ar, T& t)
{
    ar.loadValue( t );
}

// ######################################################################
//! Saving SizeTags to JSON
template <class T>
inline void CEREAL_SAVE_FUNCTION_NAME( NLJSONOutputArchive &, cereal::SizeTag<T> const & )
{
// nothing to do here, we don't explicitly save the size
}

//! Loading SizeTags from JSON
template <class T>
inline void CEREAL_LOAD_FUNCTION_NAME( NLJSONInputArchive & ar, cereal::SizeTag<T> & st )
{
    ar.loadSize( st.size );
}

ASRC_NAMESPACE_END(util)
ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)

// register archives for polymorphic support
CEREAL_REGISTER_ARCHIVE(asrc::core::util::NLJSONInputArchive)
CEREAL_REGISTER_ARCHIVE(asrc::core::util::NLJSONOutputArchive)

// tie input and output archives together
CEREAL_SETUP_ARCHIVE_TRAITS(asrc::core::util::NLJSONInputArchive, asrc::core::util::NLJSONOutputArchive)

namespace cereal{

// STLコンテナのうち、array、pair、tupleはcereal標準の関数だとobjectになるがnlohmann側の挙動に合わせてarrayで格納する。
template <class T, size_t N>
inline void CEREAL_SAVE_FUNCTION_NAME(::asrc::core::util::NLJSONOutputArchive & ar, const std::array<T, N>& array){
    cereal::size_type size=array.size();
    ar( cereal::make_size_tag( size ) );
    for( auto const & i : array )
      ar( i );
}
template <class T, size_t N>
inline void CEREAL_LOAD_FUNCTION_NAME(::asrc::core::util::NLJSONInputArchive & ar, std::array<T, N>& array){
    cereal::size_type size;
    ar( cereal::make_size_tag( size ) );
    for( auto & i : array )
      ar( i );
}
template <class T1, class T2>
inline void CEREAL_SERIALIZE_FUNCTION_NAME(::asrc::core::util::NLJSONOutputArchive & ar, std::pair<T1, T2> & pair){
    cereal::size_type size=2;
    ar(
        cereal::make_size_tag( size ),
        pair.first,
        pair.second
    );
}
template <class T1, class T2>
inline void CEREAL_SERIALIZE_FUNCTION_NAME(::asrc::core::util::NLJSONInputArchive & ar, std::pair<T1, T2> & pair){
    cereal::size_type size=2;
    ar(
        cereal::make_size_tag( size ),
        pair.first,
        pair.second
    );
}
template <class ... Types>
inline void CEREAL_SERIALIZE_FUNCTION_NAME(::asrc::core::util::NLJSONOutputArchive & ar, std::tuple<Types...>& tuple){
    cereal::size_type size=std::tuple_size_v<std::tuple<Types...>>;
    ar( cereal::make_size_tag( size ) );
    std::apply(
        [&ar](auto&& ... args){
            ar(std::forward<decltype(args)>(args)...);
        },
        tuple
    );
}
template <class ... Types>
inline void CEREAL_SERIALIZE_FUNCTION_NAME(::asrc::core::util::NLJSONInputArchive & ar, std::tuple<Types...>& tuple){
    cereal::size_type size=std::tuple_size_v<std::tuple<Types...>>;
    ar( cereal::make_size_tag( size ) );
    std::apply(
        [&ar](auto&& ... args){
            ar(std::forward<decltype(args)>(args)...);
        },
        tuple
    );
}

// std::map系はキーが全て文字列の場合はobject、それ以外は2要素arrayとしてcerealの標準の方法で入出力
template <template <typename...> class Map, typename... Args, typename = typename Map<Args...>::mapped_type>
inline void CEREAL_SAVE_FUNCTION_NAME( ::asrc::core::util::NLJSONOutputArchive & ar, Map<Args...> const & map )
{
    if constexpr (::asrc::core::traits::string_like<typename Map<Args...>::key_type>){
        for( const auto & [k,v] : map ){
            std::string key_str(k);
            ar(cereal::make_nvp<::asrc::core::util::NLJSONOutputArchive>(key_str.c_str(),v));
        }
    }else{
        ar( cereal::make_size_tag( static_cast<cereal::size_type>(map.size()) ) );

        for( const auto & i : map ){
            ar( cereal::make_map_item(i.first, i.second) );
        }
    }
}
template <template <typename...> class Map, typename... Args, typename = typename Map<Args...>::mapped_type>
inline void CEREAL_LOAD_FUNCTION_NAME( ::asrc::core::util::NLJSONInputArchive & ar, Map<Args...> & map )
{
    if constexpr (::asrc::core::traits::string_like<typename Map<Args...>::key_type>){
        const auto& j=ar.getCurrentNode();
        cereal::size_type size=j.size();

        map.clear();

        auto hint = map.begin();

        for(auto && [k, _] : j.items()){
            typename Map<Args...>::key_type key=k;
            typename Map<Args...>::mapped_type value;

            ar(cereal::make_nvp<::asrc::core::util::NLJSONInputArchive>(k.c_str(),value));
            #ifdef CEREAL_OLDER_GCC
            hint = map.insert( hint, std::make_pair(std::move(key), std::move(value)) );
            #else // NOT CEREAL_OLDER_GCC
            hint = map.emplace_hint( hint, std::move( key ), std::move( value ) );
            #endif // NOT CEREAL_OLDER_GCC
        }
    }else{
        cereal::size_type size;
        ar( cereal::make_size_tag( size ) );

        map.clear();

        auto hint = map.begin();
        for( size_t i = 0; i < size; ++i )
        {
            typename Map<Args...>::key_type key;
            typename Map<Args...>::mapped_type value;

            ar( cereal::make_map_item(key, value) );
            #ifdef CEREAL_OLDER_GCC
            hint = map.insert( hint, std::make_pair(std::move(key), std::move(value)) );
            #else // NOT CEREAL_OLDER_GCC
            hint = map.emplace_hint( hint, std::move( key ), std::move( value ) );
            #endif // NOT CEREAL_OLDER_GCC
        }
    }
}

// nl::ordered_mapはemplace_hintが使えないので個別に特殊化
template <class Key, class T, class IgnoredLess, class Allocator>
inline void CEREAL_LOAD_FUNCTION_NAME( ::asrc::core::util::NLJSONInputArchive & ar, nl::ordered_map<Key,T,IgnoredLess,Allocator> & map )
{
    if constexpr (::asrc::core::traits::string_like<Key>){
        const auto& j=ar.getCurrentNode();
        cereal::size_type size=j.size();

        map.clear();

        for(auto && [k, _] : j.items()){
            Key key=k;
            T value;

            ar(cereal::make_nvp<::asrc::core::util::NLJSONInputArchive>(k.c_str(),value));
            map.emplace_back( std::move( key ), std::move( value ) );
        }
    }else{
        cereal::size_type size;
        ar( cereal::make_size_tag( size ) );

        map.clear();

        for( size_t i = 0; i < size; ++i ){
            Key key;
            T value;

            ar( cereal::make_map_item(key, value) );
            map.emplace_back( std::move( key ), std::move( value ) );
        }
    }
}

}
