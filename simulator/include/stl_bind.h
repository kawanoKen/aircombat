/**
 * Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
 * @file
 * 
 * @brief pybind11におけるSTLコンテナのバインド
 *
 * pybind11の標準の挙動としては、STLコンテナとその要素は常にコピーされ値渡しされる。
 * これはC++クラスのメンバ変数であっても同様であり、そのままではコンテナの中身をPython側から変更することができない。
 * 
 * この対策として、pybind11にはPYBIND11_MAKE_OPAQUEマクロとpybind11::bind_vector、pybind11::bind_map関数が提供されており、
 * これらを用いると一部のSTLコンテナについて参照渡しに変更することができる。
 * ただし、それを行った場合、STLコンテナを引数に取る関数に対して変換可能な純Pythonオブジェクト(例:std::vectorにとってのlist)
 * を与えることができなくなる。
 *
 * 本ソフトウェアではこの制約を緩和してPython→C++側に参照渡しが不可能な場合には自動的にコピーによる値渡しを行えるようにした。
 * また、参照渡しに対応可能なSTLコンテナの種類を大幅に増やした。
 * その過程でpybind11のstl.hとcast.hに一部改変を加えているので、詳細はthirdParty/modification/pybind11/include/pybind11にあるそれらのファイルを参照されたい。
 *
 * @par 使用方法
 * 
 * 元のpybind11で使用していた以下のマクロ及び関数を以下のように変更すればよい。
 * - PYBIND11_MAKE_OPAQUE(Container)→ASRC_PYBIND11_MAKE_OPAQUE(Container)
 * - pybind11::bind_vector→asrc::core::bind_stl_container又はasrc::core::bind_stl_container_name
 * - pybind11::bind_map→asrc::core::bind_stl_container又はasrc::core::bind_stl_container_name
 * 
 * @note _name付きの関数はPython側に公開する際のクラス名をユーザが指定し、_nameなしの関数は自動で設定されるという差異がある。
 * 
 * 具体的には、以下のようにすればよい。
 * 
 * (1)ヘッダ側で、そのmodule中で当該コンテナが使用されるより前にASRC_PYBIND11_MAKE_OPAQUEを呼ぶ。
 * 
 * (2)ソース側で、asrc::core::bind_stl_container又はbind_stl_container_nameを呼ぶ。
 *    pybind11への登録名を自分で指定したい場合は後者を、そうでない場合は前者を用いる。
 *    これらはstl_bind.hに実装されている関数テンプレートであり、pybins11::bind_vector、pybind11::bind_mapに相当するバインド処理を、
 *    より多くのSTLコンテナについて共通のインターフェースで実行できるようにしたものである。
 *    その使用例はCommon.hのvoid bindSTLContainer(pybind11::module &m)を参照のこと。
 *
 * @attention (1)の条件によりコンテナの中身のクラスは前方宣言が必要となる可能性が高いため、クラス内クラスのように、対応できない種類のクラスがある。
 *    ただし、前方宣言できないクラスについては、多くの場合そのクラスが宣言されるヘッダファイルで呼び出せば十分である。
 *
 * @attention (2)の可変引数部分にはコンテナの中身が自作クラスである場合を除き、基本的にpybind11::module_local(true)を追加すること。
 *   自作クラスの場合、つねに参照渡しとして問題ないのであればpybind11::module_local(false)を指定してもよい。
 *
 * @attention pybind11::module_local(true)を与えたコンテナのバインドはmoduleごとに行う必要があるので、各プラグインでも同様に(1),(2)を行う必要がある。
 *   頻出しそうなコンテナはbindSTLContainer.hのvoid ::asrc::core::bindSTLContainer(pybind11::module &m)にバインド処理がまとめられているので、これをplugin側でも呼び出せばよい。
 *   追加でバインドを行う場合は各pluginで同様に(1),(2)の処理を記述すればよい。
 *
 * @par 制約
 * - 要素の追加や代入(appendや=等)に与えた引数はコピーされるので、追加・代入後に元の引数を変更しても反映されない。
 * - forward_list,multimap,unordered_multimap,multiset,unordered_multisetは未対応。
 * - collections.abc.Sequence等による判定には未対応。
 */
#pragma once
#include "util/macros/common_macros.h"
#include "traits/is_stl_container.h"
#include "traits/is_string.h"
#include "traits/is_recursively_comparable.h"
#include "traits/pybind.h"

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)

template<asrc::core::traits::stl_container Type>
struct def_common_serialization<Type>{
    static constexpr bool op_enable_if_hook = true; // これによりcls.defのオーバーロードが実体化され、以下のexecuteを呼べるようになる

    template <typename Class, typename... Extra>
    requires ( std::same_as<Type,typename Class::type> )
    void execute(Class &cl, const Extra &...extra) const{
        cl
        .def("to_json",[](const Type& v){
            nl::ordered_json ret;
            {
                ::asrc::core::util::NLJSONOutputArchive archive(ret,true);
                archive(v);
            }
            return std::move(ret);
        })
        .def("load_from_json",[](Type& v, const nl::ordered_json& j){
            ::asrc::core::util::NLJSONInputArchive archive(j,true);
            archive(v);
        })
        .def_static("from_json",[](const nl::ordered_json& j){
            Type ret;
            {
                ::asrc::core::util::NLJSONInputArchive archive(j,true);
                archive(ret);
            }
            return std::move(ret);
        })
        .def("to_binary",[](const Type& v){
            std::stringstream oss;
            {
                cereal::PortableBinaryOutputArchive archive(oss);
                archive(v);
            }
            return pybind11::bytes(oss.str());
        })
        .def("load_from_binary",[](Type& v,const pybind11::bytes& str){
            std::istringstream iss(str.cast<std::string>());
            cereal::PortableBinaryInputArchive archive(iss);
            archive(v);
        })
        .def_static("from_binary",[](const pybind11::bytes& str){
            Type ret;
            {
                std::istringstream iss(str.cast<std::string>());
                cereal::PortableBinaryInputArchive archive(iss);
                archive(ret);
            }
            return std::move(ret);
        })
        .def("save",&::asrc::core::util::save_func_exposed_to_python<Type>)
        .def("load",&::asrc::core::util::load_func_exposed_to_python<Type>)
        .def_static("static_load",&::asrc::core::util::static_load_func_exposed_to_python<Type>)
        .def_property_readonly_static("_allow_cereal_serialization_in_cpp",[](pybind11::object /* self */){return true;})
        .def(pybind11::pickle(
            //TODO C++側のアドレスは一致するがPythonのobject idが変わるのでisが使えなくなる。Unpicklerでpersistent_loadを用いる必要がある。
            [](const Type &t){//__getstate__
                std::stringstream oss;
                {
                    cereal::PortableBinaryOutputArchive archive(oss);
                    archive(t);
                }
                return pybind11::bytes(oss.str());
            },
            [](const pybind11::bytes& data){//__setstate__
                Type ret;
                {
                    std::istringstream iss(data.cast<std::string>());
                    cereal::PortableBinaryInputArchive archive(iss);
                    archive(ret);
                }
                return std::move(ret);
            }
        ))
        ;
    }
};

// forward declaration
/**
 * @brief STLコンテナをバインドするための構造体。コンテナの種類ごとに特殊化する。
 */
template<class T,typename holder_type>
struct stl_container_binder {};

// forward declaration
/**
 * @brief pybind11の本来のSTLコンテナとPythonオブジェクトの間のcaster
 */
template<typename T>
struct pybind11_default_type_caster : pybind11::detail::make_caster<T> {};

ASRC_INLINE_NAMESPACE_BEGIN(util)
    // forward declaration
    template<traits::stl_container Container>
    pybind11::object toPythonContainer(Container&& v, pybind11::return_value_policy policy=pybind11::return_value_policy::copy, pybind11::handle parent=pybind11::handle());

    /**
     * @brief C++側でSTLコンテナをPythonのlist又はtupleに変換する関数の実体
     */
    template<traits::pyobject P, traits::stl_container Container>
    requires (
        (
            std::same_as<P,pybind11::list>
            || std::same_as<P,pybind11::tuple>
        ) && (
            traits::stl_array<Container>
            || traits::stl_deque<Container>
            //|| traits::stl_forward_list<Container>
            || traits::stl_list<Container>
            || traits::stl_vector<Container>
            || traits::stl_valarray<Container>
            || traits::stl_set<Container>
            //|| traits::stl_multiset<Container>
            || traits::stl_unordered_set<Container>
            //|| traits::stl_unordered_multiset<Container>
            || traits::stl_tuple<Container>
            || traits::stl_pair<Container>
        )
    )
    P tosequence(Container&& v, pybind11::return_value_policy policy=pybind11::return_value_policy::copy, pybind11::handle parent=pybind11::handle()){
        pybind11::gil_scoped_acquire acquire;
        using uncvref=std::remove_cvref_t<Container>;
        
        if constexpr(
            traits::stl_tuple<Container>
            || traits::stl_pair<Container>
        ){
            return stl_container_binder<uncvref,std::unique_ptr<uncvref>>::template cast_as<P>(std::forward<Container>(v),policy,parent);
        }else{
            using ValueType = typename std::iterator_traits<typename uncvref::iterator>::value_type;
            using value_conv = pybind11::detail::make_caster<ValueType>;

            if (!std::is_lvalue_reference_v<Container>) {
                policy = pybind11::detail::return_value_policy_override<ValueType>::policy(policy);
            }

            P l(v.size());
            pybind11::ssize_t index=0;
            for(auto&& value : std::forward<Container>(v)){
                pybind11::object value_;
                if constexpr (traits::stl_container<ValueType>){
                    value_ = toPythonContainer(pybind11::detail::forward_like<Container>(value), policy, parent);
                }else{
                    value_ = pybind11::reinterpret_steal<pybind11::object>(
                        value_conv::cast(pybind11::detail::forward_like<Container>(value), policy, parent));
                }
                if (!value_) {
                    throw pybind11::type_error();
                }
                if constexpr (std::same_as<P,pybind11::list>){
                    PyList_SET_ITEM(l.ptr(), index++, value_.release().ptr()); // steals a reference
                }else{
                    PyTuple_SET_ITEM(l.ptr(), index++, value_.release().ptr()); // steals a reference
                }
            }
            return std::move(l);
        }
    }

    /**
     * @brief C++側でSTLコンテナをPythonのlistに変換する関数
     * 
     * @param [in] v 変換元のコンテナ
     * @param [in] policy pybind11::castのreturn_value_policyと同じ。デフォルトはcopy。
     * @param [in] parent pybind11::castのreturn_value_policyと同じ。policyがreference_internalのときは必須。
     * 
     * @code{.cpp}
     * std::vector<int> v;
     * pybind::list l = asrc::core::util::tolist(v);
     * @endcode
     * 
     * @attention return_value_policyはコンテナではなく要素に対する指定を表す。
     *    ただし、要素型がbool,数値,文字列の場合は無効化され、常にコピーされる。
     * @attention ネストされたコンテナに対して再帰的な変換を行う。
     */
    template<traits::stl_container Container>
    requires (
        traits::stl_array<Container>
        || traits::stl_deque<Container>
        //|| traits::stl_forward_list<Container>
        || traits::stl_list<Container>
        || traits::stl_vector<Container>
        || traits::stl_valarray<Container>
        || traits::stl_set<Container>
        //|| traits::stl_multiset<Container>
        || traits::stl_unordered_set<Container>
        //|| traits::stl_unordered_multiset<Container>
        || traits::stl_tuple<Container>
        || traits::stl_pair<Container>
    )
    pybind11::list tolist(Container&& v, pybind11::return_value_policy policy=pybind11::return_value_policy::copy, pybind11::handle parent=pybind11::handle()){
        return tosequence<pybind11::list>(std::forward<Container>(v),policy,parent);
    }


    /**
     * @brief C++側でSTLコンテナをPythonのtupleに変換する関数
     * 
     * @param [in] v 変換元のコンテナ
     * @param [in] policy pybind11::castのreturn_value_policyと同じ。デフォルトはcopy。
     * @param [in] parent pybind11::castのreturn_value_policyと同じ。policyがreference_internalのときは必須。
     * 
     * @code{.cpp}
     * std::vector<int> v;
     * pybind::tuple t = asrc::core::util::totuple(v);
     * @endcode
     * 
     * @attention return_value_policyはコンテナではなく要素に対する指定を表す。
     *    ただし、要素型がbool,数値,文字列の場合は無効化され、常にコピーされる。
     * @attention ネストされたコンテナに対して再帰的な変換を行う。
     */
    template<traits::stl_container Container>
    requires (
        traits::stl_array<Container>
        || traits::stl_deque<Container>
        //|| traits::stl_forward_list<Container>
        || traits::stl_list<Container>
        || traits::stl_vector<Container>
        || traits::stl_valarray<Container>
        || traits::stl_set<Container>
        //|| traits::stl_multiset<Container>
        || traits::stl_unordered_set<Container>
        //|| traits::stl_unordered_multiset<Container>
        || traits::stl_tuple<Container>
        || traits::stl_pair<Container>
    )
    pybind11::tuple totuple(Container&& v, pybind11::return_value_policy policy=pybind11::return_value_policy::copy, pybind11::handle parent=pybind11::handle()){
        return tosequence<pybind11::tuple>(std::forward<Container>(v),policy,parent);
    }


    /**
     * @brief C++側でSTLコンテナをPythonのsetに変換する関数
     * 
     * @param [in] v 変換元のコンテナ
     * @param [in] policy pybind11::castのreturn_value_policyと同じ。デフォルトはcopy。
     * @param [in] parent pybind11::castのreturn_value_policyと同じ。policyがreference_internalのときは必須。
     * 
     * @code{.cpp}
     * std::set<int> v2;
     * pybind::set s = asrc::core::util::toset(v2);
     * @endcode
     * 
     * @attention return_value_policyはコンテナではなく要素に対する指定を表す。
     *    ただし、要素型がbool,数値,文字列の場合は無効化され、常にコピーされる。
     * @attention ネストされたコンテナに対して再帰的な変換を行う。
     */
    template<traits::stl_container Container>
    requires (
        traits::stl_array<Container>
        || traits::stl_deque<Container>
        //|| traits::stl_forward_list<Container>
        || traits::stl_list<Container>
        || traits::stl_vector<Container>
        || traits::stl_valarray<Container>
        || traits::stl_set<Container>
        //|| traits::stl_multiset<Container>
        || traits::stl_unordered_set<Container>
        //|| traits::stl_unordered_multiset<Container>
        || traits::stl_tuple<Container>
        || traits::stl_pair<Container>
    )
    pybind11::set toset(Container&& v, pybind11::return_value_policy policy=pybind11::return_value_policy::copy, pybind11::handle parent=pybind11::handle()){
        pybind11::gil_scoped_acquire acquire;
        using uncvref=std::remove_cvref_t<Container>;
        
        if constexpr(
            traits::stl_tuple<Container>
            || traits::stl_pair<Container>
        ){
            return stl_container_binder<uncvref,std::unique_ptr<uncvref>>::template cast_as<pybind11::set>(std::forward<Container>(v),policy,parent);
        }else{
            using ValueType = typename std::iterator_traits<typename uncvref::iterator>::value_type;
            using value_conv = pybind11::detail::make_caster<ValueType>;

            if (!std::is_lvalue_reference_v<Container>) {
                policy = pybind11::detail::return_value_policy_override<ValueType>::policy(policy);
            }

            pybind11::set s;
            pybind11::ssize_t index=0;
            for(auto&& value : std::forward<Container>(v)){
                pybind11::object value_;
                if constexpr (traits::stl_container<ValueType>){
                    value_ = toPythonContainer(pybind11::detail::forward_like<Container>(value), policy, parent);
                }else{
                    value_ = pybind11::reinterpret_steal<pybind11::object>(
                        value_conv::cast(pybind11::detail::forward_like<Container>(value), policy, parent));
                }
                if (!value_ || !s.add(std::move(value_))) {
                    throw pybind11::type_error();
                }
            }
            return std::move(s);
        }
    }


    /**
     * @brief C++側でSTLコンテナをPythonのdictに変換する関数
     * 
     * @param [in] v 変換元のコンテナ
     * @param [in] policy pybind11::castのreturn_value_policyと同じ。デフォルトはcopy。
     * @param [in] parent pybind11::castのreturn_value_policyと同じ。policyがreference_internalのときは必須。
     * 
     * @code{.cpp}
     * std::map<std::string,int> v3;
     * pybind::dict d = asrc::core::util::todict(v3);
     * @endcode
     * 
     * @attention return_value_policyはコンテナではなく要素に対する指定を表す。
     *    ただし、要素型がbool,数値,文字列の場合は無効化され、常にコピーされる。
     * @attention ネストされたコンテナに対して再帰的な変換を行う。
     */
    template<traits::stl_container Container>
    requires (
        traits::stl_map<Container>
        //|| traits::stl_multimap<Container>
        || traits::stl_unordered_map<Container>
        //|| traits::stl_unordered_multimap<Container>
    )
    pybind11::dict todict(Container&& v, pybind11::return_value_policy policy=pybind11::return_value_policy::copy, pybind11::handle parent=pybind11::handle()){
        pybind11::gil_scoped_acquire acquire;
        using uncvref=std::remove_cvref_t<Container>;

        using KeyType = typename uncvref::key_type;
        using ValueType = typename uncvref::mapped_type;
        using key_conv = pybind11::detail::make_caster<KeyType>;
        using value_conv = pybind11::detail::make_caster<ValueType>;

        pybind11::dict d;
        pybind11::return_value_policy policy_key = policy;
        pybind11::return_value_policy policy_value = policy;
        if (!std::is_lvalue_reference<Container>::value) {
            policy_key = pybind11::detail::return_value_policy_override<KeyType>::policy(policy_key);
            policy_value = pybind11::detail::return_value_policy_override<ValueType>::policy(policy_value);
        }
        for (auto&& [key, value] : std::forward<Container>(v)) {
            pybind11::object key_, value_;
            if constexpr (traits::stl_container<KeyType>){
                key_ = toPythonContainer(pybind11::detail::forward_like<Container>(key), policy, parent);
            }else{
                key_ = pybind11::reinterpret_steal<pybind11::object>(
                    key_conv::cast(pybind11::detail::forward_like<Container>(key), policy_key, parent));
            }
            if constexpr (traits::stl_container<ValueType>){
                value_ = toPythonContainer(pybind11::detail::forward_like<Container>(value), policy, parent);
            }else{
                value_ = pybind11::reinterpret_steal<pybind11::object>(
                    value_conv::cast(pybind11::detail::forward_like<Container>(value), policy_value, parent));
            }
            if (!key_ || !value_) {
                throw pybind11::type_error();
            }
            d[std::move(key_)] = std::move(value_);
        }
        return std::move(d);


        using uncvref=std::remove_cvref_t<Container>;
        return pybind11::reinterpret_steal<pybind11::dict>(pybind11_default_type_caster<uncvref>::cast(std::forward<Container>(v),policy,parent));
    }


    /**
     * @brief C++側でSTLコンテナをPythonのコンテナに変換してpy::objectで返す関数
     * 
     * @param [in] v 変換元のコンテナ
     * @param [in] policy pybind11::castのreturn_value_policyと同じ。デフォルトはcopy。
     * @param [in] parent pybind11::castのreturn_value_policyと同じ。policyがreference_internalのときは必須。
     * 
     * @code{.cpp}
     * std::vector<int> v;
     * pybind::object o = asrc::core::util::toPythonContainer(v);
     * @endcode
     * 
     * @attention return_value_policyはコンテナではなく要素に対する指定を表す。
     *    ただし、要素型がbool,数値,文字列の場合は無効化され、常にコピーされる。
     * @attention ネストされたコンテナに対して再帰的な変換を行う。
     */
    template<traits::stl_container Container>
    pybind11::object toPythonContainer(Container&& v, pybind11::return_value_policy policy, pybind11::handle parent){
        pybind11::gil_scoped_acquire acquire;
        if constexpr (
            traits::stl_array<Container>
            || traits::stl_deque<Container>
            //|| traits::stl_forward_list<Container>
            || traits::stl_list<Container>
            || traits::stl_vector<Container>
            || traits::stl_valarray<Container>
            || traits::stl_tuple<Container>
            || traits::stl_pair<Container>
        ){
            return tolist(std::forward<Container>(v),policy,parent);
        }else if constexpr (
            traits::stl_set<Container>
            //|| traits::stl_multiset<Container>
            || traits::stl_unordered_set<Container>
            //|| traits::stl_unordered_multiset<Container>
        ){
            return toset(std::forward<Container>(v),policy,parent);
        }else if constexpr (
            traits::stl_map<Container>
            //|| traits::stl_multimap<Container>
            || traits::stl_unordered_map<Container>
            //|| traits::stl_unordered_multimap<Container>
        ){
            return todict(std::forward<Container>(v),policy,parent);
        }else{
            throw pybind11::type_error();
        }
    }

ASRC_NAMESPACE_END(util)


template<typename Vector,typename holder_type>
requires (
    (
        traits::stl_array<Vector>
        || traits::stl_deque<Vector>
        //|| traits::stl_forward_list<Vector>
        || traits::stl_list<Vector>
        || traits::stl_vector<Vector>
        || traits::stl_valarray<Vector>
    ) && traits::pybind11_bind_vector_allowed<Vector>
)
struct stl_container_binder<Vector,holder_type> {
    using Class = pybind11::class_<Vector, holder_type>;
    using ValueType = typename Vector::value_type;
    using SizeType = typename Vector::size_type;
    using DiffType = typename Vector::difference_type;

    template<typename ... Args>
    static Class bind(pybind11::handle scope, std::string const &name, Args &&...args){
        try{
            // extention of pybind11::bind_vector
            auto *vtype_info = pybind11::detail::get_type_info(typeid(ValueType));
            bool local = !vtype_info || vtype_info->module_local;

            Class ret(scope,name.c_str(),pybind11::module_local(local),std::forward<Args>(args)...);

            vector_buffer<Args...>(ret);
            ret.def(py_init<>());
            //vector_if_copy_constructible(ret);
            vector_if_equal_operator(ret);
            vector_if_insertion_operator(ret, name);
            vector_modifiers(ret);
            vector_accessor(ret);

            ret.def(
                "__bool__",
                [](const Vector &v) -> bool { return !v.empty(); },
                "Check whether the list is nonempty");

            ret.def("__len__", [](const Vector &vec) { return vec.size(); });

            ret
            .def("tolist",[](Vector& v){ return tolist(v); })
            .def("totuple",[](Vector& v){ return totuple(v); })
            .def("toset",[](Vector& v){ return toset(v); })
            .def("toPythonContainer",[](Vector& v){ return toPythonContainer(v); })
            .def(def_common_serialization<Vector>())
            ;
            return std::move(ret);
        }catch(std::exception& ex){
            if(pybind11::detail::get_type_info(typeid(Vector))){
                // already bound
                auto ret=pybind11::reinterpret_borrow<Class>(pybind11::detail::get_type_handle(typeid(Vector),true));
                assert(ret);
                return std::move(ret);
            }else{
                throw ex;
            }
        }
    }

    template<typename ... Args>
    static void vector_buffer(Class& cl){
        pybind11::detail::vector_buffer<Vector, Class, Args...>(cl);
    }

    //static void vector_if_copy_constructible(Class& cl){
    //    pybind11::detail::vector_if_copy_constructible<Vector, Class>(cl);
    //}

    static void vector_if_equal_operator(Class& cl){
        if constexpr (traits::recursively_comparable<Vector>){
            cl.def(pybind11::self == pybind11::self);
            cl.def(pybind11::self != pybind11::self);

            cl.def(
                "count",
                [](const Vector &v, const ValueType &x) { return std::count(v.begin(), v.end(), x); },
                pybind11::arg("x"),
                "Return the number of times ``x`` appears in the list");

            if constexpr (
                !(
                    traits::stl_array<Vector>
                    || traits::stl_valarray<Vector>
                )
            ){
                cl.def(
                    "remove",
                    [](Vector &v, const ValueType &x) {
                        auto p = std::find(v.begin(), v.end(), x);
                        if (p != v.end()) {
                            v.erase(p);
                        } else {
                            throw pybind11::value_error();
                        }
                    },
                    pybind11::arg("x"),
                    "Remove the first item from the list whose value is x. "
                    "It is an error if there is no such item.");
            }

            cl.def(
                "__contains__",
                [](const Vector &v, const ValueType &x) { return std::find(v.begin(), v.end(), x) != v.end(); },
                pybind11::arg("x"),
                "Return true the container contains ``x``");
        }
    }

    static constexpr bool vector_needs_copy(){
         // std::vector<bool>の特殊化対策とのこと。
         // std::listに対してはそのような特殊化は行われていないはずだが、同様に判定しておく。
        if constexpr(traits::stl_list<Vector> || traits::stl_forward_list<Vector>){
            return !std::same_as<decltype(*std::declval<Vector>().begin()),ValueType&>;
        }else{
            return !std::same_as<decltype(std::declval<Vector>()[std::declval<SizeType>()]),ValueType&>;
        }
    }

    static void vector_modifiers(Class& cl) {
        if constexpr (std::copy_constructible<ValueType>){
            auto wrap_i = [](DiffType i, SizeType n) {
                if (i < 0) {
                    i += n;
                }
                if (i < 0 || (SizeType) i >= n) {
                    throw pybind11::index_error();
                }
                return i;
            };

            if constexpr (
                !(
                    traits::stl_array<Vector>
                    || traits::stl_valarray<Vector>
                )
            ){
                cl.def(
                    "append",
                    [](Vector &v, const ValueType &value) { v.push_back(value); },
                    pybind11::arg("x"),
                    "Add an item to the end of the list");
            }

            cl.def(py_init([](const pybind11::iterable &it) {

                if(std::is_copy_constructible_v<Vector> && pybind11::isinstance<Vector>(it)){
                    // Vectorがコピーコンストラクト可能かつそれを引数として生成する場合は、
                    // コピーコンストラクタを呼ぶ。
                    if constexpr(std::is_copy_constructible_v<Vector>){
                        const Vector& value=it.template cast<std::reference_wrapper<Vector>>().get();
                        auto v = std::unique_ptr<Vector>(new Vector(value));
                        return v.release();
                    }
                }else{
                    // そうでなければ、要素ごとにコピーして生成する。
                    auto v = std::unique_ptr<Vector>(new Vector());
                    if constexpr (
                        traits::stl_array<Vector>
                        || traits::stl_valarray<Vector>
                    ){
                        auto hint=len_hint(it);
                        if(hint>0 && v->size()!=hint){
                            throw pybind11::index_error();
                        }
                        std::vector<pybind11::object> elements;
                        elements.reserve(hint);
                        for(pybind11::handle h : it){
                            elements.push_back(pybind11::reinterpret_borrow<pybind11::object>(h));
                        }
                        if(v->size() != elements.size()){
                            throw pybind11::index_error();
                        }
                        for (SizeType i = 0; i < v->size(); ++i) {
                            (*v)[i]=elements[i].template cast<ValueType>();
                        }
                    }else{
                        if constexpr (traits::stl_vector<Vector>){
                            v->reserve(len_hint(it));
                        }
                        for (pybind11::handle h : it) {
                            v->push_back(h.cast<ValueType>());
                        }
                    }
                    return v.release();
                }
            }));

            if constexpr (
                !(
                    traits::stl_array<Vector>
                    || traits::stl_valarray<Vector>
                )
            ){
                cl.def("clear", [](Vector &v) { v.clear(); }, "Clear the contents");

                cl.def(
                    "extend",
                    [](Vector &v, const Vector &src) { v.insert(v.end(), src.begin(), src.end()); },
                    pybind11::arg("L"),
                    "Extend the list by appending all the items in the given list");

                cl.def(
                    "extend",
                    [](Vector &v, const pybind11::iterable &it) {
                        const SizeType old_size = v.size();
                        if constexpr (traits::stl_vector<Vector>){
                            v.reserve(old_size + len_hint(it));
                        }
                        try {
                            for (pybind11::handle h : it) {
                                v.push_back(h.cast<ValueType>());
                            }
                        } catch (const pybind11::cast_error &) {
                            v.erase(std::next(v.begin(), old_size),v.end());
                            if constexpr (
                                traits::stl_vector<Vector>
                                || traits::stl_deque<Vector>
                            ){
                                try {
                                    v.shrink_to_fit();
                                } catch (const std::exception &) { // NOLINT(bugprone-empty-catch)
                                    // Do nothing
                                }
                            }
                            throw;
                        }
                    },
                    pybind11::arg("L"),
                    "Extend the list by appending all the items in the given list");

                cl.def(
                    "insert",
                    [](Vector &v, DiffType i, const ValueType &x) {
                        // Can't use wrap_i; i == v.size() is OK
                        if (i < 0) {
                            i += v.size();
                        }
                        if (i < 0 || (SizeType) i > v.size()) {
                            throw pybind11::index_error();
                        }
                        v.insert(std::next(v.begin(), i), x);
                    },
                    pybind11::arg("i"),
                    pybind11::arg("x"),
                    "Insert an item at a given position.");

                cl.def(
                    "pop",
                    [](Vector &v) {
                        if (v.empty()) {
                            throw pybind11::index_error();
                        }
                        ValueType t = std::move(v.back());
                        v.pop_back();
                        return t;
                    },
                    "Remove and return the last item");

                cl.def(
                    "pop",
                    [wrap_i](Vector &v, DiffType i) {
                        i = wrap_i(i, v.size());
                        auto it = std::next(v.begin(), i);
                        ValueType t = std::move(*it);
                        v.erase(it);
                        return t;
                    },
                    pybind11::arg("i"),
                    "Remove and return the item at index ``i``");
            }

            cl.def("__setitem__", [wrap_i](Vector &v, DiffType i, const ValueType &t) {
                i = wrap_i(i, v.size());
                if constexpr(
                    traits::stl_list<Vector>
                    //|| traits::stl_forward_list<Vector>
                ){
                    *std::next(v.begin(), i)=t;
                }else{
                    v[(SizeType) i] = t;
                }
            });

            /// Slicing protocol
            cl.def(
                "__getitem__",
                [](const Vector &v, const pybind11::slice &slice) -> pybind11::list {
                    size_t start = 0, stop = 0, step = 0, slicelength = 0;

                    if (!slice.compute(v.size(), &start, &stop, &step, &slicelength)) {
                        throw pybind11::error_already_set();
                    }

                    pybind11::list seq(slicelength);
                    auto self=pybind11::cast(v,pybind11::return_value_policy::reference);
                    constexpr auto policy = vector_needs_copy() ? pybind11::return_value_policy::automatic : pybind11::return_value_policy::reference_internal;

                    if constexpr(
                        traits::stl_list<Vector>
                        //|| traits::stl_forward_list<Vector>
                    ){
                        auto src = std::next(v.begin(), start);
                        for (SizeType i = 0; i < slicelength; ++i) {
                            seq[i]=pybind11::cast(*src,policy,self);
                            src = std::next(src, step);
                        }
                    }else{
                        for (SizeType i = 0; i < slicelength; ++i) {
                            seq[i]=pybind11::cast(v[start],policy,self);
                            start += step;
                        }
                    }

                    return std::move(seq);
                },
                vector_needs_copy() ? pybind11::return_value_policy::automatic : pybind11::return_value_policy::reference_internal,
                pybind11::arg("s"),
                "Retrieve list elements using a slice object");

            cl.def(
                "__setitem__",
                [](Vector &v, const pybind11::slice &slice, const pybind11::iterable &it) {
                    size_t start = 0, stop = 0, step = 0, slicelength = 0;
                    if (!slice.compute(v.size(), &start, &stop, &step, &slicelength)) {
                        throw pybind11::error_already_set();
                    }

                    if(pybind11::isinstance<Vector>(it)){
                        const Vector& value=it.template cast<std::reference_wrapper<Vector>>().get();

                        if (slicelength != value.size()) {
                            throw std::runtime_error(
                                "Left and right hand size of slice assignment have different sizes!");
                        }

                        if constexpr(
                            traits::stl_list<Vector>
                            //|| traits::stl_forward_list<Vector>
                        ){
                            auto src = value.begin();
                            auto dst = std::next(v.begin(), start);
                            for (SizeType i = 0; i < slicelength; ++i) {
                                *dst=*src;
                                src++;
                                dst = std::next(dst, step);
                            }
                        }else{
                            for (SizeType i = 0; i < slicelength; ++i) {
                                v[start] = value[i];
                                start += step;
                            }
                        }
                    }else{
                        auto hint = len_hint(it);
                        if (hint>0 && slicelength != hint) {
                            throw std::runtime_error(
                                "Left and right hand size of slice assignment have different sizes!");
                        }

                        std::vector<pybind11::object> elements;
                        elements.reserve(hint);
                        for(pybind11::handle h : it){
                            elements.push_back(pybind11::reinterpret_borrow<pybind11::object>(h));
                        }
                        if(slicelength != elements.size()){
                            throw std::runtime_error(
                                "Left and right hand size of slice assignment have different sizes!");
                        }

                        if constexpr(
                            traits::stl_list<Vector>
                            //|| traits::stl_forward_list<Vector>
                        ){
                            auto dst = std::next(v.begin(), start);
                            for (SizeType i = 0; i < slicelength; ++i) {
                                *dst=elements[i].template cast<ValueType>();
                                dst = std::next(dst, step);
                            }
                        }else{
                            for (SizeType i = 0; i < slicelength; ++i) {
                                v[start] = elements[i].template cast<ValueType>();
                                start += step;
                            }
                        }
                    }
                },
                "Assign list elements using a slice object");

            if constexpr (
                !(
                    traits::stl_array<Vector>
                    || traits::stl_valarray<Vector>
                )
            ){
                cl.def(
                    "__delitem__",
                    [wrap_i](Vector &v, DiffType i) {
                        i = wrap_i(i, v.size());
                        v.erase(std::next(v.begin(), i));
                    },
                    "Delete the list elements at index ``i``");

                cl.def(
                    "__delitem__",
                    [](Vector &v, const pybind11::slice &slice) {
                        size_t start = 0, stop = 0, step = 0, slicelength = 0;

                        if (!slice.compute(v.size(), &start, &stop, &step, &slicelength)) {
                            throw pybind11::error_already_set();
                        }

                        if (step == 1 && false) {
                            v.erase(std::next(v.begin(), start), std::next(v.begin(), start + slicelength));
                        } else {
                            for (SizeType i = 0; i < slicelength; ++i) {
                                v.erase(std::next(v.begin(), start));
                                start += step - 1;
                            }
                        }
                    },
                    "Delete list elements using a slice object");
            }
        }
    }

    static void vector_accessor(Class &cl) {
        using ItType = typename Vector::iterator;

        auto wrap_i = [](DiffType i, SizeType n) {
            if (i < 0) {
                i += n;
            }
            if (i < 0 || (SizeType) i >= n) {
                throw pybind11::index_error();
            }
            return i;
        };

        using ReturnType = std::conditional_t<vector_needs_copy(), ValueType, ValueType&>;

        cl.def(
            "__getitem__",
            [wrap_i](Vector &v, DiffType i) -> ReturnType {
                i = wrap_i(i, v.size());
                if constexpr (
                    traits::stl_list<Vector>
                    || traits::stl_forward_list<Vector>
                ){
                    return *(std::next(v.begin(), i));
                }else{
                    return v[(SizeType) i];
                }
            },
            vector_needs_copy() ? pybind11::return_value_policy::automatic : pybind11::return_value_policy::reference_internal
        );

        cl.def(
            "__iter__",
            [](Vector &v) {
                constexpr auto policy = vector_needs_copy() ? pybind11::return_value_policy::copy : pybind11::return_value_policy::reference_internal;
                return pybind11::make_iterator<policy, ItType, ItType, ReturnType>(v.begin(), v.end());
            },
            pybind11::keep_alive<0, 1>() /* Essential: keep list alive while iterator exists */
        );
    }

    static void vector_if_insertion_operator(Class &cl, std::string const &name){
        if constexpr(traits::writeable_to_stream<ValueType>){
            cl.def(
                "__repr__",
                [name](Vector &v) {
                    std::ostringstream s;
                    s << name << '[';
                    auto it=v.begin(); // std::listはoperator[]がないのでiteratorに変更
                    for (SizeType i = 0; i < v.size(); ++i) {
                        s << *it;
                        ++it;
                        if (i != v.size() - 1) {
                            s << ", ";
                        }
                    }
                    s << ']';
                    return s.str();
                },
                "Return the canonical string representation of this list.");
        }
    }
};

template<typename Map,typename holder_type>
requires (
    (
        traits::stl_map<Map>
        //|| traits::stl_multimap<Map>
        || traits::stl_unordered_map<Map>
        //|| traits::stl_unordered_multimap<Map>
    ) &&  traits::pybind11_bind_map_allowed<Map>
)
struct stl_container_binder<Map,holder_type> {
    using Class = pybind11::class_<Map, holder_type>;
    using KeyType=typename Map::key_type;
    using MappedType=typename Map::mapped_type;

    template<typename ... Args>
    static Class bind(pybind11::handle scope, std::string const &name, Args &&...args){
        try{
            auto ret=pybind11::bind_map<Map, holder_type>(scope,name,std::forward<Args>(args)...);

            if constexpr (traits::recursively_comparable<Map>){
                ret.def(pybind11::self == pybind11::self);
                ret.def(pybind11::self != pybind11::self);
            }

            ret.def(py_init([](const pybind11::iterable &it) {
                if(std::is_copy_constructible_v<Map> && pybind11::isinstance<Map>(it)){
                    // Mapがコピーコンストラクト可能かつそれを引数として生成する場合は、
                    // コピーコンストラクタを呼ぶ。
                    if constexpr(std::is_copy_constructible_v<Map>){
                        const Map& value=it.template cast<std::reference_wrapper<Map>>().get();
                        auto v = std::unique_ptr<Map>(new Map(value));
                        return v.release();
                    }
                }else if(pybind11::isinstance<pybind11::dict>(it)){
                    // そうでなければ、要素ごとにコピーして生成する。
                    pybind11::dict d=pybind11::reinterpret_borrow<pybind11::dict>(it);
                    auto v = std::unique_ptr<Map>(new Map());
                    for(auto&& [key_,value_] : d){
                        const auto& key=key_.template cast<std::reference_wrapper<KeyType>>().get();
                        const auto& value=value_.template cast<std::reference_wrapper<MappedType>>().get();
                        (*v)[key]=value;
                    }
                    return v.release();
                }else{
                    throw pybind11::type_error();
                }
            }), "Copy constructor")
            .def("todict",[](Map& v){ return todict(v); })
            .def("toPythonContainer",[](Map& v){ return toPythonContainer(v); })
            .def(def_common_serialization<Map>())
            ;
            return std::move(ret);
        }catch(std::exception& ex){
            if(pybind11::detail::get_type_info(typeid(Map))){
                // already bound
                auto ret=pybind11::reinterpret_borrow<Class>(pybind11::detail::get_type_handle(typeid(Map),true));
                assert(ret);
                return std::move(ret);
            }else{
                throw ex;
            }
        }
    }
};

template<typename Set,typename holder_type>
requires (
    traits::stl_set<Set>
    //|| traits::stl_multiset<Set>
    || traits::stl_unordered_set<Set>
    //|| traits::stl_unordered_multiset<Set>
)
struct stl_container_binder<Set,holder_type> {
    using Class = pybind11::class_<Set, holder_type>;
    using KeyType = typename Set::key_type;
    using ItemsView = pybind11::detail::items_view;

    template<typename ... Args>
    static Class bind(pybind11::handle scope, std::string const &name, Args &&...args){
        try{
            auto *tinfo =  pybind11::detail::get_type_info(typeid(KeyType));
            bool local = !tinfo || tinfo->module_local;
        
            Class ret(scope,name.c_str(),pybind11::module_local(local),std::forward<Args>(args)...);

            // Wrap ItemsView if it wasn't already wrapped
            if (!pybind11::detail::get_type_info(typeid(ItemsView))) {
                pybind11::class_<ItemsView> items_view(scope, "ItemsView", pybind11::module_local(local));
                items_view.def("__len__", &ItemsView::len);
                items_view.def("__iter__",
                            &ItemsView::iter,
                            pybind11::keep_alive<0, 1>() /* Essential: keep view alive while iterator exists */
                );
            }

            ret.def(py_init<>())
            .def(py_init([](const pybind11::iterable& it){

                if(std::is_copy_constructible_v<Set> && pybind11::isinstance<Set>(it)){
                    // Setがコピーコンストラクト可能かつそれを引数として生成する場合は、
                    // コピーコンストラクタを呼ぶ。
                    if constexpr(std::is_copy_constructible_v<Set>){
                        const Set& value=it.template cast<std::reference_wrapper<Set>>().get();
                        auto v = std::unique_ptr<Set>(new Set(value));
                        return v.release();
                    }
                }else{
                    // そうでなければ、要素ごとにコピーして生成する。
                    auto v = std::unique_ptr<Set>(new Set());
                    for (pybind11::handle h : it) {
                        v->insert(h.cast<KeyType>());
                    }
                    return v.release();
                }
            }), "Copy constructor from python iterable")
            ;

            if constexpr (traits::recursively_comparable<Set>){
                ret.def(pybind11::self == pybind11::self);
                ret.def(pybind11::self != pybind11::self);
            }

            ret.def(
                "__bool__",
                [](const Set &m) -> bool { return !m.empty(); },
                "Check whether the set is nonempty");

            ret.def(
                "__iter__",
                [](Set &m) { return pybind11::make_iterator(m.begin(), m.end()); },
                pybind11::keep_alive<0, 1>() /* Essential: keep set alive while iterator exists */
            );

            ret.def(
                "keys",
                [](Set &m) { return std::unique_ptr<ItemsView>(new pybind11::detail::ItemsViewImpl<Set>(m)); },
                pybind11::keep_alive<0, 1>() /* Essential: keep set alive while view exists */
            );

            ret.def("__contains__", [](Set &m, const KeyType &k) -> bool {
                auto it = m.find(k);
                if (it == m.end()) {
                    return false;
                }
                return true;
            });
            // Fallback for when the object is not of the key type
            ret.def("__contains__", [](Set &, const pybind11::object &) -> bool { return false; });
            // Always use a lambda in case of `using` declaration
            ret.def("__len__", [](const Set &m) { return m.size(); });

            ret.def("add", [](Set &m, const KeyType &k) {
                auto ret=m.insert(k);
                if(!ret.second){
                    throw pybind11::key_error();
                }
            });
            ret.def("add", [](Set &, const pybind11::object &) -> bool { throw pybind11::type_error(); });
            ret.def("remove", [](Set &m, const KeyType &k) {
                if(m.find(k)!=m.end()){
                    m.erase(k);
                }else{
                    throw pybind11::key_error();
                }
            });
            ret.def("remove", [](Set &, const pybind11::object &) -> bool { throw pybind11::type_error(); });
            ret.def("clear", [](Set &m) { m.clear(); });

            if constexpr (traits::writeable_to_stream<KeyType>){
                ret.def(
                    "__repr__",
                    [name](Set &m) {
                        std::ostringstream s;
                        typename Set::size_type i = 0;
                        s << name << '{';
                        for (auto&& k : m) {
                            s << k;
                            if (i != m.size() - 1) {
                                s << ", ";
                            }
                            ++i;
                        }
                        s << '}';
                        return s.str();
                    },
                    "Return the canonical string representation of this set.");
            }

            ret
            .def("tolist",[](Set& v){ return tolist(v); })
            .def("totuple",[](Set& v){ return totuple(v); })
            .def("toset",[](Set& v){ return toset(v); })
            .def("toPythonContainer",[](Set& v){ return toPythonContainer(v); })
            .def(def_common_serialization<Set>())
            ;
            return std::move(ret);
        }catch(std::exception& ex){
            if(pybind11::detail::get_type_info(typeid(Set))){
                // already bound
                auto ret=pybind11::reinterpret_borrow<Class>(pybind11::detail::get_type_handle(typeid(Set),true));
                assert(ret);
                return std::move(ret);
            }else{
                throw ex;
            }
        }
    }
};

template<template <typename...> class Tuple_, typename... Ts>
struct tuple_iterator {
    tuple_iterator(std::size_t i,const Tuple_<Ts...>& v)
    :index(i),value(v){}
    std::size_t index;
    const Tuple_<Ts...>& value;
};

template<template <typename...> class Tuple_, typename holder_type, typename... Ts>
requires (
    traits::stl_tuple<Tuple_<Ts...>>
    || traits::stl_pair<Tuple_<Ts...>>
)
struct stl_container_binder<Tuple_<Ts...>,holder_type> {
    using Tuple = Tuple_<Ts...>;
    using Class = pybind11::class_<Tuple, holder_type>;
    using Iterator=tuple_iterator<Tuple_,Ts...>;
    using SizeType = std::size_t;
    using DiffType = std::ptrdiff_t;
    static constexpr SizeType size = sizeof...(Ts);

    template<typename T>
    static bool should_be_local(){
        auto *type_info = pybind11::detail::get_type_info(typeid(T));
        return !type_info || type_info->module_local;
    }

    template<typename ... Args>
    static Class bind(pybind11::handle scope, std::string const &name, Args &&...args){
        try{
            bool local = (... || should_be_local<Ts>());

            Class cls(scope,name.c_str(),pybind11::module_local(local),std::forward<Args>(args)...);

            cls
            .def(py_init<>())
            .def(py_init<Ts...>())
            .def(
                "__bool__",
                [](const Tuple &v) -> bool { return true; },
                "Check whether the tuple is nonempty")
            .def("__len__", [](const Tuple &v) { return size; })
            ;
        
            // Register comparison-related operators and functions (if possible)
            if constexpr (traits::recursively_comparable<Tuple>){
                cls.def(pybind11::self == pybind11::self);
                cls.def(pybind11::self != pybind11::self);
            }

            auto indices = std::make_index_sequence<size>{};
            bind_tuple_impl_sequence(cls,name,indices);

            bind_tuple_iterator(scope,local,indices);

            cls.def(
                "__iter__",
                [](Tuple &v) {
                    return Iterator(0,v);
                },
                pybind11::keep_alive<0, 1>() /* Essential: keep list alive while iterator exists */
            );

            cls
            .def("tolist",[](Tuple& v){ return tolist(v); })
            .def("totuple",[](Tuple& v){ return totuple(v); })
            .def("toset",[](Tuple& v){ return toset(v); })
            .def("toPythonContainer",[](Tuple& v){ return toPythonContainer(v); })
            .def(def_common_serialization<Tuple>());

            if constexpr (traits::stl_pair<Tuple>){
                cls
                .def_readwrite("first",&Tuple::first)
                .def_readwrite("second",&Tuple::second)
                ;
            }
            return std::move(cls);
        }catch(std::exception& ex){
            if(pybind11::detail::get_type_info(typeid(Tuple))){
                // already bound
                auto cls=pybind11::reinterpret_borrow<Class>(pybind11::detail::get_type_handle(typeid(Tuple),true));
                assert(cls);
                return std::move(cls);
            }else{
                throw ex;
            }
        }
    }

    template<SizeType... Is>
    static pybind11::class_<Iterator> bind_tuple_iterator(pybind11::handle scope, bool local, std::index_sequence<Is...> indices){
        return pybind11::class_<Iterator>(scope,"(iterator)",pybind11::module_local(local))
        .def("__iter__", [](Iterator &s) -> Iterator & { return s; })
        .def("__next__",
            [](Iterator &s){
                auto self=pybind11::cast(s.value,pybind11::return_value_policy::reference);
                pybind11::object obj;
                (get_element<Is>(s.value,s.index,pybind11::return_value_policy::reference_internal,self,obj), ...);
                s.index++;
                return obj;
            }
        );
    }

    template<SizeType I, SizeType... Js>
    static bool bind_tuple_impl_index(Class& cls,std::index_sequence<Js...> indices){
        using Element = std::tuple_element_t<I,Tuple>;

        auto wrap_i = [](DiffType i) {
            if (i < 0) {
                i += size;
            }
            if (i < 0 || (SizeType) i >= size) {
                throw pybind11::index_error();
            }
            return i;
        };

        cls.def("__contains__",[](const Tuple &v, const Element &x){
            return (... || compare_element<Js>(v,x));
        },pybind11::arg("x"),"Return true the container contains ``x``");

        if constexpr (std::is_copy_constructible<Element>::value){
            cls.def("__setitem__", [indices,wrap_i](Tuple &v, SizeType index, const Element &t) {
                (assign_element<Js>(v,wrap_i(index),t), ...);
            });
        }

        return true;
    }

    template<SizeType I,class Element>
    static bool compare_element(const Tuple &v, const Element& x){
        if constexpr (traits::recursively_comparable_with<Element,std::tuple_element_t<I,Tuple>>){
            return std::get<I>(v)==x;
        }else{
            return false;
        }
    }

    template<SizeType I,class Element>
    static void assign_element(Tuple &v, SizeType index, const Element &t){
        if(I==index){
            if constexpr (std::is_convertible_v<Element,std::tuple_element_t<I,Tuple>>){
                std::get<I>(v)=t;
            }else{
                throw pybind11::type_error();
            }
        }
    }

    template<SizeType I>
    static void assign_elements(Tuple &v, const Tuple &t){
        std::get<I>(v)=std::get<I>(t);
    }
    template<SizeType I>
    static void assign_elements(Tuple &v, SizeType start, SizeType step, SizeType slicelength, const std::vector<pybind11::object> &t){
        if(I>=start && (I-start)%step==0){
            SizeType index=(I-start)/step;
            if(index<slicelength){
                std::get<I>(v)=t[index].cast<std::tuple_element_t<I,Tuple>>();
            }
        }
    }

    template<SizeType I, traits::pyobject P, typename T>
    requires (
        std::same_as<std::remove_cvref_t<T>,Tuple>
        && (
            std::same_as<P,pybind11::list>
            || std::same_as<P,pybind11::tuple>
            || std::same_as<P,pybind11::set>
            || std::same_as<P,pybind11::object>
            || std::same_as<P,pybind11::handle>
        )
    )
    static void cast_element_impl(T&& v, SizeType start, SizeType step, pybind11::return_value_policy policy, pybind11::handle parent, P& obj){
        pybind11::object element;
        using ValueType = std::tuple_element_t<I,Tuple>;
        if constexpr (traits::stl_container<ValueType>){
            element = toPythonContainer(pybind11::detail::forward_like<T>(std::get<I>(std::forward<T>(v))), policy, parent);
        }else{
            element = pybind11::cast(std::get<I>(std::forward<T>(v)),policy,parent);
        }
        if constexpr (std::same_as<P,pybind11::list>){
            SizeType index=(I-start)/step;
            PyList_SET_ITEM(obj.ptr(), index, element.release().ptr());
        }else if constexpr (std::same_as<P,pybind11::tuple>){
            SizeType index=(I-start)/step;
            PyTuple_SET_ITEM(obj.ptr(), index, element.release().ptr());
        }else if constexpr (std::same_as<P,pybind11::set>){
            obj.add(std::move(element));
        }else if constexpr (std::same_as<P,pybind11::object>){
            obj=std::move(element);
        }else{// if constexpr (std::same_as<P,pybind11::handle>){
            obj=element.release();
        }
    }

    template<traits::pyobject P, typename T, SizeType... Is>
    requires (
        std::same_as<std::remove_cvref_t<T>,Tuple>
        && (
            std::same_as<P,pybind11::list>
            || std::same_as<P,pybind11::tuple>
            || std::same_as<P,pybind11::set>
            || std::same_as<P,pybind11::object>
            || std::same_as<P,pybind11::handle>
        )
    )
    static void cast_as_impl(T&& v, pybind11::return_value_policy policy, pybind11::handle parent, P& obj,std::index_sequence<Is...> indices){
        (cast_element_impl<Is>(std::forward<T>(v),0,1,policy,parent,obj), ...);
    }

    template<traits::pyobject P, typename T>
    requires (
        std::same_as<std::remove_cvref_t<T>,Tuple>
        && (
            std::same_as<P,pybind11::list>
            || std::same_as<P,pybind11::tuple>
            || std::same_as<P,pybind11::set>
            || std::same_as<P,pybind11::object>
            || std::same_as<P,pybind11::handle>
        )
    )
    static P cast_as(T&& v, pybind11::return_value_policy policy, pybind11::handle parent){
        auto indices = std::make_index_sequence<size>{};
        if constexpr (
            std::same_as<P,pybind11::list>
            || std::same_as<P,pybind11::tuple>
        ){
            P obj(size);
            cast_as_impl(std::forward<T>(v),policy,parent,obj,indices);
            return std::move(obj);
        }else if constexpr(std::same_as<P,pybind11::set>){
            P obj;
            cast_as_impl(std::forward<T>(v),policy,parent,obj,indices);
            return std::move(obj);
        }else{
            pybind11::tuple obj(size);
            cast_as_impl(std::forward<T>(v),policy,parent,obj,indices);
            if constexpr (std::same_as<P,pybind11::object>){
                return std::move(obj);
            }else{
                return obj.release();
            }
        }
    }

    template<SizeType I, typename T>
    requires (std::same_as<std::remove_cvref_t<T>,Tuple>)
    static void get_element(T&& v, SizeType index, pybind11::return_value_policy policy, pybind11::handle parent, pybind11::object& obj){
        if(I==index){
            cast_element_impl<I>(std::forward<T>(v),0,1,policy,parent,obj);
        }
    }

    template<SizeType I, traits::pyobject P, typename T>
    requires (
        std::same_as<std::remove_cvref_t<T>,Tuple>
        && (
            std::same_as<P,pybind11::list>
            || std::same_as<P,pybind11::tuple>
            || std::same_as<P,pybind11::set>
            || std::same_as<P,pybind11::object>
            || std::same_as<P,pybind11::handle>
        )
    )
    static void get_elements(T&& v, SizeType start, SizeType step, SizeType slicelength, pybind11::return_value_policy policy, pybind11::handle parent, P& obj){
        if(I>=start && (I-start)%step==0){
            SizeType index=(I-start)/step;
            if(index<slicelength){
                return cast_element_impl<I>(std::forward<T>(v),start,step,policy,parent,obj);
            }
        }
    }

    template<SizeType I>
    static void write_element_to_stream(Tuple &v, std::ostream &s){
        if constexpr (traits::writeable_to_stream<std::tuple_element_t<I,Tuple>>){
            s<<std::get<I>(v);
            if (I != size - 1) {
                s << ", ";
            }
        }else{
            throw pybind11::type_error();
        }
    }

    template<SizeType... Is>
    static Class& bind_tuple_impl_sequence(Class& cls, std::string const &name,std::index_sequence<Is...> indices){
        bool ret = (... && bind_tuple_impl_index<Is>(cls,indices));

        auto wrap_i = [](DiffType i) {
            if (i < 0) {
                i += size;
            }
            if (i < 0 || (SizeType) i >= size) {
                throw pybind11::index_error();
            }
            return i;
        };

        if constexpr (std::conjunction_v<std::is_copy_constructible<Ts>...>){
            cls.def(py_init([](const pybind11::iterable& it){

                if(std::is_copy_constructible_v<Tuple> && pybind11::isinstance<Tuple>(it)){
                    // Tupleがコピーコンストラクト可能かつそれを引数として生成する場合は、
                    // コピーコンストラクタを呼ぶ。
                    if constexpr(std::is_copy_constructible_v<Tuple>){
                        const Tuple& value=it.template cast<std::reference_wrapper<Tuple>>().get();
                        auto v = std::unique_ptr<Tuple>(new Tuple(value));
                        return v.release();
                    }
                }else{
                    // そうでなければ、要素ごとにコピーして生成する。
                    auto v = std::unique_ptr<Tuple>(new Tuple());
                    auto hint = len_hint(it);
                    if(hint>0 && hint!=size){
                        throw pybind11::index_error();
                    }
                    std::vector<pybind11::object> elements;
                    elements.reserve(hint);
                    for(pybind11::handle h : it){
                        elements.push_back(pybind11::reinterpret_borrow<pybind11::object>(h));
                    }
                    if(size != elements.size()){
                        throw std::runtime_error(
                            "Left and right hand size of slice assignment have different sizes!");
                    }
                    
                    (assign_elements<Is>(*v,0,1,size,elements), ...);

                    return v.release();
                }
            }), "Copy constructor from python iterable");
            cls.def("__setitem__",
                [](Tuple &v, const pybind11::slice &slice, const pybind11::iterable &it) {
                    size_t start = 0, stop = 0, step = 0, slicelength = 0;
                    if (!slice.compute(size, &start, &stop, &step, &slicelength)) {
                        throw pybind11::error_already_set();
                    }

                    if(pybind11::isinstance<Tuple>(it)){
                        const Tuple& value=it.template cast<std::reference_wrapper<Tuple>>().get();

                        if (slicelength != size) {
                            throw std::runtime_error(
                                "Left and right hand size of slice assignment have different sizes!");
                        }

                        (assign_elements<Is>(v,value), ...);
                    }else{
                        auto hint = len_hint(it);
                        if (hint>0 && slicelength != hint) {
                            throw std::runtime_error(
                                "Left and right hand size of slice assignment have different sizes!");
                        }

                        std::vector<pybind11::object> elements;
                        elements.reserve(hint);
                        for(pybind11::handle h : it){
                            elements.push_back(pybind11::reinterpret_borrow<pybind11::object>(h));
                        }
                        if(slicelength != elements.size()){
                            throw std::runtime_error(
                                "Left and right hand size of slice assignment have different sizes!");
                        }
                    
                        (assign_elements<Is>(v,start,step,slicelength,elements), ...);
                    }
                },
                "Assign tuple elements using a slice object");
        }

        cls.def(
            "__getitem__",
            [wrap_i](const Tuple &v, SizeType i){
                auto self=pybind11::cast(v,pybind11::return_value_policy::reference);
                pybind11::object obj;
                (get_element<Is>(v,wrap_i(i),pybind11::return_value_policy::reference_internal,self,obj), ...);
                return obj;
            }
        );

        cls.def(
            "__getitem__",[](const Tuple &v, const pybind11::slice &slice) -> pybind11::list {
                size_t start = 0, stop = 0, step = 0, slicelength = 0;

                if (!slice.compute(size, &start, &stop, &step, &slicelength)) {
                    throw pybind11::error_already_set();
                }
                auto self=pybind11::cast(v,pybind11::return_value_policy::reference);
                pybind11::list obj(slicelength);
                (get_elements<Is>(v,start,step,slicelength,pybind11::return_value_policy::reference_internal,self,obj), ...);
                return obj;
            },
            pybind11::arg("s"),
            "Retrieve pair elements using a slice object");

        if constexpr (std::conjunction_v<traits::is_writeable_to_stream<Ts>...>){
            cls.def(
                "__repr__",
                [name](Tuple &v) {
                    std::ostringstream s;
                    SizeType i = 0;
                    s << name << '(';
                    (write_element_to_stream<Is>(v,s), ...);
                    s << ')';
                    return s.str();
                },
                "Return the canonical string representation of this set.");
        }

        return cls;
    }
};

//
// pybind11のデフォルトのcasterを共通のIFで呼べるように継承しておく。
//

template<typename T1, typename T2>
struct pybind11_default_type_caster<std::pair<T1,T2>> : pybind11::detail::tuple_caster<std::pair, T1, T2> {};

template<typename ... Ts>
struct pybind11_default_type_caster<std::tuple<Ts...>> : pybind11::detail::tuple_caster<std::tuple, Ts...> {};

template<>
struct pybind11_default_type_caster<std::tuple<>> : pybind11::detail::tuple_caster<std::tuple> {
public:
    // PEP 484 specifies this syntax for an empty tuple
    static constexpr auto name = pybind11::detail::const_name("tuple[()]");
};

template <typename Type, typename Alloc>
struct pybind11_default_type_caster<std::vector<Type, Alloc>> : pybind11::detail::list_caster<std::vector<Type, Alloc>, Type> {};

template <typename Type, typename Alloc>
struct pybind11_default_type_caster<std::deque<Type, Alloc>> : pybind11::detail::list_caster<std::deque<Type, Alloc>, Type> {};

template <typename Type, typename Alloc>
struct pybind11_default_type_caster<std::list<Type, Alloc>> : pybind11::detail::list_caster<std::list<Type, Alloc>, Type> {};

template <typename Type, size_t Size>
struct pybind11_default_type_caster<std::array<Type, Size>>
    : pybind11::detail::array_caster<std::array<Type, Size>, Type, false, Size> {};

template <typename Type>
struct pybind11_default_type_caster<std::valarray<Type>> : pybind11::detail::array_caster<std::valarray<Type>, Type, true> {};

template <typename Key, typename Compare, typename Alloc>
struct pybind11_default_type_caster<std::set<Key, Compare, Alloc>>
    : pybind11::detail::set_caster<std::set<Key, Compare, Alloc>, Key> {};

template <typename Key, typename Hash, typename Equal, typename Alloc>
struct pybind11_default_type_caster<std::unordered_set<Key, Hash, Equal, Alloc>>
    : pybind11::detail::set_caster<std::unordered_set<Key, Hash, Equal, Alloc>, Key> {};

template <typename Key, typename Value, typename Compare, typename Alloc>
struct pybind11_default_type_caster<std::map<Key, Value, Compare, Alloc>>
    : pybind11::detail::map_caster<std::map<Key, Value, Compare, Alloc>, Key, Value> {};

template <typename Key, typename Value, typename Hash, typename Equal, typename Alloc>
struct pybind11_default_type_caster<std::unordered_map<Key, Value, Hash, Equal, Alloc>>
    : pybind11::detail::map_caster<std::unordered_map<Key, Value, Hash, Equal, Alloc>, Key, Value> {};

template <traits::stl_container Container, bool withPlaceHolder>
struct opaque_stl_container_type_caster_base : pybind11::detail::type_caster_base<Container> {
};
template <traits::stl_container Container>
struct opaque_stl_container_type_caster_base<Container,true> : pybind11::detail::type_caster_base<Container> {
    protected:
    Container value;
};

/**
 * @brief STLコンテナを参照渡しでPython側とやり取りするためのtype_caster
 */
template <traits::stl_container Container>
struct opaque_stl_container_type_caster
    : opaque_stl_container_type_caster_base<Container,traits::stl_pair<Container> || traits::stl_tuple<Container>>
{
    using base_caster = pybind11::detail::type_caster_base<Container>;
    using default_caster = pybind11_default_type_caster<Container>;

    static constexpr auto name = base_caster::name;

    bool load(pybind11::handle src, bool convert) {
        if(this->base_caster::load(src,convert)){
            loaded_by_base=true;
            return true;
        }else if(def.load(src,convert)){
            loaded_by_base=false;
            return true;
        }else{
            return false;
        }
    }

    template <typename T>
    using cast_op_type = pybind11::detail::cast_op_type<T>;

    operator Container *()
    {
        if(loaded_by_base){
            return this->base_caster::operator Container*();
        }else{
            if constexpr (traits::stl_pair<Container> || traits::stl_tuple<Container>){
                this->value=def.operator Container();
                return &this->value;
            }else{
                return def.operator Container*();
            }
        }
    }
    operator Container &()
    {
        if(loaded_by_base){
            return this->base_caster::operator Container&();
        }else{
            if constexpr (traits::stl_pair<Container> || traits::stl_tuple<Container>){
                this->value=def.operator Container();
                return this->value;
            }else{
                return def.operator Container&();
            }
        }
    }

protected:
    default_caster def;
    bool loaded_by_base;
};                        

/**
 * @brief STLコンテナを参照渡しにする場合にPython側にクラスを公開するための関数
 * 
 * @details pybind11::bind_vector と pybind11::bind_map の一般化に相当する。
 * 
 *          こちらのオーバーロードはPython側でのクラス名をユーザが指定できる。
 * 
 * @tparam Container 対象のSTLコンテナ
 * @tparam holder_type Pythonオブジェクトとして保持する際のポインタクラス
 * @param [in] scope 公開する先のオブジェクト。モジュールやクラスオブジェクトが該当する。
 * @param [in] name Python側でのクラス名
 * @param [in] args pybind11::class__ の引数に転送するextras (pybind11::module_local() 等)
 */
template<asrc::core::traits::stl_container Container, typename holder_type = std::unique_ptr<Container>, typename ... Args>
requires (
    pybind11::detail::is_default_type_caster_disabled<Container>()
    && std::derived_from<
        pybind11::detail::type_caster<Container>,
        asrc::core::opaque_stl_container_type_caster<Container>
    >
)
pybind11::class_<Container, holder_type> bind_stl_container_name(pybind11::handle scope, std::string const &name, Args &&...args){
    using binder=stl_container_binder<Container,holder_type>;

    try{
        return binder::bind(scope,name,std::forward<Args>(args)...);
    }catch(std::exception& ex){
        if(pybind11::detail::get_type_info(typeid(Container))){
            // already bound
            auto ret=pybind11::reinterpret_borrow<typename binder::Class>(pybind11::detail::get_type_handle(typeid(Container),true));
            assert(ret);
            return std::move(ret);
        }else{
            throw ex;
        }
    }
}

/**
 * @brief STLコンテナを参照渡しにする場合にPython側にクラスを公開するための関数
 * 
 * @details pybind11::bind_vector と pybind11::bind_map の一般化に相当する。
 * 
 *          こちらのオーバーロードはPython側に見せるクラス名を Container の完全修飾名とする。
 *          つまり、::や<>を含むためPython側でのインスタンス生成は不可能となる。
 * 
 * @tparam Container 対象のSTLコンテナ
 * @tparam holder_type Pythonオブジェクトとして保持する際のポインタクラス
 * @param [in] scope 公開する先のオブジェクト。モジュールやクラスオブジェクトが該当する。
 * @param [in] args pybind11::class__ の引数に転送するextras (pybind11::module_local() 等)
 */
template<asrc::core::traits::stl_container Container, typename holder_type = std::unique_ptr<Container>, typename ... Args>
pybind11::class_<Container, holder_type> bind_stl_container(pybind11::handle scope, Args &&...args){
    return bind_stl_container_name<Container,holder_type,Args...>(scope,cereal::util::demangle(typeid(Container).name()),std::forward<Args>(args)...);
}

ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)
namespace detail {
    template <::asrc::core::traits::stl_container Container>
    class type_caster<Container, PYBIND11_ENABLE_IF_DEFAULT_TYPE_CASTER_DISABLED(Container)>
        : public ::asrc::core::opaque_stl_container_type_caster<Container> {};
}
PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)

/**
 * @brief PYBIND11_MAKE_OPAQUEの代替となるASRC_PYBIND11_MAKE_OPAQUEマクロの提供
 * 
 * @param ... STLコンテナの型名(std::vector<int> 等)をそのまま入れる。ネストされたコンテナでもよい。
 *
 * @details
 *      もとのPYBIND11_MAKE_OPAQUEは純PythonオブジェクトからSTLコンテナ(例:listからstd::vector)へのキャストが無効化される。
 *      しかしながら、std::vectorを引数に取るC++関数にPython側からlistを渡せないのは不便であるため、
 *      純PythonオブジェクトからSTLコンテナへの変換機能も残したASRC_PYBIND11_MAKE_OPAQUEマクロを提供する。
 *
 *      - Python→C++へのキャストはsrcの値がC++型だった場合は参照渡し、そうでなければコピーして値渡しとする。
 *      - C++→PythonへのキャストはSTLコンテナのまま、return_value_policyに従い参照渡し又は値渡しとする。
 */
#define ASRC_PYBIND11_MAKE_OPAQUE(...) PYBIND11_DISABLE_DEFAULT_TYPE_CASTER(__VA_ARGS__)
