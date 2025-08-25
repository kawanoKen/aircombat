/**
 * Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
 * @file
 * 
 * @brief Python側に公開可能とするクラスの基底クラスを定義する。
 *
 * @details このクラスを継承すると以下のような特性を有する。
 * 
 * @par protectedなコンストラクタを使用可能
 * 
 *    あるクラスTのコンストラクタがpublicでない場合、make_sharedを使用できない。
 *    Tを継承したクラス内クラス(make_shared_enabler)をprotectedで定義し、そのコンストラクタをpublicとすることで、
 *    make_shared_enablerに対してはmake_sharedを使用できるようになる。
 *    これを用いると、shared_ptrを返すstaticメンバ関数(create)のみをpublicメンバとして、
 *    shared_ptrとしてのインスタンス化をユーザに強制することも可能となる。
 *    
 * @par インターフェースクラスを多重継承した派生クラスをC++側、Python側の両方で定義することが可能
 * 
 *    Python側での仮想関数オーバーライドを簡便に実現するためには"trampoline"クラスを用いることになるが、
 *    trampolineクラスは元のC++クラスをテンプレートパラメータにとってこれを継承する形で実装される。
 *    多重継承に対応するために各trampolineクラスは元のC++クラスをvirtual継承する必要があったが、
 *    virtual継承された派生クラスはstatic_castで基底クラスからダウンキャスト不可能であるため、
 *    pybind11のcastに失敗してしまう。
 *    この対策として、trampolineクラスが使用されるのはPython側でインスタンスが生成されたときのみであることを利用して、
 *    __init__で渡されるselfに相当するpybind11::handleをC++側のprivateメンバとして保持しておきpybind11::castではそれを返すようにした。
 *
 * @par 他のpybind11モジュール内でpybind11::module_local()とともに公開された派生クラスへのダウンキャストが可能
 * 
 *    pybind11::castによる自動ダウンキャストは、それを呼び出したモジュール上で見えているクラスにしか行われない。
 *    そのため、他のモジュールでpybind11::module_local()とされた派生クラスのインスタンスが入っている基底クラスポインタを
 *    適切にダウンキャストしてpybind11::objectを得るためには、派生クラスが公開されたモジュール上で実際のダウンキャスト処理を行わせる必要がある。
 *    これを実現するために、各モジュール上に実体が生成されるような仮想メンバ関数を介してcastを行うような実装とした。
 */
#pragma once
#include "util/macros/common_macros.h"
#include <memory>
#include <concepts>
#include <pybind11/pybind11.h>
#include <cereal/details/util.hpp>
#include "py_init_hook.h"

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)

template<typename type_, typename... options>
class PyClassInfoBase{
    public:
    using PyClassType=pybind11::class_<type_,options...>;
    template <typename... Extra>
    static PyClassType class_(pybind11::handle scope, const char *name, const Extra &...extra) {
        return std::move(PyClassType(scope, name, extra...));
    }
};

/**
 * @brief 多態性を不要とするクラスをPythonに公開する際に継承すべき基底クラス
 */
template<class Base>
class PYBIND11_EXPORT NonPolymorphicPythonableBase{
    public:
    /**
     * @brief Tの型名を表す文字列を返す。
     */
    template<class T=Base> using PyClassInfo=::asrc::core::PyClassInfoBase<T>;
    std::string getDemangledName() const{
        return cereal::util::demangle(typeid(Base).name());
    }
};

/**
 * @brief 多態性を持ったクラスをPythonに公開する際に継承すべき基底クラス
 */
template<class Base>
class PYBIND11_EXPORT PolymorphicPythonableBase{
    public:
    using PtrBaseType = Base; //!< ユーザ定義の、実質的な基底クラス。
    using Type = PolymorphicPythonableBase<Base>;
    PolymorphicPythonableBase():_internal_enforce_type_caster_base(false){}
    virtual ~PolymorphicPythonableBase()=default;

    protected:
    /**
     * @internal
     * @brief [For internal use] 派生クラスTのコンストラクタがpublicでない場合にmake_shared<T>を呼び出し可能とするためのenabler。
     */
    template<std::derived_from<PtrBaseType> T>
    struct make_shared_enabler: T{
        template<typename... Args>
        explicit make_shared_enabler(Args&&... args): T(std::forward<Args>(args)...){}

        virtual const std::type_info* _get_type_info() const{
            return &typeid(T);
        }
        virtual std::string getDemangledName() const{
            return cereal::util::demangle(typeid(T).name());
        }
    };
    /**
     * @internal
     * @brief [For internal use] 派生クラスTのコンストラクタがpublicでない場合にmake_shared<T::TrampolineType>を呼び出し可能とするためのenabler。
     */
    template<std::derived_from<PtrBaseType> T>
    struct make_shared_enabler_for_trampoline: T::template TrampolineType<>{
        template<typename... Args>
        explicit make_shared_enabler_for_trampoline(Args&&... args): T(std::forward<Args>(args)...), T::template TrampolineType<>(std::forward<Args>(args)...){}

        virtual const std::type_info* _get_type_info() const{
            return &typeid(typename T::template TrampolineType<>);
        }
        virtual std::string getDemangledName() const{
            return cereal::util::demangle(typeid(T).name());
        }
    };

    public:
    /**
     * @brief インスタンス生成用のstatic関数。 asrc::core::py_init で使用される。
     * 
     * @details
     *      引数はコンストラクタが受け入れる任意の組み合わせを与えることができる。
     */
    template<std::derived_from<PtrBaseType> T=PtrBaseType, typename... Args>
    requires (
        std::constructible_from<typename T::template TrampolineType<>::template make_shared_enabler_for_trampoline<T>,Args...>
    )
    static std::shared_ptr<T> create(Args&&... args){
        if constexpr (std::constructible_from<T,Args...>){
            // public constructor
            return std::make_shared<T>(std::forward<Args>(args)...);
        }else if constexpr (std::constructible_from<typename T::template make_shared_enabler<T>,Args...>){
            // non-public constructor
            return std::make_shared<typename T::template make_shared_enabler<T>>(std::forward<Args>(args)...);
        }else{
            // pure virtual class
            std::string demangledName = cereal::util::demangle(typeid(T).name());
            throw std::runtime_error("You tried to call constructor of abstract class '"+demangledName+"'. from Python side.");
        }
    }
    /**
     * @brief Trampolineクラスのインスタンス生成用のstatic関数。 asrc::core::py_init で使用される。
     * 
     * @details
     *      引数はコンストラクタが受け入れる任意の組み合わせを与えることができる。
     */
    template<std::derived_from<PtrBaseType> T=PtrBaseType, typename... Args>
    requires (
        std::constructible_from<typename T::template TrampolineType<>::template make_shared_enabler_for_trampoline<T>,Args...>
    )
    static std::shared_ptr<typename T::template TrampolineType<>> create_for_trampoline(Args&&... args){
        if constexpr (std::constructible_from<T,Args...>){
            // public constructor
            return std::make_shared<typename T::template TrampolineType<>>(std::forward<Args>(args)...);
        }else{
            // non-public constructor
            return std::make_shared<typename T::template TrampolineType<>::template make_shared_enabler_for_trampoline<T>>(std::forward<Args>(args)...);
        }
    }

    /**
     * @internal
     * @brief [For internal use] 自身のtype_infoを返す。
     */
    virtual const std::type_info* _get_type_info() const{
        return &typeid(*this);
    }
    /**
     * @brief 自身の型名を表す文字列を返す。
     */
    virtual std::string getDemangledName() const{
        return cereal::util::demangle(typeid(*this).name());
    }
    /**
     * @internal
     * @brief [For internal use] 自身を表すpybind11::handleのsetter。py_initで使用される。
     */
    void setPyHandle(const pybind11::handle& h) const{
        if(h && !_internal_handle){
            pybind11::detail::type_caster_base<PtrBaseType> caster;
            if(caster.load(h,false) && this==(PtrBaseType*)(caster.value)){
                _internal_handle=h;
            }else{
                throw std::runtime_error("Wrong call of PolymorphicPythonableBase::setPyHandle()");
            }
        }
    }
    /**
     * @internal
     * @brief [For internal use] 
     */
    virtual pybind11::handle _call_cast(pybind11::return_value_policy policy, pybind11::handle parent) const{
        if(
            policy == pybind11::return_value_policy::copy
            || policy == pybind11::return_value_policy::move
        ){
            // new instance
            _internal_enforce_type_caster_base=true;
            auto ret=pybind11::cast(this,policy,parent).release();
            _internal_enforce_type_caster_base=false;
            return std::move(ret);
        }else{
            // existing instance
            if(_internal_handle){
                return pybind11::reinterpret_borrow<pybind11::object>(_internal_handle).release();
            }else{
                _internal_enforce_type_caster_base=true;
                auto ret=pybind11::cast(this,policy,parent).release();
                _internal_enforce_type_caster_base=false;
                return std::move(ret);
            }
        }
    }
    /**
     * @internal
     * @brief [For internal use] 
     */
    virtual pybind11::handle _call_cast_holder(const std::shared_ptr<const PtrBaseType>& holder) const{
        // policy = return_value_policy::take_ownership;
        if(_internal_handle){
            return pybind11::reinterpret_borrow<pybind11::object>(_internal_handle).release();
        }else{
            _internal_enforce_type_caster_base=true;
            auto ret=pybind11::cast(holder).release();
            _internal_enforce_type_caster_base=false;
            return std::move(ret);
        }
    }
    /**
     * @internal
     * @brief [For internal use] 
     */
    bool _type_caster_base_enforced() const{
        return _internal_enforce_type_caster_base;
    }
    private:
    mutable pybind11::handle _internal_handle; //!< 自身を表すpybind11::handle。asrc::core::py_initを用いてPython側で生成された場合にセットされる。
    mutable bool _internal_enforce_type_caster_base; //!< 
};

template<class T>
concept non_polymorphic_pythonable = std::derived_from<T,NonPolymorphicPythonableBase<T>>;
template<class T>
concept polymorphic_pythonable = std::derived_from<T,PolymorphicPythonableBase<typename T::PtrBaseType>>;

/**
 * @brief PolymorphicPythonableBaseを継承したクラスTに対して任意のArgs...を引数にとる__init__を登録する。
 * 
 * @details
 *      コンストラクタがpublicでない場合や、抽象クラスである場合にも同じ形で呼び出し可能。
 *      trampolineクラスとの呼び分けも対応している。 
 * 
 *      execute 中の__init__は
 *      pybind11::detail::initimpl::factory<CFunc, AFunc, CReturn(CArgs...), AReturn(AArgs...)>
 *      が行う処理に相当する。
 * 
 *      呼び出すコンストラクタは、Python側での継承の有無によって以下の2種類が使用される。
 *      無-> T::template create<T,Args...>
 *      有-> T::template create_for_trampoline<T,Args...>
 * 
 *      コンストラクタの呼び出し後、selfを表すpybind11::handleをC++側に伝える処理を追加している。
 *      この1行のためにpybind11::initのカスタマイズ用フックを用意した。
*/
template<polymorphic_pythonable T, typename... Args>
struct py_init_constructor_impl<T,Args...> {
    template <typename Class, typename... Extra>
    requires ( std::same_as<T,typename Class::type> )
    void execute(Class &cl, const Extra &...extra) const{
        cl.def(
            "__init__",
            [](pybind11::detail::value_and_holder &v_h, Args... args) {
                if (Py_TYPE(v_h.inst) == v_h.type->type) {
                    pybind11::detail::initimpl::construct<Class>(v_h, T::template create<T,Args...>(std::forward<Args>(args)...), false);
                } else {
                    pybind11::detail::initimpl::construct<Class>(v_h, T::template create_for_trampoline<T,Args...>(std::forward<Args>(args)...), true);
                }

                // selfをsetPyHandleでC++インスタンス側に渡す
                v_h.value_ptr<T>()->setPyHandle((PyObject*)(v_h.inst)); // この行を追加するためにこの構造体を定義した
            },
            pybind11::detail::is_new_style_constructor(),
            extra...);
    }
};

/**
 * @brief std::weak_ptr<T> をPython側に公開する処理を記述する構造体。
 */
template<class T>
struct weak_ptr_exposer {
    /**
     * @brief std::weak_ptr<T> をPython側に公開する。
     * 
     * @attention 公開したい場合は派生クラスでオーバーライドすること。
     * 
     * @param [in,out] m 公開する先のオブジェクト。モジュールやクラスオブジェクトが該当する。
     * @param [in] className Python側でのクラス名
     * @param [in] extras pybind11::class__ の引数に転送するextras (pybind11::module_local() 等)
     */
    template<typename Parent, typename ... Extras>
    static void expose(Parent& m, const char* className, Extras&& ... extras){
        // Make specialized override if you want to expose std::weak_ptr<T> to Python. 
    }
};

/**
 * @brief Tのシリアライゼーション用関数をPython側に公開する処理を記述する構造体。
 */
template<class T>
struct def_common_serialization {
    static constexpr bool op_enable_if_hook = true; //!< これによりcl.defのオーバーロードが実体化され、以下のexecuteを呼べるようになる

    /**
     * @brief Tのシリアライゼーション用関数をPython側に公開する。
     * 
     * @attention 公開したい場合は派生クラスでオーバーライドすること。
     * 
     * @param [in,out] cl Tに対応する pybind11::class_ オブジェクト
     * @param [in] extras cl.def の引数に転送するextras (pybind11::return_value_policy::copy 等)
     */
    template <typename Class, typename... Extra>
    void execute(Class &cl, const Extra &...extra) const{
        // Make specialized overload if you want to serialize some member variables to Python. 
    }
};

/**
 * @brief NonPolymorphicPythonableBase 又は PolymorphicPythonableBase から派生したクラスの共通部分をPython側に公開する関数
 * 
 * @details
 *      pybind11::class_() の呼び出しと同様にクラスオブジェクトを返すので、
 *      pybind11の本来のクラス公開と同様にチェーン方式でメソッドや属性の定義を記述できる。
 */
template<typename T,typename Parent, typename ... Extras>
auto expose_common_class(Parent& m, const char* className, Extras&& ... extras){
    weak_ptr_exposer<T>::expose(m,className,std::forward<Extras>(extras)...);

    using Info=typename T::template PyClassInfo<T>;
    return Info::class_(m,className,pybind11::multiple_inheritance(),std::forward<Extras>(extras)...)
    .def(def_common_serialization<T>())
    ;
}
/**
 * @brief NonPolymorphicPythonableBase 又は PolymorphicPythonableBase から派生したインターフェースクラスの共通部分をPython側に公開する関数
 * 
 * @details
 *      pybind11::class_() の呼び出しと同様にクラスオブジェクトを返すので、
 *      pybind11の本来のクラス公開と同様にチェーン方式でメソッドや属性の定義を記述できる。
 */
template<typename T,typename Parent, typename ... Extras>
auto expose_interface_class(Parent& m, const char* className, Extras&& ... extras){
    return expose_common_class<T>(m,className,std::forward<Extras>(extras)...)
    .def(py_init<>())
    ;
}


ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)

PYBIND11_NAMESPACE_BEGIN(PYBIND11_NAMESPACE)

    /**
     * @brief make_shared_enablerへのダウンキャストを防止するために、その手前で止めるようなpolymorphic_type_hookが必要となる。
     */
    template<class itype>
    struct polymorphic_type_hook<
        itype,
        std::enable_if_t<::asrc::core::polymorphic_pythonable<itype>,void>
    > {
        static const void *get(const itype *src, const std::type_info *&type) {
            type = src ? src->_get_type_info() : nullptr;
            return dynamic_cast<const void *>(src);
        }
    };

namespace detail {
    /**
     * @brief PolymorphicPythonableBase用のtype_caster
     * 
     * @details
     *      Python側で生成されたインスタンスをPython側に戻す際に、
     *      予めメンバ変数として保持しているpybind11::handleを返すようにしたもの。
     */
    template<class type>
    class type_caster<
        type,
        std::enable_if_t<::asrc::core::polymorphic_pythonable<intrinsic_t<type>>,void>
    > : public type_caster_base<type> {
        using base_caster = type_caster_base<type>;
        using itype = intrinsic_t<type>;
        public:
        static handle cast(const itype &src, return_value_policy policy, handle parent) {
            return base_caster::cast(src,policy,parent);
        }
        static handle cast(itype &&src, return_value_policy policy, handle parent) {
            // policy = return_value_policy::move;
            return base_caster::cast(src,policy,parent);
        }
        static handle cast(const itype *src, return_value_policy policy, handle parent) {
            if (src == nullptr) {
                return none().release();
            }
            // srcのメンバ関数を呼び出すことで、srcの実体が定義されたモジュール上でのcastになり、
            // pybind11::module_local()でローカルクラスとして公開されたクラスのpybind11::handleも正しい型情報で取得できるようになる。
            if(src->_type_caster_base_enforced()){
                return base_caster::cast(src,policy,parent);
            }else{
                return src->_call_cast(policy,parent);
            }
        }
        static handle cast_holder(const itype *src, const std::shared_ptr<type>& holder) {
            if (src == nullptr) {
                return none().release();
            }
            // srcのメンバ関数を呼び出すことで、srcの実体が定義されたモジュール上でのcastになり、
            // pybind11::module_local()でローカルクラスとして公開されたクラスのpybind11::handleも正しい型情報で取得できるようになる。
            return src->_call_cast_holder(holder);
        }
    };

    /**
     * @brief shared pointerのtype_casterも上書きする。
     * 元のholder_casterではtype_caster<T>でなくtype_caster_base<T>の関数を直接呼び出してしまうので、
     * ↑で特殊化したtype_casterを経由するように変更する。
     */
    template<::asrc::core::polymorphic_pythonable type>
    class type_caster<std::shared_ptr<type>> : public copyable_holder_caster<type, std::shared_ptr<type>> {
        using holder_type = std::shared_ptr<type>;
        public:
        static handle cast(const holder_type &src, return_value_policy, handle) {
            const auto *ptr = holder_helper<holder_type>::get(src);
            if(src->_type_caster_base_enforced()){
                return type_caster_base<type>::cast_holder(ptr, &src);
            }else{
                return type_caster<type>::cast_holder(ptr, src); // ↑のtype_caster経由で*srcのメンバ関数を呼ぶ
            }
        }
    };

    // PolymorphicPythonableBaseはholderとしてshared_ptrを前提としているため、unique_ptrのcastは対応しない。

}
PYBIND11_NAMESPACE_END(PYBIND11_NAMESPACE)
