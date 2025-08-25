/**
 * Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
 * @file
 * 
 * @brief 全翻訳単位で必ずインクルードすべきヘッダファイル。各プラグインでも同様に必ずインクルードすること。
 */
#pragma once
#if defined(_MSC_VER)
#pragma warning(disable : 4251)
#pragma warning(disable : 4267)
#pragma warning(disable : 4554)
#pragma warning(disable : 4996)
#endif
#include <memory>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <Eigen/Core>
#include <nlohmann/json.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/eigen.h>
#include <pybind11/eigen/tensor.h>
#include "util/macros/macros.h"
#include "Pythonable.h"
namespace py=pybind11;
namespace nl=nlohmann;

#if defined(MODULE_BUILD_IDENTIFIER)
#define ASRC_PLUGIN_NAMESPACE_IMPL(m, b) m::b
#define ASRC_PLUGIN_NAMESPACE_BEGIN_IMPL(m, b) namespace m { inline namespace b {
#define ASRC_PLUGIN_NAMESPACE ASRC_PLUGIN_NAMESPACE_IMPL(MODULE_NAME, MODULE_BUILD_IDENTIFIER)
#define ASRC_PLUGIN_NAMESPACE_BEGIN ASRC_PLUGIN_NAMESPACE_BEGIN_IMPL(MODULE_NAME, MODULE_BUILD_IDENTIFIER)
#define ASRC_PLUGIN_NAMESPACE_END }}
#else
#define ASRC_PLUGIN_NAMESPACE_IMPL(m) m
#define ASRC_PLUGIN_NAMESPACE_BEGIN_IMPL(m) namespace m {
/**
 * @brief このモジュールの、BUILD_IDENTIFIERに依存し得る名前空間を得る。ASRC_PLUGIN_NAMESPACE::fooのように使用可能。
 */
#define ASRC_PLUGIN_NAMESPACE ASRC_PLUGIN_NAMESPACE_IMPL(MODULE_NAME)
/**
 * @brief このモジュールの、BUILD_IDENTIFIERに依存し得る名前空間を開始する。
 */
#define ASRC_PLUGIN_NAMESPACE_BEGIN ASRC_PLUGIN_NAMESPACE_BEGIN_IMPL(MODULE_NAME)
/**
 * @brief このモジュールの、BUILD_IDENTIFIERに依存し得る名前空間を閉じる。
 */
#define ASRC_PLUGIN_NAMESPACE_END }
#endif

#define ASRC_PLUGIN_BII_NAMESPACE_IMPL(m) m
#define ASRC_PLUGIN_BII_NAMESPACE_BEGIN_IMPL(m) namespace m {
/**
 * @brief このモジュールの、BUILD_IDENTIFIERに依存しない名前空間を得る。ASRC_PLUGIN_BII_NAMESPACE::fooのように使用可能。
 */
#define ASRC_PLUGIN_BII_NAMESPACE ASRC_PLUGIN_BII_NAMESPACE_IMPL(MODULE_NAME)
/**
 * @brief このモジュールの、BUILD_IDENTIFIERに依存しない名前空間を開始する。
 */
#define ASRC_PLUGIN_BII_NAMESPACE_BEGIN ASRC_PLUGIN_BII_NAMESPACE_BEGIN_IMPL(MODULE_NAME)
/**
 * @brief このモジュールの、BUILD_IDENTIFIERに依存しない名前空間を閉じる。
 */
#define ASRC_PLUGIN_BII_NAMESPACE_END }

/**
 * @brief name をこのモジュールの名前空間を用いた完全修飾名を文字列化する。
 */
#define ASRC_SYMBOL_FULLNAME_WITH_PLUGIN_NAMESPACE(name) PYBIND11_TOSTRING(ASRC_PLUGIN_NAMESPACE::name)

/**
 * @brief name をこのモジュールの名前空間(BUILD_IDENTIFIERを除く)を用いた完全修飾名を文字列化する。
 */
#define ASRC_SYMBOL_FULLNAME_WITH_PLUGIN_BII_NAMESPACE(name) PYBIND11_TOSTRING(ASRC_PLUGIN_BII_NAMESPACE::name)

/**
 * @internal
 * @brief [For internal use] ASRC_EXPORT_PYTHON_MODULEの実体。モジュール名の設定等を行う。
 */
#define ASRC_EXPORT_PYTHON_MODULE_NAME(rawName,buildName, module, factoryHelper) \
    std::shared_ptr<::asrc::core::FactoryHelper> getModuleFactoryHelper(){\
        static std::shared_ptr<::asrc::core::FactoryHelper> _internal=std::make_shared<::asrc::core::FactoryHelper>(PYBIND11_TOSTRING(rawName));\
        return _internal;\
    }\
    PYBIND11_MODULE(PYBIND11_CONCAT(lib, buildName), module) \
    { \
        module.doc() = PYBIND11_TOSTRING(buildName); \
        std::shared_ptr<::asrc::core::FactoryHelper> factoryHelper=getModuleFactoryHelper(); \
        module.attr("factoryHelper") = factoryHelper; \

/**
 * @~japanese-en
 * @brief pybind11モジュールとしてのこの共有ライブラリのエントリポイント関数を生成するマクロ。
 *        pybind11の元々のPYBIND11_MODULEマクロと同様に、エントリポイント関数の定義が開かれるので、実際の処理をそのまま記述する。
 * 
 * @param module 関数内で使用できる pybind11::module_ 型の変数名。
 * @param factoryHelper 関数内で使用できる std::shared_ptr<asrc::core::FactoryHelper> 型の変数名。
 * 
 * @details 
 *      Pythonで"import FooBar"と書くと、FooBar.soがロードされる。
 *      他方、C++で"FooBar"という名前でリンクする場合libFooBar.soが必要となる。
 * 
 *      本シミュレータでは、ユーザがPython側でもC++側でも同じく"FooBar"という名前で
 *      インポートやリンクが可能となるように、以下のような形式としている。
 *      - "FooBar"という名前のPythonパッケージディレクトリの下に"libFooBar"という名前のpybind11モジュールを
 *      "libFooBar.so"という名前で配置する
 *      - FooBarパッケージの__init__.pyで"from libFooBar import *"によりlibFooBar.soの中身をFooBarにインポートする
 * 
 *      なお、インポート処理は依存するプラグインのロード等を含めて、
 *      @code{.py}
 *      ASRCAISim1.common_init_for_plugin (sys.modules[__name__], globals())
 *      @endcode
 *      を呼び出すことで自動化されているので、プラグイン作成者は.soファイルの配置までを行っておけばよい。
 * 
 *      使用する際は、libFooBar.soのエントリポイントとなるソースファイル(Main.cpp等)において、
 *      @code{.cpp}
 *      ASRC_EXPORT_PYTHON_MODULE(module, factoryHelper)
 *          module.def(...);
 *      }
 *      @endcode
 *      のようにモジュールの初期化処理を記述する。
 * 
 * @~english
 * @brief Macros to open the definition of the entrypoint of this shared library as a pybind11 module.
 *        Same as pybind11's original PYBIND11_MODULE macro, this opens the definition of the entrypoint and write your code under this macro.
 * 
 * @param module Name of variable of type pybind11::module_ available in the function
 * @param factoryHelper Name of variable of type std::shared_ptr<asrc::core::FactoryHelper> available in the function
 * 
 * @details
 *      In Python, "import FooBar" loads FooBar.so.
 *      But in C++, linking "FooBar" requires libFooBar.so.
 *      In order to allow users to import/link by the same name "FooBar" in both languages,
 *      we set Pybind11 module name as "libFooBar" and put the .so file under "FooBar" package directory
 *      and its __init__.py loads the symbols in the module by "from libFooBar import *" into the "FooBar" scope.
 * 
 *      The procedure of .so import is automatically performed by calling
 *      @code{.py}
 *      ASRCAISim1.common_init_for_plugin (sys.modules[__name__], globals())
 *      @endcode
 *      including import of dependency plugins, thus developers of plugins does not import .so manually.
 * 
 *      To use this macro, in the entrypoint source file (Main.cpp etc.) of libFooBar.so, write module initialization function as:
 *      @code{.cpp}
 *      ASRC_EXPORT_PYTHON_MODULE(module, factoryHelper)
 *          module.def(...);
 *      }
 *      @endcode
 */
#if defined(MODULE_BUILD_IDENTIFIER)
#define ASRC_INTERNAL_ACTUAL_MODULE_NAME(a,b) PYBIND11_CONCAT(a,b)
#define ASRC_EXPORT_PYTHON_MODULE(module, factoryHelper) ASRC_EXPORT_PYTHON_MODULE_NAME(MODULE_NAME,ASRC_INTERNAL_ACTUAL_MODULE_NAME(MODULE_NAME,MODULE_BUILD_IDENTIFIER), module, factoryHelper)
#else
#define ASRC_EXPORT_PYTHON_MODULE(module, factoryHelper) ASRC_EXPORT_PYTHON_MODULE_NAME(MODULE_NAME, MODULE_NAME, module, factoryHelper)
#endif

/**
 * @name pybind11におけるSTLコンテナのバインド
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
///@{
//
// opacityの宣言
//
//基本型

ASRC_PYBIND11_MAKE_OPAQUE(std::vector<bool>);
ASRC_PYBIND11_MAKE_OPAQUE(std::vector<std::uint8_t>);
ASRC_PYBIND11_MAKE_OPAQUE(std::vector<std::uint16_t>);
ASRC_PYBIND11_MAKE_OPAQUE(std::vector<std::uint32_t>);
ASRC_PYBIND11_MAKE_OPAQUE(std::vector<std::uint64_t>);
ASRC_PYBIND11_MAKE_OPAQUE(std::vector<std::int8_t>);
ASRC_PYBIND11_MAKE_OPAQUE(std::vector<std::int16_t>);
ASRC_PYBIND11_MAKE_OPAQUE(std::vector<std::int32_t>);
ASRC_PYBIND11_MAKE_OPAQUE(std::vector<std::int64_t>);
ASRC_PYBIND11_MAKE_OPAQUE(std::vector<float>);
ASRC_PYBIND11_MAKE_OPAQUE(std::vector<double>);
ASRC_PYBIND11_MAKE_OPAQUE(std::map<std::string,bool>);
ASRC_PYBIND11_MAKE_OPAQUE(std::map<std::string,std::uint8_t>);
ASRC_PYBIND11_MAKE_OPAQUE(std::map<std::string,std::uint16_t>);
ASRC_PYBIND11_MAKE_OPAQUE(std::map<std::string,std::uint32_t>);
ASRC_PYBIND11_MAKE_OPAQUE(std::map<std::string,std::uint64_t>);
ASRC_PYBIND11_MAKE_OPAQUE(std::map<std::string,std::int8_t>);
ASRC_PYBIND11_MAKE_OPAQUE(std::map<std::string,std::int16_t>);
ASRC_PYBIND11_MAKE_OPAQUE(std::map<std::string,std::int32_t>);
ASRC_PYBIND11_MAKE_OPAQUE(std::map<std::string,std::int64_t>);
ASRC_PYBIND11_MAKE_OPAQUE(std::map<std::string,float>);
ASRC_PYBIND11_MAKE_OPAQUE(std::map<std::string,double>);

//std::string
ASRC_PYBIND11_MAKE_OPAQUE(std::vector<std::string>);
ASRC_PYBIND11_MAKE_OPAQUE(std::map<std::string,std::string>);

//nl::json
ASRC_PYBIND11_MAKE_OPAQUE(std::vector<nl::json>);
ASRC_PYBIND11_MAKE_OPAQUE(std::map<std::string,nl::json>);
ASRC_PYBIND11_MAKE_OPAQUE(std::vector<nl::ordered_json>);
ASRC_PYBIND11_MAKE_OPAQUE(std::map<std::string,nl::ordered_json>);

//boost::uuids::uuid
ASRC_PYBIND11_MAKE_OPAQUE(std::vector<boost::uuids::uuid>);
ASRC_PYBIND11_MAKE_OPAQUE(std::map<std::string,boost::uuids::uuid>);

//Eigen
ASRC_PYBIND11_MAKE_OPAQUE(std::vector<Eigen::VectorXd>);
ASRC_PYBIND11_MAKE_OPAQUE(std::vector<Eigen::VectorXf>);
ASRC_PYBIND11_MAKE_OPAQUE(std::vector<Eigen::VectorXi>);
ASRC_PYBIND11_MAKE_OPAQUE(std::vector<Eigen::Vector4d>);
ASRC_PYBIND11_MAKE_OPAQUE(std::vector<Eigen::Vector4f>);
ASRC_PYBIND11_MAKE_OPAQUE(std::vector<Eigen::Vector4i>);
ASRC_PYBIND11_MAKE_OPAQUE(std::vector<Eigen::Vector3d>);
ASRC_PYBIND11_MAKE_OPAQUE(std::vector<Eigen::Vector3f>);
ASRC_PYBIND11_MAKE_OPAQUE(std::vector<Eigen::Vector3i>);
ASRC_PYBIND11_MAKE_OPAQUE(std::vector<Eigen::Vector2d>);
ASRC_PYBIND11_MAKE_OPAQUE(std::vector<Eigen::Vector2f>);
ASRC_PYBIND11_MAKE_OPAQUE(std::vector<Eigen::Vector2i>);
ASRC_PYBIND11_MAKE_OPAQUE(std::vector<Eigen::MatrixXd>);
ASRC_PYBIND11_MAKE_OPAQUE(std::vector<Eigen::MatrixXf>);
ASRC_PYBIND11_MAKE_OPAQUE(std::vector<Eigen::MatrixXi>);
ASRC_PYBIND11_MAKE_OPAQUE(std::vector<Eigen::Matrix4d>);
ASRC_PYBIND11_MAKE_OPAQUE(std::vector<Eigen::Matrix4f>);
ASRC_PYBIND11_MAKE_OPAQUE(std::vector<Eigen::Matrix4i>);
ASRC_PYBIND11_MAKE_OPAQUE(std::vector<Eigen::Matrix3d>);
ASRC_PYBIND11_MAKE_OPAQUE(std::vector<Eigen::Matrix3f>);
ASRC_PYBIND11_MAKE_OPAQUE(std::vector<Eigen::Matrix3i>);
ASRC_PYBIND11_MAKE_OPAQUE(std::vector<Eigen::Matrix2d>);
ASRC_PYBIND11_MAKE_OPAQUE(std::vector<Eigen::Matrix2f>);
ASRC_PYBIND11_MAKE_OPAQUE(std::vector<Eigen::Matrix2i>);
ASRC_PYBIND11_MAKE_OPAQUE(std::map<std::string,Eigen::VectorXd>);
ASRC_PYBIND11_MAKE_OPAQUE(std::map<std::string,Eigen::VectorXf>);
ASRC_PYBIND11_MAKE_OPAQUE(std::map<std::string,Eigen::VectorXi>);
ASRC_PYBIND11_MAKE_OPAQUE(std::map<std::string,Eigen::Vector4d>);
ASRC_PYBIND11_MAKE_OPAQUE(std::map<std::string,Eigen::Vector4f>);
ASRC_PYBIND11_MAKE_OPAQUE(std::map<std::string,Eigen::Vector4i>);
ASRC_PYBIND11_MAKE_OPAQUE(std::map<std::string,Eigen::Vector3d>);
ASRC_PYBIND11_MAKE_OPAQUE(std::map<std::string,Eigen::Vector3f>);
ASRC_PYBIND11_MAKE_OPAQUE(std::map<std::string,Eigen::Vector3i>);
ASRC_PYBIND11_MAKE_OPAQUE(std::map<std::string,Eigen::Vector2d>);
ASRC_PYBIND11_MAKE_OPAQUE(std::map<std::string,Eigen::Vector2f>);
ASRC_PYBIND11_MAKE_OPAQUE(std::map<std::string,Eigen::Vector2i>);
ASRC_PYBIND11_MAKE_OPAQUE(std::map<std::string,Eigen::MatrixXd>);
ASRC_PYBIND11_MAKE_OPAQUE(std::map<std::string,Eigen::MatrixXf>);
ASRC_PYBIND11_MAKE_OPAQUE(std::map<std::string,Eigen::MatrixXi>);
ASRC_PYBIND11_MAKE_OPAQUE(std::map<std::string,Eigen::Matrix4d>);
ASRC_PYBIND11_MAKE_OPAQUE(std::map<std::string,Eigen::Matrix4f>);
ASRC_PYBIND11_MAKE_OPAQUE(std::map<std::string,Eigen::Matrix4i>);
ASRC_PYBIND11_MAKE_OPAQUE(std::map<std::string,Eigen::Matrix3d>);
ASRC_PYBIND11_MAKE_OPAQUE(std::map<std::string,Eigen::Matrix3f>);
ASRC_PYBIND11_MAKE_OPAQUE(std::map<std::string,Eigen::Matrix3i>);
ASRC_PYBIND11_MAKE_OPAQUE(std::map<std::string,Eigen::Matrix2d>);
ASRC_PYBIND11_MAKE_OPAQUE(std::map<std::string,Eigen::Matrix2f>);
ASRC_PYBIND11_MAKE_OPAQUE(std::map<std::string,Eigen::Matrix2i>);

//Others
//for EntityManager
ASRC_PYBIND11_MAKE_OPAQUE(std::map<std::int32_t,boost::uuids::uuid>);
ASRC_PYBIND11_MAKE_OPAQUE(std::map<boost::uuids::uuid,std::int32_t>);
//for SimulationManager
ASRC_PYBIND11_MAKE_OPAQUE(std::map<std::string,std::map<std::string,double>>);
//for Fighter
ASRC_PYBIND11_MAKE_OPAQUE(std::vector<std::vector<std::string>>);
ASRC_PYBIND11_MAKE_OPAQUE(std::deque<std::pair<double,nl::json>>);
///@}

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)

/**
 * @brief 自作クラスではない要素型を持つSTLコンテナのうち頻出のもののバインドを行う。
 * 
 * @details
 *      py::module_local(true)とするものはモジュールごとにコンパイルする必要があるため、各プラグインのエントリポイントからも呼び出す。
 *      各モジュール上で実行されるようにするために、テンプレート関数として実装している。
 */
template<typename Dummy=void>
void PYBIND11_EXPORT bindSTLContainer(py::module &m){
    //基本型
    bind_stl_container<std::vector<bool>>(m,py::module_local(true));
    bind_stl_container<std::vector<std::uint8_t>>(m,py::module_local(true));
    bind_stl_container<std::vector<std::uint16_t>>(m,py::module_local(true));
    bind_stl_container<std::vector<std::uint32_t>>(m,py::module_local(true));
    bind_stl_container<std::vector<std::uint64_t>>(m,py::module_local(true));
    bind_stl_container<std::vector<std::int8_t>>(m,py::module_local(true));
    bind_stl_container<std::vector<std::int16_t>>(m,py::module_local(true));
    bind_stl_container<std::vector<std::int32_t>>(m,py::module_local(true));
    bind_stl_container<std::vector<std::int64_t>>(m,py::module_local(true));
    bind_stl_container<std::vector<float>>(m,py::module_local(true));
    bind_stl_container<std::vector<double>>(m,py::module_local(true));
    bind_stl_container<std::map<std::string,bool>>(m,py::module_local(true));
    bind_stl_container<std::map<std::string,std::uint8_t>>(m,py::module_local(true));
    bind_stl_container<std::map<std::string,std::uint16_t>>(m,py::module_local(true));
    bind_stl_container<std::map<std::string,std::uint32_t>>(m,py::module_local(true));
    bind_stl_container<std::map<std::string,std::uint64_t>>(m,py::module_local(true));
    bind_stl_container<std::map<std::string,std::int8_t>>(m,py::module_local(true));
    bind_stl_container<std::map<std::string,std::int16_t>>(m,py::module_local(true));
    bind_stl_container<std::map<std::string,std::int32_t>>(m,py::module_local(true));
    bind_stl_container<std::map<std::string,std::int64_t>>(m,py::module_local(true));
    bind_stl_container<std::map<std::string,float>>(m,py::module_local(true));
    bind_stl_container<std::map<std::string,double>>(m,py::module_local(true));

    //std::string
    bind_stl_container<std::vector<std::string>>(m,py::module_local(true));
    bind_stl_container<std::map<std::string,std::string>>(m,py::module_local(true));

    //nl::json
    bind_stl_container<std::vector<nl::json>>(m,py::module_local(true));
    bind_stl_container<std::map<std::string,nl::json>>(m,py::module_local(true));
    bind_stl_container<std::vector<nl::ordered_json>>(m,py::module_local(true));
    bind_stl_container<std::map<std::string,nl::ordered_json>>(m,py::module_local(true));

    //Eigen
    bind_stl_container<std::vector<Eigen::VectorXd>>(m,py::module_local(true));
    bind_stl_container<std::vector<Eigen::VectorXf>>(m,py::module_local(true));
    bind_stl_container<std::vector<Eigen::VectorXi>>(m,py::module_local(true));
    bind_stl_container<std::vector<Eigen::Vector4d>>(m,py::module_local(true));
    bind_stl_container<std::vector<Eigen::Vector4f>>(m,py::module_local(true));
    bind_stl_container<std::vector<Eigen::Vector4i>>(m,py::module_local(true));
    bind_stl_container<std::vector<Eigen::Vector3d>>(m,py::module_local(true));
    bind_stl_container<std::vector<Eigen::Vector3f>>(m,py::module_local(true));
    bind_stl_container<std::vector<Eigen::Vector3i>>(m,py::module_local(true));
    bind_stl_container<std::vector<Eigen::Vector2d>>(m,py::module_local(true));
    bind_stl_container<std::vector<Eigen::Vector2f>>(m,py::module_local(true));
    bind_stl_container<std::vector<Eigen::Vector2i>>(m,py::module_local(true));
    bind_stl_container<std::vector<Eigen::MatrixXd>>(m,py::module_local(true));
    bind_stl_container<std::vector<Eigen::MatrixXf>>(m,py::module_local(true));
    bind_stl_container<std::vector<Eigen::MatrixXi>>(m,py::module_local(true));
    bind_stl_container<std::vector<Eigen::Matrix4d>>(m,py::module_local(true));
    bind_stl_container<std::vector<Eigen::Matrix4f>>(m,py::module_local(true));
    bind_stl_container<std::vector<Eigen::Matrix4i>>(m,py::module_local(true));
    bind_stl_container<std::vector<Eigen::Matrix3d>>(m,py::module_local(true));
    bind_stl_container<std::vector<Eigen::Matrix3f>>(m,py::module_local(true));
    bind_stl_container<std::vector<Eigen::Matrix3i>>(m,py::module_local(true));
    bind_stl_container<std::vector<Eigen::Matrix2d>>(m,py::module_local(true));
    bind_stl_container<std::vector<Eigen::Matrix2f>>(m,py::module_local(true));
    bind_stl_container<std::vector<Eigen::Matrix2i>>(m,py::module_local(true));
    bind_stl_container<std::map<std::string,Eigen::VectorXd>>(m,py::module_local(true));
    bind_stl_container<std::map<std::string,Eigen::VectorXf>>(m,py::module_local(true));
    bind_stl_container<std::map<std::string,Eigen::VectorXi>>(m,py::module_local(true));
    bind_stl_container<std::map<std::string,Eigen::Vector4d>>(m,py::module_local(true));
    bind_stl_container<std::map<std::string,Eigen::Vector4f>>(m,py::module_local(true));
    bind_stl_container<std::map<std::string,Eigen::Vector4i>>(m,py::module_local(true));
    bind_stl_container<std::map<std::string,Eigen::Vector3d>>(m,py::module_local(true));
    bind_stl_container<std::map<std::string,Eigen::Vector3f>>(m,py::module_local(true));
    bind_stl_container<std::map<std::string,Eigen::Vector3i>>(m,py::module_local(true));
    bind_stl_container<std::map<std::string,Eigen::Vector2d>>(m,py::module_local(true));
    bind_stl_container<std::map<std::string,Eigen::Vector2f>>(m,py::module_local(true));
    bind_stl_container<std::map<std::string,Eigen::Vector2i>>(m,py::module_local(true));
    bind_stl_container<std::map<std::string,Eigen::MatrixXd>>(m,py::module_local(true));
    bind_stl_container<std::map<std::string,Eigen::MatrixXf>>(m,py::module_local(true));
    bind_stl_container<std::map<std::string,Eigen::MatrixXi>>(m,py::module_local(true));
    bind_stl_container<std::map<std::string,Eigen::Matrix4d>>(m,py::module_local(true));
    bind_stl_container<std::map<std::string,Eigen::Matrix4f>>(m,py::module_local(true));
    bind_stl_container<std::map<std::string,Eigen::Matrix4i>>(m,py::module_local(true));
    bind_stl_container<std::map<std::string,Eigen::Matrix3d>>(m,py::module_local(true));
    bind_stl_container<std::map<std::string,Eigen::Matrix3f>>(m,py::module_local(true));
    bind_stl_container<std::map<std::string,Eigen::Matrix3i>>(m,py::module_local(true));
    bind_stl_container<std::map<std::string,Eigen::Matrix2d>>(m,py::module_local(true));
    bind_stl_container<std::map<std::string,Eigen::Matrix2f>>(m,py::module_local(true));
    bind_stl_container<std::map<std::string,Eigen::Matrix2i>>(m,py::module_local(true));

    //Others
    //for EntityManager
    bind_stl_container<std::map<std::int32_t,boost::uuids::uuid>>(m,py::module_local(true));
    bind_stl_container<std::map<boost::uuids::uuid,std::int32_t>>(m,py::module_local(true));
    //for SimulationManager
    bind_stl_container<std::map<std::string,std::map<std::string,double>>>(m,py::module_local(true));
    //for Fighter
    bind_stl_container<std::vector<std::vector<std::string>>>(m,py::module_local(true));
    bind_stl_container<std::deque<std::pair<double,nl::json>>>(m,py::module_local(true));
}

ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
