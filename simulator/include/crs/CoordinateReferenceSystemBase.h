/**
 * Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
 * @file
 * 
 * @brief 座標系を表す基底クラス。シミュレーション登場物とは言い難いものの、Entityを継承して実装する。
 * 
 * @note 完全ではないがISO 19111に準拠することを目指している。
 *       シミュレータの本体部分としては必要最小限の座標系のみを定義する。
 *       より多様な座標系(PROJ等の外部ライブラリも使用可)はプラグインによって追加する。
 */
#pragma once
#include "../Common.h"
#include "../Factory.h"
#include "../Entity.h"
#include "../Quaternion.h"
#include "../MathUtility.h"
#include "../Coordinate.h"
#include <iostream>
#include <sstream>
#include <tuple>
#include <shared_mutex>

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)

class MotionState; // forward declaration
class AffineCRS;
class TopocentricCRS;
class ProjectedCRS;

/**
 * @brief 座標系の種類を表す列挙体。
 */
enum class CRSType{
    ECEF, //!< 地球中心、地球固定の座標系。Earth-Centric Earth-Fixedの略。
    GEOGRAPHIC, //!< 緯度経度で表す地理座標系。
    ECI, //!< 地球中心だが地球とともに回転しない座標系 Earth-Centered Inertiaの略。
    PUREFLAT, //!< 仮想的な平らな地球を仮定した座標系。ISO 19111におけるEngineering CRSに相当。
    AFFINE, //!< 他の座標系からアフィン変換を行った座標系。
    PROJECTED, //!< 2次元平面に投影された座標系。
    TOPOCENTRIC, //!< 局所水平座標系。
    INVALID //!< 無効な座標系を表す。
};
DEFINE_SERIALIZE_ENUM_AS_STRING(CRSType)

//
///@name Axis order utility
///@{
/**
 * @brief 球座標系の軸順を表す文字列かどうかを判定する。
 * 
 * @details 以下の3種類の軸を表す文字が一つずつ含まれている場合にtrueとなる。
 *      1. 経度方向
 *          - 'A' Azimuth
 *      2. 緯度方向
 *          - 'E' Elevation
 *      3. 距離(以下のいずれか。意味は同じ)
 *          - 'D' Distance
 *          - 'R' Range
 */
bool PYBIND11_EXPORT checkSphericalAxisOrder(const std::string& order);

/**
 * @brief body座標系の直交座標軸順を表す文字列かどうかを判定する。
 * 
 * @details 以下の3種類の軸を表す文字が一つずつ含まれている場合にtrueとなる。
 *      1. 前後方向
 *          - 'F' FWD 前方が正の軸
 *          - 'A' AFT 後方が正の軸
 *      2. 左右方向
 *          - 'S' STBD(RIGHT) 右方が正の軸
 *          - 'R' RIGHT 右方が正の軸
 *          - 'P' PORT(LEFT) 左方が正の軸
 *          - 'L' LEFT 左方が正の軸
 *      3. 上下方向
 *          - 'U' UP 上方が正の軸
 *          - 'D' DOWN 下方が正の軸
 */
bool PYBIND11_EXPORT checkBodyCartesianAxisOrder(const std::string& order);

/**
 * @brief 局所水平(Topocentric)座標系の直交座標軸順を表す文字列かどうかを判定する。
 * 
 * @details 以下の3種類の軸を表す文字が一つずつ含まれている場合にtrueとなる。
 *      1. 南北方向
 *          - 'N' North 北が正の軸
 *          - 'S' South 南が正の軸
 *      2. 東西方向
 *          - 'E' East 東が正の軸
 *          - 'W' West 西が正の軸
 *      3. 上下方向
 *          - 'U' UP 鉛直上方が正の軸
 *          - 'D' DOWN 鉛直下方が正の軸
 */
bool PYBIND11_EXPORT checkTopocentricCartesianAxisOrder(const std::string& order);
/**
 * @brief 同じ種類の座標系の座標軸について、軸の順番と符号を入れ替える座標変換行列を返す。
 * @param [in] srcOrder 変換元の軸順
 * @param [in] dstOrder 変換先の軸順
 */
Eigen::Matrix3d PYBIND11_EXPORT getAxisSwapRotationMatrix(const std::string& srcOrder, const std::string& dstOrder);

/**
 * @brief 同じ種類の座標系の座標軸について、軸の順番と符号を入れ替える座標変換クォータニオンを返す。
 * @param [in] srcOrder 変換元の軸順
 * @param [in] dstOrder 変換先の軸順
 */
Quaternion PYBIND11_EXPORT getAxisSwapQuaternion(const std::string& srcOrder, const std::string& dstOrder);
///@}

/**
 * @class CoordinateReferenceSystem
 * @brief 座標系を表す基底クラス
 * 
 * @note Factory経由で生成可能とするためにEntityから派生している。
 */
ASRC_DECLARE_DERIVED_CLASS_WITH_TRAMPOLINE(CoordinateReferenceSystem,Entity)
    protected:
    CRSType crsType; //!< 
    mutable std::recursive_mutex mtx;
    mutable bool _isValid; //!< [For internal use] 内部変数が適切に設定されているかどうか
    mutable bool _hasBeenValidated;
    mutable std::uint64_t _updateCount; //!< [For internal use] これまでに invalidate が呼ばれた回数
    mutable std::vector<std::weak_ptr<CoordinateReferenceSystem>> derivedCRSes; //!< このCRSをbase(parent)として持つCRSのリスト
    public:
    static const std::string baseName; // baseNameを上書き
    CoordinateReferenceSystem(const nl::json& modelConfig_,const nl::json& instanceConfig_);
    CRSType getCRSType() const; //!< 座標系の種類を返す。
    // 内部状態のシリアライゼーション
    virtual void serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full) override;
    //
    public:
    virtual bool isValid() const; //!< [For internal use] 内部変数が適切に設定されているかどうかを返す。
    virtual bool hasBeenValidated() const; //!< [For internal use] 一度でも validate が呼ばれたことがあるかどうかを返す。
    virtual void validate() override; //!< [For internal use] Entity から継承。派生クラスでは基本的にconstメンバの方をオーバーライドすること。
    virtual void validate() const; //!< [For internal use] const版の validate 。派生クラスでは基本的にこちらをオーバーライドすること。
    std::uint64_t getUpdateCount() const; //!< [For internal use] これまでに invalidate が呼ばれた回数を返す。
    bool isSameCRS(const std::shared_ptr<CoordinateReferenceSystem>& other) const; //!< 引数が自身と同一かどうかを返す。shared_from_thisがEntityへのポインタとなってしまうためこの関数を用意する。
    virtual bool isEquivalentCRS(const std::shared_ptr<CoordinateReferenceSystem>& other) const; //!< 引数が自身と等価な(座標変換が恒等式となる)CRSかどうかを返す。派生クラスでカスタマイズ可能。
    protected:
    /**
     * @internal
     * @brief [For internal use] shared_from_this()をnon-const Tにキャストして返すラッパー
     */
    template<std::derived_from<CoordinateReferenceSystem> T=CoordinateReferenceSystem>
    std::shared_ptr<T> non_const_this() const{
        auto non_const_base=std::const_pointer_cast<PtrBaseType>(shared_from_this());
        if constexpr(std::same_as<T,CoordinateReferenceSystem>){
            return std::static_pointer_cast<T>(non_const_base);
        }else{
            return std::dynamic_pointer_cast<T>(non_const_base);
        }
    }
    virtual void validateImpl(bool isFirstTime) const; //!< [For internal use] 内部状態の更新処理を行う実体。
    virtual void invalidate() const; //!< [For internal use] 座標演算に必要な内部状態が変化したときに呼び出す。
    virtual void addDerivedCRS(const std::weak_ptr<CoordinateReferenceSystem>& derived) const; //!< [For internal use] 自身から派生した座標系を追加する。
    virtual void removeDerivedCRS(const std::weak_ptr<CoordinateReferenceSystem>& derived) const; //!< [For internal use] 自身から派生した座標系を削除する。

    public:
    ///@name 座標系の種類の判定
    ///@{
    virtual bool isEarthFixed() const; //!< 地球に対して固定されているかどうか
    virtual bool isGeocentric() const; //!< 地球を中心とした座標系かどうか
    virtual bool isGeographic() const; //!< 地理座標系(lat-lon-alt)かどうか
    virtual bool isDerived() const; //!< DerivedCRSかどうか(他の座標系から派生したものかどうか)
    virtual bool isTopocentric() const; //!< 局所水平座標系かどうか
    virtual bool isProjected() const; //!< 投影された座標系かどうか(ProjectedCRS又はProjectedCRSをbaseとしたDerivedCRSが該当)
    ///@}
    ///@name Coordinate system axisの種類
    ///@{
    virtual bool isCartesian() const=0; //!< 直交座標系(geocentric XYZ又はnorthing/southing,easting/westing,ellipsoidal height)
    virtual bool isSpherical() const=0; //!< 球座標系(Azimuth-Elevation-Range)
    virtual bool isEllipsoidal() const=0; //!< 楕円座標系(latitude-Longitude-Altitude)
    ///@}
    ///@name 親座標系の取得(共通インターフェースとする)
    ///@{
	virtual std::shared_ptr<CoordinateReferenceSystem> getBaseCRS() const; //!< すぐ上位の親座標系。親がいなければ自分自身。
	virtual std::shared_ptr<CoordinateReferenceSystem> getNonDerivedBaseCRS() const; //!< 最上位の親座標系。親がいなければ自分自身。
	virtual std::shared_ptr<ProjectedCRS> getBaseProjectedCRS() const; //!< 親座標系を辿ってProjectedCRSに当たった場合、そのProjectedCRS。なければnullptr。自分自身がProjectedCRSの場合は自分自身。
    ///@}
    ///@name 中間座標系の取得
    ///@{
	virtual std::shared_ptr<CoordinateReferenceSystem> getIntermediateCRS() const; //!< 親子関係にない他のCRSとの間で座標変換を行う際に経由する中間座標系。なければ自分自身。
    ///@}

    ///@name CRSインスタンス間の変換可否
    ///@{
    public:
    virtual bool isTransformableTo(const std::shared_ptr<CoordinateReferenceSystem>& other) const=0; //!< 自身からotherへの座標変換が可能かどうかを返す。
    virtual bool isTransformableFrom(const std::shared_ptr<CoordinateReferenceSystem>& other) const=0; //!< otherから自身への座標変換が可能かどうかを返す。
    ///@}

    /** @name 座標変換(共通インターフェース)
     * ある絶対位置座標locationにおいて観測された何らかの座標値valueを座標変換するための関数群。
     * 
     * 多くの場合は変換結果が観測点の位置に依存しないため省略したインターフェースも設ける。省略した場合は原点における観測値として扱う。
     * 観測点に依存する例としては、回転する座標系において原点以外の点にある物体の速度を観測した場合や、座標軸が一様でない場合(測地座標系や投影座標系等)が挙げられる。
     * 
     * 実際の変換処理は派生クラス側でEigen::Vector3dを引数にとる省略無しの関数で実装することを想定している。
     * 
     * ユーザ側は、Coordinate型の引数をとるstatic関数を利用することを推奨する。
     */
    ///@{
    public:

    /**
     * @brief 位置locationで観測された値valueをdstCRSにおける座標に変換する
     * @param [in] value    変換対象の座標
     * @param [in] location valueが観測された位置
     * @param [in] dstCRS   変換先のCRS
     * 
     * @details value自身も Coordinate 型のためlocationの情報を持っているが、こちらの関数オーバーロードでは引数のlocationが優先して使用される。
     */
    Coordinate transformTo(const Coordinate& value, const Coordinate& location, const std::shared_ptr<CoordinateReferenceSystem>& dstCRS) const;

    /**
     * @brief 位置locationで観測された値valueをこのCRSにおける座標に変換する
     * @param [in] value    変換対象の座標
     * @param [in] location valueが観測された位置
     * 
     * @details value自身も Coordinate 型のためlocationの情報を持っているが、こちらの関数オーバーロードでは引数のlocationが優先して使用される。
     */
    Coordinate transformFrom(const Coordinate& value, const Coordinate& location) const;

    /**
     * @brief valueをdstCRSにおける座標に変換する
     * @param [in] value    変換対象の座標
     * @param [in] dstCRS   変換先のCRS
     * 
     * @details value自身が保持しているlocationの情報を使用する。
     */
    Coordinate transformTo(const Coordinate& value, const std::shared_ptr<CoordinateReferenceSystem>& dstCRS) const;

    /**
     * @brief valueをこのCRSにおける座標に変換する
     * @param [in] value    変換対象の座標
     * 
     * @details value自身が保持しているlocationの情報を使用する。
     */
    Coordinate transformFrom(const Coordinate& value) const;

    /**
     * @brief このCRSにおいて位置locationで観測された値valueをdstCRSにおける座標に変換する
     * @param [in] value          変換対象の座標
     * @param [in] location       valueが観測された位置
     * @param [in] time           valueが観測された時刻
     * @param [in] dstCRS         変換先のCRS
     * @param [in] coordinateType 座標の種類
     * 
     * @details 派生クラスはこれをオーバーライドして実際の座標変換処理を記述すること。
     * 
     */
    virtual Coordinate transformTo(const Eigen::Vector3d& value, const Eigen::Vector3d& location, const Time& time, const std::shared_ptr<CoordinateReferenceSystem>& dstCRS, const CoordinateType& coordinateType) const=0;

    /**
     * @brief srcCRSにおいて位置locationで観測された値valueをこのCRSにおける座標に変換する
     * @param [in] value          変換対象の座標
     * @param [in] location       valueが観測された位置
     * @param [in] time           valueが観測された時刻
     * @param [in] srcCRS         変換元のCRS
     * @param [in] coordinateType 座標の種類
     * 
     * @details 派生クラスはこれをオーバーライドして実際の座標変換処理を記述すること。
     */
    virtual Coordinate transformFrom(const Eigen::Vector3d& value, const Eigen::Vector3d& location, const Time& time, const std::shared_ptr<CoordinateReferenceSystem>& srcCRS, const CoordinateType& coordinateType) const=0;

    /**
     * @brief このCRSにおいて観測された値valueをdstCRSにおける座標に変換する
     * @param [in] value          変換対象の座標
     * @param [in] time           valueが観測された時刻
     * @param [in] dstCRS         変換先のCRS
     * @param [in] coordinateType 座標の種類
     * 
     * @details locationは原点(0,0,0)とみなす。
     */
    Coordinate transformTo(const Eigen::Vector3d& value, const Time& time, const std::shared_ptr<CoordinateReferenceSystem>& dstCRS, const CoordinateType& coordinateType) const;

    /**
     * @brief srcCRSにおいて観測された値valueをこのCRSにおける座標に変換する
     * @param [in] value          変換対象の座標
     * @param [in] time           valueが観測された時刻
     * @param [in] srcCRS         変換元のCRS
     * @param [in] coordinateType 座標の種類
     * 
     * @details locationは原点(0,0,0)とみなす。
     */
    Coordinate transformFrom(const Eigen::Vector3d& value, const Time& time, const std::shared_ptr<CoordinateReferenceSystem>& srcCRS, const CoordinateType& coordinateType) const;
    ///@}

    ///@name 座標変換用クォータニオンの取得
    ///@{
    Quaternion getQuaternionTo(const Coordinate& location, const std::shared_ptr<CoordinateReferenceSystem>& dstCRS) const; //!< 基準点locationにおいてこのCRSの座標をdstCRSの座標に変換するクォータニオンを得る。
    Quaternion getQuaternionFrom(const Coordinate& location) const; //!< 基準点locationにおいてlocationが属する座標系からこのCRSの座標に変換するクォータニオンを得る。
    virtual Quaternion getQuaternionTo(const Eigen::Vector3d& location, const Time& time, const std::shared_ptr<CoordinateReferenceSystem>& dstCRS) const=0; //!< ある時刻timeの基準点locationにおいてこのCRSの座標をdstCRSの座標に変換するクォータニオンを得る。
    virtual Quaternion getQuaternionFrom(const Eigen::Vector3d& location, const Time& time, const std::shared_ptr<CoordinateReferenceSystem>& srcCRS) const=0; //!< ある時刻timeの基準点locationにおいてsrcCRSの座標をこのCRSの座標に変換するクォータニオンを得る。
    Quaternion transformQuaternionTo(const Quaternion& q, const Coordinate& location, const std::shared_ptr<CoordinateReferenceSystem>& dstCRS) const; //!< 基準点locationにおいてこのCRSからの変換を表すクォータニオンをdstCRSからの変換を表すクォータニオンに変換する。
    Quaternion transformQuaternionFrom(const Quaternion& q, const Coordinate& location) const; //!< 基準点locationにおいてlocationが属する座標系からの変換を表すクォータニオンをこのCRSからの変換を表すクォータニオンに変換する。
    virtual Quaternion transformQuaternionTo(const Quaternion& q, const Eigen::Vector3d& location, const Time& time, const std::shared_ptr<CoordinateReferenceSystem>& dstCRS) const; //!< ある時刻timeの基準点locationにおいてこのCRSからの変換を表すクォータニオンをdstCRSからの変換を表すクォータニオンに変換する。
    virtual Quaternion transformQuaternionFrom(const Quaternion& q, const Eigen::Vector3d& location, const Time& time, const std::shared_ptr<CoordinateReferenceSystem>& srcCRS) const; //!< ある時刻timeの基準点locationにおいてsrcCRSからの変換を表すクォータニオンをこのCRSからの変換を表すクォータニオンに変換する。
    ///@}

    /// @name「高度」の取得
    ///@{
    public:
    double getHeight(const Coordinate& location) const; //!< 楕円体高度(幾何高度)を返す。ellipsoid (geometric) height
    double getGeoidHeight(const Coordinate& location) const; //!< 標高(ジオイド高度)を返す。geoid height (elevation)
    virtual double getHeight(const Eigen::Vector3d& location, const Time& time) const=0; //!< 楕円体高度(幾何高度)を返す。ellipsoid (geometric) height
    virtual double getGeoidHeight(const Eigen::Vector3d& location, const Time& time) const=0; //!< 標高(ジオイド高度)を返す。geoid height (elevation)
    ///@}

    /** @name Cartesian、Spherical、Elliposidalの不定性を取り除くための関数
     * 一部のCRSクラスは座標軸の種類を選べるため、Eigen::Vector3dで扱う際にその種類を明示できるようにしておく。
     * 派生クラスで座標軸の変換方法を定義する。未定義の場合は変換不可として例外を投げる。
     */
    ///@{
    Eigen::Vector3d toCartesian(const Eigen::Vector3d& v) const; //!< 自身の座標軸で表された引数を、cartesianな座標軸での値に変換する
    Eigen::Vector3d toSpherical(const Eigen::Vector3d& v) const; //!< 自身の座標軸で表された引数を、sphericalな座標軸での値に変換する
    Eigen::Vector3d toEllipsoidal(const Eigen::Vector3d& v) const; //!< 自身の座標軸で表された引数を、ellipsoidalな座標軸での値に変換する
    Eigen::Vector3d fromCartesian(const Eigen::Vector3d& v) const; //!< cartesianな座標軸で表された引数を、自身の座標軸での値に変換する
    Eigen::Vector3d fromSpherical(const Eigen::Vector3d& v) const; //!< sphericalな座標軸で表された引数を、自身の座標軸での値に変換する
    Eigen::Vector3d fromEllipsoidal(const Eigen::Vector3d& v) const; //!< ellipsoidalな座標軸で表された引数を、自身の座標軸での値に変換する
    virtual Eigen::Vector3d cartesianToSpherical(const Eigen::Vector3d& v) const; //!< cartesian -> sphericalの変換処理を記述する。変換可能な派生クラスではオーバーライドすること。
    virtual Eigen::Vector3d cartesianToEllipsoidal(const Eigen::Vector3d& v) const; //!< cartesian -> ellipsoidalの変換処理を記述する。変換可能な派生クラスではオーバーライドすること。
    virtual Eigen::Vector3d sphericalToCartesian(const Eigen::Vector3d& v) const; //!< spherical -> cartesianの変換処理を記述する。変換可能な派生クラスではオーバーライドすること。
    virtual Eigen::Vector3d sphericalToEllipsoidal(const Eigen::Vector3d& v) const; //!< spherical -> ellipsoidalの変換処理を記述する。変換可能な派生クラスではオーバーライドすること。
    virtual Eigen::Vector3d ellipsoidalToCartesian(const Eigen::Vector3d& v) const; //!< ellipsoidal -> cartesianの変換処理を記述する。変換可能な派生クラスではオーバーライドすること。
    virtual Eigen::Vector3d ellipsoidalToSpherical(const Eigen::Vector3d& v) const; //!< ellipsoidal -> sphericalの変換処理を記述する。変換可能な派生クラスではオーバーライドすること。
    ///@}

    /** @name 局所水平座標系(TopocentricCRS)との座標変換
     * TopocentricCRSは様々な位置を原点として頻繁に使用されるため、都度CRSインスタンスとして生成するのはコストが大きい。
     * 
     * そのため、TopocentricCRSインスタンスを生成せずに等価な座標変換を行う関数を各CRSに設ける。
     * originの指定はfromでもこのCRS上の座標値とする。valueとlocationは他の関数と同様にTopocentricCRS上の座標値である。
     */
    ///@{
    public:
    /**
     * @brief このCRS上の点originを原点とするTopocentricCRSを生成して返す。
     * 
     * @param [in] origin     原点の座標
     * @param [in] axisOrder  生成するCRSの座標軸順を表す文字列("NED"等)
     * @param [in] onSurface  高度方向の原点を楕円体表面上にとるかどうか。
     * @param [in] isEpisodic 生成するCRSの寿命を現在のエピソード中に限るかどうか。
     */
    std::shared_ptr<TopocentricCRS> createTopocentricCRS(const Coordinate& origin, const std::string& axisOrder, bool onSurface, bool isEpisodic) const;

    /**
     * @brief このCRS上の点originを原点とするTopocentricCRSを生成して返す。
     * 
     * @param [in] origin     原点の座標 (このCRS上の絶対位置ベクトルとして与える)
     * @param [in] time       変換の基準時刻
     * @param [in] axisOrder  生成するCRSの座標軸順を表す文字列("NED"等)
     * @param [in] onSurface  高度方向の原点を楕円体表面上にとるかどうか。
     * @param [in] isEpisodic 生成するCRSの寿命を現在のエピソード中に限るかどうか。
     * 
     * @details 派生クラスではこちらをオーバーライドする。
     */
    virtual std::shared_ptr<TopocentricCRS> createTopocentricCRS(const Eigen::Vector3d& origin, const Time& time, const std::string& axisOrder, bool onSurface, bool isEpisodic) const;

    /**
     * @brief このCRS上の点originを原点とするTopocentricCRSへの回転を表すクォータニオンを返す。
     * 
     * @details CRSインスタンスを生成したくない場合に用いる。
     * 
     * @param [in] origin     原点の座標
     * @param [in] axisOrder  生成するCRSの座標軸順を表す文字列("NED"等)
     */
    Quaternion getEquivalentQuaternionToTopocentricCRS(const Coordinate& origin, const std::string& axisOrder) const;

    /**
     * @brief このCRS上の点originを原点とするTopocentricCRSへの回転を表すクォータニオンを返す。
     * 
     * @details CRSインスタンスを生成したくない場合に用いる。派生クラスではこちらをオーバーライドする。
     * 
     * @param [in] origin     原点の座標
     * @param [in] time       変換の基準時刻
     * @param [in] axisOrder  生成するCRSの座標軸順を表す文字列("NED"等)
     */
    virtual Quaternion getEquivalentQuaternionToTopocentricCRS(const Eigen::Vector3d& origin, const Time& time, const std::string& axisOrder) const=0;

    /**
     * @brief 位置locationで観測された値valueを、このCRS上の点originを原点とするTopocentricCRSにおける座標に変換する
     * 
     * @param [in] value    変換対象の座標
     * @param [in] location valueが観測された位置
     * @param [in] origin     TopocentricCRSの原点の座標
     * @param [in] axisOrder  TopocentricCRSの座標軸順を表す文字列("NED"等)
     * @param [in] onSurface  TopocentricCRSの高度方向の原点を楕円体表面上にとるかどうか。
     * 
     * @details CRSインスタンスを生成したくない場合に用いる。
     */
    Eigen::Vector3d transformToTopocentricCRS(const Coordinate& value, const Coordinate& location, const Coordinate& origin, const std::string& axisOrder, bool onSurface) const;

    /**
     * @brief valueを、このCRS上の点originを原点とするTopocentricCRSにおける座標に変換する
     * 
     * @param [in] value    変換対象の座標
     * @param [in] origin     TopocentricCRSの原点の座標
     * @param [in] axisOrder  TopocentricCRSの座標軸順を表す文字列("NED"等)
     * @param [in] onSurface  TopocentricCRSの高度方向の原点を楕円体表面上にとるかどうか。
     * 
     * @details CRSインスタンスを生成したくない場合に用いる。value自身が保持しているlocationの情報を使用する。
     */
    Eigen::Vector3d transformToTopocentricCRS(const Coordinate& value, const Coordinate& origin, const std::string& axisOrder, bool onSurface) const;

    /**
     * @brief このCRSにおいて位置locationで観測された値valueを、このCRS上の点originを原点とするTopocentricCRSにおける座標に変換する
     * @param [in] value          変換対象の座標
     * @param [in] location       valueが観測された位置
     * @param [in] time           valueが観測された時刻
     * @param [in] coordinateType 座標の種類
     * @param [in] origin     TopocentricCRSの原点の座標
     * @param [in] axisOrder  TopocentricCRSの座標軸順を表す文字列("NED"等)
     * @param [in] onSurface  TopocentricCRSの高度方向の原点を楕円体表面上にとるかどうか。
     * 
     * @details 派生クラスはこれをオーバーライドして実際の座標変換処理を記述すること。
     * 
     */
    virtual Eigen::Vector3d transformToTopocentricCRS(const Eigen::Vector3d& value, const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType, const Eigen::Vector3d& origin, const std::string& axisOrder, bool onSurface) const;

    /**
     * @brief このCRSにおいて観測された値valueを、このCRS上の点originを原点とするTopocentricCRSにおける座標に変換する
     * @param [in] value          変換対象の座標
     * @param [in] time           valueが観測された時刻
     * @param [in] coordinateType 座標の種類
     * @param [in] origin     TopocentricCRSの原点の座標
     * @param [in] axisOrder  TopocentricCRSの座標軸順を表す文字列("NED"等)
     * @param [in] onSurface  TopocentricCRSの高度方向の原点を楕円体表面上にとるかどうか。
     * 
     * @details locationは原点(0,0,0)とみなす。
     */
    virtual Eigen::Vector3d transformToTopocentricCRS(const Eigen::Vector3d& value, const Time& time, const CoordinateType& coordinateType, const Eigen::Vector3d& origin, const std::string& axisOrder, bool onSurface) const;

    /**
     * @brief このCRS上の点originを原点とするTopocentricCRSにおいて位置locationで観測された値valueをこのCRSにおける座標に変換する
     * @param [in] value          変換対象の座標
     * @param [in] location       valueが観測された位置
     * @param [in] time           valueが観測された時刻
     * @param [in] coordinateType 座標の種類
     * @param [in] origin     TopocentricCRSの原点の座標
     * @param [in] axisOrder  TopocentricCRSの座標軸順を表す文字列("NED"等)
     * @param [in] onSurface  TopocentricCRSの高度方向の原点を楕円体表面上にとるかどうか。
     * 
     * @details 派生クラスはこれをオーバーライドして実際の座標変換処理を記述すること。
     */
    virtual Eigen::Vector3d transformFromTopocentricCRS(const Eigen::Vector3d& value, const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType, const Eigen::Vector3d& origin, const std::string& axisOrder, bool onSurface) const;

    /**
     * @brief このCRS上の点originを原点とするTopocentricCRSにおいて観測された値valueをこのCRSにおける座標に変換する
     * @param [in] value          変換対象の座標
     * @param [in] time           valueが観測された時刻
     * @param [in] coordinateType 座標の種類
     * @param [in] origin     TopocentricCRSの原点の座標
     * @param [in] axisOrder  TopocentricCRSの座標軸順を表す文字列("NED"等)
     * @param [in] onSurface  TopocentricCRSの高度方向の原点を楕円体表面上にとるかどうか。
     * 
     * @details locationは原点(0,0,0)とみなす。
     */
    virtual Eigen::Vector3d transformFromTopocentricCRS(const Eigen::Vector3d& value, const Time& time, const CoordinateType& coordinateType, const Eigen::Vector3d& origin, const std::string& axisOrder, bool onSurface) const;
    ///@}

    public:
    /**
     * @brief このCRSをparentとして持つMotionStateに対応するAffineCRSを生成して返す。
     */
    virtual std::shared_ptr<AffineCRS> createAffineCRS(const MotionState& motion, const std::string& axisOrder, bool isEpisodic) const;

    /// @name 他のインスタンスのprotectedメンバ関数を呼び出すためのstaticメンバ関数
    ///@{
    protected:
    static void validateImpl(const std::shared_ptr<const CoordinateReferenceSystem>& base,bool isFirstTime); //!< [For internal use] base->validateImpl(isFirstTime) を呼び出す。
    static void invalidate(const std::shared_ptr<const CoordinateReferenceSystem>& base); //!< [For internal use] base->invalidate() を呼び出す。
    static void addDerivedCRS(const std::shared_ptr<const CoordinateReferenceSystem>& base,const std::weak_ptr<CoordinateReferenceSystem>& derived); //!< [For internal use] base->addDerivedCRS(derived) を呼び出す。
    static void removeDerivedCRS(const std::shared_ptr< const CoordinateReferenceSystem>& base,const std::weak_ptr<CoordinateReferenceSystem>& derived); //!< [For internal use] base->removeDerivedCRS(derived) を呼び出す。
    ///@}
};

ASRC_DECLARE_DERIVED_TRAMPOLINE(CoordinateReferenceSystem)
    virtual bool isValid() const override{
        PYBIND11_OVERRIDE(bool,Base,isValid);
    }
    virtual bool hasBeenValidated() const override{
        PYBIND11_OVERRIDE(bool,Base,hasBeenValidated);
    }
    virtual bool isEquivalentCRS(const std::shared_ptr<CoordinateReferenceSystem>& other) const override{
        PYBIND11_OVERRIDE(bool,Base,isEquivalentCRS,other);
    }
    /*
    virtual void validateImpl(bool isFirstTime) const override{
        PYBIND11_OVERRIDE(void,Base,validateImpl,isFirstTime);
    }
    virtual void invalidate() const override{
        PYBIND11_OVERRIDE(void,Base,invalidate);
    }
    virtual void addDerivedCRS(const std::weak_ptr<CoordinateReferenceSystem>& derived) const override{
        PYBIND11_OVERRIDE(void,Base,addDerivedCRS,derived);
    }
    virtual void removeDerivedCRS(const std::weak_ptr<CoordinateReferenceSystem>& derived) const override{
        PYBIND11_OVERRIDE(void,Base,removeDerivedCRS,derived);
    }
    */
    //座標系の種類
    virtual bool isEarthFixed() const override{
        PYBIND11_OVERRIDE(bool,Base,isEarthFixed);
    }
    virtual bool isGeocentric() const override{
        PYBIND11_OVERRIDE(bool,Base,isGeocentric);
    }
    virtual bool isGeographic() const override{
        PYBIND11_OVERRIDE(bool,Base,isGeographic);
    }
    virtual bool isDerived() const override{
        PYBIND11_OVERRIDE(bool,Base,isDerived);
    }
    virtual bool isTopocentric() const override{
        PYBIND11_OVERRIDE(bool,Base,isTopocentric);
    }
    virtual bool isProjected() const override{
        PYBIND11_OVERRIDE(bool,Base,isProjected);
    }
    //親座標系の取得(共通インターフェースとする)
	virtual std::shared_ptr<CoordinateReferenceSystem> getBaseCRS() const override{
        PYBIND11_OVERRIDE(std::shared_ptr<CoordinateReferenceSystem>,Base,getBaseCRS);
    }
	virtual std::shared_ptr<CoordinateReferenceSystem> getNonDerivedBaseCRS() const override{
        PYBIND11_OVERRIDE(std::shared_ptr<CoordinateReferenceSystem>,Base,getNonDerivedBaseCRS);
    }
	//virtual std::shared_ptr<ProjectedCRS> getBaseProjectedCRS() const override{
    //    PYBIND11_OVERRIDE(std::shared_ptr<ProjectedCRS>,Base,getBaseProjectedCRS);
    //}
    //中間座標系の取得
	virtual std::shared_ptr<CoordinateReferenceSystem> getIntermediateCRS() const override{
        PYBIND11_OVERRIDE(std::shared_ptr<CoordinateReferenceSystem>,Base,getIntermediateCRS);
    }
    //Coordinate system axisの種類
    virtual bool isCartesian() const override{
        PYBIND11_OVERRIDE_PURE(bool,Base,isCartesian);
    }
    virtual bool isSpherical() const override{
        PYBIND11_OVERRIDE_PURE(bool,Base,isSpherical);
    }
    virtual bool isEllipsoidal() const override{
        PYBIND11_OVERRIDE_PURE(bool,Base,isEllipsoidal);
    }
    // CRSインスタンス間の変換可否
    virtual bool isTransformableTo(const std::shared_ptr<CoordinateReferenceSystem>& other) const override{
        PYBIND11_OVERRIDE_PURE(bool,Base,isTransformableTo,other);
    }
    virtual bool isTransformableFrom(const std::shared_ptr<CoordinateReferenceSystem>& other) const override{
        PYBIND11_OVERRIDE_PURE(bool,Base,isTransformableFrom,other);
    }
    // 座標変換
    using Base::transformTo;
    using Base::transformFrom;
    using Base::getQuaternionTo;
    using Base::getQuaternionFrom;
    using Base::transformQuaternionTo;
    using Base::transformQuaternionFrom;
    virtual Coordinate transformTo(const Eigen::Vector3d& value, const Eigen::Vector3d& location, const Time& time, const std::shared_ptr<CoordinateReferenceSystem>& dstCRS, const CoordinateType& coordinateType) const override{
        PYBIND11_OVERRIDE_PURE(Coordinate,Base,transformTo,value,location,time,dstCRS,coordinateType);
    }
    virtual Coordinate transformFrom(const Eigen::Vector3d& value, const Eigen::Vector3d& location, const Time& time, const std::shared_ptr<CoordinateReferenceSystem>& srcCRS, const CoordinateType& coordinateType) const override{
        PYBIND11_OVERRIDE_PURE(Coordinate,Base,transformFrom,value,location,time,srcCRS,coordinateType);
    }
    virtual Quaternion getQuaternionTo(const Eigen::Vector3d& location, const Time& time, const std::shared_ptr<CoordinateReferenceSystem>& dstCRS) const override{
        PYBIND11_OVERRIDE_PURE(Quaternion,Base,getQuaternionTo,location,time,dstCRS);
    }
    virtual Quaternion getQuaternionFrom(const Eigen::Vector3d& location, const Time& time, const std::shared_ptr<CoordinateReferenceSystem>& srcCRS) const override{
        PYBIND11_OVERRIDE_PURE(Quaternion,Base,getQuaternionFrom,location,time,srcCRS);
    }
    virtual Quaternion transformQuaternionTo(const Quaternion& q, const Eigen::Vector3d& location, const Time& time, const std::shared_ptr<CoordinateReferenceSystem>& dstCRS) const override{
        PYBIND11_OVERRIDE(Quaternion,Base,transformQuaternionTo,q,location,time,dstCRS);
    }
    virtual Quaternion transformQuaternionFrom(const Quaternion& q, const Eigen::Vector3d& location, const Time& time, const std::shared_ptr<CoordinateReferenceSystem>& srcCRS) const override{
        PYBIND11_OVERRIDE(Quaternion,Base,transformQuaternionFrom,q,location,time,srcCRS);
    }
    //「高度」の取得
    using Base::getHeight;
    using Base::getGeoidHeight;
    virtual double getHeight(const Eigen::Vector3d& location, const Time& time) const override{
        PYBIND11_OVERRIDE_PURE(double,Base,getHeight,location,time);
    }
    virtual double getGeoidHeight(const Eigen::Vector3d& location, const Time& time) const override{
        PYBIND11_OVERRIDE_PURE(double,Base,getGeoidHeight,location,time);
    }
    // Cartesian、Spherical、Elliposidalの不定性を取り除くための関数
    // 一部のCRSクラスは座標軸の種類を選べるため、Eigen::Vector3dで扱う際にその種類を明示できるようにしておく。
    // 派生クラスで座標軸の変換方法を定義する。未定義の場合は変換不可として例外を投げる。
    virtual Eigen::Vector3d cartesianToSpherical(const Eigen::Vector3d& v) const override{
        PYBIND11_OVERRIDE(Eigen::Vector3d,Base,cartesianToSpherical,v);
    }
    virtual Eigen::Vector3d cartesianToEllipsoidal(const Eigen::Vector3d& v) const override{
        PYBIND11_OVERRIDE(Eigen::Vector3d,Base,cartesianToEllipsoidal,v);
    }
    virtual Eigen::Vector3d sphericalToCartesian(const Eigen::Vector3d& v) const override{
        PYBIND11_OVERRIDE(Eigen::Vector3d,Base,sphericalToCartesian,v);
    }
    virtual Eigen::Vector3d sphericalToEllipsoidal(const Eigen::Vector3d& v) const override{
        PYBIND11_OVERRIDE(Eigen::Vector3d,Base,sphericalToEllipsoidal,v);
    }
    virtual Eigen::Vector3d ellipsoidalToCartesian(const Eigen::Vector3d& v) const override{
        PYBIND11_OVERRIDE(Eigen::Vector3d,Base,ellipsoidalToCartesian,v);
    }
    virtual Eigen::Vector3d ellipsoidalToSpherical(const Eigen::Vector3d& v) const override{
        PYBIND11_OVERRIDE(Eigen::Vector3d,Base,ellipsoidalToSpherical,v);
    }
    // このCRS上の点originを原点とするtopocentricなCRSを生成して返す。
    //using Base::createTopocentricCRS;
    //virtual std::shared_ptr<TopocentricCRS> createTopocentricCRS(const Eigen::Vector3d& origin, const Time& time, const std::string& axisOrder, bool onSurface, bool isEpisodic) const override{
    //    PYBIND11_OVERRIDE(std::shared_ptr<TopocentricCRS>,Base,createTopocentricCRS,origin,time,axisOrder,onSurface,isEpisodic);
    //}
    using Base::getEquivalentQuaternionToTopocentricCRS;
    using Base::transformToTopocentricCRS;
    using Base::transformFromTopocentricCRS;
    virtual Quaternion getEquivalentQuaternionToTopocentricCRS(const Eigen::Vector3d& origin, const Time& time, const std::string& axisOrder) const override{
        PYBIND11_OVERRIDE_PURE(Quaternion,Base,getEquivalentQuaternionToTopocentricCRS,origin,time,axisOrder);
    }
    virtual Eigen::Vector3d transformToTopocentricCRS(const Eigen::Vector3d& value, const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType, const Eigen::Vector3d& origin, const std::string& axisOrder, bool onSurface) const override{
        PYBIND11_OVERRIDE(Eigen::Vector3d,Base,transformToTopocentricCRS,value,location,time,coordinateType,origin,axisOrder,onSurface);
    }
    virtual Eigen::Vector3d transformToTopocentricCRS(const Eigen::Vector3d& value, const Time& time, const CoordinateType& coordinateType, const Eigen::Vector3d& origin, const std::string& axisOrder, bool onSurface) const override{
        PYBIND11_OVERRIDE(Eigen::Vector3d,Base,transformToTopocentricCRS,value,time,coordinateType,origin,axisOrder,onSurface);
    }
    virtual Eigen::Vector3d transformFromTopocentricCRS(const Eigen::Vector3d& value, const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType, const Eigen::Vector3d& origin, const std::string& axisOrder, bool onSurface) const override{
        PYBIND11_OVERRIDE(Eigen::Vector3d,Base,transformFromTopocentricCRS,value,location,time,coordinateType,origin,axisOrder,onSurface);
    }
    virtual Eigen::Vector3d transformFromTopocentricCRS(const Eigen::Vector3d& value, const Time& time, const CoordinateType& coordinateType, const Eigen::Vector3d& origin, const std::string& axisOrder, bool onSurface) const override{
        PYBIND11_OVERRIDE(Eigen::Vector3d,Base,transformFromTopocentricCRS,value,time,coordinateType,origin,axisOrder,onSurface);
    }
    // このCRSをparentとして持つMotionStateに対応するAffineCRSを生成して返す。
    //virtual std::shared_ptr<AffineCRS> createAffineCRS(const MotionState& motion, const std::string& axisOrder, bool isEpisodic) const override{
    //    PYBIND11_OVERRIDE(std::shared_ptr<AffineCRS>,Base,createAffineCRS,motion,axisOrder,isEpisodic);
    //}
};

void exportCoordinateReferenceSystemBase(py::module &m, const std::shared_ptr<asrc::core::FactoryHelper>& factoryHelper);

ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
