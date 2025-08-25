/**
 * Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
 * @file
 * 
 * @brief 地球を中心とした座標系を表すクラス
 */
#pragma once
#include "../Common.h"
#include "CoordinateReferenceSystemBase.h"
#include "AffineCRS.h"
#include "TopocentricCRS.h"

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)

class GeographicCRS;
class EarthCenteredInetialCRS;

/**
 * @class GeodeticCRS
 * @brief 地球を中心とした座標系を表すクラス。
 * 
 * @details
 *      Geocentric又はGeographicに分岐する。
 *      ISO 19111では、GeocentricなCRSはGeodeticCRSクラスがそのまま使われ、GeographicなCRSはサブクラスとしてGeographicCRSが用いられる。
 *      本シミュレータでもそれに準じて、GeographicCRSのインスタンスでないGeodeticCRSインスタンスはGeocentricであるものとみなす。
 *
 *      また、GeocentricなCRSは、いわゆるECEF座標系に相当する直交座標系と、
 *      極座標(geocentric latitude, geocentric longitude, geocentric radius)の2種類が許容される。
 *      一般的なECEF座標系では、+X軸を赤道上の本初子午線方向、+Z軸を北極方向とする右手系が用いられる。
 *
 */
ASRC_DECLARE_DERIVED_CLASS_WITH_TRAMPOLINE(GeodeticCRS,CoordinateReferenceSystem)
    public:
    GeodeticCRS(const nl::json& modelConfig_,const nl::json& instanceConfig_);
    // 内部状態のシリアライゼーション
    virtual void serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full) override;
    protected:
    virtual void validateImpl(bool isFirstTime) const override;
    public:
    //座標系の種類
    virtual bool isEarthFixed() const override; //地球に対して固定されているかどうか
    virtual bool isGeocentric() const override; //地球を中心とした座標系かどうか
    //中間座標系の取得
	virtual std::shared_ptr<CoordinateReferenceSystem> getIntermediateCRS() const override; //親子関係にない他のCRSとの間で座標変換を行う際に経由する中間座標系。GeodeticCRSにとっては対応するECEFが該当。
    //Coordinate system axisの種類
    virtual bool isCartesian() const override;
    virtual bool isSpherical() const override;
    virtual bool isEllipsoidal() const override;
    //「高度」の取得
    public:
    using BaseType::getHeight;
    using BaseType::getGeoidHeight;
    virtual double getHeight(const Eigen::Vector3d& location, const Time& time) const override; //楕円体高度(幾何高度)を返す。ellipsoid (geometric) height
    virtual double getGeoidHeight(const Eigen::Vector3d& location, const Time& time) const override; //標高(ジオイド高度)を返す。geoid height (elevation)
    // CRSインスタンス間の変換可否
    virtual bool isTransformableTo(const std::shared_ptr<CoordinateReferenceSystem>& other) const override;
    virtual bool isTransformableFrom(const std::shared_ptr<CoordinateReferenceSystem>& other) const override;
    // 座標変換
    using BaseType::transformTo;
    using BaseType::transformFrom;
    using BaseType::getQuaternionTo;
    using BaseType::getQuaternionFrom;
    virtual Coordinate transformTo(const Eigen::Vector3d& value, const Eigen::Vector3d& location, const Time& time, const std::shared_ptr<CoordinateReferenceSystem>& dstCRS, const CoordinateType& coordinateType) const override;
    virtual Coordinate transformFrom(const Eigen::Vector3d& value, const Eigen::Vector3d& location, const Time& time, const std::shared_ptr<CoordinateReferenceSystem>& srcCRS, const CoordinateType& coordinateType) const override;
    virtual Quaternion getQuaternionTo(const Eigen::Vector3d& location, const Time& time, const std::shared_ptr<CoordinateReferenceSystem>& dstCRS) const override;
    virtual Quaternion getQuaternionFrom(const Eigen::Vector3d& location, const Time& time, const std::shared_ptr<CoordinateReferenceSystem>& srcCRS) const override;

    // Cartesian⇔Sphericalの不定性を取り除くための関数
    virtual Eigen::Vector3d cartesianToSpherical(const Eigen::Vector3d& v) const override; //Cartesian (ECEF)で表された引数をSpherical (geocentric lat-lon-radius)での値に変換する
    virtual Eigen::Vector3d sphericalToCartesian(const Eigen::Vector3d& v) const override; //Spherical (geocentric lat-lon-radius)で表された引数をCartesian (ECEF)での値に変換する

    // このCRS上の点originを原点とするtopocentricなCRSを生成して返す。
    using BaseType::getEquivalentQuaternionToTopocentricCRS;
    virtual Quaternion getEquivalentQuaternionToTopocentricCRS(const Eigen::Vector3d& origin, const Time& time, const std::string& axisOrder) const override;//CRSとして生成せずにクォータニオンとして得る。

    //
    // 以下は本クラスで追加定義するメンバ
    //

    /**
     * @brief 同じdatumでgeographicなCRSを返す。
     */
    virtual std::shared_ptr<GeographicCRS> getGeographicCRS() const;

    /**
     * @brief 同じdatumでgeocentricかつEarth-fixedなCRS(Cartesian ECEF)を返す。
     */
    virtual std::shared_ptr<GeodeticCRS> getECEF() const;

    /**
     * pbrief 同じdatumでgeocentricかつEarth-fixedなCRS(Spherical ECEF: geocentric lat-lon-radius)を返す。
     */
    virtual std::shared_ptr<GeodeticCRS> getSphericalECEF() const;

    /**
     * @brief 同じdatumでgeocentricかつinertialなCRS(ECI)を返す。
     */
    virtual std::shared_ptr<EarthCenteredInetialCRS> getECI() const;

    /** @name 同じdatumのgeographicなCRSとの間の座標変換
     */
    ///@{
    /**
     * @brief 位置locationで観測された値valueを、同じdatumのgeographicなCRSにおける座標に変換する。
     * 
     * @param [in] value    変換対象の座標
     * @param [in] location valueが観測された位置
     */
    Coordinate transformToGeographicCRS(const Coordinate& value, const Coordinate& location) const;

    /**
     * @brief 位置locationで観測された値valueを、同じdatumのgeographicなCRSにおける座標に変換する。
     * 
     * @param [in] value    変換対象の座標
     * 
     * @details value自身が保持しているlocationの情報を使用する。
     */
    Coordinate transformToGeographicCRS(const Coordinate& value) const;

    /**
     * @brief 同じdatumのgeographicなCRSにおける位置locationで観測された値valueを、このCRSにおける座標に変換する。
     * 
     * @param [in] value    変換対象の座標
     * @param [in] location valueが観測された位置
     * 
     * @attention 引数がCoordinateの場合はその値自身がcrsを持っておりこのCRSへの直接変換が可能なのでこの関数はあまり意味がない。
     */
    Coordinate transformFromGeographicCRS(const Coordinate& value, const Coordinate& location) const;

    /**
     * @brief 同じdatumのgeographicなCRSにおいて観測された値valueを、このCRSにおける座標に変換する。
     * 
     * @param [in] value    変換対象の座標
     * 
     * @details value自身が保持しているlocationの情報を使用する。
     * 
     * @attention 引数がCoordinateの場合はその値自身がcrsを持っておりこのCRSへの直接変換が可能なのでこの関数はあまり意味がない。
     */
    Coordinate transformFromGeographicCRS(const Coordinate& value) const;

    /**
     * @brief このCRSにおいて位置locationで観測された値valueを、同じdatumのgeographicなCRSにおける座標に変換する
     * @param [in] value          変換対象の座標
     * @param [in] location       valueが観測された位置
     * @param [in] time           valueが観測された時刻
     * @param [in] coordinateType 座標の種類
     * 
     * @details このクラスはEPSG:9602に基づくECEF→Geographicの変換を記述している。
     *          派生クラスはこれをオーバーライドして実際の座標変換処理を記述すること。
     */
    virtual Coordinate transformToGeographicCRS(const Eigen::Vector3d& value, const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType) const;

    /**
     * @brief 同じdatumのgeographicなCRSにおいて位置locationで観測された値valueを、このCRSにおける座標に変換する
     * @param [in] value          変換対象の座標
     * @param [in] location       valueが観測された位置
     * @param [in] time           valueが観測された時刻
     * @param [in] coordinateType 座標の種類
     * 
     * @details このクラスはEPSG:9602に基づくGeographic→ECEFの変換を記述している。
     *          派生クラスはこれをオーバーライドして実際の座標変換処理を記述すること。
     */
    virtual Coordinate transformFromGeographicCRS(const Eigen::Vector3d& value, const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType) const;

    /**
     * @brief このCRSにおいて観測された値valueを、同じdatumのgeographicなCRSにおける座標に変換する
     * @param [in] value          変換対象の座標
     * @param [in] time           valueが観測された時刻
     * @param [in] coordinateType 座標の種類
     * 
     * @details このクラスはEPSG:9602に基づくECEF→Geographicの変換を記述している。
     *          派生クラスはこれをオーバーライドして実際の座標変換処理を記述すること。
     * 
     *          locationは原点(0,0,0)とみなす。
     */
    virtual Coordinate transformToGeographicCRS(const Eigen::Vector3d& value, const Time& time, const CoordinateType& coordinateType) const;

    /**
     * @brief 同じdatumのgeographicなCRSにおいて位置locationで観測された値valueを、このCRSにおける座標に変換する
     * @param [in] value          変換対象の座標
     * @param [in] time           valueが観測された時刻
     * @param [in] coordinateType 座標の種類
     * 
     * @details このクラスはEPSG:9602に基づくGeographic→ECEFの変換を記述している。
     *          派生クラスはこれをオーバーライドして実際の座標変換処理を記述すること。
     * 
     *          locationは原点(0,0,0)とみなす。
     */
    virtual Coordinate transformFromGeographicCRS(const Eigen::Vector3d& value, const Time& time, const CoordinateType& coordinateType) const;
    ///@}

    /** @name 同じdatumのECEFとの間の座標変換
     */
    ///@{
    /**
     * @brief 位置locationで観測された値valueを、同じdatumのECEFにおける座標に変換する。
     * 
     * @param [in] value    変換対象の座標
     * @param [in] location valueが観測された位置
     */
    Coordinate transformToECEF(const Coordinate& value, const Coordinate& location) const;

    /**
     * @brief 位置locationで観測された値valueを、同じdatumのECEFにおける座標に変換する。
     * 
     * @param [in] value    変換対象の座標
     * 
     * @details value自身が保持しているlocationの情報を使用する。
     */
    Coordinate transformToECEF(const Coordinate& value) const;

    /**
     * @brief 同じdatumのECEFにおける位置locationで観測された値valueを、このCRSにおける座標に変換する。
     * 
     * @param [in] value    変換対象の座標
     * @param [in] location valueが観測された位置
     * 
     * @attention 引数がCoordinateの場合はその値自身がcrsを持っておりこのCRSへの直接変換が可能なのでこの関数はあまり意味がない。
     */
    Coordinate transformFromECEF(const Coordinate& value, const Coordinate& location) const;

    /**
     * @brief 同じdatumのECEFにおいて観測された値valueを、このCRSにおける座標に変換する。
     * 
     * @param [in] value    変換対象の座標
     * 
     * @details value自身が保持しているlocationの情報を使用する。
     * 
     * @attention 引数がCoordinateの場合はその値自身がcrsを持っておりこのCRSへの直接変換が可能なのでこの関数はあまり意味がない。
     */
    Coordinate transformFromECEF(const Coordinate& value) const;

    /**
     * @brief このCRSにおいて位置locationで観測された値valueを、同じdatumのECEFにおける座標に変換する
     * @param [in] value          変換対象の座標
     * @param [in] location       valueが観測された位置
     * @param [in] time           valueが観測された時刻
     * @param [in] coordinateType 座標の種類
     * 
     * @details 派生クラスはこれをオーバーライドして実際の座標変換処理を記述すること。
     */
    virtual Coordinate transformToECEF(const Eigen::Vector3d& value, const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType) const;

    /**
     * @brief 同じdatumのECEFにおいて位置locationで観測された値valueを、このCRSにおける座標に変換する
     * @param [in] value          変換対象の座標
     * @param [in] location       valueが観測された位置
     * @param [in] time           valueが観測された時刻
     * @param [in] coordinateType 座標の種類
     * 
     * @details 派生クラスはこれをオーバーライドして実際の座標変換処理を記述すること。
     */
    virtual Coordinate transformFromECEF(const Eigen::Vector3d& value, const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType) const;

    /**
     * @brief このCRSにおいて観測された値valueを、同じdatumのECEFにおける座標に変換する
     * @param [in] value          変換対象の座標
     * @param [in] time           valueが観測された時刻
     * @param [in] coordinateType 座標の種類
     * 
     * @details 派生クラスはこれをオーバーライドして実際の座標変換処理を記述すること。
     * 
     *          locationは原点(0,0,0)とみなす。
     */
    virtual Coordinate transformToECEF(const Eigen::Vector3d& value, const Time& time, const CoordinateType& coordinateType) const;

    /**
     * @brief 同じdatumのECEFにおいて位置locationで観測された値valueを、このCRSにおける座標に変換する
     * @param [in] value          変換対象の座標
     * @param [in] time           valueが観測された時刻
     * @param [in] coordinateType 座標の種類
     * 
     * @details 派生クラスはこれをオーバーライドして実際の座標変換処理を記述すること。
     * 
     *          locationは原点(0,0,0)とみなす。
     */
    virtual Coordinate transformFromECEF(const Eigen::Vector3d& value, const Time& time, const CoordinateType& coordinateType) const;
    ///@}

    /** @name 同じdatumのSpherical ECEFとの間の座標変換
     */
    ///@{
    /**
     * @brief 位置locationで観測された値valueを、同じdatumのSpherical ECEFにおける座標に変換する。
     * 
     * @param [in] value    変換対象の座標
     * @param [in] location valueが観測された位置
     */
    Coordinate transformToSphericalECEF(const Coordinate& value, const Coordinate& location) const;

    /**
     * @brief 位置locationで観測された値valueを、同じdatumのSpherical ECEFにおける座標に変換する。
     * 
     * @param [in] value    変換対象の座標
     * 
     * @details value自身が保持しているlocationの情報を使用する。
     */
    Coordinate transformToSphericalECEF(const Coordinate& value) const;

    /**
     * @brief 同じdatumのSpherical ECEFにおける位置locationで観測された値valueを、このCRSにおける座標に変換する。
     * 
     * @param [in] value    変換対象の座標
     * @param [in] location valueが観測された位置
     * 
     * @attention 引数がCoordinateの場合はその値自身がcrsを持っておりこのCRSへの直接変換が可能なのでこの関数はあまり意味がない。
     */
    Coordinate transformFromSphericalECEF(const Coordinate& value, const Coordinate& location) const;

    /**
     * @brief 同じdatumのSpherical ECEFにおいて観測された値valueを、このCRSにおける座標に変換する。
     * 
     * @param [in] value    変換対象の座標
     * 
     * @details value自身が保持しているlocationの情報を使用する。
     * 
     * @attention 引数がCoordinateの場合はその値自身がcrsを持っておりこのCRSへの直接変換が可能なのでこの関数はあまり意味がない。
     */
    Coordinate transformFromSphericalECEF(const Coordinate& value) const;

    /**
     * @brief このCRSにおいて位置locationで観測された値valueを、同じdatumのSpherical ECEFにおける座標に変換する
     * @param [in] value          変換対象の座標
     * @param [in] location       valueが観測された位置
     * @param [in] time           valueが観測された時刻
     * @param [in] coordinateType 座標の種類
     * 
     * @details 派生クラスはこれをオーバーライドして実際の座標変換処理を記述すること。
     */
    virtual Coordinate transformToSphericalECEF(const Eigen::Vector3d& value, const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType) const;

    /**
     * @brief 同じdatumのSpherical ECEFにおいて位置locationで観測された値valueを、このCRSにおける座標に変換する
     * @param [in] value          変換対象の座標
     * @param [in] location       valueが観測された位置
     * @param [in] time           valueが観測された時刻
     * @param [in] coordinateType 座標の種類
     * 
     * @details 派生クラスはこれをオーバーライドして実際の座標変換処理を記述すること。
     */
    virtual Coordinate transformFromSphericalECEF(const Eigen::Vector3d& value, const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType) const;

    /**
     * @brief このCRSにおいて観測された値valueを、同じdatumのSpherical ECEFにおける座標に変換する
     * @param [in] value          変換対象の座標
     * @param [in] time           valueが観測された時刻
     * @param [in] coordinateType 座標の種類
     * 
     * @details 派生クラスはこれをオーバーライドして実際の座標変換処理を記述すること。
     * 
     *          locationは原点(0,0,0)とみなす。
     */
    virtual Coordinate transformToSphericalECEF(const Eigen::Vector3d& value, const Time& time, const CoordinateType& coordinateType) const;

    /**
     * @brief 同じdatumのSpherical ECEFにおいて位置locationで観測された値valueを、このCRSにおける座標に変換する
     * @param [in] value          変換対象の座標
     * @param [in] time           valueが観測された時刻
     * @param [in] coordinateType 座標の種類
     * 
     * @details 派生クラスはこれをオーバーライドして実際の座標変換処理を記述すること。
     * 
     *          locationは原点(0,0,0)とみなす。
     */
    virtual Coordinate transformFromSphericalECEF(const Eigen::Vector3d& value, const Time& time, const CoordinateType& coordinateType) const;
    ///@}

    /** @name 同じdatumのECIとの間の座標変換
     */
    ///@{
    /**
     * @brief 位置locationで観測された値valueを、同じdatumのECIにおける座標に変換する。
     * 
     * @param [in] value    変換対象の座標
     * @param [in] location valueが観測された位置
     */
    Coordinate transformToECI(const Coordinate& value, const Coordinate& location) const;

    /**
     * @brief 位置locationで観測された値valueを、同じdatumのECIにおける座標に変換する。
     * 
     * @param [in] value    変換対象の座標
     * 
     * @details value自身が保持しているlocationの情報を使用する。
     */
    Coordinate transformToECI(const Coordinate& value) const;

    /**
     * @brief 同じdatumのECIにおける位置locationで観測された値valueを、このCRSにおける座標に変換する。
     * 
     * @param [in] value    変換対象の座標
     * @param [in] location valueが観測された位置
     * 
     * @attention 引数がCoordinateの場合はその値自身がcrsを持っておりこのCRSへの直接変換が可能なのでこの関数はあまり意味がない。
     */
    Coordinate transformFromECI(const Coordinate& value, const Coordinate& location) const;

    /**
     * @brief 同じdatumのECIにおいて観測された値valueを、このCRSにおける座標に変換する。
     * 
     * @param [in] value    変換対象の座標
     * 
     * @details value自身が保持しているlocationの情報を使用する。
     * 
     * @attention 引数がCoordinateの場合はその値自身がcrsを持っておりこのCRSへの直接変換が可能なのでこの関数はあまり意味がない。
     */
    Coordinate transformFromECI(const Coordinate& value) const;

    /**
     * @brief このCRSにおいて位置locationで観測された値valueを、同じdatumのECIにおける座標に変換する
     * @param [in] value          変換対象の座標
     * @param [in] location       valueが観測された位置
     * @param [in] time           valueが観測された時刻
     * @param [in] coordinateType 座標の種類
     * 
     * @details 派生クラスはこれをオーバーライドして実際の座標変換処理を記述すること。
     */
    virtual Coordinate transformToECI(const Eigen::Vector3d& value, const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType) const;

    /**
     * @brief 同じdatumのECIにおいて位置locationで観測された値valueを、このCRSにおける座標に変換する
     * @param [in] value          変換対象の座標
     * @param [in] location       valueが観測された位置
     * @param [in] time           valueが観測された時刻
     * @param [in] coordinateType 座標の種類
     * 
     * @details 派生クラスはこれをオーバーライドして実際の座標変換処理を記述すること。
     */
    virtual Coordinate transformFromECI(const Eigen::Vector3d& value, const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType) const;

    /**
     * @brief このCRSにおいて観測された値valueを、同じdatumのECIにおける座標に変換する
     * @param [in] value          変換対象の座標
     * @param [in] time           valueが観測された時刻
     * @param [in] coordinateType 座標の種類
     * 
     * @details 派生クラスはこれをオーバーライドして実際の座標変換処理を記述すること。
     * 
     *          locationは原点(0,0,0)とみなす。
     */
    virtual Coordinate transformToECI(const Eigen::Vector3d& value, const Time& time, const CoordinateType& coordinateType) const;

    /**
     * @brief 同じdatumのECIにおいて位置locationで観測された値valueを、このCRSにおける座標に変換する
     * @param [in] value          変換対象の座標
     * @param [in] time           valueが観測された時刻
     * @param [in] coordinateType 座標の種類
     * 
     * @details 派生クラスはこれをオーバーライドして実際の座標変換処理を記述すること。
     * 
     *          locationは原点(0,0,0)とみなす。
     */
    virtual Coordinate transformFromECI(const Eigen::Vector3d& value, const Time& time, const CoordinateType& coordinateType) const;
    ///@}

    /** @name [For internal use] このCRSと同じdatumでgeographic座標とのECEF座標の相互変換
     */
    ///@{
    /**
     * @brief [For internal use] このCRSと同じdatumのgeographicなCRSにおいて位置locationで観測された値valueを、ECEFにおける座標に変換する
     * @param [in] value          変換対象の座標
     * @param [in] location       valueが観測された位置
     * @param [in] coordinateType 座標の種類
     * 
     * @details このクラスはEPSG:9602に基づくGeographic→ECEFの変換を記述している。
     *          派生クラスはこれをオーバーライドして実際の座標変換処理を記述すること。
     */
    virtual Eigen::Vector3d transformGeographicCoordinateToECEF(const Eigen::Vector3d& value, const Eigen::Vector3d& location, const CoordinateType& coordinateType) const;

    /**
     * @brief [For internal use] このCRSと同じdatumのgeographicなCRSにおいて位置locationで観測された値valueを、ECEFにおける座標に変換する
     * @param [in] value          変換対象の座標
     * @param [in] coordinateType 座標の種類
     * 
     * @details このクラスはEPSG:9602に基づくGeographic→ECEFの変換を記述している。
     *          派生クラスはこれをオーバーライドして実際の座標変換処理を記述すること。
     * 
     *          locationは原点(0,0,0)とみなす。
     */
    virtual Eigen::Vector3d transformGeographicCoordinateToECEF(const Eigen::Vector3d& value, const CoordinateType& coordinateType) const;
    
    /**
     * @brief [For internal use] このCRSと同じdatumのECEFにおいて位置locationで観測された値valueを、geographicなCRSにおける座標に変換する
     * @param [in] value          変換対象の座標
     * @param [in] location       valueが観測された位置
     * @param [in] coordinateType 座標の種類
     * 
     * @details このクラスはEPSG:9602に基づくECEF→Geographicの変換を記述している。
     *          派生クラスはこれをオーバーライドして実際の座標変換処理を記述すること。
     */
    virtual Eigen::Vector3d transformECEFCoordinateToGeographicCRS(const Eigen::Vector3d& value, const Eigen::Vector3d& location, const CoordinateType& coordinateType) const;

    /**
     * @brief [For internal use] このCRSと同じdatumのECEFにおいて位置locationで観測された値valueを、geographicなCRSにおける座標に変換する
     * @param [in] value          変換対象の座標
     * @param [in] coordinateType 座標の種類
     * 
     * @details このクラスはEPSG:9602に基づくECEF→Geographicの変換を記述している。
     *          派生クラスはこれをオーバーライドして実際の座標変換処理を記述すること。
     * 
     *          locationは原点(0,0,0)とみなす。
     */
    virtual Eigen::Vector3d transformECEFCoordinateToGeographicCRS(const Eigen::Vector3d& value, const CoordinateType& coordinateType) const;
    ///@}

    /**
     * @brief 引数に与えたGeodeticCRSに座標変換可能かどうかを返す。
     * 
     * @attention このクラスでは異なるdatumへの具体的な変換処理を提供しないので必要ならば派生クラスで実装すること。
     */
    virtual bool isTransformableToAnotherGeodeticCRS(const std::shared_ptr<GeodeticCRS>& other) const;

    /**
     * @brief 引数に与えたGeodeticCRSから座標変換可能かどうかを返す。
     * 
     * @attention このクラスでは異なるdatumへの具体的な変換処理を提供しないので必要ならば派生クラスで実装すること。
     */
    virtual bool isTransformableFromAnotherGeodeticCRS(const std::shared_ptr<GeodeticCRS>& other) const;

    /** @name 異なるdatumのECEFとの間の座標変換
     *  @attention このクラスでは具体的な変換処理を提供しないので必要ならば派生クラスで実装すること。
     */
    ///@{
    /**
     * @brief このCRSにおいて位置locationで観測された値valueを、dstCRSにおける座標に変換する
     * @param [in] value          変換対象の座標
     * @param [in] location       valueが観測された位置
     * @param [in] time           valueが観測された時刻
     * @param [in] dstCRS         変換先のCRS
     * @param [in] coordinateType 座標の種類
     * 
     * @details このクラスはEPSG:9602に基づくECEF→Geographicの変換を記述している。
     *          派生クラスはこれをオーバーライドして実際の座標変換処理を記述すること。
     */
    virtual Coordinate transformToAnotherGeodeticCRS(const Eigen::Vector3d& value, const Eigen::Vector3d& location, const Time& time, const std::shared_ptr<GeodeticCRS>& dstCRS, const CoordinateType& coordinateType) const;

    /**
     * @brief srcCRSにおいて位置locationで観測された値valueを、このCRSにおける座標に変換する
     * @param [in] value          変換対象の座標
     * @param [in] location       valueが観測された位置
     * @param [in] time           valueが観測された時刻
     * @param [in] srcCRS         変換元のCRS
     * @param [in] coordinateType 座標の種類
     * 
     * @details このクラスはEPSG:9602に基づくECEF→Geographicの変換を記述している。
     *          派生クラスはこれをオーバーライドして実際の座標変換処理を記述すること。
     */
    virtual Coordinate transformFromAnotherGeodeticCRS(const Eigen::Vector3d& value, const Eigen::Vector3d& location, const Time& time, const std::shared_ptr<GeodeticCRS>& srcCRS, const CoordinateType& coordinateType) const;

    /**
     * @brief このCRSにおいて観測された値valueを、dstCRSにおける座標に変換する
     * @param [in] value          変換対象の座標
     * @param [in] time           valueが観測された時刻
     * @param [in] dstCRS         変換先のCRS
     * @param [in] coordinateType 座標の種類
     * 
     * @details このクラスはEPSG:9602に基づくECEF→Geographicの変換を記述している。
     *          派生クラスはこれをオーバーライドして実際の座標変換処理を記述すること。
     * 
     *          locationは原点(0,0,0)とみなす。
     */
    virtual Coordinate transformToAnotherGeodeticCRS(const Eigen::Vector3d& value, const Time& time, const std::shared_ptr<GeodeticCRS>& dstCRS, const CoordinateType& coordinateType) const;

    /**
     * @brief srcCRSにおいて観測された値valueを、このCRSにおける座標に変換する
     * @param [in] value          変換対象の座標
     * @param [in] time           valueが観測された時刻
     * @param [in] srcCRS         変換元のCRS
     * @param [in] coordinateType 座標の種類
     * 
     * @details このクラスはEPSG:9602に基づくECEF→Geographicの変換を記述している。
     *          派生クラスはこれをオーバーライドして実際の座標変換処理を記述すること。
     * 
     *          locationは原点(0,0,0)とみなす。
     */
    virtual Coordinate transformFromAnotherGeodeticCRS(const Eigen::Vector3d& value, const Time& time, const std::shared_ptr<GeodeticCRS>& srcCRS, const CoordinateType& coordinateType) const;
    Quaternion getQuaternionToAnotherGeodeticCRS(const Coordinate& location, const std::shared_ptr<GeodeticCRS>& dstCRS) const; //!< 基準点locationにおいてこのCRSの座標をdstCRSの座標に変換するクォータニオンを得る。
    Quaternion getQuaternionFromAnotherGeodeticCRS(const Coordinate& location) const; //!< 基準点locationにおいてlocationが属する座標系からこのCRSの座標に変換するクォータニオンを得る。
    virtual Quaternion getQuaternionToAnotherGeodeticCRS(const Eigen::Vector3d& location, const Time& time, const std::shared_ptr<GeodeticCRS>& dstCRS) const; //!< ある時刻timeの基準点locationにおいてこのCRSの座標をdstCRSの座標に変換するクォータニオンを得る。
    virtual Quaternion getQuaternionFromAnotherGeodeticCRS(const Eigen::Vector3d& location, const Time& time, const std::shared_ptr<GeodeticCRS>& srcCRS) const; //!< ある時刻timeの基準点locationにおいてsrcCRSの座標をこのCRSの座標に変換するクォータニオンを得る。
    ///@}

    const double& semiMajorAxis() const; //!< 長半径(赤道半径) [m] を返す。
    const double& inverseFlattening() const; //!< 扁平率の逆数 [m] を返す。
    protected:
    void setGeographicCRS(const std::shared_ptr<GeographicCRS>& geog_) const; //!< [For internal use] 自身と同じdatumのGeographicCRSとしてgeog_を設定する。
    static void setGeographicCRS(const std::shared_ptr<const GeodeticCRS>& base, const std::shared_ptr<GeographicCRS>& geog_); //!< [For internal use] baseと同じdatumのGeographicCRSとしてgeog_を設定する。
    void setECEF(const std::shared_ptr<GeodeticCRS>& ecef_) const; //!< [For internal use] 自身と同じdatumのECEFとしてecef_を設定する。
    static void setECEF(const std::shared_ptr<const GeodeticCRS>& base, const std::shared_ptr<GeodeticCRS>& ecef_); //!< [For internal use] baseと同じdatumのECEFとしてecef_を設定する。
    void setSphericalECEF(const std::shared_ptr<GeodeticCRS>& spherical_ecef_) const; //!< [For internal use] 自身と同じdatumのSpherical ECEFとしてspherical_ecef_を設定する。
    static void setSphericalECEF(const std::shared_ptr<const GeodeticCRS>& base, const std::shared_ptr<GeodeticCRS>& spherical_ecef_); //!< [For internal use] baseと同じdatumのSpherical ECEFとしてspherical_ecef_を設定する。
    void setECI(const std::shared_ptr<EarthCenteredInetialCRS>& eci_) const; //!< [For internal use] 自身と同じdatumのECIとしてeci_を設定する。
    static void setECI(const std::shared_ptr<const GeodeticCRS>& base, const std::shared_ptr<EarthCenteredInetialCRS>& eci_); //!< [For internal use] baseと同じdatumのECIとしてeci_を設定する。

    mutable std::weak_ptr<GeographicCRS> _geog; //!< 同じdatumのGeographicCRS
    mutable std::shared_ptr<GeodeticCRS> _ecef; //!< 同じdatumのECEF
    mutable std::weak_ptr<GeodeticCRS> _spherical_ecef; //!< 同じdatumのspheciral ECEF
    mutable std::weak_ptr<EarthCenteredInetialCRS> _eci; //!< 同じdatumのECI
    mutable bool _autoGenerated; //!< 他のGeodeticCRSインスタンスから自動生成されたかどうか
    mutable bool _isSpherical; //!< 球座標系(Azimuth-Elevation-Range)かどうか
    mutable double _semiMajorAxis; //!< 長半径(赤道半径) [m]
    mutable double _inverseFlattening; //!< 扁平率の逆数 [m]
};

/**
 * @class GeographicCRS
 * @brief 緯度、経度、高度で位置を表現する測地座標系を表すCRS。
 * 
 * @details
 *      緯度は測地系緯度（楕円体表面における法線と赤道面のなす角度）を指す。
 */
ASRC_DECLARE_DERIVED_CLASS_WITHOUT_TRAMPOLINE(GeographicCRS,GeodeticCRS)
    public:
    GeographicCRS(const nl::json& modelConfig_,const nl::json& instanceConfig_);
    //座標系の種類
    virtual bool isGeocentric() const override; //地球を中心とした座標系かどうか
    virtual bool isGeographic() const override; //地理座標系(lat-lon-alt)かどうか
    //Coordinate system axisの種類
    virtual bool isCartesian() const override;
    virtual bool isSpherical() const override;
    virtual bool isEllipsoidal() const override;
    //「高度」の取得
    public:
    using CoordinateReferenceSystem::getHeight;
    using CoordinateReferenceSystem::getGeoidHeight;
    virtual double getHeight(const Eigen::Vector3d& location, const Time& time) const override; //楕円体高度(幾何高度)を返す。ellipsoid (geometric) height
    virtual double getGeoidHeight(const Eigen::Vector3d& location, const Time& time) const override; //標高(ジオイド高度)を返す。geoid height (elevation)
    // 座標変換
    using BaseType::getQuaternionTo;
    using BaseType::getQuaternionFrom;
    virtual Quaternion getQuaternionTo(const Eigen::Vector3d& location, const Time& time, const std::shared_ptr<CoordinateReferenceSystem>& dstCRS) const override; //無効化
    virtual Quaternion getQuaternionFrom(const Eigen::Vector3d& location, const Time& time, const std::shared_ptr<CoordinateReferenceSystem>& srcCRS) const override; //無効化
    // 同じdatumでgeographicなCRSを返す。
    virtual std::shared_ptr<GeographicCRS> getGeographicCRS() const override;
    // 同じdatumでgeocentricなCRSを返す。
    virtual std::shared_ptr<GeodeticCRS> getECEF() const override;
    // 同じdatumでgeocentricかつEarth-fixedなCRS(Spherical ECEF: geocentric lat-lon-radius)を返す。
    virtual std::shared_ptr<GeodeticCRS> getSphericalECEF() const override;
    // 同じdatumでgeocentricかつinertialなCRS(ECI)を返す。
    virtual std::shared_ptr<EarthCenteredInetialCRS> getECI() const override;
    // 同じdatumでgeographicなCRSとの間で座標変換する。
    using BaseType::transformToGeographicCRS;
    using BaseType::transformFromGeographicCRS;
    virtual Coordinate transformToGeographicCRS(const Eigen::Vector3d& value, const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType) const override;
    virtual Coordinate transformFromGeographicCRS(const Eigen::Vector3d& value, const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType) const override;

    // 同じdatumのECEFとの間で座標変換する。
    using BaseType::transformToECEF;
    using BaseType::transformFromECEF;
    virtual Coordinate transformToECEF(const Eigen::Vector3d& value, const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType) const override;
    virtual Coordinate transformFromECEF(const Eigen::Vector3d& value, const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType) const override;

    // 同じdatumのSpherical ECEFとの間で座標変換する。
    using BaseType::transformToSphericalECEF;
    using BaseType::transformFromSphericalECEF;
    virtual Coordinate transformToSphericalECEF(const Eigen::Vector3d& value, const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType) const override;
    virtual Coordinate transformFromSphericalECEF(const Eigen::Vector3d& value, const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType) const override;

    // 同じdatumのECIとの間で座標変換する。
    using BaseType::transformToECI;
    using BaseType::transformFromECI;
    virtual Coordinate transformToECI(const Eigen::Vector3d& value, const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType) const override;
    virtual Coordinate transformFromECI(const Eigen::Vector3d& value, const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType) const override;

    using BaseType::getQuaternionToAnotherGeodeticCRS;
    using BaseType::getQuaternionFromAnotherGeodeticCRS;
    virtual Quaternion getQuaternionToAnotherGeodeticCRS(const Eigen::Vector3d& location, const Time& time, const std::shared_ptr<GeodeticCRS>& dstCRS) const override; //無効化
    virtual Quaternion getQuaternionFromAnotherGeodeticCRS(const Eigen::Vector3d& location, const Time& time, const std::shared_ptr<GeodeticCRS>& srcCRS) const override; //無効化
    // このCRS上の点originを原点とするtopocentricなCRSを生成して返す。
    using BaseType::createTopocentricCRS;
    using BaseType::getEquivalentQuaternionToTopocentricCRS;
    virtual std::shared_ptr<TopocentricCRS> createTopocentricCRS(const Eigen::Vector3d& origin, const Time& time, const std::string& axisOrder, bool onSurface, bool isEpisodic) const override;
    virtual Quaternion getEquivalentQuaternionToTopocentricCRS(const Eigen::Vector3d& origin, const Time& time, const std::string& axisOrder) const override;//CRSとして生成せずにクォータニオンとして得る。
    // このCRSをparentとして持つMotionStateに対応するAffineCRSを生成して返す。
    virtual std::shared_ptr<AffineCRS> createAffineCRS(const MotionState& motion, const std::string& axisOrder, bool isEpisodic) const override;
};

/**
 * @class EarthCenteredInetialCRS
 * @brief EME2000やICRF等、地球中心かつ回転しない座標系。大抵の用途では慣性系とみなせる。
 * 
 * @details
 *      時刻を指定することで地球の回転状態を特定でき、ECEF等のGeodeticCRSに変換できる。
 *      測地系ISO 19111の外にあるため、ECEFとの相互変換のみを定義し、それ以外とはECEFを介して変換するものとする。
 */
ASRC_DECLARE_DERIVED_CLASS_WITH_TRAMPOLINE(EarthCenteredInetialCRS,GeodeticCRS)
    public:
    EarthCenteredInetialCRS(const nl::json& modelConfig_,const nl::json& instanceConfig_);
    virtual bool isEarthFixed() const override;
    //「高度」の取得
    public:
    using BaseType::getHeight;
    using BaseType::getGeoidHeight;
    virtual double getHeight(const Eigen::Vector3d& location, const Time& time) const override; //楕円体高度(幾何高度)を返す。ellipsoid (geometric) height
    virtual double getGeoidHeight(const Eigen::Vector3d& location, const Time& time) const override; //標高(ジオイド高度)を返す。geoid height (elevation)
    // 同じdatumでgeographicなCRSを返す。
    virtual std::shared_ptr<GeographicCRS> getGeographicCRS() const override;
    // 同じdatumでgeocentricなCRSを返す。
    virtual std::shared_ptr<GeodeticCRS> getECEF() const override;
    // 同じdatumでgeocentricかつEarth-fixedなCRS(Spherical ECEF: geocentric lat-lon-radius)を返す。
    virtual std::shared_ptr<GeodeticCRS> getSphericalECEF() const override;
    // 同じdatumでgeocentricかつinertialなCRS(ECI)を返す。
    virtual std::shared_ptr<EarthCenteredInetialCRS> getECI() const override;

    virtual bool isTransformableToAnotherGeodeticCRS(const std::shared_ptr<GeodeticCRS>& other) const override;
    virtual bool isTransformableFromAnotherGeodeticCRS(const std::shared_ptr<GeodeticCRS>& other) const override;
    // 同じdatumでgeographicなCRSとの間で座標変換する。
    using BaseType::transformToGeographicCRS;
    using BaseType::transformFromGeographicCRS;
    virtual Coordinate transformToGeographicCRS(const Eigen::Vector3d& value, const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType) const override;
    virtual Coordinate transformFromGeographicCRS(const Eigen::Vector3d& value, const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType) const override;

    // 同じdatumのECEFとの間で座標変換する。
    using BaseType::transformToECEF;
    using BaseType::transformFromECEF;
    virtual Coordinate transformToECEF(const Eigen::Vector3d& value, const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType) const override;
    virtual Coordinate transformFromECEF(const Eigen::Vector3d& value, const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType) const override;

    // 同じdatumのSpherical ECEFとの間で座標変換する。
    using BaseType::transformToSphericalECEF;
    using BaseType::transformFromSphericalECEF;
    virtual Coordinate transformToSphericalECEF(const Eigen::Vector3d& value, const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType) const override;
    virtual Coordinate transformFromSphericalECEF(const Eigen::Vector3d& value, const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType) const override;

    // 同じdatumのECIとの間で座標変換する。
    using BaseType::transformToECI;
    using BaseType::transformFromECI;
    virtual Coordinate transformToECI(const Eigen::Vector3d& value, const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType) const override;
    virtual Coordinate transformFromECI(const Eigen::Vector3d& value, const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType) const override;

    // 同じdatumでECI座標をECEF座標に変換する。(internal use)
    // 派生クラスでオーバーライドして変換処理を記述すること。
    virtual Eigen::Vector3d transformECICoordinateToECEF(const Eigen::Vector3d& value, const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType) const=0;
    virtual Eigen::Vector3d transformECICoordinateToECEF(const Eigen::Vector3d& value, const Time& time, const CoordinateType& coordinateType) const=0;
    // 同じdatumでECEF座標をECI座標に変換する。(internal use)
    // 派生クラスでオーバーライドして変換処理を記述すること。
    virtual Eigen::Vector3d transformECEFCoordinateToECI(const Eigen::Vector3d& value, const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType) const=0;
    virtual Eigen::Vector3d transformECEFCoordinateToECI(const Eigen::Vector3d& value, const Time& time, const CoordinateType& coordinateType) const=0;

    using BaseType::getQuaternionToAnotherGeodeticCRS;
    using BaseType::getQuaternionFromAnotherGeodeticCRS;
    virtual Quaternion getQuaternionToAnotherGeodeticCRS(const Eigen::Vector3d& location, const Time& time, const std::shared_ptr<GeodeticCRS>& dstCRS) const override;
    virtual Quaternion getQuaternionFromAnotherGeodeticCRS(const Eigen::Vector3d& location, const Time& time, const std::shared_ptr<GeodeticCRS>& srcCRS) const override;
    Quaternion getQuaternionToECEF(const Coordinate& location) const;
    virtual Quaternion getQuaternionToECEF(const Eigen::Vector3d& location, const Time& time) const;
    virtual Quaternion getQuaternionFromECEF(const Eigen::Vector3d& location, const Time& time) const;

    // このCRS上の点originを原点とするtopocentricなCRSを生成して返す。
    using BaseType::getEquivalentQuaternionToTopocentricCRS;
    virtual Quaternion getEquivalentQuaternionToTopocentricCRS(const Eigen::Vector3d& origin, const Time& time, const std::string& axisOrder) const override;//CRSとして生成せずにクォータニオンとして得る。
};

/**
 * @class IdealSteadyRotationECI
 * @brief 回転軸で等速で自転する地球を仮定したECIを表すクラス。ECIの具象クラスの例として提供する。
 * 
 * @details
 *      ECEFと座標軸が一致する時刻(epoch)と、角速度ベクトルをECEF座標で与える。
 *      原点オフセットも無視する。
 */
ASRC_DECLARE_DERIVED_CLASS_WITH_TRAMPOLINE(IdealSteadyRotationECI,EarthCenteredInetialCRS)
    public:
    using BaseType::BaseType;
    virtual void initialize() override;
    // 内部状態のシリアライゼーション
    virtual void serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full) override;
    // 同じdatumでECI座標をECEF座標に変換する。(internal use)
    // 派生クラスでオーバーライドして変換処理を記述すること。
    virtual Eigen::Vector3d transformECICoordinateToECEF(const Eigen::Vector3d& value, const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType) const override;
    virtual Eigen::Vector3d transformECICoordinateToECEF(const Eigen::Vector3d& value, const Time& time, const CoordinateType& coordinateType) const override;
    // 同じdatumでECEF座標をECI座標に変換する。(internal use)
    // 派生クラスでオーバーライドして変換処理を記述すること。
    virtual Eigen::Vector3d transformECEFCoordinateToECI(const Eigen::Vector3d& value, const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType) const override;
    virtual Eigen::Vector3d transformECEFCoordinateToECI(const Eigen::Vector3d& value, const Time& time, const CoordinateType& coordinateType) const override;

    using BaseType::getQuaternionToECEF;
    virtual Quaternion getQuaternionToECEF(const Eigen::Vector3d& location, const Time& time) const override;
    virtual Quaternion getQuaternionFromECEF(const Eigen::Vector3d& location, const Time& time) const override;
    protected:
    Epoch epoch;
    Eigen::Vector3d axis; //ECEF座標系でみた角速度ベクトルの向き
};

ASRC_DECLARE_DERIVED_TRAMPOLINE(GeodeticCRS)
    //Coordinate system axisの種類
    virtual bool isCartesian() const override{
        PYBIND11_OVERRIDE(bool,Base,isCartesian);
    }
    virtual bool isSpherical() const override{
        PYBIND11_OVERRIDE(bool,Base,isSpherical);
    }
    virtual bool isEllipsoidal() const override{
        PYBIND11_OVERRIDE(bool,Base,isEllipsoidal);
    }
    //「高度」の取得
    using Base::getHeight;
    using Base::getGeoidHeight;
    virtual double getHeight(const Eigen::Vector3d& location, const Time& time) const override{
        PYBIND11_OVERRIDE(double,Base,getHeight,location,time);
    }
    virtual double getGeoidHeight(const Eigen::Vector3d& location, const Time& time) const override{
        PYBIND11_OVERRIDE(double,Base,getGeoidHeight,location,time);
    }
    // CRSインスタンス間の変換可否
    virtual bool isTransformableTo(const std::shared_ptr<CoordinateReferenceSystem>& other) const override{
        PYBIND11_OVERRIDE(bool,Base,isTransformableTo,other);
    }
    virtual bool isTransformableFrom(const std::shared_ptr<CoordinateReferenceSystem>& other) const override{
        PYBIND11_OVERRIDE(bool,Base,isTransformableFrom,other);
    }
    // 座標変換
    using Base::transformTo;
    using Base::transformFrom;
    using Base::getQuaternionTo;
    using Base::getQuaternionFrom;
    virtual Coordinate transformTo(const Eigen::Vector3d& value, const Eigen::Vector3d& location, const Time& time, const std::shared_ptr<CoordinateReferenceSystem>& dstCRS, const CoordinateType& coordinateType) const override{
        PYBIND11_OVERRIDE(Coordinate,Base,transformTo,value,location,time,dstCRS,coordinateType);
    }
    virtual Coordinate transformFrom(const Eigen::Vector3d& value, const Eigen::Vector3d& location, const Time& time, const std::shared_ptr<CoordinateReferenceSystem>& srcCRS, const CoordinateType& coordinateType) const override{
        PYBIND11_OVERRIDE(Coordinate,Base,transformFrom,value,location,time,srcCRS,coordinateType);
    }
    virtual Quaternion getQuaternionTo(const Eigen::Vector3d& location, const Time& time, const std::shared_ptr<CoordinateReferenceSystem>& dstCRS) const override{
        PYBIND11_OVERRIDE(Quaternion,Base,getQuaternionTo,location,time,dstCRS);
    }
    virtual Quaternion getQuaternionFrom(const Eigen::Vector3d& location, const Time& time, const std::shared_ptr<CoordinateReferenceSystem>& srcCRS) const override{
        PYBIND11_OVERRIDE(Quaternion,Base,getQuaternionFrom,location,time,srcCRS);
    }
    // このCRS上の点originを原点とするtopocentricなCRSを生成して返す。
    using Base::getEquivalentQuaternionToTopocentricCRS;
    virtual Quaternion getEquivalentQuaternionToTopocentricCRS(const Eigen::Vector3d& origin, const Time& time, const std::string& axisOrder) const override{
        PYBIND11_OVERRIDE(Quaternion,Base,getEquivalentQuaternionToTopocentricCRS,origin,time,axisOrder);
    }
    //
    // 以下は本クラスで追加定義するメンバ
    //
    virtual bool isEarthFixed() const override{
        PYBIND11_OVERRIDE(bool,Base,isEarthFixed);
    }
    virtual bool isGeocentric() const override{
        PYBIND11_OVERRIDE(bool,Base,isGeocentric);
    }
    virtual bool isGeographic() const override{
        PYBIND11_OVERRIDE(bool,Base,isGeographic);
    }
    // 同じdatumでgeographicなCRSを返す。
    virtual std::shared_ptr<GeographicCRS> getGeographicCRS() const override{
        PYBIND11_OVERRIDE(std::shared_ptr<GeographicCRS>,Base,getGeographicCRS);
    }
    // 同じdatumでgeocentricかつEarth-fixedなCRS(ECEF)を返す。
    virtual std::shared_ptr<GeodeticCRS> getECEF() const override{
        PYBIND11_OVERRIDE(std::shared_ptr<GeodeticCRS>,Base,getECEF);
    }
    // 同じdatumでgeocentricかつEarth-fixedなCRS(Spherical ECEF: geocentric lat-lon-radius)を返す。
    virtual std::shared_ptr<GeodeticCRS> getSphericalECEF() const override{
        PYBIND11_OVERRIDE(std::shared_ptr<GeodeticCRS>,Base,getSphericalECEF);
    }
    // 同じdatumでgeocentricかつinertialなCRS(ECI)を返す。
    virtual std::shared_ptr<EarthCenteredInetialCRS> getECI() const override{
        PYBIND11_OVERRIDE(std::shared_ptr<EarthCenteredInetialCRS>,Base,getECI);
    }
    // 同じdatumでgeographicなCRSに座標変換する。
    using Base::transformToGeographicCRS;
    using Base::transformFromGeographicCRS;
    virtual Coordinate transformToGeographicCRS(const Eigen::Vector3d& value, const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType) const override{
        PYBIND11_OVERRIDE(Coordinate,Base,transformToGeographicCRS,value,location,time,coordinateType);
    }
    virtual Coordinate transformFromGeographicCRS(const Eigen::Vector3d& value, const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType) const override{
        PYBIND11_OVERRIDE(Coordinate,Base,transformFromGeographicCRS,value,location,time,coordinateType);
    }
    virtual Coordinate transformToGeographicCRS(const Eigen::Vector3d& value, const Time& time, const CoordinateType& coordinateType) const override{
        PYBIND11_OVERRIDE(Coordinate,Base,transformToGeographicCRS,value,time,coordinateType);
    }
    virtual Coordinate transformFromGeographicCRS(const Eigen::Vector3d& value, const Time& time, const CoordinateType& coordinateType) const override{
        PYBIND11_OVERRIDE(Coordinate,Base,transformFromGeographicCRS,value,time,coordinateType);
    }
    // 同じdatumのECEFに座標変換する。
    using Base::transformToECEF;
    using Base::transformFromECEF;
    virtual Coordinate transformToECEF(const Eigen::Vector3d& value, const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType) const override{
        PYBIND11_OVERRIDE(Coordinate,Base,transformToECEF,value,location,time,coordinateType);
    }
    virtual Coordinate transformFromECEF(const Eigen::Vector3d& value, const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType) const override{
        PYBIND11_OVERRIDE(Coordinate,Base,transformFromECEF,value,location,time,coordinateType);
    }
    virtual Coordinate transformToECEF(const Eigen::Vector3d& value, const Time& time, const CoordinateType& coordinateType) const override{
        PYBIND11_OVERRIDE(Coordinate,Base,transformToECEF,value,time,coordinateType);
    }
    virtual Coordinate transformFromECEF(const Eigen::Vector3d& value, const Time& time, const CoordinateType& coordinateType) const override{
        PYBIND11_OVERRIDE(Coordinate,Base,transformFromECEF,value,time,coordinateType);
    }
    // 同じdatumのSpherical ECEFに座標変換する。
    using Base::transformToSphericalECEF;
    using Base::transformFromSphericalECEF;
    virtual Coordinate transformToSphericalECEF(const Eigen::Vector3d& value, const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType) const override{
        PYBIND11_OVERRIDE(Coordinate,Base,transformToSphericalECEF,value,location,time,coordinateType);
    }
    virtual Coordinate transformFromSphericalECEF(const Eigen::Vector3d& value, const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType) const override{
        PYBIND11_OVERRIDE(Coordinate,Base,transformFromSphericalECEF,value,location,time,coordinateType);
    }
    virtual Coordinate transformToSphericalECEF(const Eigen::Vector3d& value, const Time& time, const CoordinateType& coordinateType) const override{
        PYBIND11_OVERRIDE(Coordinate,Base,transformToSphericalECEF,value,time,coordinateType);
    }
    virtual Coordinate transformFromSphericalECEF(const Eigen::Vector3d& value, const Time& time, const CoordinateType& coordinateType) const override{
        PYBIND11_OVERRIDE(Coordinate,Base,transformFromSphericalECEF,value,time,coordinateType);
    }
    // 同じdatumのECIに座標変換する。
    using Base::transformToECI;
    using Base::transformFromECI;
    virtual Coordinate transformToECI(const Eigen::Vector3d& value, const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType) const override{
        PYBIND11_OVERRIDE(Coordinate,Base,transformToECI,value,location,time,coordinateType);
    }
    virtual Coordinate transformFromECI(const Eigen::Vector3d& value, const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType) const override{
        PYBIND11_OVERRIDE(Coordinate,Base,transformFromECI,value,location,time,coordinateType);
    }
    virtual Coordinate transformToECI(const Eigen::Vector3d& value, const Time& time, const CoordinateType& coordinateType) const override{
        PYBIND11_OVERRIDE(Coordinate,Base,transformToECI,value,time,coordinateType);
    }
    virtual Coordinate transformFromECI(const Eigen::Vector3d& value, const Time& time, const CoordinateType& coordinateType) const override{
        PYBIND11_OVERRIDE(Coordinate,Base,transformFromECI,value,time,coordinateType);
    }
    // 同じdatumでgeographic座標をECEF座標に変換する。
    virtual Eigen::Vector3d transformGeographicCoordinateToECEF(const Eigen::Vector3d& value, const Eigen::Vector3d& location, const CoordinateType& coordinateType) const override{
        PYBIND11_OVERRIDE(Eigen::Vector3d,Base,transformGeographicCoordinateToECEF,value,location,coordinateType);
    }
    virtual Eigen::Vector3d transformGeographicCoordinateToECEF(const Eigen::Vector3d& value, const CoordinateType& coordinateType) const override{
        PYBIND11_OVERRIDE(Eigen::Vector3d,Base,transformGeographicCoordinateToECEF,value,coordinateType);
    }
    // 同じdatumでECEF座標をgeographic座標に変換する。
    virtual Eigen::Vector3d transformECEFCoordinateToGeographicCRS(const Eigen::Vector3d& value, const Eigen::Vector3d& location, const CoordinateType& coordinateType) const override{
        PYBIND11_OVERRIDE(Eigen::Vector3d,Base,transformECEFCoordinateToGeographicCRS,value,location,coordinateType);
    }
    virtual Eigen::Vector3d transformECEFCoordinateToGeographicCRS(const Eigen::Vector3d& value, const CoordinateType& coordinateType) const override{
        PYBIND11_OVERRIDE(Eigen::Vector3d,Base,transformECEFCoordinateToGeographicCRS,value,coordinateType);
    }
    // 異なるdatumのGeodeticCRSに座標変換する。
    virtual bool isTransformableToAnotherGeodeticCRS(const std::shared_ptr<GeodeticCRS>& other) const override{
        PYBIND11_OVERRIDE(bool,Base,isTransformableToAnotherGeodeticCRS,other);
    }
    virtual bool isTransformableFromAnotherGeodeticCRS(const std::shared_ptr<GeodeticCRS>& other) const override{
        PYBIND11_OVERRIDE(bool,Base,isTransformableFromAnotherGeodeticCRS,other);
    }
    using Base::transformToAnotherGeodeticCRS;
    using Base::transformFromAnotherGeodeticCRS;
    using Base::getQuaternionToAnotherGeodeticCRS;
    using Base::getQuaternionFromAnotherGeodeticCRS;
    virtual Coordinate transformToAnotherGeodeticCRS(const Eigen::Vector3d& value, const Eigen::Vector3d& location, const Time& time, const std::shared_ptr<GeodeticCRS>& dstCRS, const CoordinateType& coordinateType) const override{
        PYBIND11_OVERRIDE(Coordinate,Base,transformToAnotherGeodeticCRS,value,location,time,dstCRS,coordinateType);
    }
    virtual Coordinate transformFromAnotherGeodeticCRS(const Eigen::Vector3d& value, const Eigen::Vector3d& location, const Time& time, const std::shared_ptr<GeodeticCRS>& srcCRS, const CoordinateType& coordinateType) const override{
        PYBIND11_OVERRIDE(Coordinate,Base,transformFromAnotherGeodeticCRS,value,location,time,srcCRS,coordinateType);
    }
    virtual Coordinate transformToAnotherGeodeticCRS(const Eigen::Vector3d& value, const Time& time, const std::shared_ptr<GeodeticCRS>& dstCRS, const CoordinateType& coordinateType) const override{
        PYBIND11_OVERRIDE(Coordinate,Base,transformToAnotherGeodeticCRS,value,time,dstCRS,coordinateType);
    }
    virtual Coordinate transformFromAnotherGeodeticCRS(const Eigen::Vector3d& value, const Time& time, const std::shared_ptr<GeodeticCRS>& srcCRS, const CoordinateType& coordinateType) const override{
        PYBIND11_OVERRIDE(Coordinate,Base,transformFromAnotherGeodeticCRS,value,time,srcCRS,coordinateType);
    }
    virtual Quaternion getQuaternionToAnotherGeodeticCRS(const Eigen::Vector3d& location, const Time& time, const std::shared_ptr<GeodeticCRS>& dstCRS) const override{
        PYBIND11_OVERRIDE(Quaternion,Base,getQuaternionToAnotherGeodeticCRS,location,time,dstCRS);
    }
    virtual Quaternion getQuaternionFromAnotherGeodeticCRS(const Eigen::Vector3d& location, const Time& time, const std::shared_ptr<GeodeticCRS>& srcCRS) const override{
        PYBIND11_OVERRIDE(Quaternion,Base,getQuaternionFromAnotherGeodeticCRS,location,time,srcCRS);
    }
};

ASRC_DECLARE_DERIVED_TRAMPOLINE(EarthCenteredInetialCRS)
    // 同じdatumでECI座標をECEF座標に変換する。(internal use)
    // 派生クラスでオーバーライドして変換処理を記述すること。
    virtual Eigen::Vector3d transformECICoordinateToECEF(const Eigen::Vector3d& value, const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType) const override{
        PYBIND11_OVERRIDE_PURE(Eigen::Vector3d,Base,transformECICoordinateToECEF,value,location,time,coordinateType);
    }
    virtual Eigen::Vector3d transformECICoordinateToECEF(const Eigen::Vector3d& value, const Time& time, const CoordinateType& coordinateType) const override{
        PYBIND11_OVERRIDE_PURE(Eigen::Vector3d,Base,transformECICoordinateToECEF,value,time,coordinateType);
    }
    // 同じdatumでECEF座標をECI座標に変換する。(internal use)
    // 派生クラスでオーバーライドして変換処理を記述すること。
    virtual Eigen::Vector3d transformECEFCoordinateToECI(const Eigen::Vector3d& value, const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType) const override{
        PYBIND11_OVERRIDE_PURE(Eigen::Vector3d,Base,transformECEFCoordinateToECI,value,location,time,coordinateType);
    }
    virtual Eigen::Vector3d transformECEFCoordinateToECI(const Eigen::Vector3d& value, const Time& time, const CoordinateType& coordinateType) const override{
        PYBIND11_OVERRIDE_PURE(Eigen::Vector3d,Base,transformECEFCoordinateToECI,value,time,coordinateType);
    }
    using Base::getQuaternionToECEF;
    virtual Quaternion getQuaternionToECEF(const Eigen::Vector3d& location, const Time& time) const override{
        PYBIND11_OVERRIDE(Quaternion,Base,getQuaternionToECEF,location,time);
    }
    virtual Quaternion getQuaternionFromECEF(const Eigen::Vector3d& location, const Time& time) const override{
        PYBIND11_OVERRIDE(Quaternion,Base,getQuaternionFromECEF,location,time);
    }
};

ASRC_DECLARE_DERIVED_TRAMPOLINE(IdealSteadyRotationECI)
    // 同じdatumでECI座標をECEF座標に変換する。(internal use)
    // 派生クラスでオーバーライドして変換処理を記述すること。
    virtual Eigen::Vector3d transformECICoordinateToECEF(const Eigen::Vector3d& value, const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType) const override{
        PYBIND11_OVERRIDE(Eigen::Vector3d,Base,transformECICoordinateToECEF,value,location,time,coordinateType);
    }
    virtual Eigen::Vector3d transformECICoordinateToECEF(const Eigen::Vector3d& value, const Time& time, const CoordinateType& coordinateType) const override{
        PYBIND11_OVERRIDE(Eigen::Vector3d,Base,transformECICoordinateToECEF,value,time,coordinateType);
    }
    // 同じdatumでECEF座標をECI座標に変換する。(internal use)
    // 派生クラスでオーバーライドして変換処理を記述すること。
    virtual Eigen::Vector3d transformECEFCoordinateToECI(const Eigen::Vector3d& value, const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType) const override{
        PYBIND11_OVERRIDE(Eigen::Vector3d,Base,transformECEFCoordinateToECI,value,location,time,coordinateType);
    }
    virtual Eigen::Vector3d transformECEFCoordinateToECI(const Eigen::Vector3d& value, const Time& time, const CoordinateType& coordinateType) const override{
        PYBIND11_OVERRIDE(Eigen::Vector3d,Base,transformECEFCoordinateToECI,value,time,coordinateType);
    }
};

void exportGeodeticCRS(py::module &m, const std::shared_ptr<asrc::core::FactoryHelper>& factoryHelper);

ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
