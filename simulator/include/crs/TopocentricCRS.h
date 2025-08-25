/**
 * Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
 * @file
 * 
 * @brief 局所水平座標系(通常はENU又はNED)を表すCRS
 */
#pragma once
#include "../Common.h"
#include "DerivedCRS.h"
#include "../MotionState.h"

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)

/**
 * @class TopocentricCRS
 * @brief 局所水平座標系(通常はENU又はNED)を表すCRS
 * 
 * @details
 *      球座標系として扱うことも可能とし、その場合の座標軸の順序は、A-E-R(Azimuth-Elevation-Range)とする。
 *      直交座標→球座標の対応は航空分野に準じて、AzimuthはNorthから時計回り(East側)を正、Elevationは水平面からUP側を正とする。
 * 
 *      body-orientedな座標系として、FWDを水平面に投影した直線を水平面上の座標軸方向とすることも可能とする。
 *      body-orientedとした場合の直交座標→球座標の対応は航空分野に準じて、AzimuthはFWDから時計回り(STBD側)を正、ElevationはUP側を正とする。
 */
ASRC_DECLARE_DERIVED_CLASS_WITH_TRAMPOLINE(TopocentricCRS,DerivedCRS)
    public:
    TopocentricCRS(const nl::json& modelConfig_,const nl::json& instanceConfig_);
    virtual void initialize() override;
    // 内部状態のシリアライゼーション
    virtual void serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full) override;
    protected:
    virtual void validateImpl(bool isFirstTime) const override;
    virtual void setBaseCRS(const std::shared_ptr<CoordinateReferenceSystem>& newBaseCRS) const override;
    public:
    //Coordinate system axisの種類
    virtual bool isCartesian() const override;
    virtual bool isSpherical() const override;
    virtual bool isEllipsoidal() const override; //false
    //座標系の種類
    virtual bool isEarthFixed() const override; //地球に対して固定されているかどうか
    virtual bool isTopocentric() const override; //地球に対して固定されているかどうか
    virtual bool isProjected() const override; //投影された座標系かどうか(ProjectedCRS又はProjectedCRSをbaseとしたDerivedCRSが該当)
    //中間座標系の取得
	virtual std::shared_ptr<CoordinateReferenceSystem> getIntermediateCRS() const override; //親子関係にない他のCRSとの間で座標変換を行う際に経由する中間座標系。TopocentricCRSにとってはnonDerivedBaseCRSのintermediateCRS。
    // 座標変換
    using BaseType::transformTo;
    using BaseType::transformFrom;
    using BaseType::getQuaternionTo;
    using BaseType::getQuaternionFrom;
    virtual Coordinate transformTo(const Eigen::Vector3d& value, const Eigen::Vector3d& location, const Time& time, const std::shared_ptr<CoordinateReferenceSystem>& dstCRS, const CoordinateType& coordinateType) const override;
    virtual Coordinate transformFrom(const Eigen::Vector3d& value, const Eigen::Vector3d& location, const Time& time, const std::shared_ptr<CoordinateReferenceSystem>& srcCRS, const CoordinateType& coordinateType) const override;
    virtual Quaternion getQuaternionTo(const Eigen::Vector3d& location, const Time& time, const std::shared_ptr<CoordinateReferenceSystem>& dstCRS) const override;
    virtual Quaternion getQuaternionFrom(const Eigen::Vector3d& location, const Time& time, const std::shared_ptr<CoordinateReferenceSystem>& srcCRS) const override;
    //親座標系との座標変換
    using BaseType::transformToBaseCRS;
    using BaseType::transformFromBaseCRS;
    using BaseType::transformToNonDerivedBaseCRS;
    using BaseType::transformFromNonDerivedBaseCRS;
    using BaseType::transformToIntermediateCRS;
    using BaseType::transformFromIntermediateCRS;
    virtual Coordinate transformToBaseCRS(const Eigen::Vector3d& value,const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType) const override;
    virtual Coordinate transformFromBaseCRS(const Eigen::Vector3d& value,const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType) const override;
    virtual Coordinate transformToNonDerivedBaseCRS(const Eigen::Vector3d& value,const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType) const override;
    virtual Coordinate transformFromNonDerivedBaseCRS(const Eigen::Vector3d& value,const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType) const override;
    virtual Coordinate transformToIntermediateCRS(const Eigen::Vector3d& value,const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType) const override;
    virtual Coordinate transformFromIntermediateCRS(const Eigen::Vector3d& value,const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType) const override;

    // Cartesian⇔Sphericalの不定性を取り除くための関数
    virtual Eigen::Vector3d cartesianToSpherical(const Eigen::Vector3d& v) const override; //Cartesian (axisOrder)で表された引数をSpherical (Azimuth-Elevation-Range)での値に変換する
    virtual Eigen::Vector3d sphericalToCartesian(const Eigen::Vector3d& v) const override; //Spherical (Azimuth-Elevation-Range)で表された引数をCartesian (axisOrder)での値に変換する

    public:
    // このCRS上の点originを原点とするtopocentricなCRSを生成して返す。
    using BaseType::getEquivalentQuaternionToTopocentricCRS;
    virtual Quaternion getEquivalentQuaternionToTopocentricCRS(const Eigen::Vector3d& origin, const Time& time, const std::string& axisOrder) const override;//CRSとして生成せずにクォータニオンとして得る。

    //
    // 以下は本クラスで追加定義するメンバ
    //
    public:
    virtual std::string getAxisOrder() const; //!< 座標軸の順序("FSD"等)を返す。

    protected:
    virtual void checkCoordinateSystemType() const; //!< [For internal use] 直交座標系か球座標系かをconfigから読み込む
    virtual void checkAxisOrder() const; //!< [For internal use] 座標軸の順序として与えられた文字列が適切なものかどうか判定する。
    virtual void updateByMotionState(const MotionState& motionState_); //!< [For internal use] nonDerivedBaseCRSと同じdatumのECEF又はPureFlat上でこのCRSの状態を表すMotionStateを変更する。
    virtual void calculateMotionStates() const; //!< [For internal use] nonDerivedBaseCRSと同じdatumのECEF又はPureFlat上でこのCRSの状態を表すMotionState。
    public:
    virtual Eigen::Vector3d getEllipsoidalOrigin() const; //!< 原点の地理座標(lat-lon-alt)を返す。
    virtual Eigen::Vector3d getCartesianOrigin() const; //!< 原点のECEF座標を返す。
    protected:
    mutable bool _isSpherical; //!< 球座標系(Azimuth-Elevation-Range)かどうか
    mutable std::string axisOrder; //!< 座標軸の順序("FSD"等)
    mutable MotionState internalMotionState;//!< nonDerivedBaseCRSと同じdatumのECEF又はPureFlat上でこのCRSの状態を表すMotionState。
    mutable bool onSurface; //!< trueの場合、原点を楕円体表面になるように高度方向に平行移動する。
    mutable double lastAZ; //!< body-orientedな局所水平座標系とした場合に、FWDが鉛直軸と一致してしまったときの不定性を除くための変数
    mutable bool givenByEllipsoidalOrigin; //!< 生成時に原点の地理座標(lat-lon-alt)を与えられたかどうか
    mutable bool givenByCartesianOrigin; //!< 生成時に原点のECEF座標を与えられたかどうか
    mutable bool givenByMotionState; //!< 生成時に原点のMotionStateを与えられたかどうか
    mutable Eigen::Vector3d ellipsoidalOrigin; // in lat[deg], lon[deg], alt[m]
    mutable Eigen::Vector3d cartesianOrigin; // in X[m], Y[m], Z[m]/
};

ASRC_DECLARE_DERIVED_TRAMPOLINE(TopocentricCRS)
    // Coordinate system axisの種類
    virtual bool isCartesian() const override{
        PYBIND11_OVERRIDE(bool,Base,isCartesian);
    }
    virtual bool isSpherical() const override{
        PYBIND11_OVERRIDE(bool,Base,isSpherical);
    }
    virtual bool isEllipsoidal() const override{
        PYBIND11_OVERRIDE(bool,Base,isEllipsoidal);
    }
    // 座標変換
    using Base::getQuaternionTo;
    using Base::getQuaternionFrom;
    virtual Quaternion getQuaternionTo(const Eigen::Vector3d& location, const Time& time, const std::shared_ptr<CoordinateReferenceSystem>& dstCRS) const override{
        PYBIND11_OVERRIDE(Quaternion,Base,getQuaternionTo,location,time,dstCRS);
    }
    virtual Quaternion getQuaternionFrom(const Eigen::Vector3d& location, const Time& time, const std::shared_ptr<CoordinateReferenceSystem>& srcCRS) const override{
        PYBIND11_OVERRIDE(Quaternion,Base,getQuaternionFrom,location,time,srcCRS);
    }
    // 親座標系との座標変換
    using Base::transformToBaseCRS;
    using Base::transformFromBaseCRS;
    using Base::transformToNonDerivedBaseCRS;
    using Base::transformFromNonDerivedBaseCRS;
    using Base::transformToIntermediateCRS;
    using Base::transformFromIntermediateCRS;
    virtual Coordinate transformToBaseCRS(const Eigen::Vector3d& value,const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType) const override{
        PYBIND11_OVERRIDE(Coordinate,Base,transformToBaseCRS,value,location,time,coordinateType);
    }
    virtual Coordinate transformFromBaseCRS(const Eigen::Vector3d& value,const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType) const override{
        PYBIND11_OVERRIDE(Coordinate,Base,transformFromBaseCRS,value,location,time,coordinateType);
    }
    virtual Coordinate transformToNonDerivedBaseCRS(const Eigen::Vector3d& value,const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType) const override{
        PYBIND11_OVERRIDE(Coordinate,Base,transformToNonDerivedBaseCRS,value,location,time,coordinateType);
    }
    virtual Coordinate transformFromNonDerivedBaseCRS(const Eigen::Vector3d& value,const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType) const override{
        PYBIND11_OVERRIDE(Coordinate,Base,transformFromNonDerivedBaseCRS,value,location,time,coordinateType);
    }
    virtual Coordinate transformToIntermediateCRS(const Eigen::Vector3d& value,const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType) const override{
        PYBIND11_OVERRIDE(Coordinate,Base,transformToIntermediateCRS,value,location,time,coordinateType);
    }
    virtual Coordinate transformFromIntermediateCRS(const Eigen::Vector3d& value,const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType) const override{
        PYBIND11_OVERRIDE(Coordinate,Base,transformFromIntermediateCRS,value,location,time,coordinateType);
    }
    // このCRS上の点originを原点とするtopocentricなCRSを生成して返す。
    using Base::getEquivalentQuaternionToTopocentricCRS;
    virtual Quaternion getEquivalentQuaternionToTopocentricCRS(const Eigen::Vector3d& origin, const Time& time, const std::string& axisOrder) const override{
        PYBIND11_OVERRIDE(Quaternion,Base,getEquivalentQuaternionToTopocentricCRS,origin,time,axisOrder);
    }
    //
    // 以下は本クラスで追加定義するメンバ
    //
    virtual std::string getAxisOrder() const override{
        PYBIND11_OVERRIDE(std::string,Base,getAxisOrder);
    }
    virtual void updateByMotionState(const MotionState& motionState_) override{
        PYBIND11_OVERRIDE(void,Base,updateByMotionState,motionState_);
    }
    virtual void calculateMotionStates() const override{
        PYBIND11_OVERRIDE(void,Base,calculateMotionStates);
    }
    virtual Eigen::Vector3d getEllipsoidalOrigin() const override{
        PYBIND11_OVERRIDE(Eigen::Vector3d,Base,getEllipsoidalOrigin);
    }
    virtual Eigen::Vector3d getCartesianOrigin() const override{
        PYBIND11_OVERRIDE(Eigen::Vector3d,Base,getCartesianOrigin);
    }
};

void exportTopocentricCRS(py::module &m, const std::shared_ptr<asrc::core::FactoryHelper>& factoryHelper);

ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
