/**
 * Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
 * @file
 * 
 * @brief ある親座標系基準でアフィン変換により得られる相対的な座標系を表すCRS
 */
#pragma once
#include "../Common.h"
#include "DerivedCRS.h"
#include "../MotionState.h"

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)

class SimulationManagerBase;

/**
 * @class AffineCRS
 * @brief ある親座標系基準でアフィン変換により得られる相対的な座標系を表すCRS
 * 
 * @details
 *      直交座標系の場合の座標軸の順序は、船舶と航空機はF-S-D(FWD-STBD-DOWN)、車両はF-P-U(FWD-PORT-UP)が多い。省略時は"FSD"とする。
 *      球座標系として扱うことも可能とし、その場合の座標軸の順序は、A-E-R(Azimuth-Elevation-Range)とする。
 * 
 *      直交座標→球座標の対応は航空分野に準じて、AzimuthはFWDから時計回り(STBD側)を正、ElevationはUP側を正とする。
 *      球座標系とする場合のaxisOrderはMotionStateで保持する直交座標系の軸順を表すものとする。
 */
ASRC_DECLARE_DERIVED_CLASS_WITH_TRAMPOLINE(AffineCRS,DerivedCRS)
    friend class MotionState;
    friend class PhysicalAsset;
    friend class SimulationManagerBase;
    public:
    AffineCRS(const nl::json& modelConfig_,const nl::json& instanceConfig_);
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
    // 座標変換
    using BaseType::getQuaternionTo;
    using BaseType::getQuaternionFrom;
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
    MotionState getInternalMotionState() const; //!< [For internal use] baseCRS上でこのCRSの状態を表すMotionStateを返す。
    virtual void updateByMotionState(const MotionState& motionState_); //!< [For internal use] baseCRS上でこのCRSの状態を表すMotionStateを変更する。
    protected:
    mutable bool _isSpherical; //!< 球座標系(Azimuth-Elevation-Range)かどうか
    mutable std::string axisOrder; //!< 座標軸の順序("FSD"等)
    mutable MotionState internalMotionState;//!< baseCRS上でこのCRSの状態を表すMotionState
};

ASRC_DECLARE_DERIVED_TRAMPOLINE(AffineCRS)
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
};

void exportAffineCRS(py::module &m, const std::shared_ptr<asrc::core::FactoryHelper>& factoryHelper);

ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
