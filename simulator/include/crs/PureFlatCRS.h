/**
 * Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
 * @file
 * 
 * @brief 仮想的な水平な慣性座標系を表すクラス
 */
#pragma once
#include "../Common.h"
#include "CoordinateReferenceSystemBase.h"

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)

class DerivedCRS;

/**
 * @class PureFlatCRS
 * @brief 仮想的な水平な慣性座標系を表すクラス
 * 
 * @details
 *      局所水平座標系に近いが、地球の自転と曲率を無視しており、慣性力や高度方向の歪みが無視される。
 *      - 基準座標系として設定した場合のみしか使用できない。
 *      - 座標軸は直交座標系のみとする。
 *      - 暗黙に何らかのGeodeticCRS上における原点情報を持つ。
 */
ASRC_DECLARE_DERIVED_CLASS_WITH_TRAMPOLINE(PureFlatCRS,CoordinateReferenceSystem)
    friend class DerivedCRS;
    public:
    PureFlatCRS(const nl::json& modelConfig_,const nl::json& instanceConfig_);
    virtual void initialize() override;
    // 内部状態のシリアライゼーション
    virtual void serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full) override;
    //座標系の種類
    virtual bool isEarthFixed() const override; //地球に対して固定されているかどうか
    virtual bool isGeocentric() const override; //地球を中心とした座標系かどうか
    virtual bool isGeographic() const override; //地理座標系(lat-lon-alt)かどうか
    //Coordinate system axisの種類
    virtual bool isCartesian() const override; // true
    virtual bool isSpherical() const override; // false
    virtual bool isEllipsoidal() const override; // false
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
    //「高度」の取得
    public:
    using BaseType::getHeight;
    using BaseType::getGeoidHeight;
    virtual double getHeight(const Eigen::Vector3d& location, const Time& time) const override; //楕円体高度(幾何高度)を返す。ellipsoid (geometric) height
    virtual double getGeoidHeight(const Eigen::Vector3d& location, const Time& time) const override; //標高(ジオイド高度)を返す。geoid height (elevation)

    // このCRS上の点originを原点とするtopocentricなCRSを生成して返す。
    using BaseType::getEquivalentQuaternionToTopocentricCRS;
    virtual Quaternion getEquivalentQuaternionToTopocentricCRS(const Eigen::Vector3d& origin, const Time& time, const std::string& axisOrder) const override;//CRSとして生成せずにクォータニオンとして得る。

    //
    // 以下は本クラスで追加定義するメンバ
    //
    // 軸順の取得
    virtual std::string getAxisOrder() const; //!< 座標軸の順序("NED"等)を返す。
    // Topocentricとしてのインターフェース
    virtual Eigen::Vector3d getEllipsoidalOrigin() const; //!< 原点の地理座標(lat-lon-alt)を返す。
    virtual Eigen::Vector3d getCartesianOrigin() const; //!< 原点のECEF座標を返す。

    protected:
    mutable std::string axisOrder; //!< 座標軸の順序("NED"等)
    mutable bool givenByEllipsoidalOrigin; //!< 生成時に原点の地理座標(lat-lon-alt)を与えられたかどうか
    mutable bool givenByCartesianOrigin; //!< 生成時に原点のECEF座標を与えられたかどうか
    mutable Eigen::Vector3d ellipsoidalOrigin; // in lat[deg], lon[deg], alt[m]
    mutable Eigen::Vector3d cartesianOrigin; // in X[m], Y[m], Z[m]
};

ASRC_DECLARE_DERIVED_TRAMPOLINE(PureFlatCRS)
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
    //「高度」の取得
    using Base::getHeight;
    using Base::getGeoidHeight;
    virtual double getHeight(const Eigen::Vector3d& location, const Time& time) const override{
        PYBIND11_OVERRIDE(double,Base,getHeight,location,time);
    }
    virtual double getGeoidHeight(const Eigen::Vector3d& location, const Time& time) const override{
        PYBIND11_OVERRIDE(double,Base,getGeoidHeight,location,time);
    }
    // このCRS上の点originを原点とするtopocentricなCRSを生成して返す。
    using Base::getEquivalentQuaternionToTopocentricCRS;
    using Base::transformFromTopocentricCRS;
    virtual Quaternion getEquivalentQuaternionToTopocentricCRS(const Eigen::Vector3d& origin, const Time& time, const std::string& axisOrder) const override{
        PYBIND11_OVERRIDE(Quaternion,Base,getEquivalentQuaternionToTopocentricCRS,origin,time,axisOrder);
    }
    //
    // 以下は本クラスで追加定義するメンバ
    //
    // 軸順の取得
    virtual std::string getAxisOrder() const override{
        PYBIND11_OVERRIDE(std::string,Base,getAxisOrder);
    }
    // Topocentricとしてのインターフェース
    virtual Eigen::Vector3d getEllipsoidalOrigin() const override{
        PYBIND11_OVERRIDE(Eigen::Vector3d,Base,getEllipsoidalOrigin);
    }
    virtual Eigen::Vector3d getCartesianOrigin() const override{
        PYBIND11_OVERRIDE(Eigen::Vector3d,Base,getCartesianOrigin);
    }
};

void exportPureFlatCRS(py::module &m, const std::shared_ptr<asrc::core::FactoryHelper>& factoryHelper);

ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
