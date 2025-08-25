/**
 * Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
 * @file
 * 
 * @brief 2次元平面に投影された座標系(UTMなど)を表すクラス
 */
#pragma once
#include "../Common.h"
#include "DerivedCRS.h"

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)

/**
 * @class ProjectedCRS
 * @brief 2次元平面に投影された座標系(UTMなど)を表すクラス
 *
 * @details
 *      このクラス自体は具体的な投影図法まで規定しないため、これはまだ抽象クラスである。
 *      必要に応じ更に派生クラスで投影処理を実装すること。
 *      また、baseCRS==nonDerivedBaseCRSかつbaseCRS->isGeodetic()==trueであることを強制する。
 *      intermediateCRSはECEFとなる。
 */
ASRC_DECLARE_DERIVED_CLASS_WITH_TRAMPOLINE(ProjectedCRS,DerivedCRS)
    public:
    ProjectedCRS(const nl::json& modelConfig_,const nl::json& instanceConfig_);
    protected:
    virtual void validateImpl(bool isFirstTime) const override;
    public:
    //座標系の種類
    virtual bool isEarthFixed() const override; //地球に対して固定されているかどうか
    virtual bool isProjected() const override; //投影された座標系かどうか(ProjectedCRS又はProjectedCRSをbaseとしたDerivedCRSが該当)
    //Coordinate system axisの種類
    virtual bool isCartesian() const override; //true
    virtual bool isSpherical() const override; //false
    virtual bool isEllipsoidal() const override; //false
    //中間座標系の取得
	virtual std::shared_ptr<CoordinateReferenceSystem> getIntermediateCRS() const override; //親子関係にない他のCRSとの間で座標変換を行う際に経由する中間座標系。ProjectedCRSにとってはbaseCRSのintermediateCRS。
    // CRSインスタンス間の変換可否
    public:
    virtual bool isTransformableTo(const std::shared_ptr<CoordinateReferenceSystem>& other) const override;
    virtual bool isTransformableFrom(const std::shared_ptr<CoordinateReferenceSystem>& other) const override;
    // 座標変換
    using BaseType::getQuaternionTo;
    using BaseType::getQuaternionFrom;
    virtual Quaternion getQuaternionTo(const Eigen::Vector3d& location, const Time& time, const std::shared_ptr<CoordinateReferenceSystem>& dstCRS) const override;
    virtual Quaternion getQuaternionFrom(const Eigen::Vector3d& location, const Time& time, const std::shared_ptr<CoordinateReferenceSystem>& srcCRS) const override;
    // 親との座標変換
    // 派生クラスでbaseCRSとの双方向のvalue,coordinateTypeの2引数変換関数を実装すること!
    // Implement bi-directional conversion between the baseCRS with two arguments (value and coordinateType.)
    using BaseType::transformToBaseCRS;
    using BaseType::transformFromBaseCRS;
    using BaseType::transformToNonDerivedBaseCRS;
    using BaseType::transformFromNonDerivedBaseCRS;
    using BaseType::transformToIntermediateCRS;
    using BaseType::transformFromIntermediateCRS;
    virtual Coordinate transformToBaseCRS(const Eigen::Vector3d& value,const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType) const override;
    virtual Coordinate transformFromBaseCRS(const Eigen::Vector3d& value,const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType) const override;

    /**
     * @attention 派生クラスは少なくともこれをオーバーライドすること！
     */
    virtual Coordinate transformToBaseCRS(const Eigen::Vector3d& value, const Time& time, const CoordinateType& coordinateType) const override;

    /**
     * @attention 派生クラスは少なくともこれをオーバーライドすること！
     */
    virtual Coordinate transformFromBaseCRS(const Eigen::Vector3d& value, const Time& time, const CoordinateType& coordinateType) const override;

    virtual Coordinate transformToNonDerivedBaseCRS(const Eigen::Vector3d& value,const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType) const override;
    virtual Coordinate transformFromNonDerivedBaseCRS(const Eigen::Vector3d& value,const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType) const override;
    virtual Coordinate transformToNonDerivedBaseCRS(const Eigen::Vector3d& value, const Time& time, const CoordinateType& coordinateType) const override;
    virtual Coordinate transformFromNonDerivedBaseCRS(const Eigen::Vector3d& value, const Time& time, const CoordinateType& coordinateType) const override;
    virtual Coordinate transformToIntermediateCRS(const Eigen::Vector3d& value,const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType) const override;
    virtual Coordinate transformFromIntermediateCRS(const Eigen::Vector3d& value,const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType) const override;
    virtual Coordinate transformToIntermediateCRS(const Eigen::Vector3d& value, const Time& time, const CoordinateType& coordinateType) const override;
    virtual Coordinate transformFromIntermediateCRS(const Eigen::Vector3d& value, const Time& time, const CoordinateType& coordinateType) const override;

    // このCRS上の点originを原点とするtopocentricなCRSを生成して返す。
    using BaseType::getEquivalentQuaternionToTopocentricCRS;
    /**
     * @details
     *      ProjectedCRSにおいて、TopocentricCRSへの変換クォータニオンを得る場合は、
     *      ProjectedCRS上の「北」がTopocentricCRS上の北(真北)に一致するように鉛直方向まわりに回転するようなクォータニオンを返すものとする。
     * 
     *      正角図法でない場合は東西方向の軸が一致しないがそれは受容する。
     *      また、このクラスにおいては投影図の座標軸がENU相当であることを仮定する。そうでない場合は派生クラスで更なる回転を行うこと。
     */
    virtual Quaternion getEquivalentQuaternionToTopocentricCRS(const Eigen::Vector3d& origin, const Time& time, const std::string& axisOrder) const override;//CRSとして生成せずにクォータニオンとして得る。
    using BaseType::transformToTopocentricCRS;
    using BaseType::transformFromTopocentricCRS;
    virtual Eigen::Vector3d transformToTopocentricCRS(const Eigen::Vector3d& value, const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType, const Eigen::Vector3d& origin, const std::string& axisOrder, bool onSurface) const override;
    virtual Eigen::Vector3d transformFromTopocentricCRS(const Eigen::Vector3d& value, const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType, const Eigen::Vector3d& origin, const std::string& axisOrder, bool onSurface) const override;
    //
    // 以下は本クラスで追加定義するメンバ
    //
    ///@name 局所的な計量を計算するための諸量
    ///@{
    /** @name (1) 子午線収差/真北方向角 (meridian conversion/grid declination)
     *   ある位置における子午線収差角(投影図上の北と真北のなす角)を計算して[rad]で返す。真東側にずれている場合に正。
     *   例えばメルカトル図法ならば常に0である。
     */
    ///@{
    virtual double getMeridianConversionAtBaseCoordinate(const Eigen::Vector3d& baseLocation) const=0; //!< 親座標系上の位置における子午線収差を返す。
    virtual double getMeridianConversionAtIntermediateCoordinate(const Eigen::Vector3d& intermediateLocation) const; //!< 中間座標系上の位置における子午線収差を返す。
    virtual double getMeridianConversionAtProjectedCoordinate(const Eigen::Vector3d& projectedLocation) const=0; //!< 投影図上の位置における子午線収差を返す。
    ///@}

    /** @name (2) 子午線方向の縮尺係数 (meridional scale factor)
     *   ある位置における子午線方向の縮尺係数(=[投影図上の長さ]/[楕円体表面上の長さ])を計算する。
     *   例えばメルカトル図法ならば高度0において1/cos(latitude)である。
     * 
     *   高度0でない場合は緯度1°あたりの弧長が変わるため補正が必要である。
     * 
     *   引数onSurfaceがtrueのときは高度0における縮尺係数を返し、falseのときはlocationとして与えた座標の高度における補正後の縮尺係数を返す。
     */
    virtual double getMeridionalScaleFactorAtBaseCoordinate(const Eigen::Vector3d& baseLocation,bool onSurface) const=0; //!< 親座標系上の位置における子午線方向の縮尺係数を返す。
    virtual double getMeridionalScaleFactorAtIntermediateCoordinate(const Eigen::Vector3d& intermediateLocation,bool onSurface) const; //!< 中間座標系上の位置における子午線方向の縮尺係数を返す。
    virtual double getMeridionalScaleFactorAtProjectedCoordinate(const Eigen::Vector3d& projectedLocation,bool onSurface) const=0; //!< 投影図上の位置における子午線方向の縮尺係数を返す。
    ///@}

    /** @name (3) 緯線方向の縮尺係数 (parallel scale factor)
     *   ある位置における子午線方向の縮尺係数(=[投影図上の長さ]/[楕円体表面上の長さ])を計算する。
     *   例えば正角図法ならば高度0においては子午線方向の縮尺係数と等しい。
     * 
     *   引数onSurfaceがtrueのときは高度0における縮尺係数を返し、falseのときはlocationとして与えた座標の高度における補正後の縮尺係数を返す。
     */
    ///@{
    virtual double getParallelScaleFactorAtBaseCoordinate(const Eigen::Vector3d& baseLocation,bool onSurface) const=0; //!< 親座標系上の位置における緯線方向の縮尺係数を返す。
    virtual double getParallelScaleFactorAtIntermediateCoordinate(const Eigen::Vector3d& intermediateLocation,bool onSurface) const; //!< 中間座標系上の位置における緯線方向の縮尺係数を返す。
    virtual double getParallelScaleFactorAtProjectedCoordinate(const Eigen::Vector3d& projectedLocation,bool onSurface) const=0; //!< 投影図上の位置における緯線方向の縮尺係数を返す。
    ///@}

    /** @name (4) 子午線と緯線のなす角 (meridian/parallel intersection angle)
     *   投影図上のある位置における子午線方向と緯線方向のなす角を計算して[rad]で返す。
     *   例えば正角図法ならばどこでもπ/2である。
     */
    ///@{
    virtual double getIntersectionAngleAtBaseCoordinate(const Eigen::Vector3d& baseLocation) const=0; //!< 親座標系上の位置における子午線と緯線のなす角を返す。
    virtual double getIntersectionAngleAtIntermediateCoordinate(const Eigen::Vector3d& intermediateLocation) const; //!< 中間座標系上の位置における子午線と緯線のなす角を返す。
    virtual double getIntersectionAngleAtProjectedCoordinate(const Eigen::Vector3d& projectedLocation) const=0; //!< 投影図上の位置における子午線と緯線のなす角を返す。
    ///@}
    ///@}

    /**
     * @brief このCRS上の座標値を、投影図上でのENU相当(grid easting-grid northing-altitude又はX-Y-altitude)の座標値に変換するための回転クォータニオンを返す。
     * 
     * @details ProjectedCRSの共通インターフェースとして設けるもの。ENU以外の座標軸で表現したい場合には派生クラスでオーバーライドすること。
     *      主な用途としては軸順と符号の入替によって、NED等の異なる座標軸で扱いたい場合が挙げられる。
     */
    virtual Quaternion getQuaternionToENU() const;
    /**
     * @brief 投影図上でのENU相当(grid easting-grid northing-altitude又はX-Y-altitude)の座標値を、このCRS上の座標値に変換するための回転クォータニオンを返す。
     * 
     * @details ProjectedCRSの共通インターフェースとして設けるもの。ENU以外の座標軸で表現したい場合には派生クラスでオーバーライドすること。
     *      主な用途としては軸順と符号の入替によって、NED等の異なる座標軸で扱いたい場合が挙げられる。
     */
    virtual Quaternion getQuaternionFromENU() const;
};

/**
 * @class GenericUTMCRS
 * @brief ユニバーサル横メルカトル図法を表すクラス。ProjectedCRSの具象クラスの例として提供する。
 * 
 * @details
 *      座標軸はEasting-Northing-Altitudeのみでなく、TopocentricCRSと同様にNED等の軸順を入れ替えた座標系としての表現も可能とする。
 *      慣例としてのオフセット(東西方向の500km、南半球の南北方向10000km)は無視する。
 */
ASRC_DECLARE_DERIVED_CLASS_WITHOUT_TRAMPOLINE(GenericUTMCRS,ProjectedCRS)
    public:
    using BaseType::BaseType;
    virtual void initialize() override;
    // 内部状態のシリアライゼーション
    virtual void serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full) override;
    protected:
    virtual void validateImpl(bool isFirstTime) const override;
    public:
    // 親との座標変換
    using BaseType::transformToBaseCRS;
    using BaseType::transformFromBaseCRS;
    virtual Coordinate transformToBaseCRS(const Eigen::Vector3d& value, const Time& time, const CoordinateType& coordinateType) const override;
    virtual Coordinate transformFromBaseCRS(const Eigen::Vector3d& value, const Time& time, const CoordinateType& coordinateType) const override;
    // 局所的な計量を計算するための諸量
    // (1) 子午線収差/真北方向角 (meridian conversion/grid declination)
    virtual double getMeridianConversionAtBaseCoordinate(const Eigen::Vector3d& baseLocation) const override;
    virtual double getMeridianConversionAtProjectedCoordinate(const Eigen::Vector3d& projectedLocation) const override;
    // (2) 子午線方向の縮尺係数 (meridional scale factor)
    virtual double getMeridionalScaleFactorAtBaseCoordinate(const Eigen::Vector3d& baseLocation,bool onSurface) const override;
    virtual double getMeridionalScaleFactorAtProjectedCoordinate(const Eigen::Vector3d& projectedLocation,bool onSurface) const override;
    // (3) 緯線方向の縮尺係数 (parallel scale factor)
    virtual double getParallelScaleFactorAtBaseCoordinate(const Eigen::Vector3d& baseLocation,bool onSurface) const override;
    virtual double getParallelScaleFactorAtProjectedCoordinate(const Eigen::Vector3d& projectedLocation,bool onSurface) const override;
    // (4) 子午線と緯線のなす角 (meridian/parallel intersection angle)
    virtual double getIntersectionAngleAtBaseCoordinate(const Eigen::Vector3d& baseLocation) const override;
    virtual double getIntersectionAngleAtProjectedCoordinate(const Eigen::Vector3d& projectedLocation) const override;

    // このCRS上の座標値と、投影図上でのENU相当(grid easting-grid northing-altitude又はX-Y-altitude)の座標値を相互変換するための回転クォータニオンを返す。
    // ProjectedCRSの共通インターフェースとして設けるもの。ENU以外の座標軸で表現したい場合には派生クラスでオーバーライドすること。
    // 主な用途としては軸順と符号の入替によって、NED等の異なる座標軸で扱いたい場合が挙げられる。
    virtual Quaternion getQuaternionToENU() const override;
    virtual Quaternion getQuaternionFromENU() const override;
    //
    // 以下は本クラスで追加定義するメンバ
    //
    public:
    virtual Eigen::Vector3d getEllipsoidalOrigin() const; //!< 原点の地理座標(lat-lon-alt)を返す。
    virtual std::string getAxisOrder() const; //!< 座標軸の順序("FSD"等)を返す。
    protected:
    virtual void checkAxisOrder() const; //!< [For internal use] 座標軸の順序として与えられた文字列が適切なものかどうか判定する。
    mutable std::string axisOrder; //!< 座標軸の順序("NED"等)
    mutable bool onSurface; //!< trueの場合、原点を楕円体表面になるように高度方向に平行移動する。
    mutable Eigen::Vector3d ellipsoidalOrigin; //!< 原点のgeographic座標 in lat[deg], lon[deg], alt[m]
    mutable double scaleFactor; //!< 縮尺係数
    mutable int zone; //!< ゾーン番号
    mutable double lon_0; //!< 中央子午線の経度 [deg]

    mutable double semiMajorAxis; //!< 長半径(赤道半径) [m]
    mutable double inverseFlattening; //!< 扁平率の逆数 [m]
    mutable double n,A;
    mutable Eigen::VectorXd alpha,beta,delta;

    mutable Eigen::Vector3d offset; //!< 中央子午線における投影原点から本来の原点(ellipsoidalOriginに相当する点)へのオフセット。投影原点の投影後座標値として与える。
};

ASRC_DECLARE_DERIVED_TRAMPOLINE(ProjectedCRS)
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
    using Base::transformToTopocentricCRS;
    using Base::transformFromTopocentricCRS;
    virtual Quaternion getEquivalentQuaternionToTopocentricCRS(const Eigen::Vector3d& origin, const Time& time, const std::string& axisOrder) const override{
        PYBIND11_OVERRIDE(Quaternion,Base,getEquivalentQuaternionToTopocentricCRS,origin,time,axisOrder);
    }
    virtual Eigen::Vector3d transformToTopocentricCRS(const Eigen::Vector3d& value, const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType, const Eigen::Vector3d& origin, const std::string& axisOrder, bool onSurface) const override{
        PYBIND11_OVERRIDE(Eigen::Vector3d,Base,transformToTopocentricCRS,value,location,time,coordinateType,origin,axisOrder,onSurface);
    }
    virtual Eigen::Vector3d transformFromTopocentricCRS(const Eigen::Vector3d& value, const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType, const Eigen::Vector3d& origin, const std::string& axisOrder, bool onSurface) const override{
        PYBIND11_OVERRIDE(Eigen::Vector3d,Base,transformFromTopocentricCRS,value,location,time,coordinateType,origin,axisOrder,onSurface);
    }
    //
    // 以下は本クラスで追加定義するメンバ
    //
    // 局所的な計量を計算するための諸量
    // (1) 子午線収差/真北方向角 (meridian conversion/grid declination)
    virtual double getMeridianConversionAtBaseCoordinate(const Eigen::Vector3d& baseLocation) const override{
        PYBIND11_OVERRIDE_PURE(double,Base,getMeridianConversionAtBaseCoordinate,baseLocation);
    }
    virtual double getMeridianConversionAtIntermediateCoordinate(const Eigen::Vector3d& intermediateLocation) const override{
        PYBIND11_OVERRIDE(double,Base,getMeridianConversionAtIntermediateCoordinate,intermediateLocation);
    }
    virtual double getMeridianConversionAtProjectedCoordinate(const Eigen::Vector3d& projectedLocation) const override{
        PYBIND11_OVERRIDE_PURE(double,Base,getMeridianConversionAtProjectedCoordinate,projectedLocation);
    }
    // (2) 子午線方向の縮尺係数 (meridional scale factor)
    virtual double getMeridionalScaleFactorAtBaseCoordinate(const Eigen::Vector3d& baseLocation,bool onSurface) const override{
        PYBIND11_OVERRIDE_PURE(double,Base,getMeridionalScaleFactorAtBaseCoordinate,baseLocation,onSurface);
    }
    virtual double getMeridionalScaleFactorAtIntermediateCoordinate(const Eigen::Vector3d& intermediateLocation,bool onSurface) const override{
        PYBIND11_OVERRIDE(double,Base,getMeridionalScaleFactorAtIntermediateCoordinate,intermediateLocation,onSurface);
    }
    virtual double getMeridionalScaleFactorAtProjectedCoordinate(const Eigen::Vector3d& projectedLocation,bool onSurface) const override{
        PYBIND11_OVERRIDE_PURE(double,Base,getMeridionalScaleFactorAtProjectedCoordinate,projectedLocation,onSurface);
    }
    // (3) 緯線方向の縮尺係数 (parallel scale factor)
    virtual double getParallelScaleFactorAtBaseCoordinate(const Eigen::Vector3d& baseLocation,bool onSurface) const override{
        PYBIND11_OVERRIDE_PURE(double,Base,getParallelScaleFactorAtBaseCoordinate,baseLocation,onSurface);
    }
    virtual double getParallelScaleFactorAtIntermediateCoordinate(const Eigen::Vector3d& intermediateLocation,bool onSurface) const override{
        PYBIND11_OVERRIDE(double,Base,getParallelScaleFactorAtIntermediateCoordinate,intermediateLocation,onSurface);
    }
    virtual double getParallelScaleFactorAtProjectedCoordinate(const Eigen::Vector3d& projectedLocation,bool onSurface) const override{
        PYBIND11_OVERRIDE_PURE(double,Base,getParallelScaleFactorAtProjectedCoordinate,projectedLocation,onSurface);
    }
    // (4) 子午線と緯線のなす角 (meridian/parallel intersection angle)
    virtual double getIntersectionAngleAtBaseCoordinate(const Eigen::Vector3d& baseLocation) const override{
        PYBIND11_OVERRIDE_PURE(double,Base,getIntersectionAngleAtBaseCoordinate,baseLocation);
    }
    virtual double getIntersectionAngleAtIntermediateCoordinate(const Eigen::Vector3d& intermediateLocation) const override{
        PYBIND11_OVERRIDE(double,Base,getIntersectionAngleAtIntermediateCoordinate,intermediateLocation);
    }
    virtual double getIntersectionAngleAtProjectedCoordinate(const Eigen::Vector3d& projectedLocation) const override{
        PYBIND11_OVERRIDE_PURE(double,Base,getIntersectionAngleAtProjectedCoordinate,projectedLocation);
    }
    // このCRS上の座標値と、投影図上でのENU相当(grid easting-grid northing-altitude又はX-Y-altitude)の座標値を相互変換するための回転クォータニオンを返す。
    // ProjectedCRSの共通インターフェースとして設けるもの。ENU以外の座標軸で表現したい場合には派生クラスでオーバーライドすること。
    // 主な用途としては軸順と符号の入替によって、NED等の異なる座標軸で扱いたい場合が挙げられる。
    virtual Quaternion getQuaternionToENU() const override{
        PYBIND11_OVERRIDE(Quaternion,Base,getQuaternionToENU);
    }
    virtual Quaternion getQuaternionFromENU() const override{
        PYBIND11_OVERRIDE(Quaternion,Base,getQuaternionFromENU);
    }
};
ASRC_DECLARE_DERIVED_TRAMPOLINE(GenericUTMCRS)
    // 局所的な計量を計算するための諸量
    // (1) 子午線収差/真北方向角 (meridian conversion/grid declination)
    virtual double getMeridianConversionAtBaseCoordinate(const Eigen::Vector3d& baseLocation) const override{
        PYBIND11_OVERRIDE(double,Base,getMeridianConversionAtBaseCoordinate,baseLocation);
    }
    virtual double getMeridianConversionAtProjectedCoordinate(const Eigen::Vector3d& projectedLocation) const override{
        PYBIND11_OVERRIDE(double,Base,getMeridianConversionAtProjectedCoordinate,projectedLocation);
    }
    // (2) 子午線方向の縮尺係数 (meridional scale factor)
    virtual double getMeridionalScaleFactorAtBaseCoordinate(const Eigen::Vector3d& baseLocation) const override{
        PYBIND11_OVERRIDE(double,Base,getMeridionalScaleFactorAtBaseCoordinate,baseLocation);
    }
    virtual double getMeridionalScaleFactorAtProjectedCoordinate(const Eigen::Vector3d& projectedLocation) const override{
        PYBIND11_OVERRIDE(double,Base,getMeridionalScaleFactorAtProjectedCoordinate,projectedLocation);
    }
    // (3) 緯線方向の縮尺係数 (parallel scale factor)
    virtual double getParallelScaleFactorAtBaseCoordinate(const Eigen::Vector3d& baseLocation) const override{
        PYBIND11_OVERRIDE(double,Base,getParallelScaleFactorAtBaseCoordinate,baseLocation);
    }
    virtual double getParallelScaleFactorAtProjectedCoordinate(const Eigen::Vector3d& projectedLocation) const override{
        PYBIND11_OVERRIDE(double,Base,getParallelScaleFactorAtProjectedCoordinate,projectedLocation);
    }
    // (4) 子午線と緯線のなす角 (meridian/parallel intersection angle)
    virtual double getIntersectionAngleAtBaseCoordinate(const Eigen::Vector3d& baseLocation) const override{
        PYBIND11_OVERRIDE(double,Base,getIntersectionAngleAtBaseCoordinate,baseLocation);
    }
    virtual double getIntersectionAngleAtProjectedCoordinate(const Eigen::Vector3d& projectedLocation) const override{
        PYBIND11_OVERRIDE(double,Base,getIntersectionAngleAtProjectedCoordinate,projectedLocation);
    }
    //
    // 以下は本クラスで追加定義するメンバ
    //
    virtual Eigen::Vector3d getEllipsoidalOrigin() const override{
        PYBIND11_OVERRIDE(Eigen::Vector3d,Base,getEllipsoidalOrigin);
    }
};

void exportProjectedCRS(py::module &m, const std::shared_ptr<asrc::core::FactoryHelper>& factoryHelper);

ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
