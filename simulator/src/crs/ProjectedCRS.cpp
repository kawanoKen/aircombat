// Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#include "crs/ProjectedCRS.h"
#include "crs/GeodeticCRS.h"
#include "Units.h"
#include <algorithm>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <iomanip>

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)

using namespace util;

ProjectedCRS::ProjectedCRS(const nl::json& modelConfig_,const nl::json& instanceConfig_)
:DerivedCRS(modelConfig_,instanceConfig_){
    crsType=CRSType::PROJECTED;
}
void ProjectedCRS::validateImpl(bool isFirstTime) const{
    this->DerivedCRS::validateImpl(isFirstTime);
    if(isFirstTime || !hasBeenValidated()){
        // baseがDerivedCRSでないことをチェック
        auto baseCRS=getBaseCRS();
        if(getNonDerivedBaseCRS()!=baseCRS){
            throw std::runtime_error("The base CRS of ProjectedCRS must not be a DerivedCRS.");
        }
        auto geod=std::dynamic_pointer_cast<GeodeticCRS>(baseCRS);
        if(!geod || !geod->isEarthFixed()){
            throw std::runtime_error("Only ECEF or geographic CRS can be a base CRS of ProjectedCRS.");
        }
    }
}
//座標系の種類
bool ProjectedCRS::isEarthFixed() const{
    return true;
}
bool ProjectedCRS::isProjected() const{
    return true;
}
bool ProjectedCRS::isCartesian() const{
    return true;
}
bool ProjectedCRS::isSpherical() const{
    return false;
}
bool ProjectedCRS::isEllipsoidal() const{
    return false;
}
//中間座標系の取得
std::shared_ptr<CoordinateReferenceSystem> ProjectedCRS::getIntermediateCRS() const{
    return getBaseCRS()->getIntermediateCRS(); //ECEFになる
}
bool ProjectedCRS::isTransformableTo(const std::shared_ptr<CoordinateReferenceSystem>& other) const{
    return getBaseCRS()->isTransformableTo(other);
}
bool ProjectedCRS::isTransformableFrom(const std::shared_ptr<CoordinateReferenceSystem>& other) const{
    return getBaseCRS()->isTransformableFrom(other);
}
Coordinate ProjectedCRS::transformToBaseCRS(const Eigen::Vector3d& value,const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType) const{
    if(!isValid()){validate();}
    if(coordinateType==CoordinateType::POSITION_ABS){
        return transformToBaseCRS(value,time,CoordinateType::POSITION_ABS);
    }else if(coordinateType==CoordinateType::POSITION_REL){
        //相対位置は観測点からの相対位置ベクトルと解釈して、始点と終点をそれぞれ変換して差をとる。
        auto locationInBase=transformToBaseCRS(location,time,CoordinateType::POSITION_ABS)();
        return std::move(Coordinate(
            transformToBaseCRS(location+value,time,CoordinateType::POSITION_ABS)() - locationInBase,
            locationInBase,
            getBaseCRS(),
            time,
            coordinateType
        ));
    }else{
        //位置以外はECEF(とlocationを原点とするTopocentricCRS)を介して軸の向きと縮尺を考慮した変換を行う。
        auto valueInECEF=transformToIntermediateCRS(value,location,time,coordinateType)();
        auto locationInECEF=transformToIntermediateCRS(location,time,CoordinateType::POSITION_ABS)();
        return getBaseCRS()->transformFrom(valueInECEF,locationInECEF,time,getIntermediateCRS(),coordinateType);
    }
}
Coordinate ProjectedCRS::transformFromBaseCRS(const Eigen::Vector3d& value,const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType) const{
    if(!isValid()){validate();}
    if(coordinateType==CoordinateType::POSITION_ABS){
        return transformFromBaseCRS(value,time,CoordinateType::POSITION_ABS);
    }else if(coordinateType==CoordinateType::POSITION_REL){
        //相対位置は観測点からの相対位置ベクトルと解釈して、始点と終点をそれぞれ変換して差をとる。
        auto locationInSelf=transformFromBaseCRS(location,time,CoordinateType::POSITION_ABS)();
        return std::move(Coordinate(
            transformFromBaseCRS(location+value,time,CoordinateType::POSITION_ABS)() - locationInSelf,
            locationInSelf,
            non_const_this(),
            time,
            coordinateType
        ));
    }else{
        //位置以外はECEF(とlocationを原点とするTopocentricCRS)を介して軸の向きと縮尺を考慮した変換を行う。
        auto ecef=getIntermediateCRS();
        auto baseCRS=getBaseCRS();
        auto valueInECEF=ecef->transformFrom(value,location,time,baseCRS,coordinateType)();
        auto locationInECEF=ecef->transformFrom(location,location,time,baseCRS,CoordinateType::POSITION_ABS)();
        return transformFromIntermediateCRS(valueInECEF,locationInECEF,time,coordinateType);
    }
}
Coordinate ProjectedCRS::transformToBaseCRS(const Eigen::Vector3d& value, const Time& time, const CoordinateType& coordinateType) const{
    if(coordinateType==CoordinateType::POSITION_ABS){
        throw std::runtime_error("ProjectedCRS::transformToBaseCRS(const Eigen::Vector3d&,const CoordinateType&) should be overrided in the derived class. ");
    }else{
        //絶対位置以外は基準点の指定を強制する。
        throw std::runtime_error("Projection requires location of reference point, unless coordinateType==POSITION_ABS.");
    }
}
Coordinate ProjectedCRS::transformFromBaseCRS(const Eigen::Vector3d& value, const Time& time, const CoordinateType& coordinateType) const{
    if(coordinateType==CoordinateType::POSITION_ABS){
        throw std::runtime_error("ProjectedCRS::transformFromBaseCRS(const Eigen::Vector3d&,const CoordinateType&) should be overrided in the derived class. ");
    }else{
        //絶対位置以外は基準点の指定を強制する。
        throw std::runtime_error("Projection requires location of reference point, unless coordinateType==POSITION_ABS.");
    }
}
Coordinate ProjectedCRS::transformToNonDerivedBaseCRS(const Eigen::Vector3d& value,const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType) const{
    return transformToBaseCRS(value,location,time,coordinateType);
}
Coordinate ProjectedCRS::transformFromNonDerivedBaseCRS(const Eigen::Vector3d& value,const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType) const{
    return transformFromBaseCRS(value,location,time,coordinateType);
}
Coordinate ProjectedCRS::transformToNonDerivedBaseCRS(const Eigen::Vector3d& value, const Time& time, const CoordinateType& coordinateType) const{
    return transformToBaseCRS(value,time,coordinateType);
}
Coordinate ProjectedCRS::transformFromNonDerivedBaseCRS(const Eigen::Vector3d& value, const Time& time, const CoordinateType& coordinateType) const{
    return transformFromBaseCRS(value,time,coordinateType);
}
Coordinate ProjectedCRS::transformToIntermediateCRS(const Eigen::Vector3d& value, const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType) const{
    //intermediateはECEF
    if(!isValid()){validate();}
    if(coordinateType==CoordinateType::POSITION_ABS){
        return transformToIntermediateCRS(value,time,CoordinateType::POSITION_ABS);
    }else if(coordinateType==CoordinateType::POSITION_REL){
        //相対位置は観測点からの相対位置ベクトルと解釈して、始点と終点をそれぞれ変換して差をとる。
        auto locationInECEF=transformToIntermediateCRS(location,time,CoordinateType::POSITION_ABS)();
        return std::move(Coordinate(
            transformToIntermediateCRS(location+value,time,CoordinateType::POSITION_ABS)() - locationInECEF,
            locationInECEF,
            getIntermediateCRS(),
            time,
            coordinateType
        ));
    }else{
        //位置以外はlocationを原点とするTopocentricCRSを介して軸の向きと縮尺を考慮した変換を行う。
        double gamma=getMeridianConversionAtProjectedCoordinate(location);
        double h=getMeridionalScaleFactorAtProjectedCoordinate(location,false);
        double k=getParallelScaleFactorAtProjectedCoordinate(location,false);
        double theta=getIntersectionAngleAtProjectedCoordinate(location);

        // projected coordinate (Ep, Np, U) easting[m], northing[m], elliptic altitude[m]
        // topocentric coordinate (Et, Nt, U) easting[m], northing[m], elliptic altitude[m]
        double dEp_dEt = k*cos(gamma-theta+M_PI_2);
        double dEp_dNt = -h*sin(gamma);
        double dNp_dEt = k*sin(gamma-theta+M_PI_2);
        double dNp_dNt = h*cos(gamma);

        Eigen::Matrix3d Rpt({
            {dEp_dEt, dEp_dNt, 0},
            {dNp_dEt, dNp_dNt, 0},
            {      0,       0, 1}
        }); // [projected(ENU) <- topocentric(ENU)]
        Eigen::Vector3d valueInTopo = Rpt.inverse()*getQuaternionToENU().transformVector(value);
        if(coordinateType==CoordinateType::DIRECTION){
            valueInTopo.normalize();
        }
        auto ecef=getIntermediateCRS();
        auto locationInECEF = transformToIntermediateCRS(location,time,CoordinateType::POSITION_ABS)();
        auto valueInECEF=ecef->transformFromTopocentricCRS(
                valueInTopo,
                Eigen::Vector3d::Zero(),
                time,
                coordinateType,
                locationInECEF,
                "ENU",
                false
            );
        return std::move(Coordinate(
            valueInECEF,
            locationInECEF,
            ecef,
            time,
            coordinateType
        ));
    }
}
Coordinate ProjectedCRS::transformFromIntermediateCRS(const Eigen::Vector3d& value, const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType) const{
    //intermediateはECEF
    if(!isValid()){validate();}
    if(coordinateType==CoordinateType::POSITION_ABS){
        return transformFromIntermediateCRS(value,time,CoordinateType::POSITION_ABS);
    }else if(coordinateType==CoordinateType::POSITION_REL){
        //相対位置は観測点からの相対位置ベクトルと解釈して、始点と終点をそれぞれ変換して差をとる。
        auto locationInSelf=transformFromIntermediateCRS(location,time,CoordinateType::POSITION_ABS)();
        return std::move(Coordinate(
            transformFromIntermediateCRS(location+value,time,CoordinateType::POSITION_ABS)() - locationInSelf,
            locationInSelf,
            non_const_this(),
            time,
            coordinateType
        ));
    }else{
        //位置以外はlocationを原点とするTopocentricCRSを介して軸の向きと縮尺を考慮した変換を行う。
        auto ecef=getIntermediateCRS();
        auto valueInTopo=ecef->transformToTopocentricCRS(value,location,time,coordinateType,location,"ENU",false);

        double gamma=getMeridianConversionAtIntermediateCoordinate(location);
        double h=getMeridionalScaleFactorAtIntermediateCoordinate(location,false);
        double k=getParallelScaleFactorAtIntermediateCoordinate(location,false);
        double theta=getIntersectionAngleAtIntermediateCoordinate(location);

        // projected coordinate (Ep, Np, U) grid easting[m], grid northing[m], elliptic altitude[m]
        // topocentric coordinate (Et, Nt, U) true easting[m], true northing[m], elliptic altitude[m]
        double dEp_dEt = k*cos(gamma-theta+M_PI_2);
        double dEp_dNt = -h*sin(gamma);
        double dNp_dEt = k*sin(gamma-theta+M_PI_2);
        double dNp_dNt = h*cos(gamma);

        Eigen::Matrix3d Rpt({
            {dEp_dEt, dEp_dNt, 0},
            {dNp_dEt, dNp_dNt, 0},
            {      0,       0, 1}
        }); // [projected(ENU) <- topocentric(ENU)]
        auto locationInSelf=transformFromIntermediateCRS(location,time,CoordinateType::POSITION_ABS)();
        Eigen::Vector3d valueInSelf = getQuaternionFromENU().transformVector(Rpt*valueInTopo);
        if(coordinateType==CoordinateType::DIRECTION){
            valueInSelf.normalize();
        }
        return std::move(Coordinate(
            valueInSelf,
            locationInSelf,
            non_const_this(),
            time,
            coordinateType
        ));
    }
}
Coordinate ProjectedCRS::transformToIntermediateCRS(const Eigen::Vector3d& value, const Time& time, const CoordinateType& coordinateType) const{
    if(coordinateType==CoordinateType::POSITION_ABS){
        auto ecef=getIntermediateCRS();
        auto baseCRS=getBaseCRS();
        auto valueInBase=transformToBaseCRS(value,value,time,coordinateType);
        if(baseCRS==ecef){
            return valueInBase;
        }else{
            return ecef->transformFrom(valueInBase);
        }
    }else{
        //絶対位置以外は基準点の指定を強制する。
        throw std::runtime_error("Projection requires location of reference point, unless coordinateType==POSITION_ABS.");
    }
}
Coordinate ProjectedCRS::transformFromIntermediateCRS(const Eigen::Vector3d& value, const Time& time, const CoordinateType& coordinateType) const{
    if(coordinateType==CoordinateType::POSITION_ABS){
        auto ecef=getIntermediateCRS();
        auto baseCRS=getBaseCRS();
        if(baseCRS==ecef){
            return transformFromBaseCRS(value,time,coordinateType);
        }else{
            auto valueInBase=baseCRS->transformFrom(value,value,time,ecef,coordinateType)();
            return transformFromBaseCRS(valueInBase,time,coordinateType);
        }
    }else{
        //絶対位置以外は基準点の指定を強制する。
        throw std::runtime_error("Projection requires location of reference point, unless coordinateType==POSITION_ABS.");
    }
}
Quaternion ProjectedCRS::getQuaternionTo(const Eigen::Vector3d& location, const Time& time, const std::shared_ptr<CoordinateReferenceSystem>& dstCRS) const{
    if(!isValid()){validate();}
    if(!dstCRS){
        throw std::runtime_error("dstCRS is nullptr!");
    }
    if(this==dstCRS.get()){
        return Quaternion(1,0,0,0);
    }
    if(!isTransformableTo(dstCRS)){
        throw std::runtime_error("can't be transformed to "+dstCRS->getFactoryClassName());
    }
    if(auto derived=std::dynamic_pointer_cast<DerivedCRS>(dstCRS)){
        if(this==derived->getBaseProjectedCRS().get()){
            // 自身からderiveされたものは子側の関数を呼ぶ
            return derived->getQuaternionFrom(location,time,non_const_this());
        }
    }
    // そうでなければlocationを原点とするTopocentricCRSと、baseに対応するECEFの二つを間に挟む
    auto ecef=std::dynamic_pointer_cast<GeodeticCRS>(getIntermediateCRS());
    auto locationInECEF=transformTo(location,location,time,ecef,CoordinateType::POSITION_ABS)();

    auto qToECEFFromTopo=ecef->getEquivalentQuaternionToTopocentricCRS(locationInECEF,time,"ENU").conjugate();
    auto qToTopoFromSelf=getEquivalentQuaternionToTopocentricCRS(location,time,"ENU");
    auto qToECEFFromSelf=qToECEFFromTopo*qToTopoFromSelf;
    if(dstCRS==ecef){
        return qToECEFFromSelf;
    }else{
        return ecef->getQuaternionTo(locationInECEF,time,dstCRS)*qToECEFFromSelf;
    }
}
Quaternion ProjectedCRS::getQuaternionFrom(const Eigen::Vector3d& location, const Time& time, const std::shared_ptr<CoordinateReferenceSystem>& srcCRS) const{
    if(!isValid()){validate();}
    if(!srcCRS){
        throw std::runtime_error("srcCRS is nullptr!");
    }
    if(this==srcCRS.get()){
        return Quaternion(1,0,0,0);
    }
    if(!isTransformableTo(srcCRS)){
        throw std::runtime_error(getFactoryClassName()+" can't be transformed from "+srcCRS->getFactoryClassName());
    }
    if(auto derived=std::dynamic_pointer_cast<DerivedCRS>(srcCRS)){
        if(this==derived->getBaseProjectedCRS().get()){
            // 自身からderiveされたものは子側の関数を呼ぶ
            return derived->getQuaternionTo(location,time,non_const_this());
        }
    }
    // そうでなければlocationを原点とするTopocentricCRSと、baseに対応するECEFの二つを間に挟む
    auto ecef=getIntermediateCRS();
    auto locationInECEF = ecef->transformFrom(location,location,time,srcCRS,CoordinateType::POSITION_ABS)();
    auto qToECEFFromSrc = ecef->getQuaternionFrom(location,time,srcCRS);

    auto qToTopoFromECEF=ecef->getEquivalentQuaternionToTopocentricCRS(locationInECEF,time,"ENU");

    auto locationInSelf=transformFrom(location,location,time,srcCRS,CoordinateType::POSITION_ABS)();
    auto qToSelfFromTopo=getEquivalentQuaternionToTopocentricCRS(locationInSelf,time,"ENU").conjugate();
    return qToSelfFromTopo*qToTopoFromECEF*qToECEFFromSrc;
}
Quaternion ProjectedCRS::getEquivalentQuaternionToTopocentricCRS(const Eigen::Vector3d& origin, const Time& time, const std::string& axisOrder_) const{
    // CRSとして生成せずにクォータニオンとして得る。
    // 原点(origin)におけるTopocentricCRSとProjectedCRSの間で、「北」が一致するように鉛直方向を軸に回転するようなクォータニオンとする。
    // 正角図法でない場合は子午線と緯線が直交しないが、子午線を合わせることを優先する。

    // まずはENU(projected)からENU(topocentric)への回転クォータニオンを計算する。
    // U軸まわりに子午線収差角(投影図上の北と真北の差)だけ回転すればよい。
    double gamma=getMeridianConversionAtProjectedCoordinate(origin);
    Eigen::Vector3d EpInTopo(cos(gamma),-sin(gamma),0); // 投影図の緯線(Easting軸)がTopocentricCRS(ENU)でどこを向いているか
    Eigen::Vector3d NpInTopo(sin(gamma),cos(gamma),0); // 投影図の子午線(Northing軸)がTopocentricCRS(ENU)でどこを向いているか

    Quaternion qTopoFromProj = Quaternion::fromBasis(EpInTopo,NpInTopo,Eigen::Vector3d(0,0,1)); // [ENU(topo) <- ENU(projected)]

    //変換前後のaxisOrderに応じて成分を入れ替える
    std::string dstAxisOrder=axisOrder_;
    std::transform(dstAxisOrder.begin(),dstAxisOrder.end(),dstAxisOrder.begin(),
        [](unsigned char c){ return std::toupper(c);}
    );
    return (
        getAxisSwapQuaternion("ENU",dstAxisOrder)* // [Dst <- ENU(topo)]
        qTopoFromProj* // [ENU(topo) <- ENU(projected)]
        getQuaternionToENU() // [ENU(projected) <- Self]
    );
}
Eigen::Vector3d ProjectedCRS::transformToTopocentricCRS(const Eigen::Vector3d& value, const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType, const Eigen::Vector3d& origin, const std::string& axisOrder_, bool onSurface) const{
    // intermediateCRS(==ECEF)を介して変換する
    auto valueInECEF = transformToIntermediateCRS(value,location,time,coordinateType);
    auto originInECEF = transformToIntermediateCRS(origin,origin,time,CoordinateType::POSITION_ABS);
    return getIntermediateCRS()->transformToTopocentricCRS(
        valueInECEF(),
        valueInECEF.getLocation(),
        time,
        coordinateType,
        originInECEF(),
        axisOrder_,
        onSurface
    );
}
Eigen::Vector3d ProjectedCRS::transformFromTopocentricCRS(const Eigen::Vector3d& value, const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType, const Eigen::Vector3d& origin, const std::string& axisOrder_, bool onSurface) const{
    // intermediateCRS(==ECEF)を介して変換する
    auto ecef=getIntermediateCRS();
    auto originInECEF = transformToIntermediateCRS(origin,origin,time,CoordinateType::POSITION_ABS);
    auto valueInECEF = ecef->transformFromTopocentricCRS(
        value,
        location,
        time,
        coordinateType,
        originInECEF(),
        axisOrder_,
        onSurface
    );
    auto locationInECEF = ecef->transformFromTopocentricCRS(
        location,
        location,
        time,
        CoordinateType::POSITION_ABS,
        originInECEF(),
        axisOrder_,
        onSurface
    );
    return transformFrom(valueInECEF,locationInECEF,time,ecef,coordinateType)();
}
// 局所的な計量を計算するための諸量
// (1) 子午線収差/真北方向角 (meridian conversion/grid declination)
double ProjectedCRS::getMeridianConversionAtIntermediateCoordinate(const Eigen::Vector3d& intermediateLocation) const{
    auto ecef=getIntermediateCRS();
    auto baseCRS=getBaseCRS();
    if(baseCRS==ecef){
        return getMeridianConversionAtBaseCoordinate(intermediateLocation);
    }else{
        auto locationInBase=baseCRS->transformFrom(intermediateLocation,Time(),ecef,CoordinateType::POSITION_ABS)();
        return getMeridianConversionAtBaseCoordinate(locationInBase);
    }
}
// (2) 子午線方向の縮尺係数 (meridional scale factor)
double ProjectedCRS::getMeridionalScaleFactorAtIntermediateCoordinate(const Eigen::Vector3d& intermediateLocation,bool onSurface) const{
    auto ecef=getIntermediateCRS();
    auto baseCRS=getBaseCRS();
    if(baseCRS==ecef){
        return getMeridionalScaleFactorAtBaseCoordinate(intermediateLocation,onSurface);
    }else{
        auto locationInBase=baseCRS->transformFrom(intermediateLocation,Time(),ecef,CoordinateType::POSITION_ABS)();
        return getMeridionalScaleFactorAtBaseCoordinate(locationInBase,onSurface);
    }
}
// (3) 緯線方向の縮尺係数 (parallel scale factor)
double ProjectedCRS::getParallelScaleFactorAtIntermediateCoordinate(const Eigen::Vector3d& intermediateLocation,bool onSurface) const{
    auto ecef=getIntermediateCRS();
    auto baseCRS=getBaseCRS();
    if(baseCRS==ecef){
        return getParallelScaleFactorAtBaseCoordinate(intermediateLocation,onSurface);
    }else{
        auto locationInBase=baseCRS->transformFrom(intermediateLocation,Time(),ecef,CoordinateType::POSITION_ABS)();
        return getParallelScaleFactorAtBaseCoordinate(locationInBase,onSurface);
    }
}
// (4) 子午線と緯線のなす角 (meridian/parallel intersection angle)
double ProjectedCRS::getIntersectionAngleAtIntermediateCoordinate(const Eigen::Vector3d& intermediateLocation) const{
    auto ecef=getIntermediateCRS();
    auto baseCRS=getBaseCRS();
    if(baseCRS==ecef){
        return getIntersectionAngleAtBaseCoordinate(intermediateLocation);
    }else{
        auto locationInBase=baseCRS->transformFrom(intermediateLocation,Time(),ecef,CoordinateType::POSITION_ABS)();
        return getIntersectionAngleAtBaseCoordinate(locationInBase);
    }
}
// このCRS上の座標値と、投影図上でのENU相当(grid easting-grid northing-altitude又はX-Y-altitude)の座標値を相互変換するための回転クォータニオンを返す。
// ProjectedCRSの共通インターフェースとして設けるもの。ENU以外の座標軸で表現したい場合には派生クラスでオーバーライドすること。
// 主な用途としては軸順と符号の入替によって、NED等の異なる座標軸で扱いたい場合が挙げられる。
Quaternion ProjectedCRS::getQuaternionToENU() const{
    return Quaternion(1,0,0,0);
}
Quaternion ProjectedCRS::getQuaternionFromENU() const{
    return Quaternion(1,0,0,0);
}

void GenericUTMCRS::initialize(){
    BaseType::initialize();
    if(instanceConfig.contains("onSurface")){
        onSurface=instanceConfig.at("onSurface");
    }else if(modelConfig.contains("onSurface")){
        onSurface=modelConfig.at("onSurface");
    }else{
        onSurface=false;
    }
}
// 内部状態のシリアライゼーション
void GenericUTMCRS::serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full){
    BaseType::serializeInternalState(archive,full);

    if(isValid()){
        ASRC_SERIALIZE_NVP(archive
            ,axisOrder
            ,onSurface
            ,ellipsoidalOrigin
            ,scaleFactor
            ,zone
            ,lon_0
            ,semiMajorAxis
            ,inverseFlattening
            ,n
            ,A
            ,alpha
            ,beta
            ,delta
            ,offset
        )
    }
}
void GenericUTMCRS::validateImpl(bool isFirstTime) const{
    this->ProjectedCRS::validateImpl(isFirstTime);
    if(isFirstTime || !hasBeenValidated()){
        // 軸順の読み込み
        checkAxisOrder();

        auto baseCRS = getBaseCRS();
        auto geod=std::dynamic_pointer_cast<GeodeticCRS>(baseCRS); // 有効であることはProjectedCRS::validateImplで判定済

        // 縮尺係数の設定
        if(instanceConfig.contains("scaleFactor")){
            scaleFactor=instanceConfig.at("scaleFactor");
        }else if(modelConfig.contains("scaleFactor")){
            scaleFactor=modelConfig.at("scaleFactor");
        }else{
            scaleFactor=0.9996;
        }

        //原点情報の読み込み
        Eigen::Vector3d origin;
        bool isOriginGiven=false;
        if(instanceConfig.contains("origin")){
            origin=instanceConfig.at("origin");
            isOriginGiven=true;
        }else if(modelConfig.contains("origin")){
            origin=modelConfig.at("origin");
            isOriginGiven=true;
        }
        if(isOriginGiven){
            if(geod->isGeocentric()){
                origin=geod->transformECEFCoordinateToGeographicCRS(origin,CoordinateType::POSITION_ABS);
            }
        }

        // 中央子午線の設定
        // 優先度はzone > lat_0,lon_0 > origin
        bool givenByZone=false;
        bool givenByLon0=false;
        bool projectionAsLocal=false;
        double lat_0=0.0; //zoneの計算用
        if(instanceConfig.contains("zone")){
            zone=instanceConfig.at("zone");
            givenByZone=true;
        }else if(modelConfig.contains("zone")){
            zone=modelConfig.at("zone");
            givenByZone=true;
        }else{
            if(instanceConfig.contains("lon_0")){
                lon_0=instanceConfig.at("lon_0");
                givenByLon0=true;
            }else if(modelConfig.contains("lon_0")){
                lon_0=modelConfig.at("lon_0");
                givenByLon0=true;
            }else if(isOriginGiven){
                lon_0=origin(1);
                if(instanceConfig.contains("projectionAsLocal")){
                    projectionAsLocal=instanceConfig.at("projectionAsLocal");
                }else if(modelConfig.contains("projectionAsLocal")){
                    projectionAsLocal=modelConfig.at("projectionAsLocal");
                }else{
                    projectionAsLocal=false;
                }
                if(projectionAsLocal){
                    givenByLon0=true;
                }
            }else{
                throw std::runtime_error("GenericUTMCRS requires central meridian config by zone, lon_0 or origin.");
            }
            if(instanceConfig.contains("lat_0")){
                lat_0=instanceConfig.at("lat_0");
            }else if(modelConfig.contains("lat_0")){
                lat_0=modelConfig.at("lat_0");
            }else if(isOriginGiven){
                lat_0=origin(0);
            }else{
                lat_0=0.0;//未指定は赤道上でゾーン計算を行う。
            }
            if(lat_0<-80.0 || lat_0>84.0){
                throw std::runtime_error("UTM can be used between 80°S and 84°N.");
            }
        }

        if(givenByZone){
            lon_0=6*zone-183;
        }else{ //given by lon_0 or origin
            zone=int((lon_0+180.0)/6.0)+1;
            if(56.0<=lat_0 && lat_0<=64){
                // V
                if(3.0<=lon_0 && lon_0<=6.0){
                    zone=32;
                }
            }else if(72.0<=lat_0 && lat_0<=84.0){
                // X
                if(6.0<=lon_0 && lon_0<=9.0){
                    zone=31;
                }else if(9.0<=lon_0 && lon_0<=12.0){
                    zone=33;
                }else if(18.0<=lon_0 && lon_0<=21.0){
                    zone=33;
                }else if(21.0<=lon_0 && lon_0<=24.0){
                    zone=35;
                }else if(30.0<=lon_0 && lon_0<=33.0){
                    zone=35;
                }else if(33.0<=lon_0 && lon_0<=36.0){
                    zone=37;
                }
            }
            if(!givenByLon0){
                //lon_0の直接指定でなければzoneの中央子午線を用いる。
                lon_0=6*zone-183;
            }
        }

        // 原点オフセットの設定
        // 優先度はonSurface > offset > origin
        if(instanceConfig.contains("onSurface")){
            onSurface=instanceConfig.at("onSurface");
        }else if(modelConfig.contains("onSurface")){
            onSurface=modelConfig.at("onSurface");
        }else{
            onSurface=false;
        }
        bool givenByOffset=false;
        bool offsetAsLocal=false;
        if(instanceConfig.contains("offset")){
            offset=instanceConfig.at("offset");
            givenByOffset=true;
        }else if(modelConfig.contains("offset")){
            offset=modelConfig.at("offset");
            givenByOffset=true;
        }else if(isOriginGiven){
            offset=Eigen::Vector3d::Zero();
            if(instanceConfig.contains("offsetAsLocal")){
                offsetAsLocal=instanceConfig.at("offsetAsLocal");
            }else if(modelConfig.contains("offsetAsLocal")){
                offsetAsLocal=modelConfig.at("offsetAsLocal");
            }else{
                offsetAsLocal=false;
            }
        }else{
            offset=Eigen::Vector3d::Zero();
        }
    

        // 変換パラメータの計算
        // 河瀬 和重 "Gauss-Krüger投影における経緯度座標及び平面直角座標相互間の座標換算についてのより簡明な計算方法" 国土地理院時報 2011 No.121 p.109-124
        // 級数展開の次数はPROJライブラリに合わせてα,β,δいずれも6とする。
        double semiMajorAxis=geod->semiMajorAxis();
        double inverseFlattening=geod->inverseFlattening();
        n=0.5/(inverseFlattening-0.5);
        A=semiMajorAxis/(1.0+n)*(1.0+pow(n,2)/4.0+pow(n,4)/64.0+pow(n,6)/256.0);

        constexpr int a_o=6;
        alpha=Eigen::VectorXd(a_o);
        constexpr int b_o=6;
        beta=Eigen::VectorXd(b_o);
        constexpr int d_o=6;
        delta=Eigen::VectorXd(d_o);
        double nn=n;
        alpha(0) = nn * (
            1.0/2.0 + n * (
                -2.0/3.0 + n * (
                    5.0/16.0 + n * (
                        41.0/180.0 + n * (
                            -127.0/288.0 + n * (
                                7891.0/37800.0
                            )
                        )
                    )
                )
            )
        );
        beta(0) = nn * (
            1.0/2.0 + n * (
                -2.0/3.0 + n * (
                    37.0/96.0 + n * (
                        -1.0/360.0 + n * (
                            -81.0/512.0 + n * (
                                96199.0/604800.0
                            )
                        )
                    )
                )
            )
        );
        delta(0) = nn * (
            2.0 + n * (
                -2.0/3.0 + n * (
                    -2.0 + n * (
                        116.0/45.0 + n * (
                            26.0/45.0 + n * (
                                -2854.0/675.0
                            )
                        )
                    )
                )
            )
        );
        nn*=n;
        alpha(1) = nn * (
            13.0/48.0 + n * (
                -3.0/5.0 + n * (
                    557.0/1440.0 + n * (
                        281.0/630.0 + n * (
                            -1983433.0/1935360.0
                        )
                    )
                )
            )
        );
        beta(1) = nn * (
            1.0/48.0 + n * (
                1.0/15.0 + n * (
                    -437.0/1440.0 + n * (
                        46.0/105.0 + n * (
                            -1118711.0/3870720.0
                        )
                    )
                )
            )
        );
        delta(1) = nn * (
            7.0/3.0 + n * (
                -8.0/5.0 + n * (
                    -227.0/45.0 + n * (
                        2704.0/315.0 + n * (
                            2323.0/945.0
                        )
                    )
                )
            )
        );
        nn*=n;
        alpha(2) = nn * (
            61.0/240.0 + n * (
                -103.0/140.0 + n * (
                    15061.0/26880.0 + n * (
                        167603.0/181440.0
                    )
                )
            )
        );
        beta(2) = nn * (
            17.0/480.0 + n * (
                -37.0/840.0 + n * (
                    -209.0/4480.0 + n * (
                        5569.0/90720.0
                    )
                )
            )
        );
        delta(2) = nn * (
            56.0/15.0 + n * (
                -136.0/35.0 + n * (
                    -1262.0/105.0 + n * (
                        73814.0/2835.0
                    )
                )
            )
        );
        nn*=n;
        alpha(3) = nn * (
            49561.0/161280.0 + n * (
                -179.0/168.0 + n * (
                    6601661.0/7257600.0
                )
            )
        );
        beta(3) = nn * (
            4397.0/161280.0 + n * (
                -11.0/504.0 + n * (
                    -830251.0/7257600.0
                )
            )
        );
        delta(3) = nn * (
            4279.0/630.0 + n * (
                -332.0/35.0 + n * (
                    -399572.0/14175.0
                )
            )
        );
        nn*=n;
        alpha(4) = nn * (
            34729.0/80640.0 + n * (
                -3418889.0/1995840.0
            )
        );
        beta(4) = nn * (
            4583.0/161280.0 + n * (
                -108847.0/3991680.0
            )
        );
        delta(4) = nn * (
            4174.0/315.0 + n * (
                -144838.0/6237.0
            )
        );
        nn*=n;
        alpha(5) = nn * (
            212378941.0/319334400.0
        );
        beta(5) = nn * (
            20648693.0/638668800.0
        );
        delta(5) = nn * (
            601676.0/22275.0
        );

        //origin上に投影後原点を持ってくる場合はここでoffset=0でoriginを変換してoffsetを計算
        if(offsetAsLocal){
            Eigen::Vector3d value=origin;
            if(onSurface){
                value(2)-=geod->getHeight(value,Time());
            }
            offset=-transformFromBaseCRS(value,Time(),CoordinateType::POSITION_ABS)();
        }

        //ellipsoidalOriginの計算
        if(offsetAsLocal){
            ellipsoidalOrigin=origin;
            if(geod->isGeocentric()){
                ellipsoidalOrigin=geod->transformECEFCoordinateToGeographicCRS(ellipsoidalOrigin,CoordinateType::POSITION_ABS);
            }
            if(onSurface){
                ellipsoidalOrigin(2)-=geod->getHeight(ellipsoidalOrigin,Time());
            }
        }else{
            ellipsoidalOrigin<<lat_0,lon_0,0;
        }
    }
}
Coordinate GenericUTMCRS::transformToBaseCRS(const Eigen::Vector3d& value, const Time& time, const CoordinateType& coordinateType) const{
    if(coordinateType==CoordinateType::POSITION_ABS){
        Eigen::Vector3d ENU=getQuaternionToENU().transformVector(value-offset);
        double E=ENU(0);
        double N=ENU(1);
        double xi=N/(scaleFactor*A);
        double eta=E/(scaleFactor*A);
        double xi_d=xi;
        double eta_d=eta;
        for(int j=1;j<=beta.size();++j){
            double beta_sin_j=beta(j-1)*sin(2*j*xi);
            double beta_cos_j=beta(j-1)*cos(2*j*xi);
            double sinh_j=sinh(2*j*eta);
            double cosh_j=cosh(2*j*eta);
            xi_d-=beta_sin_j*cosh_j;
            eta_d-=beta_cos_j*sinh_j;
        }
        double khi=asin(sin(xi_d)/cosh(eta_d));
        double lat=khi;
        for(int j=1;j<=delta.size();++j){
            lat+=delta(j-1)*sin(2*j*khi);
        }
        lat=rad2deg(lat);
        double lon=lon_0+rad2deg(atan(sinh(eta_d)/cos(xi_d)));
        double alt=ENU(2);
        Eigen::Vector3d ret=Eigen::Vector3d(lat,lon,alt);
        auto geod=std::dynamic_pointer_cast<GeodeticCRS>(getBaseCRS()); // 有効であることはvalidateで判定済
        if(geod->isGeocentric()){
            ret=geod->transformGeographicCoordinateToECEF(ret,CoordinateType::POSITION_ABS);
        }
        return std::move(Coordinate(
            ret,
            ret,
            geod,
            time,
            coordinateType
        ));
    }else{
        //絶対位置以外は基準点の指定を強制する。
        throw std::runtime_error("Projection requires location of reference point, unless coordinateType==POSITION_ABS.");
    }
}
Coordinate GenericUTMCRS::transformFromBaseCRS(const Eigen::Vector3d& value, const Time& time, const CoordinateType& coordinateType) const{
    if(coordinateType==CoordinateType::POSITION_ABS){
        if(!isValid()){validate();}
        Eigen::Vector3d lla;
        auto geod=std::dynamic_pointer_cast<GeodeticCRS>(getBaseCRS()); // 有効であることはvalidateで判定済
        if(geod->isGeocentric()){
            lla=geod->transformECEFCoordinateToGeographicCRS(value,CoordinateType::POSITION_ABS);
        }else{
            lla=value;
        }
        double lat=deg2rad(lla(0));
        double lon=deg2rad(lla(1));
        double t=sinh(atanh(sin(lat))-(2*sqrt(n))/(1+n)*atanh((2*sqrt(n))/(1+n)*sin(lat)));

        double ttp1=sqrt(1+t*t);
        double dlon=lon-deg2rad(lon_0);

        double xi=atan(t/cos(dlon));
        double eta=atanh(sin(dlon)/ttp1);
        double E=eta;
        double N=xi;
        for(int j=1;j<=alpha.size();++j){
            double alpha_sin_j=alpha(j-1)*sin(2*j*xi);
            double alpha_cos_j=alpha(j-1)*cos(2*j*xi);
            double sinh_j=sinh(2*j*eta);
            double cosh_j=cosh(2*j*eta);
            E+=alpha_cos_j*sinh_j;
            N+=alpha_sin_j*cosh_j;
        }
        E=scaleFactor*A*E;
        N=scaleFactor*A*N;
        double alt=lla(2);

        Eigen::Vector3d ret=getQuaternionFromENU().transformVector(Eigen::Vector3d(E,N,alt))+offset;
        return std::move(Coordinate(
            ret,
            ret,
            non_const_this(),
            time,
            coordinateType
        ));
    }else{
        //絶対位置以外は基準点の指定を強制する。
        throw std::runtime_error("Projection requires location of reference point, unless coordinateType==POSITION_ABS.");
    }
}
// 局所的な計量を計算するための諸量
// (1) 子午線収差/真北方向角 (meridian conversion/grid declination)
double GenericUTMCRS::getMeridianConversionAtBaseCoordinate(const Eigen::Vector3d& baseLocation) const{
    if(!isValid()){validate();}
    Eigen::Vector3d lla;
    auto geod=std::dynamic_pointer_cast<GeodeticCRS>(getBaseCRS()); // 有効であることはvalidateで判定済
    if(geod->isGeocentric()){
        lla=geod->transformECEFCoordinateToGeographicCRS(baseLocation,CoordinateType::POSITION_ABS);
    }else{
        lla=baseLocation;
    }
    double lat=deg2rad(lla(0));
    double lon=deg2rad(lla(1));
    double t=sinh(atanh(sin(lat))-(2*sqrt(n))/(1+n)*atanh((2*sqrt(n))/(1+n)*sin(lat)));

    double ttp1=sqrt(1+t*t);
    double dlon=lon-deg2rad(lon_0);

    double xi=atan(t/cos(dlon));
    double eta=atanh(sin(dlon)/ttp1);
    double sigma=1;
    double tau=0;
    for(int j=1;j<=alpha.size();++j){
        double alpha_sin_j=alpha(j-1)*sin(2*j*xi);
        double alpha_cos_j=alpha(j-1)*cos(2*j*xi);
        double sinh_j=sinh(2*j*eta);
        double cosh_j=cosh(2*j*eta);
        sigma+=2*j*alpha_cos_j*cosh_j;
        tau+=2*j*alpha_sin_j*sinh_j;
    }
    return atan2(tau*ttp1+sigma*t*tan(dlon), sigma*ttp1-tau*t*tan(dlon));//[rad]
}
double GenericUTMCRS::getMeridianConversionAtProjectedCoordinate(const Eigen::Vector3d& projectedLocation) const{
    if(!isValid()){validate();}
    Eigen::Vector3d ENU=getQuaternionToENU().transformVector(projectedLocation-offset);
    double E=ENU(0);
    double N=ENU(1);
    double xi=N/(scaleFactor*A);
    double eta=E/(scaleFactor*A);
    double xi_d=xi;
    double eta_d=eta;
    double sigma_d=1;
    double tau_d=0;
    for(int j=1;j<=beta.size();++j){
        double beta_sin_j=beta(j-1)*sin(2*j*xi);
        double beta_cos_j=beta(j-1)*cos(2*j*xi);
        double sinh_j=sinh(2*j*eta);
        double cosh_j=cosh(2*j*eta);
        xi_d-=beta_sin_j*cosh_j;
        eta_d-=beta_cos_j*sinh_j;
        sigma_d-=2*j*beta_cos_j*cosh_j;
        tau_d+=2*j*beta_sin_j*sinh_j;
    }
    double tantan=tan(xi_d)*tanh(eta_d);
    return atan2(tau_d+sigma_d*tantan, sigma_d-tau_d*tantan) * (N>=0 ? 1 : -1);//[rad]
}
// (2) 子午線方向の縮尺係数 (meridional scale factor)
double GenericUTMCRS::getMeridionalScaleFactorAtBaseCoordinate(const Eigen::Vector3d& baseLocation,bool onSurface) const{
    if(!isValid()){validate();}
    Eigen::Vector3d lla;
    auto geod=std::dynamic_pointer_cast<GeodeticCRS>(getBaseCRS()); // 有効であることはvalidateで判定済
    if(geod->isGeocentric()){
        lla=geod->transformECEFCoordinateToGeographicCRS(baseLocation,CoordinateType::POSITION_ABS);
    }else{
        lla=baseLocation;
    }
    double lat=deg2rad(lla(0));
    double lon=deg2rad(lla(1));
    // まずは高度0における値を計算
    double t=sinh(atanh(sin(lat))-(2*sqrt(n))/(1+n)*atanh((2*sqrt(n))/(1+n)*sin(lat)));

    double ttp1=sqrt(1+t*t);
    double dlon=lon-deg2rad(lon_0);

    double xi=atan(t/cos(dlon));
    double eta=atanh(sin(dlon)/ttp1);
    double sigma=1;
    double tau=0;
    for(int j=1;j<=alpha.size();++j){
        double alpha_sin_j=alpha(j-1)*sin(2*j*xi);
        double alpha_cos_j=alpha(j-1)*cos(2*j*xi);
        double sinh_j=sinh(2*j*eta);
        double cosh_j=cosh(2*j*eta);
        sigma+=2*j*alpha_cos_j*cosh_j;
        tau+=2*j*alpha_sin_j*sinh_j;
    }
    double semiMajorAxis=geod->semiMajorAxis();
    double inverseFlattening=geod->inverseFlattening();

    double scale0=(scaleFactor*A/semiMajorAxis)*sqrt((1+pow((1-n)/(1+n)*tan(lat),2))*(sigma*sigma+tau*tau)/(t*t+pow(cos(dlon),2)));
    if(onSurface){
        return scale0;
    }else{
        // 高度補正を実施
        double alt=lla(2);
        double e2=(2-1/inverseFlattening)/inverseFlattening;
        double e2sin2lat=e2*pow(sin(lat),2);
        double nu=semiMajorAxis/sqrt(1-e2sin2lat);
        double r0=nu*(1-e2)/(1-e2sin2lat);
        return scale0*r0/(r0+alt);
    }
}
double GenericUTMCRS::getMeridionalScaleFactorAtProjectedCoordinate(const Eigen::Vector3d& projectedLocation,bool onSurface) const{
    if(!isValid()){validate();}
    Eigen::Vector3d ENU=getQuaternionToENU().transformVector(projectedLocation-offset);
    // まずは高度0における値を計算
    double E=ENU(0);
    double N=ENU(1);
    double xi=N/(scaleFactor*A);
    double eta=E/(scaleFactor*A);
    double xi_d=xi;
    double eta_d=eta;
    double sigma_d=1;
    double tau_d=0;
    for(int j=1;j<=beta.size();++j){
        double beta_sin_j=beta(j-1)*sin(2*j*xi);
        double beta_cos_j=beta(j-1)*cos(2*j*xi);
        double sinh_j=sinh(2*j*eta);
        double cosh_j=cosh(2*j*eta);
        xi_d-=beta_sin_j*cosh_j;
        eta_d-=beta_cos_j*sinh_j;
        sigma_d-=2*j*beta_cos_j*cosh_j;
        tau_d+=2*j*beta_sin_j*sinh_j;
    }
    double khi=asin(sin(xi_d)/cosh(eta_d));
    double lat=khi;
    for(int j=1;j<=delta.size();++j){
        lat+=delta(j-1)*sin(2*j*khi);
    }
    auto geod=std::dynamic_pointer_cast<GeodeticCRS>(getBaseCRS()); // 有効であることはvalidateで判定済
    double semiMajorAxis=geod->semiMajorAxis();
    double inverseFlattening=geod->inverseFlattening();
    double scale0=(scaleFactor*A/semiMajorAxis)*sqrt((1+pow((1-n)/(1+n)*tan(lat),2))*(pow(cos(xi_d),2)+pow(sinh(eta_d),2))/(sigma_d*sigma_d+tau_d*tau_d));
    if(onSurface){
        return scale0;
    }else{
        // 高度補正を実施
        double alt=ENU(2);
        double e2=(2-1/inverseFlattening)/inverseFlattening;
        double e2sin2lat=e2*pow(sin(lat),2);
        double nu=semiMajorAxis/sqrt(1-e2sin2lat);
        double r0=nu*(1-e2)/(1-e2sin2lat);
        return scale0*r0/(r0+alt);
    }
}
// (3) 緯線方向の縮尺係数 (parallel scale factor)
double GenericUTMCRS::getParallelScaleFactorAtBaseCoordinate(const Eigen::Vector3d& baseLocation,bool onSurface) const{
    // 正角図法なので高度0では子午線方向と等しいが、高度補正の係数は異なる。
    if(!isValid()){validate();}
    Eigen::Vector3d lla;
    auto geod=std::dynamic_pointer_cast<GeodeticCRS>(getBaseCRS()); // 有効であることはvalidateで判定済
    if(geod->isGeocentric()){
        lla=geod->transformECEFCoordinateToGeographicCRS(baseLocation,CoordinateType::POSITION_ABS);
    }else{
        lla=baseLocation;
    }
    double lat=deg2rad(lla(0));
    double lon=deg2rad(lla(1));
    // まずは高度0における値を計算
    double t=sinh(atanh(sin(lat))-(2*sqrt(n))/(1+n)*atanh((2*sqrt(n))/(1+n)*sin(lat)));

    double ttp1=sqrt(1+t*t);
    double dlon=lon-deg2rad(lon_0);

    double xi=atan(t/cos(dlon));
    double eta=atanh(sin(dlon)/ttp1);
    double sigma=1;
    double tau=0;
    for(int j=1;j<=alpha.size();++j){
        double alpha_sin_j=alpha(j-1)*sin(2*j*xi);
        double alpha_cos_j=alpha(j-1)*cos(2*j*xi);
        double sinh_j=sinh(2*j*eta);
        double cosh_j=cosh(2*j*eta);
        sigma+=2*j*alpha_cos_j*cosh_j;
        tau+=2*j*alpha_sin_j*sinh_j;
    }
    double semiMajorAxis=geod->semiMajorAxis();
    double inverseFlattening=geod->inverseFlattening();

    double scale0=(scaleFactor*A/semiMajorAxis)*sqrt((1+pow((1-n)/(1+n)*tan(lat),2))*(sigma*sigma+tau*tau)/(t*t+pow(cos(dlon),2)));
    if(onSurface){
        return scale0;
    }else{
        // 高度補正を実施
        double alt=lla(2);
        double e2=(2-1/inverseFlattening)/inverseFlattening;
        double e2sin2lat=e2*pow(sin(lat),2);
        double nu=semiMajorAxis/sqrt(1-e2sin2lat);
        return scale0*nu/(nu+alt);
    }
}
double GenericUTMCRS::getParallelScaleFactorAtProjectedCoordinate(const Eigen::Vector3d& projectedLocation,bool onSurface) const{
    // 正角図法なので高度0では子午線方向と等しいが、高度補正の係数は異なる。
    if(!isValid()){validate();}
    Eigen::Vector3d ENU=getQuaternionToENU().transformVector(projectedLocation-offset);
    // まずは高度0における値を計算
    double E=ENU(0);
    double N=ENU(1);
    double xi=N/(scaleFactor*A);
    double eta=E/(scaleFactor*A);
    double xi_d=xi;
    double eta_d=eta;
    double sigma_d=1;
    double tau_d=0;
    for(int j=1;j<=beta.size();++j){
        double beta_sin_j=beta(j-1)*sin(2*j*xi);
        double beta_cos_j=beta(j-1)*cos(2*j*xi);
        double sinh_j=sinh(2*j*eta);
        double cosh_j=cosh(2*j*eta);
        xi_d-=beta_sin_j*cosh_j;
        eta_d-=beta_cos_j*sinh_j;
        sigma_d-=2*j*beta_cos_j*cosh_j;
        tau_d+=2*j*beta_sin_j*sinh_j;
    }
    double khi=asin(sin(xi_d)/cosh(eta_d));
    double lat=khi;
    for(int j=1;j<=delta.size();++j){
        lat+=delta(j-1)*sin(2*j*khi);
    }
    auto geod=std::dynamic_pointer_cast<GeodeticCRS>(getBaseCRS()); // 有効であることはvalidateで判定済
    double semiMajorAxis=geod->semiMajorAxis();
    double inverseFlattening=geod->inverseFlattening();
    double scale0=(scaleFactor*A/semiMajorAxis)*sqrt((1+pow((1-n)/(1+n)*tan(lat),2))*(pow(cos(xi_d),2)+pow(sinh(eta_d),2))/(sigma_d*sigma_d+tau_d*tau_d));
    if(onSurface){
        return scale0;
    }else{
        // 高度補正を実施
        double alt=ENU(2);
        double e2=(2-1/inverseFlattening)/inverseFlattening;
        double e2sin2lat=e2*pow(sin(lat),2);
        double nu=semiMajorAxis/sqrt(1-e2sin2lat);
        return scale0*nu/(nu+alt);
    }
}
// (4) 子午線と緯線のなす角 (meridian/parallel intersection angle)
double GenericUTMCRS::getIntersectionAngleAtBaseCoordinate(const Eigen::Vector3d& baseLocation) const{
    // 正角図法なので子午線と緯線は直交。
    return M_PI_2;
}
double GenericUTMCRS::getIntersectionAngleAtProjectedCoordinate(const Eigen::Vector3d& projectedLocation) const{
    // 正角図法なので子午線と緯線は直交。
    return M_PI_2;
}
// このCRS上の座標値と、投影図上でのENU相当(grid easting-grid northing-altitude又はX-Y-altitude)の座標値を相互変換するための回転クォータニオンを返す。
// ProjectedCRSの共通インターフェースとして設けるもの。ENU以外の座標軸で表現したい場合には派生クラスでオーバーライドすること。
// 主な用途としては軸順と符号の入替によって、NED等の異なる座標軸で扱いたい場合が挙げられる。
Quaternion GenericUTMCRS::getQuaternionToENU() const{
    return getAxisSwapQuaternion(getAxisOrder(),"ENU");
}
Quaternion GenericUTMCRS::getQuaternionFromENU() const{
    return getAxisSwapQuaternion("ENU",getAxisOrder());
}

Eigen::Vector3d GenericUTMCRS::getEllipsoidalOrigin() const{
    if(!isValid()){validate();}
    return ellipsoidalOrigin;
}
std::string GenericUTMCRS::getAxisOrder() const{
    if(!isValid()){validate();}
    return axisOrder;
}
void GenericUTMCRS::checkAxisOrder() const{
    if(instanceConfig.contains("axisOrder")){
        axisOrder=instanceConfig.at("axisOrder");
    }else if(modelConfig.contains("axisOrder")){
        axisOrder=modelConfig.at("axisOrder");
    }else{
        axisOrder="ENU";
    }
    std::transform(axisOrder.begin(),axisOrder.end(),axisOrder.begin(),
        [](unsigned char c){ return std::toupper(c);}
    );
    assert(checkTopocentricCartesianAxisOrder(axisOrder));
}

void exportProjectedCRS(py::module &m, const std::shared_ptr<asrc::core::FactoryHelper>& factoryHelper){
    using namespace pybind11::literals;

    expose_entity_subclass<ProjectedCRS>(m,"ProjectedCRS")
    DEF_FUNC(ProjectedCRS,getMeridianConversionAtBaseCoordinate)
    DEF_FUNC(ProjectedCRS,getMeridianConversionAtIntermediateCoordinate)
    DEF_FUNC(ProjectedCRS,getMeridianConversionAtProjectedCoordinate)
    DEF_FUNC(ProjectedCRS,getMeridionalScaleFactorAtBaseCoordinate)
    DEF_FUNC(ProjectedCRS,getMeridionalScaleFactorAtIntermediateCoordinate)
    DEF_FUNC(ProjectedCRS,getMeridionalScaleFactorAtProjectedCoordinate)
    DEF_FUNC(ProjectedCRS,getParallelScaleFactorAtBaseCoordinate)
    DEF_FUNC(ProjectedCRS,getParallelScaleFactorAtIntermediateCoordinate)
    DEF_FUNC(ProjectedCRS,getParallelScaleFactorAtProjectedCoordinate)
    DEF_FUNC(ProjectedCRS,getIntersectionAngleAtBaseCoordinate)
    DEF_FUNC(ProjectedCRS,getIntersectionAngleAtIntermediateCoordinate)
    DEF_FUNC(ProjectedCRS,getIntersectionAngleAtProjectedCoordinate)
    ;

    expose_entity_subclass<GenericUTMCRS>(m,"GenericUTMCRS")
    ;

    FACTORY_ADD_CLASS(CoordinateReferenceSystem,GenericUTMCRS)
}

ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
