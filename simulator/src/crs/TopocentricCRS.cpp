// Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#include "crs/TopocentricCRS.h"
#include "crs/GeodeticCRS.h"
#include "crs/PureFlatCRS.h"
#include "crs/AffineCRS.h"
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

TopocentricCRS::TopocentricCRS(const nl::json& modelConfig_,const nl::json& instanceConfig_)
:DerivedCRS(modelConfig_,instanceConfig_){
    crsType=CRSType::TOPOCENTRIC;
}
void TopocentricCRS::initialize(){
    BaseType::initialize();
    // 原点の設定
    // cartesianOriginは原点のbaseCRSにおける座標。
    givenByEllipsoidalOrigin=false;
    givenByCartesianOrigin=false;
    givenByMotionState=false;
    ellipsoidalOrigin=Eigen::Vector3d::Zero(); // in lat[deg], lon[deg], alt[m]
    cartesianOrigin=Eigen::Vector3d::Zero(); // in X[m], Y[m], Z[m]
    if(instanceConfig.contains("cartesianOrigin")){
        // baseがGeodeticの場合は対応するECEF上での原点位置、
        // Affineの場合はbaseCRS上での原点位置として扱う。
        cartesianOrigin=instanceConfig.at("cartesianOrigin");
        givenByCartesianOrigin=true;
    }else if(modelConfig.contains("cartesianOrigin")){
        // baseがGeodeticの場合は対応するECEF上での原点位置、
        // Affineの場合はbaseCRS上での原点位置として扱う。
        cartesianOrigin=modelConfig.at("cartesianOrigin");
        givenByCartesianOrigin=true;
    }else if(instanceConfig.contains("ellipsoidalOrigin")){
        // baseがGeodeticの場合のみ有効
        ellipsoidalOrigin=instanceConfig.at("ellipsoidalOrigin");
        givenByEllipsoidalOrigin=true;
    }else if(modelConfig.contains("ellipsoidalOrigin")){
        ellipsoidalOrigin=modelConfig.at("ellipsoidalOrigin");
        givenByEllipsoidalOrigin=true;
    }else if(instanceConfig.contains("motion")){
        // baseがGeographicの場合は無効
        // baseCRS上での原点位置及び座標軸の回転情報として扱う。
        internalMotionState=instanceConfig.at("motion");
        givenByMotionState=true;
    }else if(modelConfig.contains("motion")){
        internalMotionState=modelConfig.at("motion");
        givenByMotionState=true;
    }else{
        throw std::runtime_error("TopocentricCRS requires 'cartesianOrigin', 'ellipsoidalOrigin', or 'motion' for instantiation.");
    }
    if(instanceConfig.contains("onSurface")){
        onSurface=instanceConfig.at("onSurface");
    }else if(modelConfig.contains("onSurface")){
        onSurface=modelConfig.at("onSurface");
    }else{
        onSurface=false;
    }
    lastAZ=0;
}
// 内部状態のシリアライゼーション
void TopocentricCRS::serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full){
    BaseType::serializeInternalState(archive,full);

    if(isValid()){
        ASRC_SERIALIZE_NVP(archive
            ,_isSpherical
            ,axisOrder
            ,internalMotionState
            ,cartesianOrigin
        )
    }
    if(full){
        ASRC_SERIALIZE_NVP(archive
            ,onSurface
            ,lastAZ
            ,givenByEllipsoidalOrigin
            ,givenByCartesianOrigin
            ,givenByMotionState
            ,ellipsoidalOrigin
        )
    }
}
void TopocentricCRS::validateImpl(bool isFirstTime) const{
    this->DerivedCRS::validateImpl(isFirstTime);
    if(isFirstTime || !hasBeenValidated()){
        // baseCRSの種別チェックは省略(Topocentricは任意のCRSをbaseにしてよい)

        // originの指定方法チェック
        auto baseCRS = getBaseCRS();
        if(givenByEllipsoidalOrigin){
            // baseがGeodeticの場合のみ有効
            if(!std::dynamic_pointer_cast<GeodeticCRS>(baseCRS)){
                throw std::runtime_error("'ellipsoidalOrigin' for TopocentricCRS config is valid only if baseCRS is GeodeticCRS.");
            }
        }else if(givenByMotionState){
            // baseがGeographicの場合は無効
            // baseCRS上での原点位置及び座標軸の回転情報として扱う。
            if(std::dynamic_pointer_cast<GeographicCRS>(baseCRS)){
                throw std::runtime_error("'motion' in TopocentricCRS config is valid only if baseCRS is not GeographicCRS.");
            }
            if(internalMotionState.getCRS()){
                // 有効なCRSを持っていれば変換
                cartesianOrigin=internalMotionState.pos(baseCRS);
            }else{
                // 持っていなければ直接値を取得
                cartesianOrigin=internalMotionState.pos();
            }
        }
        checkCoordinateSystemType();
        checkAxisOrder();
        internalMotionState.setCRS(getIntermediateCRS(),true);
    }
    // internalMotionState(のposとq)を更新する
    calculateMotionStates();
}
void TopocentricCRS::setBaseCRS(const std::shared_ptr<CoordinateReferenceSystem>& newBaseCRS) const{
    if(!isValid()){validate();}
    if(newBaseCRS){
        // 現在のnonDerivedBaseから変換できないものは禁止
        auto oldNonDerivedBase=getNonDerivedBaseCRS();
        auto newNonDerivedBase=newBaseCRS->getNonDerivedBaseCRS();
        if(oldNonDerivedBase && !oldNonDerivedBase->isTransformableTo(newNonDerivedBase)){
            throw std::runtime_error("new non-derived base CRS cannot be transformable from the current non-derived base CRS.");
        }
    }
    this->DerivedCRS::setBaseCRS(newBaseCRS);//invalidate()
    auto nonDerivedBase=getNonDerivedBaseCRS();
    if(nonDerivedBase->isGeographic() || !nonDerivedBase->isEarthFixed()){
        internalMotionState.setCRS(std::dynamic_pointer_cast<GeodeticCRS>(nonDerivedBase)->getECEF(),true);
    }else{
        internalMotionState.setCRS(nonDerivedBase,true);
    }
}
bool TopocentricCRS::isCartesian() const{
    if(!isValid()){validate();}
    return !_isSpherical;
}
bool TopocentricCRS::isSpherical() const{
    if(!isValid()){validate();}
    return _isSpherical;
}
bool TopocentricCRS::isEllipsoidal() const{
    return false;
}
//座標系の種類
bool TopocentricCRS::isEarthFixed() const{
    return true;
}
bool TopocentricCRS::isTopocentric() const{
    return true;
}
bool TopocentricCRS::isProjected() const{
    return false;
}
//中間座標系の取得
std::shared_ptr<CoordinateReferenceSystem> TopocentricCRS::getIntermediateCRS() const{
    return getNonDerivedBaseCRS()->getIntermediateCRS();
}
Coordinate TopocentricCRS::transformTo(const Eigen::Vector3d& value, const Eigen::Vector3d& location, const Time& time, const std::shared_ptr<CoordinateReferenceSystem>& dstCRS, const CoordinateType& coordinateType) const{
    if(!isValid()){validate();}
    if(!dstCRS){
        throw std::runtime_error("dstCRS is nullptr!");
    }
    if(this==std::dynamic_pointer_cast<TopocentricCRS>(dstCRS).get()){
        return std::move(Coordinate(
            value,
            location,
            dstCRS,
            time,
            coordinateType
        ));
    }
    if(!isTransformableTo(dstCRS)){
        throw std::runtime_error(getFactoryClassName()+" can't be transformed to "+dstCRS->getFactoryClassName());
    }
    if(dstCRS==getIntermediateCRS()){
        return transformToIntermediateCRS(value,location,time,coordinateType);
    }else if(dstCRS==getNonDerivedBaseCRS()){
        return transformToNonDerivedBaseCRS(value,location,time,coordinateType);
    }else if(dstCRS==getBaseCRS()){
        return transformToBaseCRS(value,location,time,coordinateType);
    }else{
        return dstCRS->transformFrom(transformToIntermediateCRS(value,location,time,coordinateType));
    }
}
Coordinate TopocentricCRS::transformFrom(const Eigen::Vector3d& value, const Eigen::Vector3d& location, const Time& time, const std::shared_ptr<CoordinateReferenceSystem>& srcCRS, const CoordinateType& coordinateType) const{
    if(!isValid()){validate();}
    if(!srcCRS){
        throw std::runtime_error("srcCRS is nullptr!");
    }
    if(this==std::dynamic_pointer_cast<TopocentricCRS>(srcCRS).get()){
        return std::move(Coordinate(
            value,
            location,
            srcCRS,
            time,
            coordinateType
        ));
    }
    if(!isTransformableFrom(srcCRS)){
        throw std::runtime_error(getFactoryClassName()+" can't be transformed from "+srcCRS->getFactoryClassName());
    }
    if(srcCRS==getIntermediateCRS()){
        return transformFromIntermediateCRS(value,location,time,coordinateType);
    }else if(srcCRS==getNonDerivedBaseCRS()){
        return transformFromNonDerivedBaseCRS(value,location,time,coordinateType);
    }else if(srcCRS==getBaseCRS()){
        return transformFromBaseCRS(value,location,time,coordinateType);
    }else{
        return transformFromIntermediateCRS(getIntermediateCRS()->transformFrom(value,location,time,srcCRS,coordinateType));
    }
}
Quaternion TopocentricCRS::getQuaternionTo(const Eigen::Vector3d& location, const Time& time, const std::shared_ptr<CoordinateReferenceSystem>& dstCRS) const{
    if(!isValid()){validate();}
    if(!dstCRS){
        throw std::runtime_error("dstCRS is nullptr!");
    }
    if(this==std::dynamic_pointer_cast<TopocentricCRS>(dstCRS).get()){
        return Quaternion(1,0,0,0);
    }
    if(!isTransformableTo(dstCRS)){
        throw std::runtime_error(getFactoryClassName()+" can't be transformed to "+dstCRS->getFactoryClassName());
    }
    // MotionState::qは左からqをかけてderived座標をbase座標に移す
    // v_base = q.transformVector(v_derived)
    auto qToIntermediateCRSFromThis=internalMotionState.q;
    if(getIntermediateCRS()==dstCRS){
        return qToIntermediateCRSFromThis;
    }else{
        auto locationInIntermediate=transformToIntermediateCRS(location,location,time,CoordinateType::POSITION_ABS)();
        auto qToDstFromIntermediateCRS=getIntermediateCRS()->getQuaternionTo(locationInIntermediate,time,dstCRS);
        return qToDstFromIntermediateCRS*qToIntermediateCRSFromThis;
    }
}
Quaternion TopocentricCRS::getQuaternionFrom(const Eigen::Vector3d& location, const Time& time, const std::shared_ptr<CoordinateReferenceSystem>& srcCRS) const{
    if(!isValid()){validate();}
    if(!srcCRS){
        throw std::runtime_error("srcCRS is nullptr!");
    }
    if(this==std::dynamic_pointer_cast<TopocentricCRS>(srcCRS).get()){
        return Quaternion(1,0,0,0);
    }
    if(!isTransformableFrom(srcCRS)){
        throw std::runtime_error(getFactoryClassName()+" can't be transformed from "+srcCRS->getFactoryClassName());
    }
    // MotionState::qは左からqをかけてderived座標をbase座標に移す
    // v_base = q.transformVector(v_derived)
    auto qToThisFromIntermediateCRS=internalMotionState.q.conjugate();
    if(getIntermediateCRS()==srcCRS){
        return qToThisFromIntermediateCRS;
    }else{
        auto qToIntermediateCRSFromSrc=getIntermediateCRS()->getQuaternionFrom(location,time,srcCRS);
        return qToThisFromIntermediateCRS*qToIntermediateCRSFromSrc;
    }
}
//親座標系との座標変換
Coordinate TopocentricCRS::transformToBaseCRS(const Eigen::Vector3d& value,const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType) const{
    auto baseCRS=getBaseCRS();
    auto intermediateCRS=getIntermediateCRS();
    auto valueInIntermediate=transformToIntermediateCRS(value,location,time,coordinateType);
    if(baseCRS==intermediateCRS){
        return valueInIntermediate;
    }else{
        return baseCRS->transformFrom(valueInIntermediate);
    }
}
Coordinate TopocentricCRS::transformFromBaseCRS(const Eigen::Vector3d& value,const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType) const{
    auto baseCRS=getBaseCRS();
    auto intermediateCRS=getIntermediateCRS();
    if(baseCRS==intermediateCRS){
        return transformFromIntermediateCRS(value,location,time,coordinateType);
    }else{
        return transformFromIntermediateCRS(baseCRS->transformTo(value,location,time,intermediateCRS,coordinateType));
    }
}
Coordinate TopocentricCRS::transformToNonDerivedBaseCRS(const Eigen::Vector3d& value,const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType) const{
    if(!isValid()){validate();}
    auto nonDerivedBase=getNonDerivedBaseCRS();
    auto intermediateCRS=getIntermediateCRS();
    auto valueInIntermediate=transformToIntermediateCRS(value,location,time,coordinateType);
    if(nonDerivedBase==intermediateCRS){
        return valueInIntermediate;
    }else{
        return nonDerivedBase->transformFrom(valueInIntermediate);
    }
}
Coordinate TopocentricCRS::transformFromNonDerivedBaseCRS(const Eigen::Vector3d& value,const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType) const{
    if(!isValid()){validate();}
    auto nonDerivedBase=getNonDerivedBaseCRS();
    auto intermediateCRS=getIntermediateCRS();
    if(nonDerivedBase==intermediateCRS){
        return transformFromIntermediateCRS(value,location,time,coordinateType);
    }else{
        return transformFromIntermediateCRS(nonDerivedBase->transformTo(value,location,time,intermediateCRS,coordinateType));
    }
}
Coordinate TopocentricCRS::transformToIntermediateCRS(const Eigen::Vector3d& value_,const Eigen::Vector3d& location_, const Time& time, const CoordinateType& coordinateType) const{
    if(!isValid()){validate();}
    Eigen::Vector3d ret;
    auto value=toCartesian(value_);
    auto location=toCartesian(location_);
    if(coordinateType==CoordinateType::POSITION_ABS){
        ret=internalMotionState.absBtoP(value);
    }else if(coordinateType==CoordinateType::POSITION_REL){
        ret=internalMotionState.relBtoP(value);
    }else if(coordinateType==CoordinateType::DIRECTION){
        ret=internalMotionState.dirBtoP(value);
    }else if(coordinateType==CoordinateType::VELOCITY){
        ret=internalMotionState.velBtoP(value,location);
    }else if(coordinateType==CoordinateType::ANGULAR_VELOCITY){
        ret=internalMotionState.omegaBtoP(value);
    }else{
        throw std::runtime_error("Invalide coordinate type.");
    }
    return Coordinate(
        ret,
        internalMotionState.absBtoP(location),
        getIntermediateCRS(),
        time,
        coordinateType
    );
}
Coordinate TopocentricCRS::transformFromIntermediateCRS(const Eigen::Vector3d& value,const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType) const{
    if(!isValid()){validate();}
    Eigen::Vector3d ret;
    if(coordinateType==CoordinateType::POSITION_ABS){
        ret=internalMotionState.absPtoB(value);
    }else if(coordinateType==CoordinateType::POSITION_REL){
        ret=internalMotionState.relPtoB(value);
    }else if(coordinateType==CoordinateType::DIRECTION){
        ret=internalMotionState.dirPtoB(value);
    }else if(coordinateType==CoordinateType::VELOCITY){
        ret=internalMotionState.velPtoB(value,location);
    }else if(coordinateType==CoordinateType::ANGULAR_VELOCITY){
        ret=internalMotionState.omegaPtoB(value);
    }else{
        throw std::runtime_error("Invalide coordinate type.");
    }
    return Coordinate(
        fromCartesian(ret),
        fromCartesian(internalMotionState.absPtoB(location)),
        non_const_this(),
        time,
        coordinateType
    );
}
void TopocentricCRS::checkCoordinateSystemType() const{
    if(instanceConfig.contains("isSpherical")){
        _isSpherical=instanceConfig.at("isSpherical");
    }else if(modelConfig.contains("isSpherical")){
        _isSpherical=modelConfig.at("isSpherical");
    }else{
        _isSpherical=false;
    }
}
void TopocentricCRS::checkAxisOrder() const{
    if(instanceConfig.contains("axisOrder")){
        axisOrder=instanceConfig.at("axisOrder");
    }else if(modelConfig.contains("axisOrder")){
        axisOrder=modelConfig.at("axisOrder");
    }else{
        axisOrder="NED";
    }
    std::transform(axisOrder.begin(),axisOrder.end(),axisOrder.begin(),
        [](unsigned char c){ return std::toupper(c);}
    );
    assert(checkBodyCartesianAxisOrder(axisOrder) || checkTopocentricCartesianAxisOrder(axisOrder));
}
std::string TopocentricCRS::getAxisOrder() const{
    if(!isValid()){validate();}
    return axisOrder;
}

Eigen::Vector3d TopocentricCRS::sphericalToCartesian(const Eigen::Vector3d& v) const{
    if(checkTopocentricCartesianAxisOrder(getAxisOrder())){
        // (Azimuth Elevation Range) -> NED -> axisOrder
        Eigen::Vector3d ned(
            v(2)*cos(v(0))*cos(v(1)),
            v(2)*sin(v(0))*cos(v(1)),
            -v(2)*sin(v(1))
        );
        return getAxisSwapQuaternion("NED",getAxisOrder()).transformVector(ned);
    }else{// if(checkBodyCartesianAxisOrder(getAxisOrder()){
        // (Azimuth Elevation Range) -> FSD -> axisOrder
        Eigen::Vector3d fsd(
            v(2)*cos(v(0))*cos(v(1)),
            v(2)*sin(v(0))*cos(v(1)),
            -v(2)*sin(v(1))
        );
        return getAxisSwapQuaternion("FSD",getAxisOrder()).transformVector(fsd);
    }
}
Eigen::Vector3d TopocentricCRS::cartesianToSpherical(const Eigen::Vector3d& v) const{
    if(checkTopocentricCartesianAxisOrder(getAxisOrder())){
        // axisOrder -> NED -> (Azimuth Elevation Range)
        auto ned=getAxisSwapQuaternion(getAxisOrder(),"NED").transformVector(v);
        double xy=sqrt(ned(0)*ned(0)+ned(1)*ned(1));
        if(xy==0){
            return Eigen::Vector3d(
                0,
                ned(2)>0 ? -M_PI_2 : M_PI_2,
                abs(ned(2))
            );
        }else{
            double el=atan2(-ned(2),xy);
            return Eigen::Vector3d(
                atan2(ned(1),ned(0)),
                el,
                sqrt(xy*xy+ned(2)*ned(2))
            );
        }
    }else{// if(checkBodyCartesianAxisOrder(getAxisOrder()){
    // axisOrder -> FSD -> (Azimuth Elevation Range)
        auto fsd=getAxisSwapQuaternion(getAxisOrder(),"FSD").transformVector(v);
        double xy=sqrt(fsd(0)*fsd(0)+fsd(1)*fsd(1));
        if(xy==0){
            return Eigen::Vector3d(
                0,
                fsd(2)>0 ? -M_PI_2 : M_PI_2,
                abs(fsd(2))
            );
        }else{
            double el=atan2(-fsd(2),xy);
            return Eigen::Vector3d(
                atan2(fsd(1),fsd(0)),
                el,
                sqrt(xy*xy+fsd(2)*fsd(2))
            );
        }
    }
}

Eigen::Vector3d TopocentricCRS::getEllipsoidalOrigin() const{
    if(!isValid()){validate();}
    return ellipsoidalOrigin;
}
Eigen::Vector3d TopocentricCRS::getCartesianOrigin() const{
    if(!isValid()){validate();}
    return cartesianOrigin;
}
void TopocentricCRS::updateByMotionState(const MotionState& motionState_){
    MotionState motionStateInBase=motionState_.transformTo(getBaseCRS());
    internalMotionState.setTime(motionState_.time);
    if(!givenByEllipsoidalOrigin){
        cartesianOrigin=motionStateInBase.pos();
    }
    invalidate();
}
void TopocentricCRS::calculateMotionStates() const{
    auto baseCRS = getBaseCRS();
    auto intermediateCRS = getIntermediateCRS();
    auto nonDerivedBaseCRS = getNonDerivedBaseCRS();

    Eigen::Vector3d cartesianOriginInIntermediateCRS = cartesianOrigin;
    if(auto derived = std::dynamic_pointer_cast<DerivedCRS>(baseCRS)){
        // baseがDerivedCRSの場合には、cartesianOriginをbaseCRSにおける原点位置と解釈する。
        // ellipsoidalOriginはDerivedCRSをbaseに指定する意味がないので無視する。
        cartesianOriginInIntermediateCRS = derived->transformTo(
            cartesianOrigin,
            cartesianOrigin,
            internalMotionState.time,
            intermediateCRS,
            CoordinateType::POSITION_ABS
        )();
    }
    
    // ENUへの変換
    if(auto geod = std::dynamic_pointer_cast<GeodeticCRS>(intermediateCRS)){
        if(givenByEllipsoidalOrigin){
            // こちらの分岐に入るときはbase自身がGeodeticCRS
            cartesianOriginInIntermediateCRS = geod->transformGeographicCoordinateToECEF(ellipsoidalOrigin,CoordinateType::POSITION_ABS);
            cartesianOrigin = cartesianOriginInIntermediateCRS;
        }else{
            ellipsoidalOrigin = geod->transformECEFCoordinateToGeographicCRS(cartesianOriginInIntermediateCRS,CoordinateType::POSITION_ABS);
        }
        // EPSG:9836
        double lat_0=deg2rad(ellipsoidalOrigin(0));
        double lon_0=deg2rad(ellipsoidalOrigin(1));
        Eigen::Matrix3d Rt({
            {-sin(lon_0), -sin(lat_0)*cos(lon_0), cos(lat_0)*cos(lon_0)},
            { cos(lon_0), -sin(lat_0)*sin(lon_0), cos(lat_0)*sin(lon_0)},
            {          0,             cos(lat_0),            sin(lat_0)}
        });
        // pos_ecef = Rt * pos_topo + cartesianOriginInIntermediateCRS
        // pos_base = this.q.transformVector(pos_this) + this.pos
        internalMotionState.setPos(cartesianOriginInIntermediateCRS);
        internalMotionState.setQ(Quaternion::fromRotationMatrix(Rt));
    }else if(auto flat = std::dynamic_pointer_cast<PureFlatCRS>(intermediateCRS)){
        // PureFlatからの変換はPureFlatのaxisOrderを考慮する必要がある。
        internalMotionState.setPos(cartesianOriginInIntermediateCRS);
        internalMotionState.setQ(getAxisSwapQuaternion("ENU",flat->getAxisOrder()));
    }else{
        throw std::runtime_error("Non-derived (top-level) base CRS of TopocentricCRS must be Geodetic or PureFlat CRS.");
    }

    //高度方向のオフセット
    if(onSurface){
        auto altOffset=internalMotionState.q.transformVector(
            Eigen::Vector3d(0,0,intermediateCRS->getHeight(internalMotionState.pos(),internalMotionState.time))
        );// [intermediateCRS<-ENU]
        internalMotionState.setPos(internalMotionState.pos()-altOffset);
    }

    //ENUから指定の座標軸への変換
    if(checkTopocentricCartesianAxisOrder(getAxisOrder())){
        if(getAxisOrder()!="ENU"){
            internalMotionState.setQ( //[intermediateCRS <- axisOrder] =
                internalMotionState.q * // [intermediateCRS<-ENU] *
                getAxisSwapQuaternion(getAxisOrder(),"ENU") // [ENU <- axisOrder]
            );
        }
    }else{// if(checkBodyCartesianAxisOrder(getAxisOrder()){
        if(auto affine = std::dynamic_pointer_cast<AffineCRS>(baseCRS)){
            // baseのFWD方向のベクトルを水平面に射影した方向を「前方」に相当する基準軸Fとした局所水平座標系
            // 「左方」に相当するP=U×Fの3軸を導出する必要がある。
            Eigen::Vector3d fwdInBase=getAxisSwapQuaternion("FPU",affine->getAxisOrder()).transformVector(Eigen::Vector3d(1,0,0));
            Eigen::Vector3d fwdInIntermediateCRS=affine->transformTo(
                fwdInBase,
                Eigen::Vector3d::Zero(),
                internalMotionState.time,
                intermediateCRS,
                CoordinateType::DIRECTION
            )();
            Eigen::Vector3d fwdInENU=internalMotionState.relPtoB(fwdInIntermediateCRS);
            double xy=sqrt(fwdInENU(0)*fwdInENU(0)+fwdInENU(1)*fwdInENU(1));
            if(xy>0){
                lastAZ=atan2(fwdInENU(1),fwdInENU(0));
            } // xy==0の場合は前回の値を使う。
            fwdInENU=Eigen::Vector3d(cos(lastAZ),sin(lastAZ),0);
            Eigen::Vector3d portInENU(-sin(lastAZ),cos(lastAZ),0);
            Eigen::Vector3d upInENU(0,0,1);
            auto qFPUtoENU=Quaternion::fromBasis(fwdInENU,portInENU,upInENU); // [ENU <- FPU]

            internalMotionState.setQ( //[intermediateCRS <- axisOrder] =
                internalMotionState.q * //[intermediateCRS <- ENU] *
                qFPUtoENU.conjugate() * //[ENU <- FPU] *
                getAxisSwapQuaternion(getAxisOrder(),"FPU") //[FPU <- axisOrder] *
            );
        }else{
            throw std::runtime_error("Body-oriented TopocentricCRS can be valid only if the base CRS is AffineCRS.");
        }
    }
    // TopocentricはEarth-fixedなのでinternalMotionStateの速度と角速度は0となる
    internalMotionState.setVel(Eigen::Vector3d::Zero());
    internalMotionState.setOmega(Eigen::Vector3d::Zero());
}
Quaternion TopocentricCRS::getEquivalentQuaternionToTopocentricCRS(const Eigen::Vector3d& origin, const Time& time, const std::string& axisOrder_) const{
    //CRSとして生成せずにクォータニオンとして得る。
    std::string dstAxisOrder=axisOrder_;
    std::transform(dstAxisOrder.begin(),dstAxisOrder.end(),dstAxisOrder.begin(),
        [](unsigned char c){ return std::toupper(c);}
    );
    auto intermediateCRS = getIntermediateCRS();
    if(!intermediateCRS){
        throw std::runtime_error("intermediateCRS is nullptr!");
    }
    Coordinate cartesianOriginInIntermediateCRS = transformToIntermediateCRS(origin,origin,time,CoordinateType::POSITION_ABS);
    if(checkTopocentricCartesianAxisOrder(dstAxisOrder)){
        return (
            intermediateCRS->getEquivalentQuaternionToTopocentricCRS(cartesianOriginInIntermediateCRS,dstAxisOrder) * //[new Topo <- IntermediateCRS]
            getQuaternionFrom(cartesianOriginInIntermediateCRS).conjugate() //[IntermediateCRS <- this]
        );
    }else{
        throw std::runtime_error("Body-oriented TopocentricCRS cannot be created from TopocentricCRS.");
    }
}

void exportTopocentricCRS(py::module &m, const std::shared_ptr<asrc::core::FactoryHelper>& factoryHelper){
    using namespace pybind11::literals;

    expose_entity_subclass<TopocentricCRS>(m,"TopocentricCRS")
    DEF_FUNC(TopocentricCRS,getEllipsoidalOrigin)
    DEF_FUNC(TopocentricCRS,getCartesianOrigin)
    ;

    FACTORY_ADD_CLASS(CoordinateReferenceSystem,TopocentricCRS)
}

ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
