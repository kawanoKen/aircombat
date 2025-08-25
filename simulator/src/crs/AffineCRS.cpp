// Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#include "crs/AffineCRS.h"
#include "Units.h"
#include <algorithm>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <iomanip>
#include "SimulationManager.h"

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)

using namespace util;

AffineCRS::AffineCRS(const nl::json& modelConfig_,const nl::json& instanceConfig_)
:DerivedCRS(modelConfig_,instanceConfig_){
    crsType=CRSType::AFFINE;
}
void AffineCRS::initialize(){
    BaseType::initialize();
    if(instanceConfig.contains("motion")){
        internalMotionState=instanceConfig.at("motion");
    }else if(modelConfig.contains("motion")){
        internalMotionState=modelConfig.at("motion");
    }else{
        throw std::runtime_error("AffineCRS requires 'motion' in configs.");
    }
}
// 内部状態のシリアライゼーション
void AffineCRS::serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full){
    BaseType::serializeInternalState(archive,full);

    if(isValid()){
        ASRC_SERIALIZE_NVP(archive
            ,_isSpherical
            ,axisOrder
            ,internalMotionState
        )
    }
}
void AffineCRS::validateImpl(bool isFirstTime) const{
    this->DerivedCRS::validateImpl(isFirstTime);
    if(isFirstTime || !hasBeenValidated()){
        auto base=getBaseCRS();
        // baseCRSの種別チェック
        // Geographicからの派生は禁止。ただし、GeographicをbaseとするProjectedからの派生は可。
        if(base->isGeographic()){
            throw std::runtime_error("AffineCRS cannot be derived from a GeographicCRS.");
        }

        checkCoordinateSystemType();
        checkAxisOrder();
        //internalMotionStateのcrsをbaseとする
        internalMotionState.setCRS(base,true);
    }
}
void AffineCRS::setBaseCRS(const std::shared_ptr<CoordinateReferenceSystem>& newBaseCRS) const{
    if(!isValid()){validate();}
    if(newBaseCRS){
        // ・geographicは禁止
        // ・projectedとnon-projectedを跨ぐ変更は禁止
        if(isProjected()){
            if(!newBaseCRS->isProjected()){
                throw std::runtime_error("A projected AffineCRS cannot change its base CRS to a non-projected CRS.");
            }
            if(getBaseProjectedCRS()!=newBaseCRS->getBaseProjectedCRS()){
                throw std::runtime_error("A projected AffineCRS cannot change its base CRS unless a new base CRS is on the chain between it and its base ProjectedCRS.");
            }
        }else{
            if(newBaseCRS->isProjected()){
                throw std::runtime_error("A non-projected AffineCRS cannot change its base CRS to a projected CRS.");
            }else if(newBaseCRS->isGeographic()){
                throw std::runtime_error("AffineCRS (except for TopocentricCRS) cannot be derived from a GeographicCRS.");
            }
        }
    }
    this->DerivedCRS::setBaseCRS(newBaseCRS);//invalidate()
    auto base=getBaseCRS();
    internalMotionState.setCRS(base,true);
}
bool AffineCRS::isCartesian() const{
    if(!isValid()){validate();}
    return !_isSpherical;
}
bool AffineCRS::isSpherical() const{
    if(!isValid()){validate();}
    return _isSpherical;
}
bool AffineCRS::isEllipsoidal() const{
    return false;
}
Quaternion AffineCRS::getQuaternionTo(const Eigen::Vector3d& location, const Time& time, const std::shared_ptr<CoordinateReferenceSystem>& dstCRS) const{
    if(!isValid()){validate();}
    if(!dstCRS){
        throw std::runtime_error("dstCRS is nullptr!");
    }
    if(this==std::dynamic_pointer_cast<AffineCRS>(dstCRS).get()){
        return Quaternion(1,0,0,0);
    }
    if(!isTransformableTo(dstCRS)){
        throw std::runtime_error(getFactoryClassName()+" can't be transformed to "+dstCRS->getFactoryClassName());
    }
    // MotionState::qは左からqをかけてderived座標をbase座標に移す
    // v_base = q.transformVector(v_derived)
    auto qToBaseFromThis=internalMotionState.q;
    if(getBaseCRS()==dstCRS){
        return qToBaseFromThis;
    }else{
        auto locationInBase=transformToBaseCRS(location,location,time,CoordinateType::POSITION_ABS)();
        auto qToDstFromBase=getBaseCRS()->getQuaternionTo(locationInBase,time,dstCRS);
        return qToDstFromBase*qToBaseFromThis;
    }
}
Quaternion AffineCRS::getQuaternionFrom(const Eigen::Vector3d& location, const Time& time, const std::shared_ptr<CoordinateReferenceSystem>& srcCRS) const{
    if(!isValid()){validate();}
    if(!srcCRS){
        throw std::runtime_error("srcCRS is nullptr!");
    }
    if(this==std::dynamic_pointer_cast<AffineCRS>(srcCRS).get()){
        return Quaternion(1,0,0,0);
    }
    if(!isTransformableFrom(srcCRS)){
        throw std::runtime_error(getFactoryClassName()+" can't be transformed from "+srcCRS->getFactoryClassName());
    }
    // MotionState::qは左からqをかけてderived座標をbase座標に移す
    // v_base = q.transformVector(v_derived)
    auto qToThisFromBase=internalMotionState.q.conjugate();
    if(getBaseCRS()==srcCRS){
        return qToThisFromBase;
    }else{
        auto qToBaseFromSrc=getBaseCRS()->getQuaternionFrom(location,time,srcCRS);
        return qToThisFromBase*qToBaseFromSrc;
    }
}

//親座標系との座標変換
Coordinate AffineCRS::transformToBaseCRS(const Eigen::Vector3d& value_,const Eigen::Vector3d& location_, const Time& time, const CoordinateType& coordinateType) const{
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
        internalMotionState.getCRS(),
        time,
        coordinateType
    );
}
Coordinate AffineCRS::transformFromBaseCRS(const Eigen::Vector3d& value,const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType) const{
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
Coordinate AffineCRS::transformToNonDerivedBaseCRS(const Eigen::Vector3d& value,const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType) const{
    auto baseCRS=getBaseCRS();
    if(auto derived=std::dynamic_pointer_cast<DerivedCRS>(baseCRS)){
        return derived->transformToNonDerivedBaseCRS(transformToBaseCRS(value,location,time,coordinateType));
    }else{//baseCRS==nonDerivedBaseCRS
        return transformToBaseCRS(value,location,time,coordinateType);
    }
}
Coordinate AffineCRS::transformFromNonDerivedBaseCRS(const Eigen::Vector3d& value,const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType) const{
    auto baseCRS=getBaseCRS();
    if(auto derived=std::dynamic_pointer_cast<DerivedCRS>(baseCRS)){
        return transformFromBaseCRS(derived->transformFromNonDerivedBaseCRS(value,location,time,coordinateType));
    }else{//baseCRS==nonDerivedBaseCRS
        return transformFromBaseCRS(value,location,time,coordinateType);
    }
}
Coordinate AffineCRS::transformToIntermediateCRS(const Eigen::Vector3d& value,const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType) const{
    // AffineCRSの場合はbase==intermediate
    return transformToBaseCRS(value,location,time,coordinateType);
}
Coordinate AffineCRS::transformFromIntermediateCRS(const Eigen::Vector3d& value,const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType) const{
    // AffineCRSの場合はbase==intermediate
    return transformFromBaseCRS(value,location,time,coordinateType);
}
void AffineCRS::checkCoordinateSystemType() const{
    if(instanceConfig.contains("isSpherical")){
        _isSpherical=instanceConfig.at("isSpherical");
    }else if(modelConfig.contains("isSpherical")){
        _isSpherical=modelConfig.at("isSpherical");
    }else{
        _isSpherical=false;
    }
}
void AffineCRS::checkAxisOrder() const{
    if(instanceConfig.contains("axisOrder")){
        axisOrder=instanceConfig.at("axisOrder");
    }else if(modelConfig.contains("axisOrder")){
        axisOrder=modelConfig.at("axisOrder");
    }else{
        axisOrder="FSD";
    }
    std::transform(axisOrder.begin(),axisOrder.end(),axisOrder.begin(),
        [](unsigned char c){ return std::toupper(c);}
    );
    assert(checkBodyCartesianAxisOrder(axisOrder));
}
std::string AffineCRS::getAxisOrder() const{
    if(!isValid()){validate();}
    return axisOrder;
}

Eigen::Vector3d AffineCRS::cartesianToSpherical(const Eigen::Vector3d& v) const{
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
Eigen::Vector3d AffineCRS::sphericalToCartesian(const Eigen::Vector3d& v) const{
    // (Azimuth Elevation Range) -> FSD -> axisOrder
    Eigen::Vector3d fsd(
        v(2)*cos(v(0))*cos(v(1)),
        v(2)*sin(v(0))*cos(v(1)),
        -v(2)*sin(v(1))
    );
    return getAxisSwapQuaternion("FSD",getAxisOrder()).transformVector(fsd);
}
MotionState AffineCRS::getInternalMotionState() const{
    return internalMotionState;
}
void AffineCRS::updateByMotionState(const MotionState& motionState_){
    internalMotionState=motionState_.transformTo(getBaseCRS());
    invalidate();
}
Quaternion AffineCRS::getEquivalentQuaternionToTopocentricCRS(const Eigen::Vector3d& origin, const Time& time, const std::string& axisOrder_) const{
    //CRSとして生成せずにクォータニオンとして得る。
    std::string dstAxisOrder=axisOrder_;
    std::transform(dstAxisOrder.begin(),dstAxisOrder.end(),dstAxisOrder.begin(),
        [](unsigned char c){ return std::toupper(c);}
    );
    auto intermediateCRS=getIntermediateCRS();
    Coordinate cartesianOriginInIntermediateCRS = transformToIntermediateCRS(origin,time,CoordinateType::POSITION_ABS);
    auto qToIntermediateCRSFromSelf=getQuaternionFrom(cartesianOriginInIntermediateCRS).conjugate();// [intermediateCRS <- this] 
    if(checkTopocentricCartesianAxisOrder(dstAxisOrder)){
        return (
            intermediateCRS->getEquivalentQuaternionToTopocentricCRS(cartesianOriginInIntermediateCRS(),time,dstAxisOrder) * //[new Topo <- intermediateCRS]
            qToIntermediateCRSFromSelf // [intermediateCRS <- this]
        );
    }else if(checkBodyCartesianAxisOrder(dstAxisOrder)){
        // baseのFWD方向のベクトルを水平面に射影した方向を「前方」に相当する基準軸Fとした局所水平座標系
        // 「左方」に相当するP=U×Fの3軸を導出する必要がある。
        auto qToENUFromIntermediateCRS = intermediateCRS->getEquivalentQuaternionToTopocentricCRS(cartesianOriginInIntermediateCRS(),time,"ENU");//[ENU <- intermediateCRS]
        Eigen::Vector3d fwdInBody=getAxisSwapQuaternion("FPU",getAxisOrder()).transformVector(Eigen::Vector3d(1,0,0));
        Eigen::Vector3d fwdInIntermediateCRS=transformTo(fwdInBody,time,intermediateCRS,CoordinateType::DIRECTION)();
        Eigen::Vector3d fwdInENU=qToENUFromIntermediateCRS.transformVector(fwdInIntermediateCRS);
        double xy=sqrt(fwdInENU(0)*fwdInENU(0)+fwdInENU(1)*fwdInENU(1));
        double az=0.0;
        if(xy>0){
            az=atan2(fwdInENU(1),fwdInENU(0));
        }
        fwdInENU=Eigen::Vector3d(cos(az),sin(az),0);
        Eigen::Vector3d portInENU(-sin(az),cos(az),0);
        Eigen::Vector3d upInENU(0,0,1);
        auto qToENUFromFPU=Quaternion::fromBasis(fwdInENU,portInENU,upInENU); // [ENU <- FPU]
        return ( //[dstAxisOrder <- this] =
            getAxisSwapQuaternion("FPU",dstAxisOrder) * //[dstAxisOrder <- FPU] *
            qToENUFromFPU.conjugate() * //[FPU <- ENU] *
            qToENUFromIntermediateCRS * //[ENU <- intermediateCRS] *
            qToIntermediateCRSFromSelf //[intermediateCRS <- this]
        );
    }else{
        throw std::runtime_error("'"+axisOrder_+"' is not a valid axisOrder of TopocentricCRS.");
    }
}

void exportAffineCRS(py::module &m, const std::shared_ptr<asrc::core::FactoryHelper>& factoryHelper){
    using namespace pybind11::literals;

    expose_entity_subclass<AffineCRS>(m,"AffineCRS")
    DEF_FUNC(AffineCRS,getAxisOrder)
    //DEF_FUNC(AffineCRS,updateByMotionState)
    ;

    FACTORY_ADD_CLASS(CoordinateReferenceSystem,AffineCRS)
}

ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
