// Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#include "crs/PureFlatCRS.h"
#include "crs/DerivedCRS.h"
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

PureFlatCRS::PureFlatCRS(const nl::json& modelConfig_,const nl::json& instanceConfig_)
:CoordinateReferenceSystem(modelConfig_,instanceConfig_){
    crsType=CRSType::PUREFLAT;
}
void PureFlatCRS::initialize(){
    BaseType::initialize();
    // 座標軸の設定
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
    assert(checkTopocentricCartesianAxisOrder(axisOrder));

    //原点の設定
    givenByEllipsoidalOrigin=false;
    givenByCartesianOrigin=false;
    ellipsoidalOrigin=Eigen::Vector3d::Zero(); // in lat[deg], lon[deg], alt[m]
    cartesianOrigin=Eigen::Vector3d::Zero(); // in X[m], Y[m], Z[m]
    if(instanceConfig.contains("cartesianOrigin")){
        cartesianOrigin=instanceConfig.at("cartesianOrigin");
        givenByCartesianOrigin=true;
    }else if(modelConfig.contains("cartesianOrigin")){
        cartesianOrigin=modelConfig.at("cartesianOrigin");
        givenByCartesianOrigin=true;
    }else if(instanceConfig.contains("ellipsoidalOrigin")){
        ellipsoidalOrigin=instanceConfig.at("ellipsoidalOrigin");
        givenByEllipsoidalOrigin=true;
    }else if(modelConfig.contains("ellipsoidalOrigin")){
        ellipsoidalOrigin=modelConfig.at("ellipsoidalOrigin");
        givenByEllipsoidalOrigin=true;
    }
}
// 内部状態のシリアライゼーション
void PureFlatCRS::serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full){
    BaseType::serializeInternalState(archive,full);

    if(full){
        ASRC_SERIALIZE_NVP(archive
            ,axisOrder
            ,givenByEllipsoidalOrigin
            ,givenByCartesianOrigin
            ,ellipsoidalOrigin
            ,cartesianOrigin
        )
    };
}
//座標系の種類
bool PureFlatCRS::isEarthFixed() const{
    return true;
}
bool PureFlatCRS::isGeocentric() const{
    return true;
}
bool PureFlatCRS::isGeographic() const{
    return false;
}
//Coordinate system axisの種類
bool PureFlatCRS::isCartesian() const{
    return true;
}
bool PureFlatCRS::isSpherical() const{
    return false;
}
bool PureFlatCRS::isEllipsoidal() const{
    return false;
}
bool PureFlatCRS::isTransformableTo(const std::shared_ptr<CoordinateReferenceSystem>& other) const{
    if(!other){
        return false;
    }
    if(this==std::dynamic_pointer_cast<PureFlatCRS>(other).get()){
        return true;
    }
    if(auto derived = std::dynamic_pointer_cast<DerivedCRS>(other)){
        return this==std::dynamic_pointer_cast<PureFlatCRS>(derived->getNonDerivedBaseCRS()).get();
    }else{
        return false;
    }
}
bool PureFlatCRS::isTransformableFrom(const std::shared_ptr<CoordinateReferenceSystem>& other) const{
    if(!other){
        return false;
    }
    if(this==std::dynamic_pointer_cast<PureFlatCRS>(other).get()){
        return true;
    }
    if(auto derived = std::dynamic_pointer_cast<DerivedCRS>(other)){
        return this==std::dynamic_pointer_cast<PureFlatCRS>(derived->getNonDerivedBaseCRS()).get();
    }else{
        return false;
    }
}
Coordinate PureFlatCRS::transformTo(const Eigen::Vector3d& value, const Eigen::Vector3d& location, const Time& time, const std::shared_ptr<CoordinateReferenceSystem>& dstCRS, const CoordinateType& coordinateType) const{
    if(!isValid()){validate();}
    if(!dstCRS){
        throw std::runtime_error("dstCRS is nullptr!");
    }
    if(this==std::dynamic_pointer_cast<PureFlatCRS>(dstCRS).get()){
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
    if(auto derived=std::dynamic_pointer_cast<DerivedCRS>(dstCRS)){
        assert(this==std::dynamic_pointer_cast<PureFlatCRS>(derived->getNonDerivedBaseCRS()).get());
        return derived->transformFromNonDerivedBaseCRS(value,location,time,coordinateType);
    }else{
        throw std::runtime_error(getFactoryClassName()+" can't be transformed to "+dstCRS->getFactoryClassName());
    }
}
Coordinate PureFlatCRS::transformFrom(const Eigen::Vector3d& value, const Eigen::Vector3d& location, const Time& time, const std::shared_ptr<CoordinateReferenceSystem>& srcCRS, const CoordinateType& coordinateType) const{
    if(!isValid()){validate();}
    if(!srcCRS){
        throw std::runtime_error("srcCRS is nullptr!");
    }
    if(this==std::dynamic_pointer_cast<PureFlatCRS>(srcCRS).get()){
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
    if(auto derived=std::dynamic_pointer_cast<DerivedCRS>(srcCRS)){
        assert(this==std::dynamic_pointer_cast<PureFlatCRS>(derived->getNonDerivedBaseCRS()).get());
        return derived->transformToNonDerivedBaseCRS(value,location,time,coordinateType);
    }else{
        throw std::runtime_error(getFactoryClassName()+" can't be transformed from "+srcCRS->getFactoryClassName());
    }
}
Quaternion PureFlatCRS::getQuaternionTo(const Eigen::Vector3d& location, const Time& time, const std::shared_ptr<CoordinateReferenceSystem>& dstCRS) const{
    if(!isValid()){validate();}
    if(!dstCRS){
        throw std::runtime_error("dstCRS is nullptr!");
    }
    if(this==std::dynamic_pointer_cast<PureFlatCRS>(dstCRS).get()){
        return Quaternion(1,0,0,0);
    }
    if(!isTransformableTo(dstCRS)){
        throw std::runtime_error(getFactoryClassName()+" can't be transformed to "+dstCRS->getFactoryClassName());
    }
    if(auto derived=std::dynamic_pointer_cast<DerivedCRS>(dstCRS)){
        if(this!=std::dynamic_pointer_cast<PureFlatCRS>(derived->getNonDerivedBaseCRS()).get()){
            throw std::runtime_error("PureFlatCRS can calculate quaternion only to/from AffineCRS or TopocentricCRS whose non-derived base is it.");
        }
        if(derived->isProjected()){
            throw std::runtime_error("Quaternion between a projected and a non-projected CRS cannot be calculated.");
        }
        return derived->getQuaternionFrom(location,time,non_const_this());
    }else{
        throw std::runtime_error("PureFlatCRS can calculate quaternion only to/from AffineCRS or TopocentricCRS whose non-derived base is it.");
    }
}
Quaternion PureFlatCRS::getQuaternionFrom(const Eigen::Vector3d& location, const Time& time, const std::shared_ptr<CoordinateReferenceSystem>& srcCRS) const{
    if(!isValid()){validate();}
    if(!srcCRS){
        throw std::runtime_error("srcCRS is nullptr!");
    }
    if(this==std::dynamic_pointer_cast<PureFlatCRS>(srcCRS).get()){
        return Quaternion(1,0,0,0);
    }
    if(!isTransformableFrom(srcCRS)){
        throw std::runtime_error(getFactoryClassName()+" can't be transformed from "+srcCRS->getFactoryClassName());
    }
    if(auto derived=std::dynamic_pointer_cast<DerivedCRS>(srcCRS)){
        if(this!=std::dynamic_pointer_cast<PureFlatCRS>(derived->getNonDerivedBaseCRS()).get()){
            throw std::runtime_error("PureFlatCRS can calculate quaternion only to/from AffineCRS or TopocentricCRS whose non-derived base is it.");
        }
        if(derived->isProjected()){
            throw std::runtime_error("Quaternion between a projected and a non-projected CRS cannot be calculated.");
        }
        return derived->getQuaternionTo(location,time,non_const_this());
    }else{
        throw std::runtime_error("PureFlatCRS can calculate quaternion only to/from AffineCRS or TopocentricCRS whose non-derived base is it.");
    }
}
double PureFlatCRS::getHeight(const Eigen::Vector3d& location, const Time& time) const{
    //楕円体高度(幾何高度)を返す。ellipsoid (geometric) height
    if(!isValid()){validate();}
    for(int i=0; i<3; ++i){
        if(axisOrder.at(i)=='U'){
            return location(i);
        }else if(axisOrder.at(i)=='D'){
            return -location(i);
        }
    }
    //コンストラクタでチェックしているので派生クラス等で後から書き換えていなければここには到達しない。
    throw std::runtime_error("'"+axisOrder+"' is an invalid axisOrder.");
}
double PureFlatCRS::getGeoidHeight(const Eigen::Vector3d& location, const Time& time) const{
    //標高(ジオイド高度)を返す。geoid height (elevation)
    //PureFlatCRSは楕円体面とジオイド面が一致するものとする。
    return getHeight(location,time);
}
std::string PureFlatCRS::getAxisOrder() const{
    if(!isValid()){validate();}
    return axisOrder;
}
Eigen::Vector3d PureFlatCRS::getEllipsoidalOrigin() const{
    if(!isValid()){validate();}
    return ellipsoidalOrigin;
}
Eigen::Vector3d PureFlatCRS::getCartesianOrigin() const{
    if(!isValid()){validate();}
    return cartesianOrigin;
}
Quaternion PureFlatCRS::getEquivalentQuaternionToTopocentricCRS(const Eigen::Vector3d& origin, const Time& time, const std::string& axisOrder_) const{
    //CRSとして生成せずにクォータニオンとして得る。
    std::string dstAxisOrder=axisOrder_;
    std::transform(dstAxisOrder.begin(),dstAxisOrder.end(),dstAxisOrder.begin(),
        [](unsigned char c){ return std::toupper(c);}
    );
    if(checkTopocentricCartesianAxisOrder(dstAxisOrder)){
        if(dstAxisOrder==getAxisOrder()){
            return Quaternion(1,0,0,0);
        }else{
            return getAxisSwapQuaternion(getAxisOrder(),dstAxisOrder); // [dstAxisOrder <- this]
        }
    }else{// if(checkBodyCartesianAxisOrder(dstAxisOrder){
        throw std::runtime_error("Body-oriented TopocentricCRS cannot be created from PureFlatCRS.");
    }
}

void exportPureFlatCRS(py::module &m, const std::shared_ptr<asrc::core::FactoryHelper>& factoryHelper){
    using namespace pybind11::literals;

    expose_entity_subclass<PureFlatCRS>(m,"PureFlatCRS")
    DEF_FUNC(PureFlatCRS,getAxisOrder)
    DEF_FUNC(PureFlatCRS,getEllipsoidalOrigin)
    DEF_FUNC(PureFlatCRS,getCartesianOrigin)
    ;

    FACTORY_ADD_CLASS(CoordinateReferenceSystem,PureFlatCRS)
}

ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
