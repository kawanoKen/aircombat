// Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#include "crs/CoordinateReferenceSystemBase.h"
#include "crs/AffineCRS.h"
#include "crs/TopocentricCRS.h"
#include "crs/ProjectedCRS.h"
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

//
// Axis order utility
//
bool checkSphericalAxisOrder(const std::string& order){
    // Azimuth, Elevation, Distance(Range)の3種類の軸を含んでいるかどうかを判定する。
    if(order.size()!=3){
        return false;
    }
    bool found[3] = {false,false,false}; // Azimuth, Elevation, Distance(Range)
    for(int i=0; i<3; ++i){
        if(order.at(i)=='A'){ // Azimuth
            found[0]=true;
        }else if(order.at(i)=='E'){ // Elevation
            found[1]=true;
        }else if(order.at(i)=='D'){ // Distance
            found[2]=true;
        }else if(order.at(i)=='R'){ // Range
            found[2]=true;
        }
    }
    return found[0] && found[1] && found[2];
}
bool checkBodyCartesianAxisOrder(const std::string& order){
    // FWD/AFT, STBD/PORT, UP/DOWNの3種類の軸を含んでいるかどうかを判定する。
    // 左右を表すSTBD/PORTはRIGHT(R)/LEFT(L)も許容する。
    if(order.size()!=3){
        return false;
    }
    bool found[3] = {false,false,false}; // FWD/AFT, STBD/PORT, UP/DOWN
    for(int i=0; i<3; ++i){
        if(order.at(i)=='F'){ // FWD
            found[0]=true;
        }else if(order.at(i)=='A'){ // AFT
            found[0]=true;
        }else if(order.at(i)=='S'){ // STBD(RIGHT)
            found[1]=true;
        }else if(order.at(i)=='P'){ // PORT(LEFT)
            found[1]=true;
        }else if(order.at(i)=='R'){ // STBD(RIGHT)
            found[1]=true;
        }else if(order.at(i)=='L'){ // PORT(LEFT)
            found[1]=true;
        }else if(order.at(i)=='U'){ // UP
            found[2]=true;
        }else if(order.at(i)=='D'){ // DOWN
            found[2]=true;
        }
    }
    return found[0] && found[1] && found[2];
}
bool checkTopocentricCartesianAxisOrder(const std::string& order){
    // North/South, East/West, Up/Downの3種類の軸を含んでいるかどうかを判定する。
    if(order.size()!=3){
        return false;
    }
    bool found[3] = {false,false,false}; // North/South, East/West, Up/Down
    for(int i=0; i<3; ++i){
        if(order.at(i)=='N'){ // North
            found[0]=true;
        }else if(order.at(i)=='S'){ // South
            found[0]=true;
        }else if(order.at(i)=='E'){ // East
            found[1]=true;
        }else if(order.at(i)=='W'){ // West
            found[1]=true;
        }else if(order.at(i)=='U'){ // UP
            found[2]=true;
        }else if(order.at(i)=='D'){ // DOWN
            found[2]=true;
        }
    }
    return found[0] && found[1] && found[2];
}
Eigen::Matrix3d getAxisSwapRotationMatrix(const std::string& srcOrder_, const std::string& dstOrder_){
    static const std::map<char,char> bodyCartesianReverseAxis={
        {'F','A'},
        {'A','F'},
        {'S','P'},
        {'P','S'},
        {'U','D'},
        {'D','U'}
    };
    static const std::map<char,char> topocentricReverseAxis={
        {'N','S'},
        {'S','N'},
        {'E','W'},
        {'W','E'},
        {'U','D'},
        {'D','U'}
    };
    std::string srcOrder=srcOrder_;
    std::string dstOrder=dstOrder_;
    if(checkBodyCartesianAxisOrder(srcOrder)){
        if(!checkBodyCartesianAxisOrder(dstOrder)){
            throw std::runtime_error("'"+srcOrder+"' and '"+dstOrder+"' are not same type of axisOrder.");
        }
        //RをSに、LをPに変換する。
        for(int i=0; i<3; ++i){
            if(srcOrder.at(i)=='R'){
                srcOrder.at(i)='S';
            }else if(srcOrder.at(i)=='L'){
                srcOrder.at(i)='P';
            }
            if(dstOrder.at(i)=='R'){
                dstOrder.at(i)='S';
            }else if(dstOrder.at(i)=='L'){
                dstOrder.at(i)='P';
            }
        }
        Eigen::Matrix3d ret=Eigen::Matrix3d::Zero();
        for(int i=0; i<3; ++i){
            for(int j=0; j<3; ++j){
                if(dstOrder.at(i)==srcOrder.at(j)){
                    ret(i,j)=1;
                }else if(dstOrder.at(i)==bodyCartesianReverseAxis.at(srcOrder.at(j))){
                    ret(i,j)=-1;
                }
            }
        }
        return ret;
    }else if(checkTopocentricCartesianAxisOrder(srcOrder)){
        if(!checkTopocentricCartesianAxisOrder(srcOrder)){
            throw std::runtime_error("'"+srcOrder+"' and '"+dstOrder+"' are not same type of axisOrder.");
        }
        Eigen::Matrix3d ret=Eigen::Matrix3d::Zero();
        for(int i=0; i<3; ++i){
            for(int j=0; j<3; ++j){
                if(dstOrder.at(i)==srcOrder.at(j)){
                    ret(i,j)=1;
                }else if(dstOrder.at(i)==topocentricReverseAxis.at(srcOrder.at(j))){
                    ret(i,j)=-1;
                }
            }
        }
        return ret;
    }else{
        throw std::runtime_error("Given axisOrder '"+srcOrder+"' is not valid as bodyCartesian or topocentric axisOrder.");
    }
}
Quaternion getAxisSwapQuaternion(const std::string& srcOrder, const std::string& dstOrder){
    return Quaternion::fromRotationMatrix(getAxisSwapRotationMatrix(std::move(srcOrder),std::move(dstOrder)));
}

const std::string CoordinateReferenceSystem::baseName="CoordinateReferenceSystem"; // baseNameを上書き

CoordinateReferenceSystem::CoordinateReferenceSystem(const nl::json& modelConfig_,const nl::json& instanceConfig_)
:Entity(modelConfig_,instanceConfig_){
    crsType=CRSType::INVALID;
    _isValid=false;
    _hasBeenValidated=false;
    _updateCount=0;
}
CRSType CoordinateReferenceSystem::getCRSType() const{
    return crsType;
}
// 内部状態のシリアライゼーション
void CoordinateReferenceSystem::serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full){
    BaseType::serializeInternalState(archive,full);

    ASRC_SERIALIZE_NVP(archive
        ,_isValid
        ,_hasBeenValidated
        ,_updateCount
    )
    if(asrc::core::util::isInputArchive(archive)){
        std::lock_guard<std::recursive_mutex> lock(mtx);
        derivedCRSes.clear(); // derivedCRSesはderived側でaddDerivedCRSを呼び出すことで復元する
    }
}
bool CoordinateReferenceSystem::isValid() const{
    return _isValid;
}
bool CoordinateReferenceSystem::hasBeenValidated() const{
    return _hasBeenValidated;
}
void CoordinateReferenceSystem::validate() {
    BaseType::validate();
    validateImpl(!hasBeenValidated());
}
void CoordinateReferenceSystem::validate() const{
    const_cast<CoordinateReferenceSystem*>(this)->validate();
}
std::uint64_t CoordinateReferenceSystem::getUpdateCount() const{
    return _updateCount;
}
bool CoordinateReferenceSystem::isSameCRS(const std::shared_ptr<CoordinateReferenceSystem>& other) const{
    //引数が自身と同一かどうかを返す。shared_from_thisがEntityへのポインタとなってしまうためこの関数を用意する。
    return this==other.get();
}
bool CoordinateReferenceSystem::isEquivalentCRS(const std::shared_ptr<CoordinateReferenceSystem>& other) const{
    //引数が自身と等価な(座標変換が恒等式となる)CRSかどうかを返す。派生クラスでカスタマイズ可能。
    return this==other.get();
}
void CoordinateReferenceSystem::validateImpl(bool isFirstTime) const{
    _isValid=true;
    _hasBeenValidated=true;
}
void CoordinateReferenceSystem::invalidate() const{
    _isValid=false;
    _hasBeenValidated=false;
    _updateCount++;
    std::lock_guard<std::recursive_mutex> lock(mtx);
    for(auto&& it = derivedCRSes.begin(); it != derivedCRSes.end();){
        auto&& wDerived = *it;
        if(wDerived.expired()){
            it=derivedCRSes.erase(it);
        }else{
            invalidate(wDerived.lock());
            ++it;
        }
    }
}
void CoordinateReferenceSystem::addDerivedCRS(const std::weak_ptr<CoordinateReferenceSystem>& derived) const{
    std::lock_guard<std::recursive_mutex> lock(mtx);
    if(!derived.expired()){
        derivedCRSes.push_back(derived);
        if(!isValid()){
            invalidate(derived.lock());
        }
    }
}
void CoordinateReferenceSystem::removeDerivedCRS(const std::weak_ptr<CoordinateReferenceSystem>& derived) const{
    std::lock_guard<std::recursive_mutex> lock(mtx);
    if(!derived.expired()){
        auto id=derived.lock()->getEntityID();
        for(auto&& it = derivedCRSes.begin(); it != derivedCRSes.end();){
            auto&& wDerived = *it;
            if(wDerived.expired()){
                it=derivedCRSes.erase(it);
            }else{
                if(wDerived.lock()->getEntityID()==id){
                    it=derivedCRSes.erase(it);
                }else{
                    ++it;
                }
            }
        }
    }
}
//座標系の種類
bool CoordinateReferenceSystem::isEarthFixed() const{
    return false;
}
bool CoordinateReferenceSystem::isGeocentric() const{
    return false;
}
bool CoordinateReferenceSystem::isGeographic() const{
    return false;
}
bool CoordinateReferenceSystem::isDerived() const{
    return false;
}
bool CoordinateReferenceSystem::isTopocentric() const{
    return false;
}
bool CoordinateReferenceSystem::isProjected() const{
    return false;
}
//親座標系の取得
std::shared_ptr<CoordinateReferenceSystem> CoordinateReferenceSystem::getBaseCRS() const{
    return non_const_this();
}
std::shared_ptr<CoordinateReferenceSystem> CoordinateReferenceSystem::getNonDerivedBaseCRS() const{
    return non_const_this();
}
std::shared_ptr<ProjectedCRS> CoordinateReferenceSystem::getBaseProjectedCRS() const{
    return nullptr;
}
//中間座標系の取得
std::shared_ptr<CoordinateReferenceSystem> CoordinateReferenceSystem::getIntermediateCRS() const{
    return non_const_this();
}
// 座標変換
Coordinate CoordinateReferenceSystem::transformTo(const Coordinate& value, const Coordinate& location, const std::shared_ptr<CoordinateReferenceSystem>& dstCRS) const{
    if(location.getType()!=CoordinateType::POSITION_ABS){
        throw std::runtime_error("Given location is not a 'absolute location' coordinate.");
    }
    return transformTo(
        value(non_const_this()),
        location(non_const_this()),
        value.getTime(),
        dstCRS,
        value.getType()
    );
}
Coordinate CoordinateReferenceSystem::transformFrom(const Coordinate& value, const Coordinate& location) const{
    if(location.getType()!=CoordinateType::POSITION_ABS){
        throw std::runtime_error("Given location is not a 'absolute location' coordinate.");
    }
    return transformFrom(value(),location(value.getCRS()),value.getTime(),value.getCRS(),value.getType());
}
Coordinate CoordinateReferenceSystem::transformTo(const Coordinate& value, const std::shared_ptr<CoordinateReferenceSystem>& dstCRS) const{
    Coordinate valueInSelf=value.transformTo(non_const_this());
    return transformTo(
        valueInSelf(),
        valueInSelf.getLocation(),
        valueInSelf.getTime(),
        dstCRS,
        valueInSelf.getType()
    );
}
Coordinate CoordinateReferenceSystem::transformFrom(const Coordinate& value) const{
    return transformFrom(value(),value.getLocation(),value.getTime(),value.getCRS(),value.getType());
}
Coordinate CoordinateReferenceSystem::transformTo(const Eigen::Vector3d& value, const Time& time, const std::shared_ptr<CoordinateReferenceSystem>& dstCRS, const CoordinateType& coordinateType) const{
    return transformTo(value,Eigen::Vector3d::Zero(),time,dstCRS,coordinateType);
}
Coordinate CoordinateReferenceSystem::transformFrom(const Eigen::Vector3d& value, const Time& time, const std::shared_ptr<CoordinateReferenceSystem>& srcCRS, const CoordinateType& coordinateType) const{
    return transformFrom(value,Eigen::Vector3d::Zero(),time,srcCRS,coordinateType);
}
void CoordinateReferenceSystem::validateImpl(const std::shared_ptr<const CoordinateReferenceSystem>& base,bool isFirstTime){
    base->validateImpl(isFirstTime);
}
void CoordinateReferenceSystem::invalidate(const std::shared_ptr<const CoordinateReferenceSystem>& base){
    base->invalidate();
}
void CoordinateReferenceSystem::addDerivedCRS(const std::shared_ptr<const CoordinateReferenceSystem>& base,const std::weak_ptr<CoordinateReferenceSystem>& derived){
    base->addDerivedCRS(derived);
}
void CoordinateReferenceSystem::removeDerivedCRS(const std::shared_ptr<const CoordinateReferenceSystem>& base,const std::weak_ptr<CoordinateReferenceSystem>& derived){
    base->removeDerivedCRS(derived);
}
Quaternion CoordinateReferenceSystem::getQuaternionTo(const Coordinate& location, const std::shared_ptr<CoordinateReferenceSystem>& dstCRS) const{
    // 基準点locationにおいてこのCRSの座標をdstCRSの座標に変換するクォータニオンを得る。
    return getQuaternionTo(location(non_const_this()),location.getTime(),dstCRS);
}
Quaternion CoordinateReferenceSystem::getQuaternionFrom(const Coordinate& location) const{
    // 基準点locationにおいてlocationが属する座標系の座標をこのCRSの座標に変換するクォータニオンを得る。
    return getQuaternionFrom(location(),location.getTime(),location.getCRS());
}
Quaternion CoordinateReferenceSystem::transformQuaternionTo(const Quaternion& q, const Coordinate& location, const std::shared_ptr<CoordinateReferenceSystem>& dstCRS) const{
    // 基準点locationにおいてこのCRSからの変換を表すクォータニオンをdstCRSからの変換を表すクォータニオンに変換する。
    return transformQuaternionTo(q,location(non_const_this()),location.getTime(),dstCRS);
}
Quaternion CoordinateReferenceSystem::transformQuaternionFrom(const Quaternion& q, const Coordinate& location) const{
    // 基準点locationにおいてlocationが属する座標系からの変換を表すクォータニオンをこのCRSからの変換を表すクォータニオンに変換する。
    return transformQuaternionFrom(q,location(),location.getTime(),location.getCRS());
}
Quaternion CoordinateReferenceSystem::transformQuaternionTo(const Quaternion& q, const Eigen::Vector3d& location, const Time& time, const std::shared_ptr<CoordinateReferenceSystem>& dstCRS) const{
    // 基準点locationにおいてこのCRSからの変換を表すクォータニオンをdstCRSからの変換を表すクォータニオンに変換する。
    // q [another <- self]
    // return [another <- dstCRS] = [another <- self] * [self <- dstCRS]
    return q * getQuaternionTo(location,time,dstCRS).conjugate();
}
Quaternion CoordinateReferenceSystem::transformQuaternionFrom(const Quaternion& q, const Eigen::Vector3d& location, const Time& time, const std::shared_ptr<CoordinateReferenceSystem>& srcCRS) const{
    // 基準点locationにおいてsrcCRSからの変換を表すクォータニオンをこのCRSからの変換を表すクォータニオンに変換する。
    // q [another <- srcCRS]
    // return [another <- self] = [another <- srcCRS] * [srcCRS <- self]
    return q * getQuaternionFrom(location,time,srcCRS).conjugate();
}
double CoordinateReferenceSystem::getHeight(const Coordinate& location) const{
    if(location.getType()!=CoordinateType::POSITION_ABS){
        throw std::runtime_error("Given value is not a 'absolute location' coordinate.");
    }
    return getHeight(location(non_const_this()),location.getTime());
}
double CoordinateReferenceSystem::getGeoidHeight(const Coordinate& location) const{
    if(location.getType()!=CoordinateType::POSITION_ABS){
        throw std::runtime_error("Given value is not a 'absolute location' coordinate.");
    }
    return getGeoidHeight(location(non_const_this()),location.getTime());
}

Eigen::Vector3d CoordinateReferenceSystem::toCartesian(const Eigen::Vector3d& v) const{
    if(isCartesian()){
        return v;
    }else if(isSpherical()){
        return sphericalToCartesian(v);
    }else{
        return ellipsoidalToCartesian(v);
    }
}
Eigen::Vector3d CoordinateReferenceSystem::toSpherical(const Eigen::Vector3d& v) const{
    if(isCartesian()){
        return cartesianToSpherical(v);
    }else if(isSpherical()){
        return v;
    }else{
        return ellipsoidalToSpherical(v);
    } 
}
Eigen::Vector3d CoordinateReferenceSystem::toEllipsoidal(const Eigen::Vector3d& v) const{
    if(isCartesian()){
        return cartesianToEllipsoidal(v);
    }else if(isSpherical()){
        return sphericalToEllipsoidal(v);
    }else{
        return v;
    } 
}
Eigen::Vector3d CoordinateReferenceSystem::fromCartesian(const Eigen::Vector3d& v) const{
    if(isCartesian()){
        return v;
    }else if(isSpherical()){
        return cartesianToSpherical(v);
    }else{
        return cartesianToEllipsoidal(v);
    } 
}
Eigen::Vector3d CoordinateReferenceSystem::fromSpherical(const Eigen::Vector3d& v) const{
    if(isCartesian()){
        return sphericalToCartesian(v);
    }else if(isSpherical()){
        return v;
    }else{
        return sphericalToEllipsoidal(v);
    } 
}
Eigen::Vector3d CoordinateReferenceSystem::fromEllipsoidal(const Eigen::Vector3d& v) const{
    if(isCartesian()){
        return ellipsoidalToCartesian(v);
    }else if(isSpherical()){
        return ellipsoidalToSpherical(v);
    }else{
        return v;
    } 
}
Eigen::Vector3d CoordinateReferenceSystem::cartesianToSpherical(const Eigen::Vector3d& v) const{
    throw std::runtime_error("cartesian to spherical conversion is not defined for "+getDemangledName());
}
Eigen::Vector3d CoordinateReferenceSystem::cartesianToEllipsoidal(const Eigen::Vector3d& v) const{
    throw std::runtime_error("cartesian to ellipsoidal conversion is not defined for "+getDemangledName());
}
Eigen::Vector3d CoordinateReferenceSystem::sphericalToCartesian(const Eigen::Vector3d& v) const{
    throw std::runtime_error("spherical to cartesian conversion is not defined for "+getDemangledName());
}
Eigen::Vector3d CoordinateReferenceSystem::sphericalToEllipsoidal(const Eigen::Vector3d& v) const{
    throw std::runtime_error("spherical to ellipsoidal conversion is not defined for "+getDemangledName());
}
Eigen::Vector3d CoordinateReferenceSystem::ellipsoidalToCartesian(const Eigen::Vector3d& v) const{
    throw std::runtime_error("ellipsoidal to cartesian conversion is not defined for "+getDemangledName());
}
Eigen::Vector3d CoordinateReferenceSystem::ellipsoidalToSpherical(const Eigen::Vector3d& v) const{
    throw std::runtime_error("ellipsoidal to spherical conversion is not defined for "+getDemangledName());
}

// このCRS上の点originを原点とするtopocentricなCRSを生成して返す。
std::shared_ptr<TopocentricCRS> CoordinateReferenceSystem::createTopocentricCRS(const Coordinate& origin, const std::string& axisOrder, bool onSurface, bool isEpisodic) const{
    return createTopocentricCRS(origin(non_const_this()),origin.getTime(),axisOrder,onSurface,isEpisodic);
}
std::shared_ptr<TopocentricCRS> CoordinateReferenceSystem::createTopocentricCRS(const Eigen::Vector3d& origin, const Time& time, const std::string& axisOrder, bool onSurface, bool isEpisodic) const{
    nl::json mc=nl::json::object();
    nl::json ic={
        {"base",{
            {"entityIdentifier",getEntityID()},
        }},
        {"cartesianOrigin",origin},
        {"time",time},
        {"axisOrder",axisOrder},
        {"onSurface",onSurface}
    };
    return createEntityByClassName<TopocentricCRS>(isEpisodic,"CoordinateReferenceSystem","TopocentricCRS",mc,ic);
}
Quaternion CoordinateReferenceSystem::getEquivalentQuaternionToTopocentricCRS(const Coordinate& origin, const std::string& axisOrder) const{
    //CRSとして生成せずにクォータニオンとして得る。
    if(origin.getType()!=CoordinateType::POSITION_ABS){
        throw std::runtime_error("Given value is not a 'absolute location' coordinate.");
    }
    if(!origin.getCRS()){
        throw std::runtime_error("origin.crs is nullptr!");
    }
    return getEquivalentQuaternionToTopocentricCRS(origin(non_const_this()),origin.getTime(),axisOrder);
}
Eigen::Vector3d CoordinateReferenceSystem::transformToTopocentricCRS(const Coordinate& value, const Coordinate& location, const Coordinate& origin, const std::string& axisOrder, bool onSurface) const{
    return transformToTopocentricCRS(
        value(non_const_this()),
        location(non_const_this()),
        value.getTime(),
        value.getType(),
        origin(non_const_this()),
        axisOrder,
        onSurface
    );
}
Eigen::Vector3d CoordinateReferenceSystem::transformToTopocentricCRS(const Coordinate& value, const Coordinate& origin, const std::string& axisOrder, bool onSurface) const{
    Coordinate valueInSelf=value.transformTo(non_const_this());
    return transformToTopocentricCRS(
        valueInSelf(),
        valueInSelf.getLocation(),
        valueInSelf.getTime(),
        valueInSelf.getType(),
        origin(non_const_this()),
        axisOrder,
        onSurface
    );
}
Eigen::Vector3d CoordinateReferenceSystem::transformToTopocentricCRS(const Eigen::Vector3d& value, const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType, const Eigen::Vector3d& origin, const std::string& axisOrder, bool onSurface) const{
    auto intermediateCRS = getNonDerivedBaseCRS()->getIntermediateCRS();
    if(non_const_this() == intermediateCRS){
        // 高度オフセットの計算(ENU経由)
        auto qToSelfFromENU=getEquivalentQuaternionToTopocentricCRS(origin,time,"ENU").conjugate();
        Eigen::Vector3d actualOriginInSelf;
        if(onSurface){
            auto altOffset=qToSelfFromENU.transformVector(
                Eigen::Vector3d(0,0,getHeight(origin,time))
            );// [Self <- ENU]
            actualOriginInSelf=origin-fromCartesian(altOffset);
        }else{
            actualOriginInSelf=origin;
        }
        // クォータニオンの計算
        Quaternion qToSelfFromTopo=getEquivalentQuaternionToTopocentricCRS(origin,time,axisOrder).conjugate();

        // 変換の実行(MotionStateを利用)
        MotionState topoMotionState(
            non_const_this(),
            time,
            actualOriginInSelf,
            Eigen::Vector3d::Zero(),
            Eigen::Vector3d::Zero(),
            qToSelfFromTopo,
            "FSD"
        );
        if(coordinateType==CoordinateType::POSITION_ABS){
            return topoMotionState.absPtoB(value);
        }else if(coordinateType==CoordinateType::POSITION_REL){
            return topoMotionState.relPtoB(value);
        }else if(coordinateType==CoordinateType::DIRECTION){
            return topoMotionState.dirPtoB(value);
        }else if(coordinateType==CoordinateType::VELOCITY){
            return topoMotionState.velPtoB(value,location);
        }else if(coordinateType==CoordinateType::ANGULAR_VELOCITY){
            return topoMotionState.omegaPtoB(value);
        }else{
            throw std::runtime_error("Invalide coordinate type.");
        }
    }else{
        auto valueInIntermediate = transformTo(value,location,time,intermediateCRS,coordinateType);
        auto originInIntermediate = transformTo(origin,origin,time,intermediateCRS,CoordinateType::POSITION_ABS);
        return intermediateCRS->transformToTopocentricCRS(
            valueInIntermediate(),
            valueInIntermediate.getLocation(),
            time,
            coordinateType,
            originInIntermediate(),
            axisOrder,
            onSurface
        );
    }

}
Eigen::Vector3d CoordinateReferenceSystem::transformToTopocentricCRS(const Eigen::Vector3d& value, const Time& time, const CoordinateType& coordinateType, const Eigen::Vector3d& origin, const std::string& axisOrder, bool onSurface) const{
    return transformToTopocentricCRS(
        value,
        Eigen::Vector3d::Zero(),
        time,
        coordinateType,
        origin,
        axisOrder,
        onSurface
    );
}
Eigen::Vector3d CoordinateReferenceSystem::transformFromTopocentricCRS(const Eigen::Vector3d& value, const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType, const Eigen::Vector3d& origin, const std::string& axisOrder, bool onSurface) const{
    auto intermediateCRS = getNonDerivedBaseCRS()->getIntermediateCRS();
    if(non_const_this() == intermediateCRS){
        // 高度オフセットの計算(ENU経由)
        auto qToSelfFromENU=getEquivalentQuaternionToTopocentricCRS(origin,time,"ENU").conjugate();
        Eigen::Vector3d actualOriginInSelf;
        if(onSurface){
            auto altOffset=qToSelfFromENU.transformVector(
                Eigen::Vector3d(0,0,getHeight(origin,time))
            );// [Self <- ENU]
            actualOriginInSelf=origin-fromCartesian(altOffset);
        }else{
            actualOriginInSelf=origin;
        }
        // クォータニオンの計算
        Quaternion qToSelfFromTopo=getEquivalentQuaternionToTopocentricCRS(origin,time,axisOrder).conjugate();

        // 変換の実行(MotionStateを利用)
        MotionState topoMotionState(
            non_const_this(),
            time,
            actualOriginInSelf,
            Eigen::Vector3d::Zero(),
            Eigen::Vector3d::Zero(),
            qToSelfFromTopo,
            "FSD"
        );
        if(coordinateType==CoordinateType::POSITION_ABS){
            return topoMotionState.absBtoP(value);
        }else if(coordinateType==CoordinateType::POSITION_REL){
            return topoMotionState.relBtoP(value);
        }else if(coordinateType==CoordinateType::DIRECTION){
            return topoMotionState.dirBtoP(value);
        }else if(coordinateType==CoordinateType::VELOCITY){
            return topoMotionState.velBtoP(value,location);
        }else if(coordinateType==CoordinateType::ANGULAR_VELOCITY){
            return topoMotionState.omegaBtoP(value);
        }else{
            throw std::runtime_error("Invalide coordinate type.");
        }
    }else{
        auto originInIntermediate = transformTo(origin,origin,time,intermediateCRS,CoordinateType::POSITION_ABS);
        auto valueInIntermediate = intermediateCRS->transformFromTopocentricCRS(
            value,
            location,
            time,
            coordinateType,
            originInIntermediate(),
            axisOrder,
            onSurface
        );
        auto locationInIntermediate = intermediateCRS->transformFromTopocentricCRS(
            location,
            location,
            time,
            CoordinateType::POSITION_ABS,
            originInIntermediate(),
            axisOrder,
            onSurface
        );
        return transformFrom(valueInIntermediate,locationInIntermediate,time,intermediateCRS,coordinateType)();
    }
}
Eigen::Vector3d CoordinateReferenceSystem::transformFromTopocentricCRS(const Eigen::Vector3d& value, const Time& time, const CoordinateType& coordinateType, const Eigen::Vector3d& origin, const std::string& axisOrder, bool onSurface) const{
    return transformFromTopocentricCRS(
        value,
        Eigen::Vector3d::Zero(),
        time,
        coordinateType,
        origin,
        axisOrder,
        onSurface
    );
}

// このCRSをparentとして持つMotionStateに対応するAffineCRSを生成して返す。
std::shared_ptr<AffineCRS> CoordinateReferenceSystem::createAffineCRS(const MotionState& motion, const std::string& axisOrder, bool isEpisodic) const{
    nl::json mc=nl::json::object();
    nl::json ic={
        {"base",{
            {"entityIdentifier",getEntityID()},
        }},
        {"motion",motion},
        {"axisOrder",axisOrder}
    };
    return createEntityByClassName<AffineCRS>(isEpisodic,"CoordinateReferenceSystem","AffineCRS",mc,ic);
}

void exportCoordinateReferenceSystemBase(py::module &m, const std::shared_ptr<asrc::core::FactoryHelper>& factoryHelper){
    using namespace pybind11::literals;

    expose_enum_value_helper(
        expose_enum_class<CRSType>(m,"CRSType")
        ,"ECEF"
        ,"GEOGRAPHIC"
        ,"ECI"
        ,"PUREFLAT"
        ,"AFFINE"
        ,"PROJECTED"
        ,"TOPOCENTRIC"
        ,"INVALID"
    );

    FACTORY_ADD_BASE(CoordinateReferenceSystem,{"AnyOthers"},{"AnyOthers"})

    m.def("checkSphericalAxisOrder",&checkSphericalAxisOrder);
    m.def("checkBodyCartesianAxisOrder",&checkBodyCartesianAxisOrder);
    m.def("checkTopocentricCartesianAxisOrder",&checkTopocentricCartesianAxisOrder);
    m.def("getAxisSwapRotationMatrix",&getAxisSwapRotationMatrix);
    m.def("getAxisSwapQuaternion",&getAxisSwapQuaternion);

    expose_entity_subclass<CoordinateReferenceSystem>(m,"CoordinateReferenceSystem")
    DEF_FUNC(CoordinateReferenceSystem,isValid)
    DEF_FUNC(CoordinateReferenceSystem,hasBeenValidated)
    DEF_FUNC(CoordinateReferenceSystem,getUpdateCount)
    DEF_FUNC(CoordinateReferenceSystem,isSameCRS)
    DEF_FUNC(CoordinateReferenceSystem,isEquivalentCRS)
    //DEF_FUNC(CoordinateReferenceSystem,validateImpl)
    //DEF_FUNC(CoordinateReferenceSystem,invalidate)
    //DEF_FUNC(CoordinateReferenceSystem,addDerivedCRS)
    //DEF_FUNC(CoordinateReferenceSystem,removeDerivedCRS)
    DEF_FUNC(CoordinateReferenceSystem,isEarthFixed)
    DEF_FUNC(CoordinateReferenceSystem,isGeocentric)
    DEF_FUNC(CoordinateReferenceSystem,isGeographic)
    DEF_FUNC(CoordinateReferenceSystem,isDerived)
    DEF_FUNC(CoordinateReferenceSystem,isProjected)
    DEF_FUNC(CoordinateReferenceSystem,getBaseCRS)
    DEF_FUNC(CoordinateReferenceSystem,getNonDerivedBaseCRS)
    DEF_FUNC(CoordinateReferenceSystem,getBaseProjectedCRS)
    DEF_FUNC(CoordinateReferenceSystem,isCartesian)
    DEF_FUNC(CoordinateReferenceSystem,isSpherical)
    DEF_FUNC(CoordinateReferenceSystem,isEllipsoidal)
    DEF_FUNC(CoordinateReferenceSystem,isTransformableTo)
    DEF_FUNC(CoordinateReferenceSystem,isTransformableFrom)
    .def("transformTo",py::overload_cast<const Coordinate&,const Coordinate&,const std::shared_ptr<CoordinateReferenceSystem>&>(&CoordinateReferenceSystem::transformTo,py::const_))
    .def("transformFrom",py::overload_cast<const Coordinate&,const Coordinate&>(&CoordinateReferenceSystem::transformFrom,py::const_))
    .def("transformTo",py::overload_cast<const Coordinate&,const std::shared_ptr<CoordinateReferenceSystem>&>(&CoordinateReferenceSystem::transformTo,py::const_))
    .def("transformFrom",py::overload_cast<const Coordinate&>(&CoordinateReferenceSystem::transformFrom,py::const_))
    .def("transformTo",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const Time&,const std::shared_ptr<CoordinateReferenceSystem>&,const CoordinateType&>(&CoordinateReferenceSystem::transformTo,py::const_))
    .def("transformFrom",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const Time&,const std::shared_ptr<CoordinateReferenceSystem>&,const CoordinateType&>(&CoordinateReferenceSystem::transformFrom,py::const_))
    .def("transformTo",py::overload_cast<const Eigen::Vector3d&,const Time&,const std::shared_ptr<CoordinateReferenceSystem>&,const CoordinateType&>(&CoordinateReferenceSystem::transformTo,py::const_))
    .def("transformFrom",py::overload_cast<const Eigen::Vector3d&,const Time&,const std::shared_ptr<CoordinateReferenceSystem>&,const CoordinateType&>(&CoordinateReferenceSystem::transformFrom,py::const_))
    .def("getQuaternionTo",py::overload_cast<const Coordinate&,const std::shared_ptr<CoordinateReferenceSystem>&>(&CoordinateReferenceSystem::getQuaternionTo,py::const_))
    .def("getQuaternionFrom",py::overload_cast<const Coordinate&>(&CoordinateReferenceSystem::getQuaternionFrom,py::const_))
    .def("getQuaternionTo",py::overload_cast<const Eigen::Vector3d&,const Time&,const std::shared_ptr<CoordinateReferenceSystem>&>(&CoordinateReferenceSystem::getQuaternionTo,py::const_))
    .def("getQuaternionFrom",py::overload_cast<const Eigen::Vector3d&,const Time&,const std::shared_ptr<CoordinateReferenceSystem>&>(&CoordinateReferenceSystem::getQuaternionFrom,py::const_))
    .def("transformQuaternionTo",py::overload_cast<const Quaternion&,const Coordinate&,const std::shared_ptr<CoordinateReferenceSystem>&>(&CoordinateReferenceSystem::transformQuaternionTo,py::const_))
    .def("transformQuaternionFrom",py::overload_cast<const Quaternion&,const Coordinate&>(&CoordinateReferenceSystem::transformQuaternionFrom,py::const_))
    .def("transformQuaternionTo",py::overload_cast<const Quaternion&,const Eigen::Vector3d&,const Time&,const std::shared_ptr<CoordinateReferenceSystem>&>(&CoordinateReferenceSystem::transformQuaternionTo,py::const_))
    .def("transformQuaternionFrom",py::overload_cast<const Quaternion&,const Eigen::Vector3d&,const Time&,const std::shared_ptr<CoordinateReferenceSystem>&>(&CoordinateReferenceSystem::transformQuaternionFrom,py::const_))
    .def("getHeight",py::overload_cast<const Coordinate&>(&CoordinateReferenceSystem::getHeight,py::const_))
    .def("getGeoidHeight",py::overload_cast<const Coordinate&>(&CoordinateReferenceSystem::getGeoidHeight,py::const_))
    .def("getHeight",py::overload_cast<const Eigen::Vector3d&, const Time&>(&CoordinateReferenceSystem::getHeight,py::const_))
    .def("getGeoidHeight",py::overload_cast<const Eigen::Vector3d&, const Time&>(&CoordinateReferenceSystem::getGeoidHeight,py::const_))
    DEF_FUNC(CoordinateReferenceSystem,toCartesian)
    DEF_FUNC(CoordinateReferenceSystem,toSpherical)
    DEF_FUNC(CoordinateReferenceSystem,toEllipsoidal)
    DEF_FUNC(CoordinateReferenceSystem,fromCartesian)
    DEF_FUNC(CoordinateReferenceSystem,fromSpherical)
    DEF_FUNC(CoordinateReferenceSystem,fromEllipsoidal)
    DEF_FUNC(CoordinateReferenceSystem,cartesianToSpherical)
    DEF_FUNC(CoordinateReferenceSystem,cartesianToEllipsoidal)
    DEF_FUNC(CoordinateReferenceSystem,sphericalToCartesian)
    DEF_FUNC(CoordinateReferenceSystem,sphericalToEllipsoidal)
    DEF_FUNC(CoordinateReferenceSystem,ellipsoidalToCartesian)
    DEF_FUNC(CoordinateReferenceSystem,ellipsoidalToSpherical)
    .def("createTopocentricCRS",[](CoordinateReferenceSystem& v,const Coordinate& origin, const std::string& axisOrder, bool onSurface, bool isEpisodic){
        return v.createTopocentricCRS(origin,axisOrder,onSurface,isEpisodic);
    })
    .def("createTopocentricCRS",[](CoordinateReferenceSystem& v,const Eigen::Vector3d& origin,const Time& time, const std::string& axisOrder, bool onSurface, bool isEpisodic){
        return v.createTopocentricCRS(origin,time,axisOrder,onSurface,isEpisodic);
    })
    .def("getEquivalentQuaternionToTopocentricCRS",py::overload_cast<const Coordinate&,const std::string&>(&CoordinateReferenceSystem::getEquivalentQuaternionToTopocentricCRS,py::const_))
    .def("getEquivalentQuaternionToTopocentricCRS",py::overload_cast<const Eigen::Vector3d&, const Time&,const std::string&>(&CoordinateReferenceSystem::getEquivalentQuaternionToTopocentricCRS,py::const_))
    .def("transformToTopocentricCRS",[](CoordinateReferenceSystem& v,const Coordinate& value, const Coordinate& location, const Coordinate& origin, const std::string& axisOrder, bool onSurface){
        return v.transformToTopocentricCRS(value,location,origin,axisOrder,onSurface);
    })
    .def("transformToTopocentricCRS",[](CoordinateReferenceSystem& v,const Coordinate& value, const Coordinate& origin, const std::string& axisOrder, bool onSurface){
        return v.transformToTopocentricCRS(value,origin,axisOrder,onSurface);
    })
    .def("transformToTopocentricCRS",[](CoordinateReferenceSystem& v,const Eigen::Vector3d& value, const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType, const Eigen::Vector3d& origin, const std::string& axisOrder, bool onSurface){
        return v.transformToTopocentricCRS(value,location,time,coordinateType,origin,axisOrder,onSurface);
    })
    .def("transformToTopocentricCRS",[](CoordinateReferenceSystem& v,const Eigen::Vector3d& value, const Time& time, const CoordinateType& coordinateType, const Eigen::Vector3d& origin, const std::string& axisOrder, bool onSurface){
        return v.transformToTopocentricCRS(value,time,coordinateType,origin,axisOrder,onSurface);
    })
    .def("transformFromTopocentricCRS",[](CoordinateReferenceSystem& v,const Eigen::Vector3d& value, const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType, const Eigen::Vector3d& origin, const std::string& axisOrder, bool onSurface){
        return v.transformFromTopocentricCRS(value,location,time,coordinateType,origin,axisOrder,onSurface);
    })
    .def("transformFromTopocentricCRS",[](CoordinateReferenceSystem& v,const Eigen::Vector3d& value, const Time& time, const CoordinateType& coordinateType, const Eigen::Vector3d& origin, const std::string& axisOrder, bool onSurface){
        return v.transformFromTopocentricCRS(value,time,coordinateType,origin,axisOrder,onSurface);
    })
    DEF_FUNC(CoordinateReferenceSystem,createAffineCRS)
    ;
}

ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
