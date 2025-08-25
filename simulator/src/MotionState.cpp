// Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#include "MotionState.h"

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)

MotionState::MotionState()
:crs(nullptr),crsUpdateCount(-1),_isValid(false),
 _pos(Eigen::Vector3d::Zero(),nullptr,Time(),CoordinateType::POSITION_ABS),
 _vel(Eigen::Vector3d::Zero(),nullptr,Time(),CoordinateType::VELOCITY),
 _omega(Eigen::Vector3d::Zero(),nullptr,Time(),CoordinateType::ANGULAR_VELOCITY),
 _q(1,0,0,0),qToFSDFromNED(1,0,0,0),
 _az(0),_el(0),alt(0),_time(),_axisOrder("FSD"),
 pos(_pos),vel(_vel),q(_q),omega(_omega),time(_time),axisOrder(_axisOrder){
}
MotionState::MotionState(const std::shared_ptr<CoordinateReferenceSystem>& crs_,const Time& time_,const Eigen::Vector3d& pos_,const Eigen::Vector3d& vel_,const Eigen::Vector3d& omega_,const Quaternion& q_, const std::string& axisOrder_)
:crs(crs_),crsUpdateCount(-1),_isValid(false),
 _pos(pos_,pos_,crs_,time_,CoordinateType::POSITION_ABS),
 _vel(vel_,pos_,crs_,time_,CoordinateType::VELOCITY),
 _omega(omega_,pos_,crs_,time_,CoordinateType::ANGULAR_VELOCITY),
 _q(q_),qToFSDFromNED(1,0,0,0),
 _az(0),_el(0),alt(0),_time(time_),_axisOrder(axisOrder_),
 pos(_pos),vel(_vel),q(_q),omega(_omega),time(_time),axisOrder(_axisOrder){
    validate();
}
MotionState::MotionState(const nl::json& j_):MotionState(){
    load_from_json(j_);
}
MotionState::MotionState(const MotionState& other)
:crs(other.crs),crsUpdateCount(other.crsUpdateCount),_isValid(other._isValid),
 _time(other.time),_pos(other.pos),_vel(other.vel),_q(other.q),_omega(other.omega),_axisOrder(other.axisOrder),
 qToFSDFromNED(other.qToFSDFromNED),_az(other._az),_el(other._el),alt(other.alt),
 pos(_pos),vel(_vel),q(_q),omega(_omega),time(_time),axisOrder(_axisOrder){
}
MotionState::MotionState(MotionState&& other)
:crs(other.crs),crsUpdateCount(other.crsUpdateCount),_isValid(other._isValid),
 _time(other.time),_pos(other.pos),_vel(other.vel),_q(other.q),_omega(other.omega),_axisOrder(other.axisOrder),
 qToFSDFromNED(other.qToFSDFromNED),_az(other._az),_el(other._el),alt(other.alt),
 pos(_pos),vel(_vel),q(_q),omega(_omega),time(_time),axisOrder(_axisOrder){
}
void MotionState::operator=(const MotionState& other){
    _time=other.time;
    _pos=other.pos;
    _vel=other.vel;
    _q=other.q;
    _omega=other.omega;
    _axisOrder=other.axisOrder;
    _az=other._az;
    _el=other._el;
    alt=other.alt;
    qToFSDFromNED=other.qToFSDFromNED;
    _isValid=other._isValid;
    crs=other.crs;
    crsUpdateCount=other.crsUpdateCount;
}
void MotionState::operator=(MotionState&& other){
    _time=other.time;
    _pos=other.pos;
    _vel=other.vel;
    _q=other.q;
    _omega=other.omega;
    _axisOrder=other.axisOrder;
    _az=other._az;
    _el=other._el;
    alt=other.alt;
    qToFSDFromNED=other.qToFSDFromNED;
    _isValid=other._isValid;
    crs=other.crs;
    crsUpdateCount=other.crsUpdateCount;
}
MotionState::~MotionState(){}
MotionState MotionState::copy() const{
    return std::move(MotionState(*this));
}
void MotionState::setPos(const Eigen::Vector3d& pos_){
    _pos.setValue(pos_);
    _vel.setLocation(pos_);
    _omega.setLocation(pos_);
    invalidate();
}
void MotionState::setPos(const Coordinate& pos_){
    setPos(pos_(crs));
    invalidate();
}
void MotionState::setVel(const Eigen::Vector3d& vel_){
    _vel.setValue(vel_);
    invalidate();
}
void MotionState::setVel(const Coordinate& vel_){
    setVel(vel_(crs));
    invalidate();
}
void MotionState::setQ(const Quaternion& q_){
    _q=q_;
    invalidate();
}
void MotionState::setOmega(const Eigen::Vector3d&omega_){
    _omega.setValue(omega_);
    invalidate();
}
void MotionState::setOmega(const Coordinate& omega_){
    setOmega(omega_(crs));
    invalidate();
}
void MotionState::setTime(const Time& time_){
    _time=time_;
    _pos.setTime(time);
    _vel.setTime(time);
    _omega.setTime(time);
    invalidate();
}
void MotionState::setAxisOrder(const std::string& axisOrder_){
    _axisOrder=axisOrder_;
    invalidate();
}
bool MotionState::isValid() const{
    if(crs){
        return crs->isValid() && crs->getUpdateCount()==crsUpdateCount && _isValid;
    }else{
        return _isValid;
    }
}
void MotionState::invalidate() const{
    _isValid=false;
}
void MotionState::validate() const{
    Eigen::Vector3d fwdInNED;
    if(crs){
        std::shared_ptr<CoordinateReferenceSystem> intermediateCRS=crs->getNonDerivedBaseCRS()->getIntermediateCRS(); //Topocentricの基準となる地球固定座標系(ECEF or PureFlat)
        Quaternion qToNEDFromIntermediate=intermediateCRS->getEquivalentQuaternionToTopocentricCRS(pos(intermediateCRS),time,"NED");//[NED <- intermediate]
        Quaternion qToIntermediateFromParent=crs->getQuaternionTo(pos(),time,intermediateCRS);//[intermediate <- parent]

        auto qToNEDFromBody=qToNEDFromIntermediate*qToIntermediateFromParent*q;//[NED <- this]
        fwdInNED=(
            qToNEDFromBody* //[NED <- this]
            getAxisSwapQuaternion("FSD",axisOrder) //[this <- FSD(body)]
        ).transformVector(Eigen::Vector3d(1,0,0));
    }else{
        //親CRSの指定が無い場合はNEDのTopocentricとみなす。
        fwdInNED=(
            q* //[parent(NED) <- this]
            getAxisSwapQuaternion("FSD",axisOrder) //[this <- FSD(body)]
        ).transformVector(Eigen::Vector3d(1,0,0));
    }
    double xy=sqrt(fwdInNED(0)*fwdInNED(0)+fwdInNED(1)*fwdInNED(1));
    double newEl,newAz;
    if(xy==0){
        newEl=fwdInNED(2)>0 ? -M_PI_2 : M_PI_2;
        newAz=_az;
    }else{
        newEl=atan2(-fwdInNED(2),xy);
        newAz=atan2(fwdInNED(1),fwdInNED(0));
    }
    Eigen::Vector3d fwdH(cos(newAz),sin(newAz),0);
    Eigen::Vector3d stbdH,downH;
    stbdH<<-sin(newAz),cos(newAz),0;
    downH<<0,0,1;
    qToFSDFromNED=Quaternion::fromBasis(fwdH,stbdH,downH).conjugate();//[FSD(topo) <- NED]
    _az=newAz;
    _el=newEl;
    if(crs){
        alt=crs->getHeight(pos(),time);
        crsUpdateCount=crs->getUpdateCount();
    }else{
        alt=-pos(2); //親座標系が未指定の場合は暗黙に3軸目が下向き正の高度に相当するものと解釈する。
    }
    _isValid=true;
}

//(1)座標軸の変換 (axis conversion in the parent crs)
Eigen::Vector3d MotionState::toCartesian(const Eigen::Vector3d& v) const{
    return crs ? crs->toCartesian(v) : v;
}
Eigen::Vector3d MotionState::toSpherical(const Eigen::Vector3d& v) const{
    if(crs){
        return crs->toSpherical(v);
    }else{
        //親CRSの指定が無い場合はNEDのTopocentricとみなす。
        //NED -> (Azimuth Elevation Range)
        double xy=sqrt(v(0)*v(0)+v(1)*v(1));
        if(xy==0){
            return Eigen::Vector3d(
                0,
                v(2)>0 ? -M_PI_2 : M_PI_2,
                abs(v(2))
            );
        }else{
            double el=atan2(-v(2),xy);
            return Eigen::Vector3d(
                atan2(v(1),v(0)),
                el,
                sqrt(xy*xy+v(2)*v(2))
            );
        }
    }
}
Eigen::Vector3d MotionState::toEllipsoidal(const Eigen::Vector3d& v) const{
    if(crs){
        return crs->toEllipsoidal(v);
    }else{
        throw std::runtime_error("axis conversion to ellipsoidal axis is not defined for MotionState.");
    }
}
Eigen::Vector3d MotionState::fromCartesian(const Eigen::Vector3d& v) const{
    return crs ? crs->fromCartesian(v) : v;
}
Eigen::Vector3d MotionState::fromSpherical(const Eigen::Vector3d& v) const{
    if(crs){
        return crs->fromSpherical(v);
    }else{
        //親CRSの指定が無い場合はNEDのTopocentricとみなす。
        // (Azimuth Elevation Range) -> NED
        return Eigen::Vector3d(
            v(2)*cos(v(0))*cos(v(1)),
            v(2)*sin(v(0))*cos(v(1)),
            -v(2)*sin(v(1))
        );
    }
}
Eigen::Vector3d MotionState::fromEllipsoidal(const Eigen::Vector3d& v) const{
    if(crs){
        return crs->fromEllipsoidal(v);
    }else{
        throw std::runtime_error("axis conversion from ellipsoidal axis is not defined for MotionState.");
    }
}

//(2)相対位置ベクトルの変換 (transformation of "relative" position)
Eigen::Vector3d MotionState::relBtoP(const Eigen::Vector3d &v) const{
    return relBtoP(v,Eigen::Vector3d::Zero());
}
Eigen::Vector3d MotionState::relPtoB(const Eigen::Vector3d &v) const{
    return relPtoB(v,Eigen::Vector3d::Zero());
}
Eigen::Vector3d MotionState::relHtoP(const Eigen::Vector3d &v,const std::string& dstAxisOrder,bool onSurface) const{
    return relHtoP(v,Eigen::Vector3d::Zero(),dstAxisOrder,onSurface);
}
Eigen::Vector3d MotionState::relPtoH(const Eigen::Vector3d &v,const std::string& dstAxisOrder,bool onSurface) const{
    return relPtoH(v,Eigen::Vector3d::Zero(),dstAxisOrder,onSurface);
}
Eigen::Vector3d MotionState::relHtoB(const Eigen::Vector3d &v,const std::string& dstAxisOrder,bool onSurface) const{
    return relHtoB(v,Eigen::Vector3d::Zero(),dstAxisOrder,onSurface);
}
Eigen::Vector3d MotionState::relBtoH(const Eigen::Vector3d &v,const std::string& dstAxisOrder,bool onSurface) const{
    return relBtoH(v,Eigen::Vector3d::Zero(),dstAxisOrder,onSurface);
}
Eigen::Vector3d MotionState::relBtoA(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another) const{
    return relBtoA(v,Eigen::Vector3d::Zero(),another);
}
Eigen::Vector3d MotionState::relAtoB(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another) const{
    return relAtoB(v,Eigen::Vector3d::Zero(),another);
}
Eigen::Vector3d MotionState::relPtoA(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another) const{
    return relPtoA(v,Eigen::Vector3d::Zero(),another);
}
Eigen::Vector3d MotionState::relAtoP(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another) const{
    return relAtoP(v,Eigen::Vector3d::Zero(),another);
}
Eigen::Vector3d MotionState::relHtoA(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another,const std::string& dstAxisOrder,bool onSurface) const{
    return relHtoA(v,Eigen::Vector3d::Zero(),another,dstAxisOrder,onSurface);
}
Eigen::Vector3d MotionState::relAtoH(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another,const std::string& dstAxisOrder,bool onSurface) const{
    return relAtoH(v,Eigen::Vector3d::Zero(),another,dstAxisOrder,onSurface);
}

Eigen::Vector3d MotionState::relBtoP(const Eigen::Vector3d &v,const Eigen::Vector3d &r) const{
    return fromCartesian(q.transformVector(v));
}
Eigen::Vector3d MotionState::relPtoB(const Eigen::Vector3d &v,const Eigen::Vector3d &r) const{
    return q.conjugate().transformVector(toCartesian(v));
}
Eigen::Vector3d MotionState::relHtoP(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::string& dstAxisOrder,bool onSurface) const{
    if(!crs){
        throw std::runtime_error("MotionState without CRS cannot be converted to another CRS.");
    }
    if(checkTopocentricCartesianAxisOrder(dstAxisOrder)){
        return crs->transformFromTopocentricCRS(v,r,time,CoordinateType::POSITION_REL,pos(),dstAxisOrder,onSurface);
    }else if(checkBodyCartesianAxisOrder(dstAxisOrder)){
        auto qSwap=qToFSDFromNED.conjugate()*getAxisSwapQuaternion(dstAxisOrder,"FSD");
        auto vNED=qSwap.transformVector(v);
        auto rNED=qSwap.transformVector(r);
        return crs->transformFromTopocentricCRS(vNED,rNED,time,CoordinateType::POSITION_REL,pos(),"NED",onSurface);
    }else{
        throw std::runtime_error("'"+dstAxisOrder+"' is not a valid dstAxisOrder.");
    }
}
Eigen::Vector3d MotionState::relPtoH(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::string& dstAxisOrder,bool onSurface) const{
    if(!crs){
        throw std::runtime_error("MotionState without CRS cannot be converted to another CRS.");
    }
    if(checkTopocentricCartesianAxisOrder(dstAxisOrder)){
        return crs->transformToTopocentricCRS(v,r,time,CoordinateType::POSITION_REL,pos(),dstAxisOrder,onSurface);
    }else if(checkBodyCartesianAxisOrder(dstAxisOrder)){
        auto vNED=crs->transformToTopocentricCRS(v,r,time,CoordinateType::POSITION_REL,pos(),"NED",onSurface);
        return (getAxisSwapQuaternion("FSD",dstAxisOrder)*qToFSDFromNED).transformVector(vNED);
    }else{
        throw std::runtime_error("'"+dstAxisOrder+"' is not a valid dstAxisOrder.");
    }
}
Eigen::Vector3d MotionState::relHtoB(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::string& dstAxisOrder,bool onSurface) const{
    return relPtoB(relHtoP(v,r,dstAxisOrder,onSurface),absHtoP(r,r,dstAxisOrder,onSurface));
}
Eigen::Vector3d MotionState::relBtoH(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::string& dstAxisOrder,bool onSurface) const{
    return relPtoH(relBtoP(v,r),absBtoP(r,r),dstAxisOrder,onSurface);
}
Eigen::Vector3d MotionState::relBtoA(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another) const{
    //self系⇛another系
    return relPtoA(relBtoP(v,r),absBtoP(r,r),another);
}
Eigen::Vector3d MotionState::relAtoB(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another) const{
    //another系⇛self系
    return relPtoB(relAtoP(v,r,another),absAtoP(r,r,another));
}
Eigen::Vector3d MotionState::relPtoA(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another) const{
    //parent系⇛another系
    if(!crs){
        throw std::runtime_error("MotionState without CRS cannot be converted to another CRS.");
    }
    return crs->transformTo(v,r,time,another,CoordinateType::POSITION_REL)();
}
Eigen::Vector3d MotionState::relAtoP(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another) const{
    //another系⇛parent系
    if(!crs){
        throw std::runtime_error("MotionState without CRS cannot be converted to another CRS.");
    }
    return crs->transformFrom(v,r,time,another,CoordinateType::POSITION_REL)();
}
Eigen::Vector3d MotionState::relHtoA(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another,const std::string& dstAxisOrder,bool onSurface) const{
    //horizontal系⇛another系
    return relPtoA(relHtoP(v,r,dstAxisOrder,onSurface),absHtoP(r,r,dstAxisOrder,onSurface),another);
}
Eigen::Vector3d MotionState::relAtoH(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another,const std::string& dstAxisOrder,bool onSurface) const{
    //another系⇛horizontal系
    return relPtoH(relAtoP(v,r,another),absAtoP(r,r,another),dstAxisOrder,onSurface);
}

//(3)絶対位置ベクトルの変換 (transformation of "absolute" position)
Eigen::Vector3d MotionState::absBtoP(const Eigen::Vector3d &v) const{
    return absBtoP(v,Eigen::Vector3d::Zero());
}
Eigen::Vector3d MotionState::absPtoB(const Eigen::Vector3d &v) const{
    return absPtoB(v,Eigen::Vector3d::Zero());
}
Eigen::Vector3d MotionState::absHtoP(const Eigen::Vector3d &v,const std::string& dstAxisOrder,bool onSurface) const{
    return absHtoP(v,Eigen::Vector3d::Zero(),dstAxisOrder,onSurface);
}
Eigen::Vector3d MotionState::absPtoH(const Eigen::Vector3d &v,const std::string& dstAxisOrder,bool onSurface) const{
    return absPtoH(v,Eigen::Vector3d::Zero(),dstAxisOrder,onSurface);
}
Eigen::Vector3d MotionState::absHtoB(const Eigen::Vector3d &v,const std::string& dstAxisOrder,bool onSurface) const{
    return absHtoB(v,Eigen::Vector3d::Zero(),dstAxisOrder,onSurface);
}
Eigen::Vector3d MotionState::absBtoH(const Eigen::Vector3d &v,const std::string& dstAxisOrder,bool onSurface) const{
    return absBtoH(v,Eigen::Vector3d::Zero(),dstAxisOrder,onSurface);
}
Eigen::Vector3d MotionState::absBtoA(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another) const{
    return absBtoA(v,Eigen::Vector3d::Zero(),another);
}
Eigen::Vector3d MotionState::absAtoB(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another) const{
    return absAtoB(v,Eigen::Vector3d::Zero(),another);
}
Eigen::Vector3d MotionState::absPtoA(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another) const{
    return absPtoA(v,Eigen::Vector3d::Zero(),another);
}
Eigen::Vector3d MotionState::absAtoP(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another) const{
    return absAtoP(v,Eigen::Vector3d::Zero(),another);
}
Eigen::Vector3d MotionState::absHtoA(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another,const std::string& dstAxisOrder,bool onSurface) const{
    return absHtoA(v,Eigen::Vector3d::Zero(),another,dstAxisOrder,onSurface);
}
Eigen::Vector3d MotionState::absAtoH(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another,const std::string& dstAxisOrder,bool onSurface) const{
    return absAtoH(v,Eigen::Vector3d::Zero(),another,dstAxisOrder,onSurface);
}

Eigen::Vector3d MotionState::absBtoP(const Eigen::Vector3d &v,const Eigen::Vector3d &r) const{
    return fromCartesian(q.transformVector(v)+pos());
}
Eigen::Vector3d MotionState::absPtoB(const Eigen::Vector3d &v,const Eigen::Vector3d &r) const{
    return q.conjugate().transformVector(toCartesian(v)-pos.toCartesian());
}
Eigen::Vector3d MotionState::absHtoP(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::string& dstAxisOrder,bool onSurface) const{
    if(!crs){
        throw std::runtime_error("MotionState without CRS cannot be converted to another CRS.");
    }
    if(checkTopocentricCartesianAxisOrder(dstAxisOrder)){
        return crs->transformFromTopocentricCRS(v,r,time,CoordinateType::POSITION_ABS,pos(),dstAxisOrder,onSurface);
    }else if(checkBodyCartesianAxisOrder(dstAxisOrder)){
        auto qSwap=qToFSDFromNED.conjugate()*getAxisSwapQuaternion(dstAxisOrder,"FSD");
        auto vNED=qSwap.transformVector(v);
        auto rNED=qSwap.transformVector(r);
        return crs->transformFromTopocentricCRS(vNED,rNED,time,CoordinateType::POSITION_ABS,pos(),"NED",onSurface);
    }else{
        throw std::runtime_error("'"+dstAxisOrder+"' is not a valid dstAxisOrder.");
    }
}
Eigen::Vector3d MotionState::absPtoH(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::string& dstAxisOrder,bool onSurface) const{
    if(!crs){
        throw std::runtime_error("MotionState without CRS cannot be converted to another CRS.");
    }
    if(checkTopocentricCartesianAxisOrder(dstAxisOrder)){
        return crs->transformToTopocentricCRS(v,r,time,CoordinateType::POSITION_ABS,pos(),dstAxisOrder,onSurface);
    }else if(checkBodyCartesianAxisOrder(dstAxisOrder)){
        auto vNED=crs->transformToTopocentricCRS(v,r,time,CoordinateType::POSITION_ABS,pos(),"NED",onSurface);
        return (getAxisSwapQuaternion("FSD",dstAxisOrder)*qToFSDFromNED).transformVector(vNED);
    }else{
        throw std::runtime_error("'"+dstAxisOrder+"' is not a valid dstAxisOrder.");
    }
}
Eigen::Vector3d MotionState::absHtoB(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::string& dstAxisOrder,bool onSurface) const{
    return absPtoB(absHtoP(v,r,dstAxisOrder,onSurface),absHtoP(r,r,dstAxisOrder,onSurface));
}
Eigen::Vector3d MotionState::absBtoH(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::string& dstAxisOrder,bool onSurface) const{
    return absPtoH(absBtoP(v,r),absBtoP(r,r),dstAxisOrder,onSurface);
}
Eigen::Vector3d MotionState::absBtoA(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another) const{
    //self系⇛another系
    return absPtoA(absBtoP(v,r),absBtoP(r,r),another);
}
Eigen::Vector3d MotionState::absAtoB(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another) const{
    //another系⇛self系
    return absPtoB(absAtoP(v,r,another),absAtoP(r,r,another));
}
Eigen::Vector3d MotionState::absPtoA(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another) const{
    //parent系⇛another系
    if(!crs){
        throw std::runtime_error("MotionState without CRS cannot be converted to another CRS.");
    }
    return crs->transformTo(v,r,time,another,CoordinateType::POSITION_ABS)();
}
Eigen::Vector3d MotionState::absAtoP(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another) const{
    //another系⇛parent系
    if(!crs){
        throw std::runtime_error("MotionState without CRS cannot be converted to another CRS.");
    }
    return crs->transformFrom(v,r,time,another,CoordinateType::POSITION_ABS)();
}
Eigen::Vector3d MotionState::absHtoA(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another,const std::string& dstAxisOrder,bool onSurface) const{
    //horizontal系⇛another系
    return absPtoA(absHtoP(v,r,dstAxisOrder,onSurface),absHtoP(r,r,dstAxisOrder,onSurface),another);
}
Eigen::Vector3d MotionState::absAtoH(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another,const std::string& dstAxisOrder,bool onSurface) const{
    //another系⇛horizontal系
    return absPtoH(absAtoP(v,r,another),absAtoP(r,r,another),dstAxisOrder,onSurface);
}

//(4)速度ベクトルの変換 (transformation of "observed" velocity)
Eigen::Vector3d MotionState::velBtoP(const Eigen::Vector3d &v) const{
    return velBtoP(v,Eigen::Vector3d::Zero());
}
Eigen::Vector3d MotionState::velPtoB(const Eigen::Vector3d &v) const{
    return velPtoB(v,Eigen::Vector3d::Zero());
}
Eigen::Vector3d MotionState::velHtoP(const Eigen::Vector3d &v,const std::string& dstAxisOrder,bool onSurface) const{
    return velHtoP(v,Eigen::Vector3d::Zero(),dstAxisOrder,onSurface);
}
Eigen::Vector3d MotionState::velPtoH(const Eigen::Vector3d &v,const std::string& dstAxisOrder,bool onSurface) const{
    return velPtoH(v,Eigen::Vector3d::Zero(),dstAxisOrder,onSurface);
}
Eigen::Vector3d MotionState::velHtoB(const Eigen::Vector3d &v,const std::string& dstAxisOrder,bool onSurface) const{
    return velHtoB(v,Eigen::Vector3d::Zero(),dstAxisOrder,onSurface);
}
Eigen::Vector3d MotionState::velBtoH(const Eigen::Vector3d &v,const std::string& dstAxisOrder,bool onSurface) const{
    return velBtoH(v,Eigen::Vector3d::Zero(),dstAxisOrder,onSurface);
}
Eigen::Vector3d MotionState::velBtoA(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another) const{
    return velBtoA(v,Eigen::Vector3d::Zero(),another);
}
Eigen::Vector3d MotionState::velAtoB(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another) const{
    return velAtoB(v,Eigen::Vector3d::Zero(),another);
}
Eigen::Vector3d MotionState::velPtoA(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another) const{
    return velPtoA(v,Eigen::Vector3d::Zero(),another);
}
Eigen::Vector3d MotionState::velAtoP(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another) const{
    return velAtoP(v,Eigen::Vector3d::Zero(),another);
}
Eigen::Vector3d MotionState::velHtoA(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another,const std::string& dstAxisOrder,bool onSurface) const{
    return velHtoA(v,Eigen::Vector3d::Zero(),another,dstAxisOrder,onSurface);
}
Eigen::Vector3d MotionState::velAtoH(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another,const std::string& dstAxisOrder,bool onSurface) const{
    return velAtoH(v,Eigen::Vector3d::Zero(),another,dstAxisOrder,onSurface);
}

Eigen::Vector3d MotionState::velBtoP(const Eigen::Vector3d &v,const Eigen::Vector3d &r) const{
    return fromCartesian(q.transformVector(v)+vel.toCartesian()+omega.toCartesian().cross(q.transformVector(r)));
}
Eigen::Vector3d MotionState::velPtoB(const Eigen::Vector3d &v,const Eigen::Vector3d &r) const{
    return q.conjugate().transformVector(toCartesian(v)-vel.toCartesian()-omega.toCartesian().cross(toCartesian(r)-pos.toCartesian()));
}
Eigen::Vector3d MotionState::velHtoP(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::string& dstAxisOrder,bool onSurface) const{
    if(!crs){
        throw std::runtime_error("MotionState without CRS cannot be converted to another CRS.");
    }
    if(checkTopocentricCartesianAxisOrder(dstAxisOrder)){
        return crs->transformFromTopocentricCRS(v,r,time,CoordinateType::VELOCITY,pos(),dstAxisOrder,onSurface);
    }else if(checkBodyCartesianAxisOrder(dstAxisOrder)){
        auto qSwap=qToFSDFromNED.conjugate()*getAxisSwapQuaternion(dstAxisOrder,"FSD");
        auto vNED=qSwap.transformVector(v);
        auto rNED=qSwap.transformVector(r);
        return crs->transformFromTopocentricCRS(vNED,rNED,time,CoordinateType::VELOCITY,pos(),"NED",onSurface);
    }else{
        throw std::runtime_error("'"+dstAxisOrder+"' is not a valid dstAxisOrder.");
    }
}
Eigen::Vector3d MotionState::velPtoH(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::string& dstAxisOrder,bool onSurface) const{
    if(!crs){
        throw std::runtime_error("MotionState without CRS cannot be converted to another CRS.");
    }
    if(checkTopocentricCartesianAxisOrder(dstAxisOrder)){
        return crs->transformToTopocentricCRS(v,r,time,CoordinateType::VELOCITY,pos(),dstAxisOrder,onSurface);
    }else if(checkBodyCartesianAxisOrder(dstAxisOrder)){
        auto vNED=crs->transformToTopocentricCRS(v,r,time,CoordinateType::VELOCITY,pos(),"NED",onSurface);
        return (getAxisSwapQuaternion("FSD",dstAxisOrder)*qToFSDFromNED).transformVector(vNED);
    }else{
        throw std::runtime_error("'"+dstAxisOrder+"' is not a valid dstAxisOrder.");
    }
}
Eigen::Vector3d MotionState::velHtoB(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::string& dstAxisOrder,bool onSurface) const{
    return velPtoB(velHtoP(v,r,dstAxisOrder,onSurface),absHtoP(r,r,dstAxisOrder,onSurface));
}
Eigen::Vector3d MotionState::velBtoH(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::string& dstAxisOrder,bool onSurface) const{
    return velPtoH(velBtoP(v,r),absBtoP(r,r),dstAxisOrder,onSurface);
}
Eigen::Vector3d MotionState::velBtoA(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another) const{
    //self系⇛another系
    return velPtoA(velBtoP(v,r),absBtoP(r),another);
}
Eigen::Vector3d MotionState::velAtoB(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another) const{
    //another系⇛self系
    return velPtoB(velAtoP(v,r,another),absAtoP(r,another));
}
Eigen::Vector3d MotionState::velPtoA(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another) const{
    //parent系⇛another系
    if(!crs){
        throw std::runtime_error("MotionState without CRS cannot be converted to another CRS.");
    }
    return crs->transformTo(v,r,time,another,CoordinateType::VELOCITY)();
}
Eigen::Vector3d MotionState::velAtoP(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another) const{
    //another系⇛parent系
    if(!crs){
        throw std::runtime_error("MotionState without CRS cannot be converted to another CRS.");
    }
    return crs->transformFrom(v,r,time,another,CoordinateType::VELOCITY)();
}
Eigen::Vector3d MotionState::velHtoA(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another,const std::string& dstAxisOrder,bool onSurface) const{
    //horizontal系⇛another系
    return velPtoA(velHtoP(v,r,dstAxisOrder,onSurface),absHtoP(r,dstAxisOrder,onSurface),another);
}
Eigen::Vector3d MotionState::velAtoH(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another,const std::string& dstAxisOrder,bool onSurface) const{
    //another系⇛horizontal系
    return velPtoH(velAtoP(v,r,another),absAtoP(r,another),dstAxisOrder,onSurface);
}

//(5)角速度ベクトルの変換 (transformation of "observed" angular velocity)
Eigen::Vector3d MotionState::omegaBtoP(const Eigen::Vector3d &v) const{
    return omegaBtoP(v,Eigen::Vector3d::Zero());
}
Eigen::Vector3d MotionState::omegaPtoB(const Eigen::Vector3d &v) const{
    return omegaPtoB(v,Eigen::Vector3d::Zero());
}
Eigen::Vector3d MotionState::omegaHtoP(const Eigen::Vector3d &v,const std::string& dstAxisOrder,bool onSurface) const{
    return omegaHtoP(v,Eigen::Vector3d::Zero(),dstAxisOrder,onSurface);
}
Eigen::Vector3d MotionState::omegaPtoH(const Eigen::Vector3d &v,const std::string& dstAxisOrder,bool onSurface) const{
    return omegaPtoH(v,Eigen::Vector3d::Zero(),dstAxisOrder,onSurface);
}
Eigen::Vector3d MotionState::omegaHtoB(const Eigen::Vector3d &v,const std::string& dstAxisOrder,bool onSurface) const{
    return omegaHtoB(v,Eigen::Vector3d::Zero(),dstAxisOrder,onSurface);
}
Eigen::Vector3d MotionState::omegaBtoH(const Eigen::Vector3d &v,const std::string& dstAxisOrder,bool onSurface) const{
    return omegaBtoH(v,Eigen::Vector3d::Zero(),dstAxisOrder,onSurface);
}
Eigen::Vector3d MotionState::omegaBtoA(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another) const{
    return omegaBtoA(v,Eigen::Vector3d::Zero(),another);
}
Eigen::Vector3d MotionState::omegaAtoB(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another) const{
    return omegaAtoB(v,Eigen::Vector3d::Zero(),another);
}
Eigen::Vector3d MotionState::omegaPtoA(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another) const{
    return omegaPtoA(v,Eigen::Vector3d::Zero(),another);
}
Eigen::Vector3d MotionState::omegaAtoP(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another) const{
    return omegaAtoP(v,Eigen::Vector3d::Zero(),another);
}
Eigen::Vector3d MotionState::omegaHtoA(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another,const std::string& dstAxisOrder,bool onSurface) const{
    return omegaHtoA(v,Eigen::Vector3d::Zero(),another,dstAxisOrder,onSurface);
}
Eigen::Vector3d MotionState::omegaAtoH(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another,const std::string& dstAxisOrder,bool onSurface) const{
    return omegaAtoH(v,Eigen::Vector3d::Zero(),another,dstAxisOrder,onSurface);
}

Eigen::Vector3d MotionState::omegaBtoP(const Eigen::Vector3d &v,const Eigen::Vector3d &r) const{
    return fromCartesian(q.transformVector(v)+omega.toCartesian());
}
Eigen::Vector3d MotionState::omegaPtoB(const Eigen::Vector3d &v,const Eigen::Vector3d &r) const{
    return q.conjugate().transformVector(toCartesian(v)-omega.toCartesian());
}
Eigen::Vector3d MotionState::omegaHtoP(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::string& dstAxisOrder,bool onSurface) const{
    if(!crs){
        throw std::runtime_error("MotionState without CRS cannot be converted to another CRS.");
    }
    if(checkTopocentricCartesianAxisOrder(dstAxisOrder)){
        return crs->transformFromTopocentricCRS(v,r,time,CoordinateType::ANGULAR_VELOCITY,pos(),dstAxisOrder,onSurface);
    }else if(checkBodyCartesianAxisOrder(dstAxisOrder)){
        auto qSwap=qToFSDFromNED.conjugate()*getAxisSwapQuaternion(dstAxisOrder,"FSD");
        auto vNED=qSwap.transformVector(v);
        auto rNED=qSwap.transformVector(r);
        return crs->transformFromTopocentricCRS(vNED,rNED,time,CoordinateType::ANGULAR_VELOCITY,pos(),"NED",onSurface);
    }else{
        throw std::runtime_error("'"+dstAxisOrder+"' is not a valid dstAxisOrder.");
    }
}
Eigen::Vector3d MotionState::omegaPtoH(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::string& dstAxisOrder,bool onSurface) const{
    if(!crs){
        throw std::runtime_error("MotionState without CRS cannot be converted to another CRS.");
    }
    if(checkTopocentricCartesianAxisOrder(dstAxisOrder)){
        return crs->transformToTopocentricCRS(v,r,time,CoordinateType::ANGULAR_VELOCITY,pos(),dstAxisOrder,onSurface);
    }else if(checkBodyCartesianAxisOrder(dstAxisOrder)){
        auto vNED=crs->transformToTopocentricCRS(v,r,time,CoordinateType::ANGULAR_VELOCITY,pos(),"NED",onSurface);
        return (getAxisSwapQuaternion("FSD",dstAxisOrder)*qToFSDFromNED).transformVector(vNED);
    }else{
        throw std::runtime_error("'"+dstAxisOrder+"' is not a valid dstAxisOrder.");
    }
}
Eigen::Vector3d MotionState::omegaHtoB(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::string& dstAxisOrder,bool onSurface) const{
    return omegaPtoB(omegaHtoP(v,r,dstAxisOrder,onSurface),absHtoP(r,r,dstAxisOrder,onSurface));
}
Eigen::Vector3d MotionState::omegaBtoH(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::string& dstAxisOrder,bool onSurface) const{
    return omegaPtoH(omegaBtoP(v,r),absBtoP(r,r),dstAxisOrder,onSurface);
}
Eigen::Vector3d MotionState::omegaBtoA(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another) const{
    //self系⇛another系
    return omegaPtoA(omegaBtoP(v,r),absBtoP(r,r),another);
}
Eigen::Vector3d MotionState::omegaAtoB(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another) const{
    //another系⇛self系
    return omegaPtoB(omegaAtoP(v,r,another),absAtoP(r,r,another));
}
Eigen::Vector3d MotionState::omegaPtoA(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another) const{
    //parent系⇛another系
    if(!crs){
        throw std::runtime_error("MotionState without CRS cannot be converted to another CRS.");
    }
    return crs->transformTo(v,r,time,another,CoordinateType::ANGULAR_VELOCITY)();
}
Eigen::Vector3d MotionState::omegaAtoP(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another) const{
    //another系⇛parent系
    if(!crs){
        throw std::runtime_error("MotionState without CRS cannot be converted to another CRS.");
    }
    return crs->transformFrom(v,r,time,another,CoordinateType::ANGULAR_VELOCITY)();
}
Eigen::Vector3d MotionState::omegaHtoA(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another,const std::string& dstAxisOrder,bool onSurface) const{
    //horizontal系⇛another系
    return omegaPtoA(omegaHtoP(v,r,dstAxisOrder,onSurface),absHtoP(r,r,dstAxisOrder,onSurface),another);
}
Eigen::Vector3d MotionState::omegaAtoH(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another,const std::string& dstAxisOrder,bool onSurface) const{
    //another系⇛horizontal系
    return omegaPtoH(omegaAtoP(v,r,another),absAtoP(r,r,another),dstAxisOrder,onSurface);
}

//(6)方向ベクトルの変換 (transformation of direction)
Eigen::Vector3d MotionState::dirBtoP(const Eigen::Vector3d &v) const{
    return dirBtoP(v,Eigen::Vector3d::Zero());
}
Eigen::Vector3d MotionState::dirPtoB(const Eigen::Vector3d &v) const{
    return dirPtoB(v,Eigen::Vector3d::Zero());
}
Eigen::Vector3d MotionState::dirHtoP(const Eigen::Vector3d &v,const std::string& dstAxisOrder,bool onSurface) const{
    return dirHtoP(v,Eigen::Vector3d::Zero(),dstAxisOrder,onSurface);
}
Eigen::Vector3d MotionState::dirPtoH(const Eigen::Vector3d &v,const std::string& dstAxisOrder,bool onSurface) const{
    return dirPtoH(v,Eigen::Vector3d::Zero(),dstAxisOrder,onSurface);
}
Eigen::Vector3d MotionState::dirHtoB(const Eigen::Vector3d &v,const std::string& dstAxisOrder,bool onSurface) const{
    return dirHtoB(v,Eigen::Vector3d::Zero(),dstAxisOrder,onSurface);
}
Eigen::Vector3d MotionState::dirBtoH(const Eigen::Vector3d &v,const std::string& dstAxisOrder,bool onSurface) const{
    return dirBtoH(v,Eigen::Vector3d::Zero(),dstAxisOrder,onSurface);
}
Eigen::Vector3d MotionState::dirBtoA(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another) const{
    return dirBtoA(v,Eigen::Vector3d::Zero(),another);
}
Eigen::Vector3d MotionState::dirAtoB(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another) const{
    return dirAtoB(v,Eigen::Vector3d::Zero(),another);
}
Eigen::Vector3d MotionState::dirPtoA(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another) const{
    return dirPtoA(v,Eigen::Vector3d::Zero(),another);
}
Eigen::Vector3d MotionState::dirAtoP(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another) const{
    return dirAtoP(v,Eigen::Vector3d::Zero(),another);
}
Eigen::Vector3d MotionState::dirHtoA(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another,const std::string& dstAxisOrder,bool onSurface) const{
    return dirHtoA(v,Eigen::Vector3d::Zero(),another,dstAxisOrder,onSurface);
}
Eigen::Vector3d MotionState::dirAtoH(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another,const std::string& dstAxisOrder,bool onSurface) const{
    return dirAtoH(v,Eigen::Vector3d::Zero(),another,dstAxisOrder,onSurface);
}

Eigen::Vector3d MotionState::dirBtoP(const Eigen::Vector3d &v,const Eigen::Vector3d &r) const{
    return fromCartesian(q.transformVector(v).normalized());
}
Eigen::Vector3d MotionState::dirPtoB(const Eigen::Vector3d &v,const Eigen::Vector3d &r) const{
    return q.conjugate().transformVector(toCartesian(v)).normalized();
}
Eigen::Vector3d MotionState::dirHtoP(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::string& dstAxisOrder,bool onSurface) const{
    if(!crs){
        throw std::runtime_error("MotionState without CRS cannot be converted to another CRS.");
    }
    if(checkTopocentricCartesianAxisOrder(dstAxisOrder)){
        return crs->transformFromTopocentricCRS(v,r,time,CoordinateType::DIRECTION,pos(),dstAxisOrder,onSurface);
    }else if(checkBodyCartesianAxisOrder(dstAxisOrder)){
        auto qSwap=qToFSDFromNED.conjugate()*getAxisSwapQuaternion(dstAxisOrder,"FSD");
        auto vNED=qSwap.transformVector(v);
        auto rNED=qSwap.transformVector(r);
        return crs->transformFromTopocentricCRS(vNED,rNED,time,CoordinateType::DIRECTION,pos(),"NED",onSurface);
    }else{
        throw std::runtime_error("'"+dstAxisOrder+"' is not a valid dstAxisOrder.");
    }
}
Eigen::Vector3d MotionState::dirPtoH(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::string& dstAxisOrder,bool onSurface) const{
    if(!crs){
        throw std::runtime_error("MotionState without CRS cannot be converted to another CRS.");
    }
    if(checkTopocentricCartesianAxisOrder(dstAxisOrder)){
        return crs->transformToTopocentricCRS(v,r,time,CoordinateType::DIRECTION,pos(),dstAxisOrder,onSurface);
    }else if(checkBodyCartesianAxisOrder(dstAxisOrder)){
        auto vNED=crs->transformToTopocentricCRS(v,r,time,CoordinateType::DIRECTION,pos(),"NED",onSurface);
        return (getAxisSwapQuaternion("FSD",dstAxisOrder)*qToFSDFromNED).transformVector(vNED);
    }else{
        throw std::runtime_error("'"+dstAxisOrder+"' is not a valid dstAxisOrder.");
    }
}
Eigen::Vector3d MotionState::dirHtoB(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::string& dstAxisOrder,bool onSurface) const{
    return dirPtoB(dirHtoP(v,r,dstAxisOrder,onSurface),absHtoP(r,r,dstAxisOrder,onSurface));
}
Eigen::Vector3d MotionState::dirBtoH(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::string& dstAxisOrder,bool onSurface) const{
    return dirPtoH(dirBtoP(v,r),absBtoP(r,r),dstAxisOrder,onSurface);
}
Eigen::Vector3d MotionState::dirBtoA(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another) const{
    //self系⇛another系
    return dirPtoA(dirBtoP(v,r),absBtoP(r,r),another);
}
Eigen::Vector3d MotionState::dirAtoB(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another) const{
    //another系⇛self系
    return dirPtoB(dirAtoP(v,r,another),absAtoP(r,r,another));
}
Eigen::Vector3d MotionState::dirPtoA(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another) const{
    //parent系⇛another系
    if(!crs){
        throw std::runtime_error("MotionState without CRS cannot be converted to another CRS.");
    }
    return crs->transformTo(v,r,time,another,CoordinateType::DIRECTION)();
}
Eigen::Vector3d MotionState::dirAtoP(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another) const{
    //another系⇛parent系
    if(!crs){
        throw std::runtime_error("MotionState without CRS cannot be converted to another CRS.");
    }
    return crs->transformFrom(v,r,time,another,CoordinateType::DIRECTION)();
}
Eigen::Vector3d MotionState::dirHtoA(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another,const std::string& dstAxisOrder,bool onSurface) const{
    //horizontal系⇛another系
    return dirPtoA(dirHtoP(v,r,dstAxisOrder,onSurface),absHtoP(r,r,dstAxisOrder,onSurface),another);
}
Eigen::Vector3d MotionState::dirAtoH(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another,const std::string& dstAxisOrder,bool onSurface) const{
    //another系⇛horizontal系
    return dirPtoH(dirAtoP(v,r,another),absAtoP(r,r,another),dstAxisOrder,onSurface);
}
//(7)他のCRSへの変換
std::shared_ptr<CoordinateReferenceSystem> MotionState::getCRS() const {
    return crs;
}
void MotionState::setCRS(const std::shared_ptr<CoordinateReferenceSystem>& newCRS,bool transform){
    //CRSを再設定する。座標変換の有無を第2引数で指定する。
    //元のCRSが存在しなかった場合は変換なし元の値を用いる。
    if(!newCRS){
        throw std::runtime_error("Do not give nullptr as a new CRS of MotionState!");
    }
    if(transform && bool(crs)){
        _q=crs->transformQuaternionTo(q.conjugate(),pos,newCRS).conjugate();
    }
    _pos.setCRS(newCRS,transform && bool(crs));
    _vel.setCRS(newCRS,transform && bool(crs));
    _omega.setCRS(newCRS,transform && bool(crs));
    crs=newCRS;
    validate();
}
MotionState MotionState::transformTo(const std::shared_ptr<CoordinateReferenceSystem>& newCRS) const{
    if(crs==newCRS){
        //同一ならバイパス
        return *this;
    }
    if(!crs){
        throw std::runtime_error("MotionState without CRS cannot be converted to another CRS.");
    }
    if(!newCRS){
        throw std::runtime_error("Do not give nullptr as new parent CRS of MotionState!");
    }
    return std::move(MotionState(
        newCRS,
        time,
        pos.transformTo(newCRS)(),
        vel.transformTo(newCRS)(),
        omega.transformTo(newCRS)(),
        crs->transformQuaternionTo(q.conjugate(),pos,newCRS).conjugate(),
        axisOrder
    ));
}
//(8)方位角、仰角、高度の取得
double MotionState::getAZ() const{
    // 方位角(真北を0、東向きを正)を返す
    if(!isValid()){validate();}
    return _az;
}
double MotionState::getEL() const{
    // 仰角(水平を0、鉛直上向きを正)を返す
    if(!isValid()){validate();}
    return _el;
}
double MotionState::getHeight() const{
    //楕円体高度(幾何高度)を返す。ellipsoid (geometric) height
    // validateで計算しておいたaltを返す。
    if(!isValid()){validate();}
    return alt;
}
double MotionState::getGeoidHeight() const{
    //標高(ジオイド高度)を返す。geoid height (elevation)
    if(!crs){
        return -pos(2); //親座標系が未指定の場合は暗黙に3軸目が下向き正の高度に相当し、かつジオイド面が楕円体表面と一致するものと解釈する。
    }
    return crs->getGeoidHeight(pos);
}

//(9)時刻情報を用いた外挿
MotionState MotionState::extrapolate(const double &dt) const{
    auto cartesianOmega=omega.toCartesian();
    double w=cartesianOmega.norm();
    Quaternion qAft;
    if(w*dt>1e-6){
        Quaternion dq=Quaternion::fromAngle(cartesianOmega/w,w*dt);
        qAft=(dq*q).normalized();
    }else{
        Eigen::VectorXd dq=q.dqdwi()*cartesianOmega*dt;
        qAft=(q+Quaternion(dq)).normalized();
    }
    MotionState ret=MotionState(
        crs,
        time+dt,
        pos.toCartesian()+vel.toCartesian()*dt,
        vel.toCartesian(),
        cartesianOmega,
        qAft,
        axisOrder
    );
    return ret;
}
MotionState MotionState::extrapolateTo(const Time &dstTime) const{
    return extrapolate(dstTime-time);
}

void exportMotionState(py::module &m)
{
    using namespace pybind11::literals;
    bind_stl_container<std::vector<MotionState>>(m,py::module_local(false));
    bind_stl_container<std::map<std::string,MotionState>>(m,py::module_local(false));

    expose_common_class<MotionState>(m,"MotionState")
    .def(py_init())
    .def(
        py_init<const std::shared_ptr<CoordinateReferenceSystem>&,const Time&,const Eigen::Vector3d&,const Eigen::Vector3d&,const Eigen::Vector3d&,const Quaternion&,const std::string&>(),
        "crs"_a,"time"_a,"pos"_a,"vel"_a,"omega"_a,"q"_a,"axisOrder"_a="FSD")
    .def(py_init<const MotionState&>())
    .def(py_init<const nl::json&>())
    DEF_FUNC(MotionState,copy)
    .def("setPos",py::overload_cast<const Eigen::Vector3d&>(&MotionState::setPos))
    .def("setPos",py::overload_cast<const Coordinate&>(&MotionState::setPos))
    .def("setVel",py::overload_cast<const Eigen::Vector3d&>(&MotionState::setVel))
    .def("setVel",py::overload_cast<const Coordinate&>(&MotionState::setVel))
    DEF_FUNC(MotionState,setQ)
    .def("setOmega",py::overload_cast<const Eigen::Vector3d&>(&MotionState::setOmega))
    .def("setOmega",py::overload_cast<const Coordinate&>(&MotionState::setOmega))
    DEF_FUNC(MotionState,setTime)
    DEF_FUNC(MotionState,isValid)
    DEF_FUNC(MotionState,invalidate)
    DEF_FUNC(MotionState,validate)
    //(1)座標軸の変換 (axis conversion in the parent crs)
    DEF_FUNC(MotionState,toCartesian)
    DEF_FUNC(MotionState,toSpherical)
    DEF_FUNC(MotionState,toEllipsoidal)
    DEF_FUNC(MotionState,fromCartesian)
    DEF_FUNC(MotionState,fromSpherical)
    DEF_FUNC(MotionState,fromEllipsoidal)
    //(2)相対位置ベクトルの変換 (transformation of "relative" position)
    .def("relBtoP",py::overload_cast<const Eigen::Vector3d&>(&MotionState::relBtoP,py::const_))
    .def("relPtoB",py::overload_cast<const Eigen::Vector3d&>(&MotionState::relPtoB,py::const_))
    .def("relHtoP",py::overload_cast<const Eigen::Vector3d&,const std::string&,bool>(&MotionState::relHtoP,py::const_),"v"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("relPtoH",py::overload_cast<const Eigen::Vector3d&,const std::string&,bool>(&MotionState::relPtoH,py::const_),"v"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("relHtoB",py::overload_cast<const Eigen::Vector3d&,const std::string&,bool>(&MotionState::relHtoB,py::const_),"v"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("relBtoH",py::overload_cast<const Eigen::Vector3d&,const std::string&,bool>(&MotionState::relBtoH,py::const_),"v"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("relBtoA",py::overload_cast<const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&>(&MotionState::relBtoA,py::const_))
    .def("relAtoB",py::overload_cast<const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&>(&MotionState::relAtoB,py::const_))
    .def("relPtoA",py::overload_cast<const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&>(&MotionState::relPtoA,py::const_))
    .def("relAtoP",py::overload_cast<const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&>(&MotionState::relAtoP,py::const_))
    .def("relHtoA",py::overload_cast<const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&,const std::string&,bool>(&MotionState::relHtoA,py::const_),"v"_a,"another"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("relAtoH",py::overload_cast<const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&,const std::string&,bool>(&MotionState::relAtoH,py::const_),"v"_a,"another"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("relBtoP",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&>(&MotionState::relBtoP,py::const_))
    .def("relPtoB",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&>(&MotionState::relPtoB,py::const_))
    .def("relHtoP",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::string&,bool>(&MotionState::relHtoP,py::const_),"v"_a,"r"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("relPtoH",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::string&,bool>(&MotionState::relPtoH,py::const_),"v"_a,"r"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("relHtoB",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::string&,bool>(&MotionState::relHtoB,py::const_),"v"_a,"r"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("relBtoH",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::string&,bool>(&MotionState::relBtoH,py::const_),"v"_a,"r"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("relBtoA",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&>(&MotionState::relBtoA,py::const_))
    .def("relAtoB",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&>(&MotionState::relAtoB,py::const_))
    .def("relPtoA",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&>(&MotionState::relPtoA,py::const_))
    .def("relAtoP",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&>(&MotionState::relAtoP,py::const_))
    .def("relHtoA",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&,const std::string&,bool>(&MotionState::relHtoA,py::const_),"v"_a,"r"_a,"another"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("relAtoH",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&,const std::string&,bool>(&MotionState::relAtoH,py::const_),"v"_a,"r"_a,"another"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    //(3)絶対位置ベクトルの変換 (transformation of "absolute" position)
    .def("absBtoP",py::overload_cast<const Eigen::Vector3d&>(&MotionState::absBtoP,py::const_))
    .def("absPtoB",py::overload_cast<const Eigen::Vector3d&>(&MotionState::absPtoB,py::const_))
    .def("absHtoP",py::overload_cast<const Eigen::Vector3d&,const std::string&,bool>(&MotionState::absHtoP,py::const_),"v"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("absPtoH",py::overload_cast<const Eigen::Vector3d&,const std::string&,bool>(&MotionState::absPtoH,py::const_),"v"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("absHtoB",py::overload_cast<const Eigen::Vector3d&,const std::string&,bool>(&MotionState::absHtoB,py::const_),"v"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("absBtoH",py::overload_cast<const Eigen::Vector3d&,const std::string&,bool>(&MotionState::absBtoH,py::const_),"v"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("absBtoA",py::overload_cast<const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&>(&MotionState::absBtoA,py::const_))
    .def("absAtoB",py::overload_cast<const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&>(&MotionState::absAtoB,py::const_))
    .def("absPtoA",py::overload_cast<const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&>(&MotionState::absPtoA,py::const_))
    .def("absAtoP",py::overload_cast<const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&>(&MotionState::absAtoP,py::const_))
    .def("absHtoA",py::overload_cast<const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&,const std::string&,bool>(&MotionState::absHtoA,py::const_),"v"_a,"another"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("absAtoH",py::overload_cast<const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&,const std::string&,bool>(&MotionState::absAtoH,py::const_),"v"_a,"another"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("absBtoP",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&>(&MotionState::absBtoP,py::const_))
    .def("absPtoB",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&>(&MotionState::absPtoB,py::const_))
    .def("absHtoP",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::string&,bool>(&MotionState::absHtoP,py::const_),"v"_a,"r"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("absPtoH",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::string&,bool>(&MotionState::absPtoH,py::const_),"v"_a,"r"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("absHtoB",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::string&,bool>(&MotionState::absHtoB,py::const_),"v"_a,"r"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("absBtoH",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::string&,bool>(&MotionState::absBtoH,py::const_),"v"_a,"r"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("absBtoA",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&>(&MotionState::absBtoA,py::const_))
    .def("absAtoB",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&>(&MotionState::absAtoB,py::const_))
    .def("absPtoA",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&>(&MotionState::absPtoA,py::const_))
    .def("absAtoP",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&>(&MotionState::absAtoP,py::const_))
    .def("absHtoA",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&,const std::string&,bool>(&MotionState::absHtoA,py::const_),"v"_a,"r"_a,"another"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("absAtoH",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&,const std::string&,bool>(&MotionState::absAtoH,py::const_),"v"_a,"r"_a,"another"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    //(4)速度ベクトルの変換 (transformation of "observed" velocity)
    .def("velBtoP",py::overload_cast<const Eigen::Vector3d&>(&MotionState::velBtoP,py::const_))
    .def("velPtoB",py::overload_cast<const Eigen::Vector3d&>(&MotionState::velPtoB,py::const_))
    .def("velHtoP",py::overload_cast<const Eigen::Vector3d&,const std::string&,bool>(&MotionState::velHtoP,py::const_),"v"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("velPtoH",py::overload_cast<const Eigen::Vector3d&,const std::string&,bool>(&MotionState::velPtoH,py::const_),"v"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("velHtoB",py::overload_cast<const Eigen::Vector3d&,const std::string&,bool>(&MotionState::velHtoB,py::const_),"v"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("velBtoH",py::overload_cast<const Eigen::Vector3d&,const std::string&,bool>(&MotionState::velBtoH,py::const_),"v"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("velBtoA",py::overload_cast<const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&>(&MotionState::velBtoA,py::const_))
    .def("velAtoB",py::overload_cast<const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&>(&MotionState::velAtoB,py::const_))
    .def("velPtoA",py::overload_cast<const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&>(&MotionState::velPtoA,py::const_))
    .def("velAtoP",py::overload_cast<const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&>(&MotionState::velAtoP,py::const_))
    .def("velHtoA",py::overload_cast<const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&,const std::string&,bool>(&MotionState::velHtoA,py::const_),"v"_a,"another"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("velAtoH",py::overload_cast<const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&,const std::string&,bool>(&MotionState::velAtoH,py::const_),"v"_a,"another"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("velBtoP",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&>(&MotionState::velBtoP,py::const_))
    .def("velPtoB",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&>(&MotionState::velPtoB,py::const_))
    .def("velHtoP",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::string&,bool>(&MotionState::velHtoP,py::const_),"v"_a,"r"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("velPtoH",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::string&,bool>(&MotionState::velPtoH,py::const_),"v"_a,"r"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("velHtoB",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::string&,bool>(&MotionState::velHtoB,py::const_),"v"_a,"r"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("velBtoH",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::string&,bool>(&MotionState::velBtoH,py::const_),"v"_a,"r"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("velBtoA",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&>(&MotionState::velBtoA,py::const_))
    .def("velAtoB",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&>(&MotionState::velAtoB,py::const_))
    .def("velPtoA",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&>(&MotionState::velPtoA,py::const_))
    .def("velAtoP",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&>(&MotionState::velAtoP,py::const_))
    .def("velHtoA",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&,const std::string&,bool>(&MotionState::velHtoA,py::const_),"v"_a,"r"_a,"another"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("velAtoH",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&,const std::string&,bool>(&MotionState::velAtoH,py::const_),"v"_a,"r"_a,"another"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    //(5)角速度ベクトルの変換 (transformation of "observed" angular velocity)
    .def("omegaBtoP",py::overload_cast<const Eigen::Vector3d&>(&MotionState::omegaBtoP,py::const_))
    .def("omegaPtoB",py::overload_cast<const Eigen::Vector3d&>(&MotionState::omegaPtoB,py::const_))
    .def("omegaHtoP",py::overload_cast<const Eigen::Vector3d&,const std::string&,bool>(&MotionState::omegaHtoP,py::const_),"v"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("omegaPtoH",py::overload_cast<const Eigen::Vector3d&,const std::string&,bool>(&MotionState::omegaPtoH,py::const_),"v"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("omegaHtoB",py::overload_cast<const Eigen::Vector3d&,const std::string&,bool>(&MotionState::omegaHtoB,py::const_),"v"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("omegaBtoH",py::overload_cast<const Eigen::Vector3d&,const std::string&,bool>(&MotionState::omegaBtoH,py::const_),"v"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("omegaBtoA",py::overload_cast<const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&>(&MotionState::omegaBtoA,py::const_))
    .def("omegaAtoB",py::overload_cast<const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&>(&MotionState::omegaAtoB,py::const_))
    .def("omegaPtoA",py::overload_cast<const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&>(&MotionState::omegaPtoA,py::const_))
    .def("omegaAtoP",py::overload_cast<const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&>(&MotionState::omegaAtoP,py::const_))
    .def("omegaHtoA",py::overload_cast<const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&,const std::string&,bool>(&MotionState::omegaHtoA,py::const_),"v"_a,"another"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("omegaAtoH",py::overload_cast<const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&,const std::string&,bool>(&MotionState::omegaAtoH,py::const_),"v"_a,"another"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("omegaBtoP",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&>(&MotionState::omegaBtoP,py::const_))
    .def("omegaPtoB",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&>(&MotionState::omegaPtoB,py::const_))
    .def("omegaHtoP",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::string&,bool>(&MotionState::omegaHtoP,py::const_),"v"_a,"r"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("omegaPtoH",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::string&,bool>(&MotionState::omegaPtoH,py::const_),"v"_a,"r"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("omegaHtoB",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::string&,bool>(&MotionState::omegaHtoB,py::const_),"v"_a,"r"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("omegaBtoH",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::string&,bool>(&MotionState::omegaBtoH,py::const_),"v"_a,"r"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("omegaBtoA",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&>(&MotionState::omegaBtoA,py::const_))
    .def("omegaAtoB",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&>(&MotionState::omegaAtoB,py::const_))
    .def("omegaPtoA",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&>(&MotionState::omegaPtoA,py::const_))
    .def("omegaAtoP",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&>(&MotionState::omegaAtoP,py::const_))
    .def("omegaHtoA",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&,const std::string&,bool>(&MotionState::omegaHtoA,py::const_),"v"_a,"r"_a,"another"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("omegaAtoH",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&,const std::string&,bool>(&MotionState::omegaAtoH,py::const_),"v"_a,"r"_a,"another"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    //(6)方向ベクトルの変換 (transformation of direction)
    .def("dirBtoP",py::overload_cast<const Eigen::Vector3d&>(&MotionState::dirBtoP,py::const_))
    .def("dirPtoB",py::overload_cast<const Eigen::Vector3d&>(&MotionState::dirPtoB,py::const_))
    .def("dirHtoP",py::overload_cast<const Eigen::Vector3d&,const std::string&,bool>(&MotionState::dirHtoP,py::const_),"v"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("dirPtoH",py::overload_cast<const Eigen::Vector3d&,const std::string&,bool>(&MotionState::dirPtoH,py::const_),"v"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("dirHtoB",py::overload_cast<const Eigen::Vector3d&,const std::string&,bool>(&MotionState::dirHtoB,py::const_),"v"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("dirBtoH",py::overload_cast<const Eigen::Vector3d&,const std::string&,bool>(&MotionState::dirBtoH,py::const_),"v"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("dirBtoA",py::overload_cast<const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&>(&MotionState::dirBtoA,py::const_))
    .def("dirAtoB",py::overload_cast<const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&>(&MotionState::dirAtoB,py::const_))
    .def("dirPtoA",py::overload_cast<const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&>(&MotionState::dirPtoA,py::const_))
    .def("dirAtoP",py::overload_cast<const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&>(&MotionState::dirAtoP,py::const_))
    .def("dirHtoA",py::overload_cast<const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&,const std::string&,bool>(&MotionState::dirHtoA,py::const_),"v"_a,"another"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("dirAtoH",py::overload_cast<const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&,const std::string&,bool>(&MotionState::dirAtoH,py::const_),"v"_a,"another"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("dirBtoP",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&>(&MotionState::dirBtoP,py::const_))
    .def("dirPtoB",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&>(&MotionState::dirPtoB,py::const_))
    .def("dirHtoP",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::string&,bool>(&MotionState::dirHtoP,py::const_),"v"_a,"r"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("dirPtoH",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::string&,bool>(&MotionState::dirPtoH,py::const_),"v"_a,"r"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("dirHtoB",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::string&,bool>(&MotionState::dirHtoB,py::const_),"v"_a,"r"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("dirBtoH",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::string&,bool>(&MotionState::dirBtoH,py::const_),"v"_a,"r"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("dirBtoA",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&>(&MotionState::dirBtoA,py::const_))
    .def("dirAtoB",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&>(&MotionState::dirAtoB,py::const_))
    .def("dirPtoA",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&>(&MotionState::dirPtoA,py::const_))
    .def("dirAtoP",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&>(&MotionState::dirAtoP,py::const_))
    .def("dirHtoA",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&,const std::string&,bool>(&MotionState::dirHtoA,py::const_),"v"_a,"r"_a,"another"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("dirAtoH",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&,const std::string&,bool>(&MotionState::dirAtoH,py::const_),"v"_a,"r"_a,"another"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    //(7)他のCRSへの変換
    DEF_FUNC(MotionState,getCRS)
    DEF_FUNC(MotionState,setCRS)
    DEF_FUNC(MotionState,transformTo)
    //(8)方位角、仰角、高度の取得
    DEF_FUNC(MotionState,getAZ)
    DEF_FUNC(MotionState,getEL)
    DEF_FUNC(MotionState,getHeight)
    DEF_FUNC(MotionState,getGeoidHeight)
    //(9)時刻情報を用いた外挿
    DEF_FUNC(MotionState,extrapolate)
    DEF_FUNC(MotionState,extrapolateTo)
    //DEF_READWRITE(MotionState,crs)
    .def_property_readonly("time",[](const MotionState& v){return v.time;})
    .def_property_readonly("pos",[](const MotionState& v){return v.pos;})
    .def_property_readonly("vel",[](const MotionState& v){return v.vel;})
    .def_property_readonly("q",[](const MotionState& v){return v.q;})
    .def_property_readonly("omega",[](const MotionState& v){return v.omega;})
    .def_property_readonly("axisOrder",[](const MotionState& v){return v.axisOrder;})
    .def_property_readonly("az",[](const MotionState& v){return v.getAZ();})
    .def_property_readonly("el",[](const MotionState& v){return v.getEL();})
    ;
}

ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
