// Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#include "PhysicalAsset.h"
#include "Utility.h"
#include "SimulationManager.h"
#include "Controller.h"
#include "Agent.h"
#include "Ruler.h"
#include "Units.h"
#include "crs/CoordinateReferenceSystem.h"

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)
using namespace util;

const std::string PhysicalAsset::baseName="PhysicalAsset"; // baseNameを上書き

PhysicalAsset::PhysicalAsset(const nl::json& modelConfig_,const nl::json& instanceConfig_)
:Asset(modelConfig_,instanceConfig_),
 motion(),motion_prev(),
 pos(motion.pos),vel(motion.vel),q(motion.q),omega(motion.omega),
 pos_prev(motion_prev.pos),vel_prev(motion_prev.vel),q_prev(motion_prev.q),omega_prev(motion_prev.omega){
}
void PhysicalAsset::initialize(){
    BaseType::initialize();
    isAlive_=true;
    auto fullName=getFullName();
    if(fullName.find("/")>=0){
        team=fullName.substr(0,fullName.find("/"));
        group=fullName.substr(0,fullName.rfind("/"));
        name=fullName.substr(fullName.rfind("/")+1);
    }else{
        team=group=name=fullName;
    }
    if(instanceConfig.contains("parent")){
        parent=instanceConfig.at("parent");
    }
    try{
        isBoundToParent=instanceConfig.at("isBound").get<bool>();
    }catch(...){
        isBoundToParent=false;
    }
    if(parent.expired()){
        isBoundToParent=false;
    }else{
        removeWhenKilled=parent.lock()->removeWhenKilled;
    }
    setInitialMotionStateFromJson(instanceConfig);
    observables["motion"]=motion;
    observables["isAlive"]=isAlive();
}

void PhysicalAsset::serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full){
    BaseType::serializeInternalState(archive,full);

    if(full){
        if(asrc::core::util::isInputArchive(archive)){
            auto fullName=getFullName();
            if(fullName.find("/")>=0){
                team=fullName.substr(0,fullName.find("/"));
                group=fullName.substr(0,fullName.rfind("/"));
                name=fullName.substr(fullName.rfind("/")+1);
            }else{
                team=group=name=fullName;
            }
        }

        ASRC_SERIALIZE_NVP(archive
            ,parent
            ,port
            ,children
            ,controllers
            ,bodyAxisOrder
            ,agent
        )
    }

    ASRC_SERIALIZE_NVP(archive
        ,isAlive_
        ,isBoundToParent
        ,bodyCRS
        ,motion
        ,motion_prev
        ,bodyAxisOrder
    )
}

bool PhysicalAsset::isAlive() const{
    return isAlive_;
}
std::string PhysicalAsset::getTeam() const{
    return team;
}
std::string PhysicalAsset::getGroup() const{
    return group;
}
std::string PhysicalAsset::getName() const{
    return name;
}
void PhysicalAsset::setAgent(const std::weak_ptr<Agent>& agent_,const std::string &port_){
    agent=agent_;
    port=port_;
}
void PhysicalAsset::perceive(bool inReset){
    observables["motion"]=motion;
    observables["isAlive"]=isAlive();
}
void PhysicalAsset::control(){
}
void PhysicalAsset::behave(){
    motion.setTime(manager->getTime()+interval[SimPhase::BEHAVE]*manager->getBaseTimeStep());
    motion.validate();
    if(bodyCRS){
        bodyCRS->updateByMotionState(motion);
    }
}
void PhysicalAsset::kill(){
    isAlive_=false;
    observables={
        {"isAlive",false}
    };
    commands=nl::json::object();
}
void PhysicalAsset::setManagerAccessor(const std::shared_ptr<EntityManagerAccessor>& ema){
    BaseType::setManagerAccessor(ema);
    manager=std::dynamic_pointer_cast<SimulationManagerAccessorForPhysicalAsset>(ema);
    assert(manager);
}
void PhysicalAsset::setInitialMotionStateFromJson(const nl::json& j){
    if(j.contains("motion")){
        // MotionStateを直接指定する場合
        motion.load_from_json(j.at("motion"));

        //Body座標軸の修正
        if(j.contains("bodyAxisOrder")){
            bodyAxisOrder=j.at("bodyAxisOrder");
        }else if(modelConfig.contains("bodyAxisOrder")){
            bodyAxisOrder=modelConfig.at("bodyAxisOrder");
        }else{
            bodyAxisOrder=motion.axisOrder;
        }
        motion.setAxisOrder(bodyAxisOrder);

        //親座標系の修正
        std::shared_ptr<CoordinateReferenceSystem> parentCRS;
        if(isBoundToParent){
            //親に固定されている場合はそのBody座標系でMotionStateを管理する。
            parentCRS=parent.lock()->getBodyCRS();
        }else{
            //親に固定されていない場合はSimulationManagerの基準座標系でMotionStateを管理する。
            parentCRS=manager->getRootCRS();
        }
        motion.setCRS(parentCRS,true);
    }else{
        // 各状態量を個別に指定する場合
        if(j.contains("bodyAxisOrder")){
            bodyAxisOrder=j.at("bodyAxisOrder");
        }else if(modelConfig.contains("bodyAxisOrder")){
            bodyAxisOrder=modelConfig.at("bodyAxisOrder");
        }else{
            bodyAxisOrder="FSD";
        }
        motion.setTime(manager->getTime());
        motion.setAxisOrder(bodyAxisOrder);
        std::shared_ptr<CoordinateReferenceSystem> parentCRS;
        if(isBoundToParent){
            //親に固定されている場合はそのBody座標系でMotionStateを管理する。
            parentCRS=parent.lock()->getBodyCRS();
        }else{
            //親に固定されていない場合はSimulationManagerの基準座標系でMotionStateを管理する。
            parentCRS=manager->getRootCRS();
        }
        if(j.contains("crs")){
            auto initialStateCRS=createOrGetEntity<CoordinateReferenceSystem>(j.at("crs"));
            motion.setCRS(initialStateCRS,false);
        }else{
            //初期条件の座標系を指定しなかった場合は、親があれば親の座標系、無ければRulerのlocal座標系とみなす
            if(parent.expired()){
                motion.setCRS(manager->getRuler().lock()->getLocalCRS(),false);
            }else{
                motion.setCRS(parentCRS,false);
            }
        }
        // ----座標計算ここから----
        if(j.contains("pos")){
            nl::json pos_j=getValueFromJsonKR(j,"pos",randomGen);
            Eigen::Vector3d pos_v;
            if(pos_j.is_array() && pos_j.size()==3){
                pos_v=pos_j;
            }else{
                throw std::runtime_error("PhysicalAsset::setInitialMotionStateFromJson() failed. 'pos' should be 3-D vector.");
            }
            motion.setPos(pos_v);
        }

        double azInDeg=0.0;
        if(j.contains("heading")){
            azInDeg=getValueFromJsonKR(j,"heading",randomGen);
        }else if(j.contains("azimuth")){
            azInDeg=getValueFromJsonKR(j,"azimuth",randomGen);
        }else if(j.contains("az")){
            azInDeg=getValueFromJsonKR(j,"az",randomGen);
        }
        while(azInDeg>180){azInDeg-=360;}
        while(azInDeg<=-180){azInDeg+=360;}
        double az=deg2rad(azInDeg);
        double cs=cos(az);
        double sn=sin(az);
        double cs_2=cos(az/2);
        double sn_2=sin(az/2);
        if(azInDeg==0){
            cs=1;sn=0;
            cs_2=1;sn_2=0;
        }else if(azInDeg==90){
            cs=0;sn=1;
            cs_2=sqrt(2)/2;
            sn_2=sqrt(2)/2;
        }else if(azInDeg==180){
            cs=-1;sn=0;
            cs_2=0;sn_2=1;
        }else if(azInDeg==-90){
            cs=0;sn=-1;
            cs_2=sqrt(2)/2;
            sn_2=-sqrt(2)/2;
        }
        double elInDeg=0.0;
        if(j.contains("pitch")){
            elInDeg=getValueFromJsonKR(j,"pitch",randomGen);
        }else if(j.contains("elevation")){
            elInDeg=getValueFromJsonKR(j,"elevation",randomGen);
        }else if(j.contains("el")){
            elInDeg=getValueFromJsonKR(j,"el",randomGen);
        }
        while(elInDeg>180){elInDeg-=360;}
        while(elInDeg<=-180){elInDeg+=360;}
        double el=deg2rad(elInDeg);
        double rollInDeg=0.0;
        if(j.contains("roll")){
            rollInDeg=getValueFromJsonKR(j,"roll",randomGen);
        }else if(j.contains("bank")){
            rollInDeg=getValueFromJsonKR(j,"bank",randomGen);
        }
        while(rollInDeg>180){rollInDeg-=360;}
        while(rollInDeg<=-180){rollInDeg+=360;}
        double roll=deg2rad(rollInDeg);

        Quaternion q1=Quaternion::fromAngle(Eigen::Vector3d(0,0,1),az);
        Quaternion q2=Quaternion::fromAngle(Eigen::Vector3d(0,1,0),el);
        Quaternion q3=Quaternion::fromAngle(Eigen::Vector3d(1,0,0),roll);
        Quaternion qToNEDFromFSD = q1*q2*q3;
        Quaternion qToNEDFromBody = qToNEDFromFSD*getAxisSwapQuaternion("FSD",bodyAxisOrder); // [NED <- Body]
        Quaternion qToParentFromNED = motion.getCRS()->getEquivalentQuaternionToTopocentricCRS(pos,"NED").conjugate(); // [Parent <- NED]
        Quaternion q=qToParentFromNED*qToNEDFromBody; // [Parent <- Body]
        motion.setQ(q);

        Eigen::Vector3d vel_v;
        if(j.contains("vel")){
            nl::json vel_j=getValueFromJsonKR(j,"vel",randomGen);
            if(vel_j.is_array() && vel_j.size()==3){
                vel_v=vel_j;
            }else if(vel_j.is_number()){
                double V=vel_j;
                double alphaInDeg=getValueFromJsonKRD(j,"alpha",randomGen,0.0);
                while(alphaInDeg>180){alphaInDeg-=360;}
                while(alphaInDeg<=-180){alphaInDeg+=360;}
                double alpha=deg2rad(alphaInDeg);
                double betaInDeg=getValueFromJsonKRD(j,"beta",randomGen,0.0);
                while(betaInDeg>180){betaInDeg-=360;}
                while(betaInDeg<=-180){betaInDeg+=360;}
                double beta=deg2rad(betaInDeg);
                Eigen::Vector3d velInFSD=V*Eigen::Vector3d(cos(alpha)*cos(beta),sin(beta),sin(alpha)*cos(beta));
                Eigen::Vector3d velInNED=qToNEDFromFSD.transformVector(velInFSD);
                vel_v=motion.velHtoP(
                    velInNED,
                    Eigen::Vector3d::Zero(),
                    "NED",
                    false
                );
            }else{
                throw std::runtime_error("PhysicalAsset::setInitialMotionStateFromJson() failed. 'vel' should be 3-D vector or scalar value.");
            }
        }else{
            vel_v=Eigen::Vector3d::Zero();
        }
        motion.setVel(vel_v);

        Eigen::Vector3d omega_v;
        if(j.contains("omegaB")){
            omega_v=getValueFromJsonKR(j,"omegaB",randomGen);
            omega_v=motion.omegaHtoP(
                qToNEDFromBody.transformVector(omega_v),
                Eigen::Vector3d::Zero(),
                "NED",
                false
            );
        }else if(j.contains("omegaP")){
            omega_v=getValueFromJsonKR(j,"omegaP",randomGen);
        }else{
            omega_v=Eigen::Vector3d::Zero();
        }
        motion.setOmega(omega_v);
        // ----座標計算ここまで----
        motion.setCRS(parentCRS,true);
    }
    motion_prev=motion;
    motion_prev.setVel(motion.vel());
    motion_prev.setPos(motion.pos()-motion.vel()*(interval[SimPhase::BEHAVE]*manager->getBaseTimeStep()));
}

//座標系の取り扱いに関すること (基本的にはMotionStateのメンバ関数を呼ぶだけ)
//(1)座標軸の変換 (axis conversion in the parent crs)
Eigen::Vector3d PhysicalAsset::toCartesian(const Eigen::Vector3d& v) const{
    return motion.toCartesian(v);
}
Eigen::Vector3d PhysicalAsset::toSpherical(const Eigen::Vector3d& v) const{
    return motion.toSpherical(v);
}
Eigen::Vector3d PhysicalAsset::toEllipsoidal(const Eigen::Vector3d& v) const{
    return motion.toEllipsoidal(v);
}
Eigen::Vector3d PhysicalAsset::fromCartesian(const Eigen::Vector3d& v) const{
    return motion.fromCartesian(v);
}
Eigen::Vector3d PhysicalAsset::fromSpherical(const Eigen::Vector3d& v) const{
    return motion.fromSpherical(v);
}
Eigen::Vector3d PhysicalAsset::fromEllipsoidal(const Eigen::Vector3d& v) const{
    return motion.fromEllipsoidal(v);
}
//(2)相対位置ベクトルの変換 (conversion of "relative" position)
Eigen::Vector3d PhysicalAsset::relBtoP(const Eigen::Vector3d &v) const{
    return motion.relBtoP(v);
}
Eigen::Vector3d PhysicalAsset::relPtoB(const Eigen::Vector3d &v) const{
    return motion.relPtoB(v);
}
Eigen::Vector3d PhysicalAsset::relHtoP(const Eigen::Vector3d &v,const std::string& dstAxisOrder,bool onSurface) const{
    return motion.relHtoP(v,dstAxisOrder,onSurface);
}
Eigen::Vector3d PhysicalAsset::relPtoH(const Eigen::Vector3d &v,const std::string& dstAxisOrder,bool onSurface) const{
    return motion.relPtoH(v,dstAxisOrder,onSurface);
}
Eigen::Vector3d PhysicalAsset::relHtoB(const Eigen::Vector3d &v,const std::string& dstAxisOrder,bool onSurface) const{
    return motion.relHtoB(v,dstAxisOrder,onSurface);
}
Eigen::Vector3d PhysicalAsset::relBtoH(const Eigen::Vector3d &v,const std::string& dstAxisOrder,bool onSurface) const{
    return motion.relBtoH(v,dstAxisOrder,onSurface);
}
Eigen::Vector3d PhysicalAsset::relBtoA(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another) const{
    return motion.relBtoA(v,another);
}
Eigen::Vector3d PhysicalAsset::relAtoB(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another) const{
    return motion.relAtoB(v,another);
}
Eigen::Vector3d PhysicalAsset::relPtoA(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another) const{
    return motion.relPtoA(v,another);
}
Eigen::Vector3d PhysicalAsset::relAtoP(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another) const{
    return motion.relAtoP(v,another);
}
Eigen::Vector3d PhysicalAsset::relHtoA(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another,const std::string& dstAxisOrder,bool onSurface) const{
    return motion.relHtoA(v,another,dstAxisOrder,onSurface);
}
Eigen::Vector3d PhysicalAsset::relAtoH(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another,const std::string& dstAxisOrder,bool onSurface) const{
    return motion.relAtoH(v,another,dstAxisOrder,onSurface);
}

Eigen::Vector3d PhysicalAsset::relBtoP(const Eigen::Vector3d &v,const Eigen::Vector3d &r) const{
    return motion.relBtoP(v,r);
}
Eigen::Vector3d PhysicalAsset::relPtoB(const Eigen::Vector3d &v,const Eigen::Vector3d &r) const{
    return motion.relPtoB(v,r);
}
Eigen::Vector3d PhysicalAsset::relHtoP(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::string& dstAxisOrder,bool onSurface) const{
    return motion.relHtoP(v,r,dstAxisOrder,onSurface);
}
Eigen::Vector3d PhysicalAsset::relPtoH(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::string& dstAxisOrder,bool onSurface) const{
    return motion.relPtoH(v,r,dstAxisOrder,onSurface);
}
Eigen::Vector3d PhysicalAsset::relHtoB(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::string& dstAxisOrder,bool onSurface) const{
    return motion.relHtoB(v,r,dstAxisOrder,onSurface);
}
Eigen::Vector3d PhysicalAsset::relBtoH(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::string& dstAxisOrder,bool onSurface) const{
    return motion.relBtoH(v,r,dstAxisOrder,onSurface);
}
Eigen::Vector3d PhysicalAsset::relBtoA(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another) const{
    return motion.relBtoA(v,r,another);
}
Eigen::Vector3d PhysicalAsset::relAtoB(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another) const{
    return motion.relAtoB(v,r,another);
}
Eigen::Vector3d PhysicalAsset::relPtoA(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another) const{
    return motion.relPtoA(v,r,another);
}
Eigen::Vector3d PhysicalAsset::relAtoP(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another) const{
    return motion.relAtoP(v,r,another);
}
Eigen::Vector3d PhysicalAsset::relHtoA(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another,const std::string& dstAxisOrder,bool onSurface) const{
    return motion.relHtoA(v,r,another,dstAxisOrder,onSurface);
}
Eigen::Vector3d PhysicalAsset::relAtoH(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another,const std::string& dstAxisOrder,bool onSurface) const{
    return motion.relAtoH(v,r,another,dstAxisOrder,onSurface);
}

//(3)絶対位置ベクトルの変換 (conversion of "absolute" position)
Eigen::Vector3d PhysicalAsset::absBtoP(const Eigen::Vector3d &v) const{
    return motion.absBtoP(v);
}
Eigen::Vector3d PhysicalAsset::absPtoB(const Eigen::Vector3d &v) const{
    return motion.absPtoB(v);
}
Eigen::Vector3d PhysicalAsset::absHtoP(const Eigen::Vector3d &v,const std::string& dstAxisOrder,bool onSurface) const{
    return motion.absHtoP(v,dstAxisOrder,onSurface);
}
Eigen::Vector3d PhysicalAsset::absPtoH(const Eigen::Vector3d &v,const std::string& dstAxisOrder,bool onSurface) const{
    return motion.absPtoH(v,dstAxisOrder,onSurface);
}
Eigen::Vector3d PhysicalAsset::absHtoB(const Eigen::Vector3d &v,const std::string& dstAxisOrder,bool onSurface) const{
    return motion.absHtoB(v,dstAxisOrder,onSurface);
}
Eigen::Vector3d PhysicalAsset::absBtoH(const Eigen::Vector3d &v,const std::string& dstAxisOrder,bool onSurface) const{
    return motion.absBtoH(v,dstAxisOrder,onSurface);
}
Eigen::Vector3d PhysicalAsset::absBtoA(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another) const{
    return motion.absBtoA(v,another);
}
Eigen::Vector3d PhysicalAsset::absAtoB(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another) const{
    return motion.absAtoB(v,another);
}
Eigen::Vector3d PhysicalAsset::absPtoA(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another) const{
    return motion.absPtoA(v,another);
}
Eigen::Vector3d PhysicalAsset::absAtoP(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another) const{
    return motion.absAtoP(v,another);
}
Eigen::Vector3d PhysicalAsset::absHtoA(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another,const std::string& dstAxisOrder,bool onSurface) const{
    return motion.absHtoA(v,another,dstAxisOrder,onSurface);
}
Eigen::Vector3d PhysicalAsset::absAtoH(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another,const std::string& dstAxisOrder,bool onSurface) const{
    return motion.absAtoH(v,another,dstAxisOrder,onSurface);
}

Eigen::Vector3d PhysicalAsset::absBtoP(const Eigen::Vector3d &v,const Eigen::Vector3d &r) const{
    return motion.absBtoP(v,r);
}
Eigen::Vector3d PhysicalAsset::absPtoB(const Eigen::Vector3d &v,const Eigen::Vector3d &r) const{
    return motion.absPtoB(v,r);
}
Eigen::Vector3d PhysicalAsset::absHtoP(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::string& dstAxisOrder,bool onSurface) const{
    return motion.absHtoP(v,r,dstAxisOrder,onSurface);
}
Eigen::Vector3d PhysicalAsset::absPtoH(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::string& dstAxisOrder,bool onSurface) const{
    return motion.absPtoH(v,r,dstAxisOrder,onSurface);
}
Eigen::Vector3d PhysicalAsset::absHtoB(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::string& dstAxisOrder,bool onSurface) const{
    return motion.absHtoB(v,r,dstAxisOrder,onSurface);
}
Eigen::Vector3d PhysicalAsset::absBtoH(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::string& dstAxisOrder,bool onSurface) const{
    return motion.absBtoH(v,r,dstAxisOrder,onSurface);
}
Eigen::Vector3d PhysicalAsset::absBtoA(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another) const{
    return motion.absBtoA(v,r,another);
}
Eigen::Vector3d PhysicalAsset::absAtoB(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another) const{
    return motion.absAtoB(v,r,another);
}
Eigen::Vector3d PhysicalAsset::absPtoA(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another) const{
    return motion.absPtoA(v,r,another);
}
Eigen::Vector3d PhysicalAsset::absAtoP(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another) const{
    return motion.absAtoP(v,r,another);
}
Eigen::Vector3d PhysicalAsset::absHtoA(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another,const std::string& dstAxisOrder,bool onSurface) const{
    return motion.absHtoA(v,r,another,dstAxisOrder,onSurface);
}
Eigen::Vector3d PhysicalAsset::absAtoH(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another,const std::string& dstAxisOrder,bool onSurface) const{
    return motion.absAtoH(v,r,another,dstAxisOrder,onSurface);
}

//(4)速度ベクトルの変換 (conversion of "observed" velocity)
Eigen::Vector3d PhysicalAsset::velBtoP(const Eigen::Vector3d &v) const{
    return motion.velBtoP(v);
}
Eigen::Vector3d PhysicalAsset::velPtoB(const Eigen::Vector3d &v) const{
    return motion.velPtoB(v);
}
Eigen::Vector3d PhysicalAsset::velHtoP(const Eigen::Vector3d &v,const std::string& dstAxisOrder,bool onSurface) const{
    return motion.velHtoP(v,dstAxisOrder,onSurface);
}
Eigen::Vector3d PhysicalAsset::velPtoH(const Eigen::Vector3d &v,const std::string& dstAxisOrder,bool onSurface) const{
    return motion.velPtoH(v,dstAxisOrder,onSurface);
}
Eigen::Vector3d PhysicalAsset::velHtoB(const Eigen::Vector3d &v,const std::string& dstAxisOrder,bool onSurface) const{
    return motion.velHtoB(v,dstAxisOrder,onSurface);
}
Eigen::Vector3d PhysicalAsset::velBtoH(const Eigen::Vector3d &v,const std::string& dstAxisOrder,bool onSurface) const{
    return motion.velBtoH(v,dstAxisOrder,onSurface);
}
Eigen::Vector3d PhysicalAsset::velBtoA(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another) const{
    return motion.velBtoA(v,another);
}
Eigen::Vector3d PhysicalAsset::velAtoB(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another) const{
    return motion.velAtoB(v,another);
}
Eigen::Vector3d PhysicalAsset::velPtoA(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another) const{
    return motion.velPtoA(v,another);
}
Eigen::Vector3d PhysicalAsset::velAtoP(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another) const{
    return motion.velAtoP(v,another);
}
Eigen::Vector3d PhysicalAsset::velHtoA(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another,const std::string& dstAxisOrder,bool onSurface) const{
    return motion.velHtoA(v,another,dstAxisOrder,onSurface);
}
Eigen::Vector3d PhysicalAsset::velAtoH(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another,const std::string& dstAxisOrder,bool onSurface) const{
    return motion.velAtoH(v,another,dstAxisOrder,onSurface);
}

Eigen::Vector3d PhysicalAsset::velBtoP(const Eigen::Vector3d &v,const Eigen::Vector3d &r) const{
    return motion.velBtoP(v,r);
}
Eigen::Vector3d PhysicalAsset::velPtoB(const Eigen::Vector3d &v,const Eigen::Vector3d &r) const{
    return motion.velPtoB(v,r);
}
Eigen::Vector3d PhysicalAsset::velHtoP(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::string& dstAxisOrder,bool onSurface) const{
    return motion.velHtoP(v,r,dstAxisOrder,onSurface);
}
Eigen::Vector3d PhysicalAsset::velPtoH(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::string& dstAxisOrder,bool onSurface) const{
    return motion.velPtoH(v,r,dstAxisOrder,onSurface);
}
Eigen::Vector3d PhysicalAsset::velHtoB(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::string& dstAxisOrder,bool onSurface) const{
    return motion.velHtoB(v,r,dstAxisOrder,onSurface);
}
Eigen::Vector3d PhysicalAsset::velBtoH(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::string& dstAxisOrder,bool onSurface) const{
    return motion.velBtoH(v,r,dstAxisOrder,onSurface);
}
Eigen::Vector3d PhysicalAsset::velBtoA(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another) const{
    return motion.velBtoA(v,r,another);
}
Eigen::Vector3d PhysicalAsset::velAtoB(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another) const{
    return motion.velAtoB(v,r,another);
}
Eigen::Vector3d PhysicalAsset::velPtoA(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another) const{
    return motion.velPtoA(v,r,another);
}
Eigen::Vector3d PhysicalAsset::velAtoP(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another) const{
    return motion.velAtoP(v,r,another);
}
Eigen::Vector3d PhysicalAsset::velHtoA(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another,const std::string& dstAxisOrder,bool onSurface) const{
    return motion.velHtoA(v,r,another,dstAxisOrder,onSurface);
}
Eigen::Vector3d PhysicalAsset::velAtoH(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another,const std::string& dstAxisOrder,bool onSurface) const{
    return motion.velAtoH(v,r,another,dstAxisOrder,onSurface);
}

//(5)角速度ベクトルの変換 (conversion of "observed" angular velocity)
Eigen::Vector3d PhysicalAsset::omegaBtoP(const Eigen::Vector3d &v) const{
    return motion.omegaBtoP(v);
}
Eigen::Vector3d PhysicalAsset::omegaPtoB(const Eigen::Vector3d &v) const{
    return motion.omegaPtoB(v);
}
Eigen::Vector3d PhysicalAsset::omegaHtoP(const Eigen::Vector3d &v,const std::string& dstAxisOrder,bool onSurface) const{
    return motion.omegaHtoP(v,dstAxisOrder,onSurface);
}
Eigen::Vector3d PhysicalAsset::omegaPtoH(const Eigen::Vector3d &v,const std::string& dstAxisOrder,bool onSurface) const{
    return motion.omegaPtoH(v,dstAxisOrder,onSurface);
}
Eigen::Vector3d PhysicalAsset::omegaHtoB(const Eigen::Vector3d &v,const std::string& dstAxisOrder,bool onSurface) const{
    return motion.omegaHtoB(v,dstAxisOrder,onSurface);
}
Eigen::Vector3d PhysicalAsset::omegaBtoH(const Eigen::Vector3d &v,const std::string& dstAxisOrder,bool onSurface) const{
    return motion.omegaBtoH(v,dstAxisOrder,onSurface);
}
Eigen::Vector3d PhysicalAsset::omegaBtoA(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another) const{
    return motion.omegaBtoA(v,another);
}
Eigen::Vector3d PhysicalAsset::omegaAtoB(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another) const{
    return motion.omegaAtoB(v,another);
}
Eigen::Vector3d PhysicalAsset::omegaPtoA(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another) const{
    return motion.omegaPtoA(v,another);
}
Eigen::Vector3d PhysicalAsset::omegaAtoP(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another) const{
    return motion.omegaAtoP(v,another);
}
Eigen::Vector3d PhysicalAsset::omegaHtoA(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another,const std::string& dstAxisOrder,bool onSurface) const{
    return motion.omegaHtoA(v,another,dstAxisOrder,onSurface);
}
Eigen::Vector3d PhysicalAsset::omegaAtoH(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another,const std::string& dstAxisOrder,bool onSurface) const{
    return motion.omegaAtoH(v,another,dstAxisOrder,onSurface);
}

Eigen::Vector3d PhysicalAsset::omegaBtoP(const Eigen::Vector3d &v,const Eigen::Vector3d &r) const{
    return motion.omegaBtoP(v,r);
}
Eigen::Vector3d PhysicalAsset::omegaPtoB(const Eigen::Vector3d &v,const Eigen::Vector3d &r) const{
    return motion.omegaPtoB(v,r);
}
Eigen::Vector3d PhysicalAsset::omegaHtoP(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::string& dstAxisOrder,bool onSurface) const{
    return motion.omegaHtoP(v,r,dstAxisOrder,onSurface);
}
Eigen::Vector3d PhysicalAsset::omegaPtoH(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::string& dstAxisOrder,bool onSurface) const{
    return motion.omegaPtoH(v,r,dstAxisOrder,onSurface);
}
Eigen::Vector3d PhysicalAsset::omegaHtoB(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::string& dstAxisOrder,bool onSurface) const{
    return motion.omegaHtoB(v,r,dstAxisOrder,onSurface);
}
Eigen::Vector3d PhysicalAsset::omegaBtoH(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::string& dstAxisOrder,bool onSurface) const{
    return motion.omegaBtoH(v,r,dstAxisOrder,onSurface);
}
Eigen::Vector3d PhysicalAsset::omegaBtoA(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another) const{
    return motion.omegaBtoA(v,r,another);
}
Eigen::Vector3d PhysicalAsset::omegaAtoB(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another) const{
    return motion.omegaAtoB(v,r,another);
}
Eigen::Vector3d PhysicalAsset::omegaPtoA(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another) const{
    return motion.omegaPtoA(v,r,another);
}
Eigen::Vector3d PhysicalAsset::omegaAtoP(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another) const{
    return motion.omegaAtoP(v,r,another);
}
Eigen::Vector3d PhysicalAsset::omegaHtoA(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another,const std::string& dstAxisOrder,bool onSurface) const{
    return motion.omegaHtoA(v,r,another,dstAxisOrder,onSurface);
}
Eigen::Vector3d PhysicalAsset::omegaAtoH(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another,const std::string& dstAxisOrder,bool onSurface) const{
    return motion.omegaAtoH(v,r,another,dstAxisOrder,onSurface);
}

//(6)方向ベクトルの変換 (transformation of direction)
Eigen::Vector3d PhysicalAsset::dirBtoP(const Eigen::Vector3d &v) const{
    return motion.dirBtoP(v);
}
Eigen::Vector3d PhysicalAsset::dirPtoB(const Eigen::Vector3d &v) const{
    return motion.dirPtoB(v);
}
Eigen::Vector3d PhysicalAsset::dirHtoP(const Eigen::Vector3d &v,const std::string& dstAxisOrder,bool onSurface) const{
    return motion.dirHtoP(v,dstAxisOrder,onSurface);
}
Eigen::Vector3d PhysicalAsset::dirPtoH(const Eigen::Vector3d &v,const std::string& dstAxisOrder,bool onSurface) const{
    return motion.dirPtoH(v,dstAxisOrder,onSurface);
}
Eigen::Vector3d PhysicalAsset::dirHtoB(const Eigen::Vector3d &v,const std::string& dstAxisOrder,bool onSurface) const{
    return motion.dirHtoB(v,dstAxisOrder,onSurface);
}
Eigen::Vector3d PhysicalAsset::dirBtoH(const Eigen::Vector3d &v,const std::string& dstAxisOrder,bool onSurface) const{
    return motion.dirBtoH(v,dstAxisOrder,onSurface);
}
Eigen::Vector3d PhysicalAsset::dirBtoA(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another) const{
    return motion.dirBtoA(v,another);
}
Eigen::Vector3d PhysicalAsset::dirAtoB(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another) const{
    return motion.dirAtoB(v,another);
}
Eigen::Vector3d PhysicalAsset::dirPtoA(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another) const{
    return motion.dirPtoA(v,another);
}
Eigen::Vector3d PhysicalAsset::dirAtoP(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another) const{
    return motion.dirAtoP(v,another);
}
Eigen::Vector3d PhysicalAsset::dirHtoA(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another,const std::string& dstAxisOrder,bool onSurface) const{
    return motion.dirHtoA(v,another,dstAxisOrder,onSurface);
}
Eigen::Vector3d PhysicalAsset::dirAtoH(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another,const std::string& dstAxisOrder,bool onSurface) const{
    return motion.dirAtoH(v,another,dstAxisOrder,onSurface);
}

Eigen::Vector3d PhysicalAsset::dirBtoP(const Eigen::Vector3d &v,const Eigen::Vector3d &r) const{
    return motion.dirBtoP(v,r);
}
Eigen::Vector3d PhysicalAsset::dirPtoB(const Eigen::Vector3d &v,const Eigen::Vector3d &r) const{
    return motion.dirPtoB(v,r);
}
Eigen::Vector3d PhysicalAsset::dirHtoP(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::string& dstAxisOrder,bool onSurface) const{
    return motion.dirHtoP(v,r,dstAxisOrder,onSurface);
}
Eigen::Vector3d PhysicalAsset::dirPtoH(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::string& dstAxisOrder,bool onSurface) const{
    return motion.dirPtoH(v,r,dstAxisOrder,onSurface);
}
Eigen::Vector3d PhysicalAsset::dirHtoB(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::string& dstAxisOrder,bool onSurface) const{
    return motion.dirHtoB(v,r,dstAxisOrder,onSurface);
}
Eigen::Vector3d PhysicalAsset::dirBtoH(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::string& dstAxisOrder,bool onSurface) const{
    return motion.dirBtoH(v,r,dstAxisOrder,onSurface);
}
Eigen::Vector3d PhysicalAsset::dirBtoA(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another) const{
    return motion.dirBtoA(v,r,another);
}
Eigen::Vector3d PhysicalAsset::dirAtoB(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another) const{
    return motion.dirAtoB(v,r,another);
}
Eigen::Vector3d PhysicalAsset::dirPtoA(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another) const{
    return motion.dirPtoA(v,r,another);
}
Eigen::Vector3d PhysicalAsset::dirAtoP(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another) const{
    return motion.dirAtoP(v,r,another);
}
Eigen::Vector3d PhysicalAsset::dirHtoA(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another,const std::string& dstAxisOrder,bool onSurface) const{
    return motion.dirHtoA(v,r,another,dstAxisOrder,onSurface);
}
Eigen::Vector3d PhysicalAsset::dirAtoH(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another,const std::string& dstAxisOrder,bool onSurface) const{
    return motion.dirAtoH(v,r,another,dstAxisOrder,onSurface);
}

//(7)方位角、仰角、高度の取得
double PhysicalAsset::getAZ() const{
    // 方位角(真北を0、東向きを正)を返す
    return motion.getAZ();
}
double PhysicalAsset::getEL() const{
    // 仰角(水平を0、鉛直上向きを正)を返す
    return motion.getEL();
}
double PhysicalAsset::getHeight() const{
    //楕円体高度(幾何高度)を返す。ellipsoid (geometric) height
    return motion.getHeight();
}
double PhysicalAsset::getGeoidHeight() const{
    //標高(ジオイド高度)を返す。geoid height (elevation)
    return motion.getGeoidHeight();
}

//自身の座標情報
std::shared_ptr<CoordinateReferenceSystem> PhysicalAsset::getParentCRS(){
    return motion.getCRS();
}
std::shared_ptr<AffineCRS> PhysicalAsset::getBodyCRS(){
    if(bodyCRS){
        return bodyCRS;
    }
    if(!getParentCRS()){
        return nullptr;
    }
    bodyCRS=getParentCRS()->createAffineCRS(motion,bodyAxisOrder,isEpisodic());
    return bodyCRS;
}
void PhysicalAsset::bindToParent(){
    //親があるなら固定する。親がないなら何も起こさない(このクラスとしては例外も投げない)
    if(!parent.expired()){
        isBoundToParent=true;
        auto newCRS=parent.lock()->getBodyCRS();
        motion_prev.setCRS(newCRS,true);
        motion.setCRS(newCRS,true);
        if(bodyCRS){
            bodyCRS->setBaseCRS(newCRS);
            bodyCRS->updateByMotionState(motion);
        }
    }
}
void PhysicalAsset::unbindFromParent(){
    //親があるなら固定を解除する。親がないなら何も起こさない(このクラスとしては例外も投げない)
    if(!parent.expired()){
        isBoundToParent=false;
        auto newCRS=manager->getRootCRS();
        motion_prev.setCRS(newCRS,true);
        motion.setCRS(newCRS,true);
        if(bodyCRS){
            bodyCRS->setBaseCRS(newCRS);
            bodyCRS->updateByMotionState(motion);
        }
    }
}
std::shared_ptr<EntityAccessor> PhysicalAsset::getAccessorImpl(){
    return std::make_shared<PhysicalAssetAccessor>(getShared<PhysicalAsset>(this->shared_from_this()));
}
bool PhysicalAsset::generateCommunicationBuffer(const std::string& name_,const nl::json& participants_,const nl::json& inviteOnRequest_){
    return manager->generateCommunicationBuffer(name_,participants_,inviteOnRequest_);
}
PhysicalAssetAccessor::PhysicalAssetAccessor(const std::shared_ptr<PhysicalAsset>& pa)
:AssetAccessor(pa)
,physicalAsset(pa){
}

void exportPhysicalAsset(py::module &m, const std::shared_ptr<asrc::core::FactoryHelper>& factoryHelper)
{
    using namespace pybind11::literals;
    FACTORY_ADD_BASE(PhysicalAsset,{"Controller","PhysicalAsset","Callback","!AnyOthers"},{"!Agent","AnyOthers"})

    bind_stl_container<std::map<std::string,std::shared_ptr<PhysicalAssetAccessor>>>(m,py::module_local(false));

    expose_entity_subclass<PhysicalAsset>(m,"PhysicalAsset")
    DEF_FUNC(PhysicalAsset,setInitialMotionStateFromJson)
    //(1)座標軸の変換 (axis conversion in the parent crs)
    DEF_FUNC(PhysicalAsset,toCartesian)
    DEF_FUNC(PhysicalAsset,toSpherical)
    DEF_FUNC(PhysicalAsset,toEllipsoidal)
    DEF_FUNC(PhysicalAsset,fromCartesian)
    DEF_FUNC(PhysicalAsset,fromSpherical)
    DEF_FUNC(PhysicalAsset,fromEllipsoidal)
    //(2)相対位置ベクトルの変換 (conversion of "relative" position)
    .def("relBtoP",py::overload_cast<const Eigen::Vector3d&>(&PhysicalAsset::relBtoP,py::const_))
    .def("relPtoB",py::overload_cast<const Eigen::Vector3d&>(&PhysicalAsset::relPtoB,py::const_))
    .def("relHtoP",py::overload_cast<const Eigen::Vector3d&,const std::string&,bool>(&PhysicalAsset::relHtoP,py::const_),"v"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("relPtoH",py::overload_cast<const Eigen::Vector3d&,const std::string&,bool>(&PhysicalAsset::relPtoH,py::const_),"v"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("relHtoB",py::overload_cast<const Eigen::Vector3d&,const std::string&,bool>(&PhysicalAsset::relHtoB,py::const_),"v"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("relBtoH",py::overload_cast<const Eigen::Vector3d&,const std::string&,bool>(&PhysicalAsset::relBtoH,py::const_),"v"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("relBtoA",py::overload_cast<const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&>(&PhysicalAsset::relBtoA,py::const_))
    .def("relAtoB",py::overload_cast<const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&>(&PhysicalAsset::relAtoB,py::const_))
    .def("relPtoA",py::overload_cast<const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&>(&PhysicalAsset::relPtoA,py::const_))
    .def("relAtoP",py::overload_cast<const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&>(&PhysicalAsset::relAtoP,py::const_))
    .def("relHtoA",py::overload_cast<const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&,const std::string&,bool>(&PhysicalAsset::relHtoA,py::const_),"v"_a,"another"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("relAtoH",py::overload_cast<const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&,const std::string&,bool>(&PhysicalAsset::relAtoH,py::const_),"v"_a,"another"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("relBtoP",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&>(&PhysicalAsset::relBtoP,py::const_))
    .def("relPtoB",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&>(&PhysicalAsset::relPtoB,py::const_))
    .def("relHtoP",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::string&,bool>(&PhysicalAsset::relHtoP,py::const_),"v"_a,"r"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("relPtoH",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::string&,bool>(&PhysicalAsset::relPtoH,py::const_),"v"_a,"r"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("relHtoB",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::string&,bool>(&PhysicalAsset::relHtoB,py::const_),"v"_a,"r"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("relBtoH",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::string&,bool>(&PhysicalAsset::relBtoH,py::const_),"v"_a,"r"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("relBtoA",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&>(&PhysicalAsset::relBtoA,py::const_))
    .def("relAtoB",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&>(&PhysicalAsset::relAtoB,py::const_))
    .def("relPtoA",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&>(&PhysicalAsset::relPtoA,py::const_))
    .def("relAtoP",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&>(&PhysicalAsset::relAtoP,py::const_))
    .def("relHtoA",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&,const std::string&,bool>(&PhysicalAsset::relHtoA,py::const_),"v"_a,"r"_a,"another"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("relAtoH",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&,const std::string&,bool>(&PhysicalAsset::relAtoH,py::const_),"v"_a,"r"_a,"another"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    //(3)絶対位置ベクトルの変換 (conversion of "absolute" position)
    .def("absBtoP",py::overload_cast<const Eigen::Vector3d&>(&PhysicalAsset::absBtoP,py::const_))
    .def("absPtoB",py::overload_cast<const Eigen::Vector3d&>(&PhysicalAsset::absPtoB,py::const_))
    .def("absHtoP",py::overload_cast<const Eigen::Vector3d&,const std::string&,bool>(&PhysicalAsset::absHtoP,py::const_),"v"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("absPtoH",py::overload_cast<const Eigen::Vector3d&,const std::string&,bool>(&PhysicalAsset::absPtoH,py::const_),"v"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("absHtoB",py::overload_cast<const Eigen::Vector3d&,const std::string&,bool>(&PhysicalAsset::absHtoB,py::const_),"v"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("absBtoH",py::overload_cast<const Eigen::Vector3d&,const std::string&,bool>(&PhysicalAsset::absBtoH,py::const_),"v"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("absBtoA",py::overload_cast<const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&>(&PhysicalAsset::absBtoA,py::const_))
    .def("absAtoB",py::overload_cast<const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&>(&PhysicalAsset::absAtoB,py::const_))
    .def("absPtoA",py::overload_cast<const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&>(&PhysicalAsset::absPtoA,py::const_))
    .def("absAtoP",py::overload_cast<const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&>(&PhysicalAsset::absAtoP,py::const_))
    .def("absHtoA",py::overload_cast<const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&,const std::string&,bool>(&PhysicalAsset::absHtoA,py::const_),"v"_a,"another"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("absAtoH",py::overload_cast<const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&,const std::string&,bool>(&PhysicalAsset::absAtoH,py::const_),"v"_a,"another"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("absBtoP",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&>(&PhysicalAsset::absBtoP,py::const_))
    .def("absPtoB",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&>(&PhysicalAsset::absPtoB,py::const_))
    .def("absHtoP",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::string&,bool>(&PhysicalAsset::absHtoP,py::const_),"v"_a,"r"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("absPtoH",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::string&,bool>(&PhysicalAsset::absPtoH,py::const_),"v"_a,"r"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("absHtoB",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::string&,bool>(&PhysicalAsset::absHtoB,py::const_),"v"_a,"r"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("absBtoH",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::string&,bool>(&PhysicalAsset::absBtoH,py::const_),"v"_a,"r"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("absBtoA",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&>(&PhysicalAsset::absBtoA,py::const_))
    .def("absAtoB",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&>(&PhysicalAsset::absAtoB,py::const_))
    .def("absPtoA",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&>(&PhysicalAsset::absPtoA,py::const_))
    .def("absAtoP",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&>(&PhysicalAsset::absAtoP,py::const_))
    .def("absHtoA",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&,const std::string&,bool>(&PhysicalAsset::absHtoA,py::const_),"v"_a,"r"_a,"another"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("absAtoH",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&,const std::string&,bool>(&PhysicalAsset::absAtoH,py::const_),"v"_a,"r"_a,"another"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    //(4)速度ベクトルの変換 (conversion of "observed" velocity)
    .def("velBtoP",py::overload_cast<const Eigen::Vector3d&>(&PhysicalAsset::velBtoP,py::const_))
    .def("velPtoB",py::overload_cast<const Eigen::Vector3d&>(&PhysicalAsset::velPtoB,py::const_))
    .def("velHtoP",py::overload_cast<const Eigen::Vector3d&,const std::string&,bool>(&PhysicalAsset::velHtoP,py::const_),"v"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("velPtoH",py::overload_cast<const Eigen::Vector3d&,const std::string&,bool>(&PhysicalAsset::velPtoH,py::const_),"v"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("velHtoB",py::overload_cast<const Eigen::Vector3d&,const std::string&,bool>(&PhysicalAsset::velHtoB,py::const_),"v"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("velBtoH",py::overload_cast<const Eigen::Vector3d&,const std::string&,bool>(&PhysicalAsset::velBtoH,py::const_),"v"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("velBtoA",py::overload_cast<const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&>(&PhysicalAsset::velBtoA,py::const_))
    .def("velAtoB",py::overload_cast<const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&>(&PhysicalAsset::velAtoB,py::const_))
    .def("velPtoA",py::overload_cast<const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&>(&PhysicalAsset::velPtoA,py::const_))
    .def("velAtoP",py::overload_cast<const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&>(&PhysicalAsset::velAtoP,py::const_))
    .def("velHtoA",py::overload_cast<const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&,const std::string&,bool>(&PhysicalAsset::velHtoA,py::const_),"v"_a,"another"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("velAtoH",py::overload_cast<const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&,const std::string&,bool>(&PhysicalAsset::velAtoH,py::const_),"v"_a,"another"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("velBtoP",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&>(&PhysicalAsset::velBtoP,py::const_))
    .def("velPtoB",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&>(&PhysicalAsset::velPtoB,py::const_))
    .def("velHtoP",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::string&,bool>(&PhysicalAsset::velHtoP,py::const_),"v"_a,"r"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("velPtoH",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::string&,bool>(&PhysicalAsset::velPtoH,py::const_),"v"_a,"r"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("velHtoB",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::string&,bool>(&PhysicalAsset::velHtoB,py::const_),"v"_a,"r"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("velBtoH",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::string&,bool>(&PhysicalAsset::velBtoH,py::const_),"v"_a,"r"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("velBtoA",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&>(&PhysicalAsset::velBtoA,py::const_))
    .def("velAtoB",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&>(&PhysicalAsset::velAtoB,py::const_))
    .def("velPtoA",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&>(&PhysicalAsset::velPtoA,py::const_))
    .def("velAtoP",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&>(&PhysicalAsset::velAtoP,py::const_))
    .def("velHtoA",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&,const std::string&,bool>(&PhysicalAsset::velHtoA,py::const_),"v"_a,"r"_a,"another"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("velAtoH",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&,const std::string&,bool>(&PhysicalAsset::velAtoH,py::const_),"v"_a,"r"_a,"another"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    //(5)角速度ベクトルの変換 (conversion of "observed" angular velocity)
    .def("omegaBtoP",py::overload_cast<const Eigen::Vector3d&>(&PhysicalAsset::omegaBtoP,py::const_))
    .def("omegaPtoB",py::overload_cast<const Eigen::Vector3d&>(&PhysicalAsset::omegaPtoB,py::const_))
    .def("omegaHtoP",py::overload_cast<const Eigen::Vector3d&,const std::string&,bool>(&PhysicalAsset::omegaHtoP,py::const_),"v"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("omegaPtoH",py::overload_cast<const Eigen::Vector3d&,const std::string&,bool>(&PhysicalAsset::omegaPtoH,py::const_),"v"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("omegaHtoB",py::overload_cast<const Eigen::Vector3d&,const std::string&,bool>(&PhysicalAsset::omegaHtoB,py::const_),"v"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("omegaBtoH",py::overload_cast<const Eigen::Vector3d&,const std::string&,bool>(&PhysicalAsset::omegaBtoH,py::const_),"v"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("omegaBtoA",py::overload_cast<const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&>(&PhysicalAsset::omegaBtoA,py::const_))
    .def("omegaAtoB",py::overload_cast<const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&>(&PhysicalAsset::omegaAtoB,py::const_))
    .def("omegaPtoA",py::overload_cast<const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&>(&PhysicalAsset::omegaPtoA,py::const_))
    .def("omegaAtoP",py::overload_cast<const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&>(&PhysicalAsset::omegaAtoP,py::const_))
    .def("omegaHtoA",py::overload_cast<const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&,const std::string&,bool>(&PhysicalAsset::omegaHtoA,py::const_),"v"_a,"another"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("omegaAtoH",py::overload_cast<const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&,const std::string&,bool>(&PhysicalAsset::omegaAtoH,py::const_),"v"_a,"another"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("omegaBtoP",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&>(&PhysicalAsset::omegaBtoP,py::const_))
    .def("omegaPtoB",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&>(&PhysicalAsset::omegaPtoB,py::const_))
    .def("omegaHtoP",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::string&,bool>(&PhysicalAsset::omegaHtoP,py::const_),"v"_a,"r"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("omegaPtoH",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::string&,bool>(&PhysicalAsset::omegaPtoH,py::const_),"v"_a,"r"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("omegaHtoB",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::string&,bool>(&PhysicalAsset::omegaHtoB,py::const_),"v"_a,"r"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("omegaBtoH",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::string&,bool>(&PhysicalAsset::omegaBtoH,py::const_),"v"_a,"r"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("omegaBtoA",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&>(&PhysicalAsset::omegaBtoA,py::const_))
    .def("omegaAtoB",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&>(&PhysicalAsset::omegaAtoB,py::const_))
    .def("omegaPtoA",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&>(&PhysicalAsset::omegaPtoA,py::const_))
    .def("omegaAtoP",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&>(&PhysicalAsset::omegaAtoP,py::const_))
    .def("omegaHtoA",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&,const std::string&,bool>(&PhysicalAsset::omegaHtoA,py::const_),"v"_a,"r"_a,"another"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("omegaAtoH",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&,const std::string&,bool>(&PhysicalAsset::omegaAtoH,py::const_),"v"_a,"r"_a,"another"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    //(6)方向ベクトルの変換 (transformation of direction)
    .def("dirBtoP",py::overload_cast<const Eigen::Vector3d&>(&PhysicalAsset::dirBtoP,py::const_))
    .def("dirPtoB",py::overload_cast<const Eigen::Vector3d&>(&PhysicalAsset::dirPtoB,py::const_))
    .def("dirHtoP",py::overload_cast<const Eigen::Vector3d&,const std::string&,bool>(&PhysicalAsset::dirHtoP,py::const_),"v"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("dirPtoH",py::overload_cast<const Eigen::Vector3d&,const std::string&,bool>(&PhysicalAsset::dirPtoH,py::const_),"v"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("dirHtoB",py::overload_cast<const Eigen::Vector3d&,const std::string&,bool>(&PhysicalAsset::dirHtoB,py::const_),"v"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("dirBtoH",py::overload_cast<const Eigen::Vector3d&,const std::string&,bool>(&PhysicalAsset::dirBtoH,py::const_),"v"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("dirBtoA",py::overload_cast<const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&>(&PhysicalAsset::dirBtoA,py::const_))
    .def("dirAtoB",py::overload_cast<const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&>(&PhysicalAsset::dirAtoB,py::const_))
    .def("dirPtoA",py::overload_cast<const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&>(&PhysicalAsset::dirPtoA,py::const_))
    .def("dirAtoP",py::overload_cast<const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&>(&PhysicalAsset::dirAtoP,py::const_))
    .def("dirHtoA",py::overload_cast<const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&,const std::string&,bool>(&PhysicalAsset::dirHtoA,py::const_),"v"_a,"another"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("dirAtoH",py::overload_cast<const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&,const std::string&,bool>(&PhysicalAsset::dirAtoH,py::const_),"v"_a,"another"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("dirBtoP",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&>(&PhysicalAsset::dirBtoP,py::const_))
    .def("dirPtoB",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&>(&PhysicalAsset::dirPtoB,py::const_))
    .def("dirHtoP",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::string&,bool>(&PhysicalAsset::dirHtoP,py::const_),"v"_a,"r"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("dirPtoH",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::string&,bool>(&PhysicalAsset::dirPtoH,py::const_),"v"_a,"r"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("dirHtoB",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::string&,bool>(&PhysicalAsset::dirHtoB,py::const_),"v"_a,"r"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("dirBtoH",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::string&,bool>(&PhysicalAsset::dirBtoH,py::const_),"v"_a,"r"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("dirBtoA",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&>(&PhysicalAsset::dirBtoA,py::const_))
    .def("dirAtoB",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&>(&PhysicalAsset::dirAtoB,py::const_))
    .def("dirPtoA",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&>(&PhysicalAsset::dirPtoA,py::const_))
    .def("dirAtoP",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&>(&PhysicalAsset::dirAtoP,py::const_))
    .def("dirHtoA",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&,const std::string&,bool>(&PhysicalAsset::dirHtoA,py::const_),"v"_a,"r"_a,"another"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    .def("dirAtoH",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&,const std::string&,bool>(&PhysicalAsset::dirAtoH,py::const_),"v"_a,"r"_a,"another"_a,"dstAxisOrder"_a="FSD","onSurface"_a=false)
    //(7)方位角、仰角、高度の取得
    DEF_FUNC(PhysicalAsset,getAZ)
    DEF_FUNC(PhysicalAsset,getEL)
    DEF_FUNC(PhysicalAsset,getHeight)
    DEF_FUNC(PhysicalAsset,getGeoidHeight)
    //自身の座標情報
    .def_property_readonly("pos",[](const PhysicalAsset& v){return v.pos;})
    .def_property_readonly("vel",[](const PhysicalAsset& v){return v.vel;})
    .def_property_readonly("q",[](const PhysicalAsset& v){return v.q;})
    .def_property_readonly("omega",[](const PhysicalAsset& v){return v.omega;})
    DEF_FUNC(PhysicalAsset,getParentCRS)
    DEF_FUNC(PhysicalAsset,getBodyCRS)
    .def_property_readonly("pos_prev",[](const PhysicalAsset& v){return v.pos_prev;})
    .def_property_readonly("vel_prev",[](const PhysicalAsset& v){return v.vel_prev;})
    .def_property_readonly("q_prev",[](const PhysicalAsset& v){return v.q_prev;})
    .def_property_readonly("omega_prev",[](const PhysicalAsset& v){return v.omega_prev;})
    DEF_READWRITE(PhysicalAsset,parent)
    DEF_READWRITE(PhysicalAsset,team)
    DEF_READWRITE(PhysicalAsset,group)
    DEF_READWRITE(PhysicalAsset,name)
    DEF_READWRITE(PhysicalAsset,isAlive_)
    DEF_READWRITE(PhysicalAsset,isBoundToParent)
    DEF_READWRITE(PhysicalAsset,motion)
    DEF_READWRITE(PhysicalAsset,motion_prev)
    DEF_READWRITE(PhysicalAsset,agent)
    DEF_READWRITE(PhysicalAsset,children)
    DEF_READWRITE(PhysicalAsset,controllers)
    ;
    expose_common_class<PhysicalAssetAccessor>(m,"PhysicalAssetAccessor")
    .def(py_init<std::shared_ptr<PhysicalAsset>>())
    ;
}

ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
