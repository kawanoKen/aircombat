// Copyright (c) 2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#include "PlanarFighter.h"
#include <ASRCAISim1/Utility.h>
#include <ASRCAISim1/Units.h>
#include <ASRCAISim1/SimulationManager.h>
#include <ASRCAISim1/MassPointFighter.h>

ASRC_PLUGIN_NAMESPACE_BEGIN

using namespace asrc::core;
using namespace util;

void PlanarFighter::calcMotion(double dt){
    motion_prev=motion;
    nl::json ctrl=controllers["FlightController"].lock()->commands["motion"];
    double yaw=ctrl.at("yaw");
    auto extra=getExtraDynamicsState();
    auto [vMin,vMax]=getVelocityRange(motion,extra);
    auto [pitchMin,pitchMax]=getPitchRateRange(motion,extra);
    auto [yawMin,yawMax]=getYawRateRange(motion,extra);
    yawMin=std::min(pitchMin,yawMin);
    yawMax=std::max(pitchMax,yawMax);
    yaw=std::clamp(yaw,yawMin,yawMax);
    double accel=getThrust()/m;
    Eigen::Vector3d omega=motion.relHtoP(Eigen::Vector3d(0,0,yaw),"NED",false);
    auto qToParentFromNED=motion.getCRS()->getEquivalentQuaternionToTopocentricCRS(motion.pos(),motion.time,"NED").conjugate();
    double newAZ=motion.getAZ()+yaw*dt;
    double V=motion.vel.norm();
    double wt=abs(yaw*dt);
    if(wt<1e-8){//ほぼ無回転
        auto accelVector=motion.relBtoP(Eigen::Vector3d(accel,0,0));
        motion.setPos(motion.pos()+motion.vel()*dt+accelVector*dt*dt/2.);
        auto posH=motion.absPtoH(motion.pos(),"NED",true);
        posH(2)=-motion_prev.getHeight();
        motion.setPos(motion.absHtoP(posH,"NED",true));
        Eigen::Vector3d vn=motion.vel().normalized();
        double Vaft=std::max(vMin,std::min(vMax,V+accel*dt));
        motion.setVel(Vaft*vn);
        auto velH=motion.velPtoH(motion.vel(),motion.pos(),"NED",true);
        velH(2)=0;
        motion.setVel(motion.velHtoP(velH,motion.pos(),"NED",true));
    }else{
        Eigen::Vector3d ex0=motion.vel()/V;
        Eigen::Vector3d ey0=omega.cross(ex0).normalized();
        
        Eigen::Vector3d dr=V*dt*(ex0*sinc(wt)+ey0*oneMinusCos_x2(wt)*wt)+accel*dt*dt*(ex0*(sinc(wt)-oneMinusCos_x2(wt))+ey0*wt*(sincMinusOne_x2(wt)+oneMinusCos_x2(wt)));
        motion.setPos(motion.pos()+dr);
        auto posH=motion.absPtoH(motion.pos(),"NED",true);
        posH(2)=-motion_prev.getHeight();
        motion.setPos(motion.absHtoP(posH,"NED",true));
        Eigen::Vector3d vn=(ex0*cos(wt)+ey0*sin(wt)).normalized();
        double Vaft=std::max(vMin,std::min(vMax,V+accel*dt));
        motion.setVel(Vaft*vn);
        auto velH=motion.velPtoH(motion.vel(),motion.pos(),"NED",true);
        velH(2)=0;
        motion.setVel(motion.velHtoP(velH,motion.pos(),"NED",true));
    }
    motion.setOmega(omega);
    motion.setQ(
        qToParentFromNED* //[parent <- NED]
        Quaternion::fromAngle(Eigen::Vector3d(0,0,1),newAZ)* //[NED <- FSD(body)]
        getAxisSwapQuaternion(bodyAxisOrder,"FSD") // [FSD(body) <- this]
    );
}

nl::json PlanarFlightController::getDefaultCommand(){
    return {
        {"yaw",0.0},
        {"pCmd",0.0}
    };
}
nl::json PlanarFlightController::calc(const nl::json &cmd){
    if(mode=="fromDirAndVel"){
        return calcFromDirAndVel(cmd);
    }else{
        return calcDirect(cmd);
    }
}
nl::json PlanarFlightController::calcDirect(const nl::json &cmd){
    auto p=getShared<PlanarFighter>(parent);
    auto extra=p->getExtraDynamicsState();
    auto [pitchMin,pitchMax]=p->getPitchRateRange(p->motion,extra);
    auto [yawMin,yawMax]=p->getYawRateRange(p->motion,extra);
    auto yaw=std::clamp(cmd.at("yaw").get<double>(),-1.,1.);
    yawMin=std::min(pitchMin,yawMin);
    yawMax=std::max(pitchMax,yawMax);
    if(yaw>=0){
        if(yawMax>0){
            yaw=std::max(yawMin,yaw*abs(yawMax));
        }else{
            yaw=yawMax;
        }
    }else{
        if(yawMin<0){
            yaw=std::min(yawMax,yaw*abs(yawMin));
        }else{
            yaw=yawMin;
        }
    }
    double pCmd;
    if(cmd.contains("throttle")){//0〜1
        pCmd=std::clamp(cmd.at("throttle").get<double>(),0.,1.);
    }else if(cmd.contains("accel")){//-1〜+1
        double accel=std::clamp(cmd.at("accel").get<double>(),-1.,1.);
        auto [accelMin,accelMax]=p->getAccelRange(p->motion,extra);
        if(accel>=0){
            if(accelMax>0){
                accel=std::max(accelMin,accel*abs(accelMax));
            }else{
                accel=accelMax;
            }
        }else{
            if(accelMin<0){
                accel=std::min(accelMax,accel*abs(accelMin));
            }else{
                accel=accelMin;
            }
        }
        pCmd=(accel-accelMin)/(accelMax-accelMin);
    }else{
        throw std::runtime_error("Either 'throttle' or 'accel' must be in cmd keys.");
    }
    return {
        {"yaw",0.0},
        {"pCmd",pCmd}
    };
}
nl::json PlanarFlightController::calcFromDirAndVel(const nl::json &cmd){
    auto p=getShared<PlanarFighter>(parent);
    auto extra=p->getExtraDynamicsState();
    auto [accelMin,accelMax]=p->getAccelRange(p->motion,extra);
    double pCmd;
    if(cmd.contains("dstV")){
        double V=p->motion.vel.norm();
	    double dstV=cmd.at("dstV");
        double dt=p->interval[SimPhase::BEHAVE]*manager->getBaseTimeStep();
        double dstAccel=(dstV-V)/dt;
		pCmd=std::clamp((dstAccel-accelMin)/(accelMax-accelMin),0.0,1.0);
    }else if(cmd.contains("dstAccel")){
		double dstAccel=cmd.at("dstAccel");
		pCmd=std::clamp((dstAccel-accelMin)/(accelMax-accelMin),0.0,1.0);
    }else if(cmd.contains("dstThrust")){
		double dstT=cmd.at("dstThrust");
        double Tmin=accelMin*p->m;
        double Tmax=accelMax*p->m;
		pCmd=std::clamp((dstT-Tmin)/(Tmax-Tmin),0.0,1.0);
    }else if(cmd.contains("dstThrottle")){
        double dstT=cmd.at("dstThrottle");
		pCmd=std::clamp(dstT,0.0,1.0);
    }else{
		throw std::runtime_error("Only one of dstV, dstAccel, dstThrust or dstThrottle is acceptable.");
    }

    double yaw;
	if(cmd.contains("dstDir")){
		Eigen::Vector3d dstDir=cmd.at("dstDir");
        auto dstDirH=p->motion.dirPtoH(dstDir,p->motion.pos(),"NED",true);
        dstDirH(2)=0;
        dstDir=p->motion.dirHtoP(dstDirH,p->motion.pos(),"NED",true);
        double V=p->motion.vel.norm();
        Eigen::Vector3d currentDir=p->motion.vel()/V;
		Eigen::Vector3d rotationAxis=currentDir.cross(dstDir);
		double snTheta=std::min(1.0,rotationAxis.norm());
        double csTheta=std::clamp(currentDir.dot(dstDir),-1.0,1.0);
        double eTheta=atan2(snTheta,csTheta);
        double dt=p->interval[SimPhase::BEHAVE]*manager->getBaseTimeStep();
        auto [pitchMin,pitchMax]=p->getPitchRateRange(p->motion,extra);
        auto [yawMin,yawMax]=p->getYawRateRange(p->motion,extra);
        yawMin=std::min(pitchMin,yawMin);
        yawMax=std::max(pitchMax,yawMax);
        if(rotationAxis(2)>=0){
            yaw=std::clamp(eTheta/dt,yawMin,yawMax);
        }else{
            yaw=std::clamp(-eTheta/dt,yawMin,yawMax);
        }
    }else if(cmd.contains("dstTurnRate")){
        if(cmd.at("dstTurnRate").is_number()){
            yaw=cmd.at("dstTurnRate").get<double>();
        }else{
            yaw=cmd.at("dstTurnRate").get<Eigen::Vector3d>()(2);
        }
    }else{
		throw std::runtime_error("Only one of dstDir, or dstTurnRate is acceptable.");
    }
    return {
        {"yaw",yaw},
        {"pCmd",pCmd}
    };
}

void exportPlanarFighter(py::module &m, const std::shared_ptr<asrc::core::FactoryHelper>& factoryHelper)
{
    using namespace pybind11::literals;

    expose_entity_subclass<PlanarFighter>(m,"PlanarFighter")
    DEF_READWRITE(PlanarFighter,vMin)
    DEF_READWRITE(PlanarFighter,vMax)
    DEF_READWRITE(PlanarFighter,rollMax)
    DEF_READWRITE(PlanarFighter,pitchMax)
    DEF_READWRITE(PlanarFighter,yawMax)
    ;
    FACTORY_ADD_CLASS(PhysicalAsset,PlanarFighter)

    expose_entity_subclass<PlanarFlightController>(m,"PlanarFlightController")
    ;
    FACTORY_ADD_CLASS(Controller,PlanarFlightController)
}

ASRC_PLUGIN_NAMESPACE_END
