// Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#include "MassPointFighter.h"
#include "Utility.h"
#include "Units.h"
#include "SimulationManager.h"

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)
using namespace util;

void IdealDirectPropulsion::initialize(){
    BaseType::initialize();
    tMin=getValueFromJsonKRD(modelConfig,"tMin",randomGen,-gravity*10000.0);
    tMax=getValueFromJsonKRD(modelConfig,"tMax",randomGen,gravity*10000.0);
    pCmd=0.0;
    thrust=0.0;
}

void IdealDirectPropulsion::serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full){
    BaseType::serializeInternalState(archive,full);

    if(full){
        ASRC_SERIALIZE_NVP(archive
            ,tMin
            ,tMax
        )
    }
    ASRC_SERIALIZE_NVP(archive
        ,pCmd
        ,thrust
    )
}

double IdealDirectPropulsion::getFuelFlowRate(){
    return 0.0;
}
double IdealDirectPropulsion::getThrust(){
    return thrust;
}
double IdealDirectPropulsion::calcFuelFlowRate(const nl::json& args){
    return 0.0;
}
double IdealDirectPropulsion::calcThrust(const nl::json& args){
    double arg_pCmd=args.at("pCmd");
    return tMin+arg_pCmd*(tMax-tMin);
}
void IdealDirectPropulsion::setPowerCommand(const double& pCmd_){
    pCmd=pCmd_;
    thrust=tMin+pCmd*(tMax-tMin);
}

void MassPointFighter::initialize(){
    BaseType::initialize();
    //modelConfigで指定するもの
    vMin=getValueFromJsonKRD(modelConfig.at("dynamics"),"vMin",randomGen,150.0);
    vMax=getValueFromJsonKRD(modelConfig.at("dynamics"),"vMax",randomGen,450.0);
    rollMax=deg2rad(getValueFromJsonKRD(modelConfig.at("dynamics"),"rollMax",randomGen,180.0));
    pitchMax=deg2rad(getValueFromJsonKRD(modelConfig.at("dynamics"),"pitchMax",randomGen,30.0));
    yawMax=deg2rad(getValueFromJsonKRD(modelConfig.at("dynamics"),"yawMax",randomGen,30.0));
    observables["spec"].merge_patch({
        {"dynamics",{
            {"vMin",vMin},
            {"vMax",vMax},
            {"rollMax",rollMax},
            {"pitchMax",pitchMax},
            {"yawMax",yawMax}
        }}
    });
}

void MassPointFighter::serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full){
    BaseType::serializeInternalState(archive,full);

    if(full){
        ASRC_SERIALIZE_NVP(archive
            ,vMin
            ,vMax
            ,rollMax
            ,pitchMax
            ,yawMax
        )
    }
}

double MassPointFighter::calcOptimumCruiseFuelFlowRatePerDistance(){
    return 0.0;//no fuel consideration
}
void MassPointFighter::calcMotion(double dt){
    motion_prev=motion;
    double V=motion.vel.norm();
    nl::json ctrl=controllers["FlightController"].lock()->commands["motion"];
    Eigen::Vector3d omegaB=ctrl.at("omegaB");
    double accel=getThrust()/m;
    motion.setOmega(relBtoP(omegaB));
    double theta=motion.omega.norm()*dt;
    if(theta>1e-6){
        Eigen::Vector3d ax=motion.omega.normalized()();
        Quaternion dq=Quaternion::fromAngle(ax,theta);
        motion.setQ(dq*motion.q);
    }else{
        Eigen::VectorXd dq=motion.q.dqdwi()*motion.omega()*dt;
        motion.setQ((motion.q+Quaternion(dq)).normalized());
    }
    double Vaft=std::max(vMin,std::min(vMax,V+accel*dt));
    motion.setVel(relBtoP(Eigen::Vector3d(Vaft,0,0)));
    motion.setPos(motion.pos()+motion.vel()*dt);
}

void MassPointFlightController::initialize(){
    BaseType::initialize();
    mode="fromDirAndVel";
    if(modelConfig.contains("altitudeKeeper")){
        nl::json sub=modelConfig.at("altitudeKeeper");
        altitudeKeeper=AltitudeKeeper(sub);
    }else{
        nl::json sub={
            {"pGain",-3e-1},
            {"dGain",-1e-1},
            {"minPitch",-deg2rad(10.0)},
            {"maxPitch",deg2rad(10.0)}
        };
        altitudeKeeper=AltitudeKeeper(sub);
    }
}
void MassPointFlightController::serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full){
    BaseType::serializeInternalState(archive,full);
    
    if(full){
        ASRC_SERIALIZE_NVP(archive
            ,altitudeKeeper
        )
    }
}

nl::json MassPointFlightController::getDefaultCommand(){
    return {
        {"roll",0.0},
        {"pitch",0.0},
        {"yaw",0.0},
        {"accel",0.0}
    };
}
nl::json MassPointFlightController::calc(const nl::json &cmd){
    if(mode=="fromDirAndVel"){
        return calcFromDirAndVel(cmd);
    }else{
        return calcDirect(cmd);
    }
}
nl::json MassPointFlightController::calcDirect(const nl::json &cmd){
    auto p=getShared<MassPointFighter>(parent);
    auto roll=std::clamp(cmd.at("roll").get<double>(),-1.,1.)*p->rollMax;
    auto pitch=std::clamp(cmd.at("pitch").get<double>(),-1.,1.)*p->pitchMax;
    auto yaw=std::clamp(cmd.at("yaw").get<double>(),-1.,1.)*p->yawMax;
    double pCmd;
    if(cmd.contains("throttle")){//0〜1
        pCmd=std::clamp(cmd.at("throttle").get<double>(),0.,1.);
    }else if(cmd.contains("accel")){//-1〜+1
        pCmd=(1+std::clamp(cmd.at("accel").get<double>(),-1.,1.))*0.5;
    }else{
        throw std::runtime_error("Either 'throttle' or 'accel' must be in cmd keys.");
    }
    return {
        {"omegaB",Eigen::Vector3d(roll,pitch,yaw)},
        {"pCmd",pCmd}
    };
}
nl::json MassPointFlightController::calcFromDirAndVel(const nl::json &cmd){
    auto p=getShared<MassPointFighter>(parent);
    auto eng=getShared<IdealDirectPropulsion>(p->engine);
    double Tmin=eng->tMin;
	double Tmax=eng->tMax;
    double pCmd;
    if(cmd.contains("dstV")){
        double V=p->motion.vel.norm();
	    double dstV=cmd.at("dstV");
        double dt=p->interval[SimPhase::BEHAVE]*manager->getBaseTimeStep();
        double dstAccel=(dstV-V)/dt;
        double aMin=Tmin/p->m;
        double aMax=Tmax/p->m;
		pCmd=std::clamp((dstAccel-aMin)/(aMax-aMin),0.0,1.0);
    }else if(cmd.contains("dstAccel")){
		double dstAccel=cmd.at("dstAccel");
        double aMin=Tmin/p->m;
        double aMax=Tmax/p->m;
		pCmd=std::clamp((dstAccel-aMin)/(aMax-aMin),0.0,1.0);
    }else if(cmd.contains("dstThrust")){
		double dstT=cmd.at("dstThrust");
		pCmd=std::clamp((dstT-Tmin)/(Tmax-Tmin),0.0,1.0);
    }else if(cmd.contains("dstThrottle")){
        double dstT=cmd.at("dstThrottle");
		pCmd=std::clamp(dstT,0.0,1.0);
    }else{
		throw std::runtime_error("Only one of dstV, dstAccel, dstThrust or dstThrottle is acceptable.");
    }

    Eigen::Vector3d omegaB;
	if(cmd.contains("dstDir")){
		Eigen::Vector3d dstDir=cmd.at("dstDir");
        if(cmd.contains("dstAlt")){
            dstDir=altitudeKeeper(p->motion,dstDir,cmd.at("dstAlt").get<double>());
        }
        double V=p->motion.vel.norm();
        Eigen::Vector3d currentDir=p->motion.vel()/V;
		dstDir.normalize();
		Eigen::Vector3d rotationAxis=currentDir.cross(dstDir);
        auto ex=p->dirBtoP(Eigen::Vector3d(1,0,0));
        auto ey=p->dirBtoP(Eigen::Vector3d(0,1,0));
        auto ez=p->dirBtoP(Eigen::Vector3d(0,1,1));
        auto down=p->motion.dirHtoP(Eigen::Vector3d(0,0,1),Eigen::Vector3d(0,0,0),"NED",false);
		auto horizontalY=down.cross(ex);
		double snTheta=std::min(1.0,rotationAxis.norm());
        double csTheta=std::clamp(currentDir.dot(dstDir),-1.0,1.0);
        double eTheta=atan2(snTheta,csTheta);
        double eps=1e-4;
        double dt=p->interval[SimPhase::BEHAVE]*manager->getBaseTimeStep();
		if(snTheta<eps){
            //Almost no rotation or 180deg rotation
            rotationAxis=currentDir.cross(down.cross(currentDir)).normalized();
        }else{
            rotationAxis/=snTheta;
        }
        Quaternion dq=Quaternion::fromAngle(rotationAxis,eTheta);
        Quaternion q1=dq*p->motion.q;
        ex=q1.transformVector(Eigen::Vector3d(1,0,0));// == dstDir
        ey=q1.transformVector(Eigen::Vector3d(0,1,0));
        double sinRoll=ex.dot(ey.cross(rotationAxis));
        double cosRoll=ey.dot(rotationAxis);
        double roll=atan2(sinRoll,cosRoll);
        if(abs(roll)<=M_PI_2){
            dq=Quaternion::fromAngle(ex,roll)*dq;
        }else{
            dq=Quaternion::fromAngle(ex,M_PI-roll)*dq;
        }
        if(dq.w()<0){
            dq=-dq;
        }
        double cosRot=dq.w();
        double sinRot=dq.vec().norm();
        Eigen::Vector3d ax=dq.vec().normalized();
        omegaB=p->relPtoB(ax*atan2(sinRot,cosRot)*2/dt);
        omegaB*=std::min(std::min(1.0,p->rollMax/abs(omegaB(0))),std::min(p->pitchMax/abs(omegaB(1)),p->yawMax/abs(omegaB(2))));
    }else if(cmd.contains("dstTurnRate")){
        omegaB=cmd.at("dstTurnRate");
		throw std::runtime_error("Only one of dstDir, or dstTurnRate is acceptable.");
    }

    return {
        {"omegaB",omegaB},
        {"pCmd",pCmd}
    };
}

void exportMassPointFighter(py::module &m, const std::shared_ptr<asrc::core::FactoryHelper>& factoryHelper)
{
    using namespace pybind11::literals;
    expose_entity_subclass<IdealDirectPropulsion>(m,"IdealDirectPropulsion")
    DEF_READWRITE(IdealDirectPropulsion,tMin)
    DEF_READWRITE(IdealDirectPropulsion,tMax)
    DEF_READWRITE(IdealDirectPropulsion,thrust)
    DEF_READWRITE(IdealDirectPropulsion,pCmd)
    ;
    FACTORY_ADD_CLASS(PhysicalAsset,IdealDirectPropulsion)

    expose_entity_subclass<MassPointFighter>(m,"MassPointFighter")
    DEF_READWRITE(MassPointFighter,vMin)
    DEF_READWRITE(MassPointFighter,vMax)
    DEF_READWRITE(MassPointFighter,rollMax)
    DEF_READWRITE(MassPointFighter,pitchMax)
    DEF_READWRITE(MassPointFighter,yawMax)
    ;
    FACTORY_ADD_CLASS(PhysicalAsset,MassPointFighter)

    expose_entity_subclass<MassPointFlightController>(m,"MassPointFlightController")
    ;
    FACTORY_ADD_CLASS(Controller,MassPointFlightController)
}

ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
