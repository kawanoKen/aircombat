// Copyright (c) 2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#include "CapsuleCoursePatrolAgent.h"
#include <ASRCAISim1/Utility.h>
#include <ASRCAISim1/MathUtility.h>
#include <ASRCAISim1/Units.h>
#include <ASRCAISim1/MotionState.h>
#include <ASRCAISim1/Track.h>
#include <ASRCAISim1/Fighter.h>

ASRC_PLUGIN_NAMESPACE_BEGIN

using namespace asrc::core;
using namespace util;

void CapsuleCoursePatrolAgent::initialize(){
    BaseType::initialize();
	//modelConfigの読み込み
    p1=getValueFromJsonKR(modelConfig,"p1",randomGen);
    p2=getValueFromJsonKR(modelConfig,"p2",randomGen);
    velocity=getValueFromJsonKRD(modelConfig,"velocity",randomGen,300.0);
    radius=getValueFromJsonKRD(modelConfig,"radius",randomGen,5000.0);
    deltaAZLimit=deg2rad(getValueFromJsonKRD(modelConfig,"deltaAZLimit",randomGen,45.0));
    minPitchForNormal=deg2rad(getValueFromJsonKRD(modelConfig,"minPitchForNormal",randomGen,-10.0));
    maxPitchForNormal=deg2rad(getValueFromJsonKRD(modelConfig,"maxPitchForNormal",randomGen,10.0));
	if(modelConfig.contains("altitudeKeeper")){
		nl::json sub=modelConfig.at("altitudeKeeper");
		altitudeKeeper=AltitudeKeeper(sub);
	}else{
		nl::json sub={
			{"pGain",-3e-1},
			{"dGain",-1e-1},
			{"minPitch",minPitchForNormal},
			{"maxPitch",maxPitchForNormal}
		};
	}
    minPitchForEvasion=deg2rad(getValueFromJsonKRD(modelConfig,"minPitchForEvasion",randomGen,-45.0));
    maxPitchForEvasion=deg2rad(getValueFromJsonKRD(modelConfig,"maxPitchForEvasion",randomGen,10.0));
    minAltForEvasion=getValueFromJsonKRD(modelConfig,"minAltForEvasion",randomGen,4000.0);

	Eigen::Vector3d dp=p2-p1;
	double distance=sqrt(dp(0)*dp(0)+dp(1)*dp(1));
	course_length=2*(radius*M_PI+distance);
}
void CapsuleCoursePatrolAgent::serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full){
    BaseType::serializeInternalState(archive,full);
	if(full){
        ASRC_SERIALIZE_NVP(archive
            ,p1
            ,p2
			,velocity
            ,radius
			,deltaAZLimit
			,minPitchForNormal
			,maxPitchForNormal
			,minPitchForEvasion
			,maxPitchForEvasion
			,minAltForEvasion
            ,altitudeKeeper
            ,course_length
        )
	}
}

void CapsuleCoursePatrolAgent::validate(){
    BaseType::validate();
    std::dynamic_pointer_cast<FighterAccessor>(parent)->setFlightControllerMode("fromDirAndVel");

	MotionState originalMyMotion(parent->observables.at("motion"));
	MotionState myMotion=originalMyMotion.transformTo(getLocalCRS());
}

std::pair<Eigen::Vector3d,double> CapsuleCoursePatrolAgent::getCoursePoint(const double& phase) const{
	Eigen::Vector3d dp=p2-p1;
	double distance=sqrt(dp(0)*dp(0)+dp(1)*dp(1));
	Eigen::Vector3d ex=dp.normalized();
	Eigen::Vector3d ey=Eigen::Vector3d(0,0,1).cross(ex).normalized();
	double phase2rad=course_length/radius;

	std::vector<double> anchors = {
		(radius*M_PI_2)/course_length,
		(radius*M_PI_2+distance)/course_length,
		(radius*M_PI_2*3+distance)/course_length,
		(radius*M_PI_2*3+2*distance)/course_length
	};
	if(phase<anchors[0]){
		double angle=phase*phase2rad;
		return {
			p1+radius*(-ex*cos(angle)-ey*sin(angle)),
			rad2deg(angle)
		};
	}else if(phase<anchors[1]){
		return {
			p1+(ex*(phase-anchors[0])*course_length-ey*radius),
			M_PI
		};
	}else if(phase<anchors[2]){
        double angle=(phase-anchors[1])*phase2rad;
		return {
			p2+radius*(ex*sin(angle)-ey*cos(angle)),
			M_PI_2+rad2deg(angle)
		};
	}else if(phase<anchors[3]){
		return {
			p2+(-ex*(phase-anchors[2])*course_length+ey*radius),
			M_PI
		};
	}else{
		double angle=(1.0-phase)*phase2rad;
		return {
        	p1+radius*(-ex*cos(angle)+ey*sin(angle)),
        	-rad2deg(angle)
		};
	}
}
double CapsuleCoursePatrolAgent::getCoursePhase(const Eigen::Vector3d& pos) const{
	Eigen::Vector3d dp=p2-p1;
	double distance=sqrt(dp(0)*dp(0)+dp(1)*dp(1));
	Eigen::Vector3d ex=dp.normalized();
	Eigen::Vector3d ey=Eigen::Vector3d(0,0,1).cross(ex).normalized();
	double rad2phase=radius/course_length;

	std::vector<double> anchors = {
		(radius*M_PI_2)/course_length,
		(radius*M_PI_2+distance)/course_length,
		(radius*M_PI_2*3+distance)/course_length,
		(radius*M_PI_2*3+2*distance)/course_length
	};

	double x=ex.dot(pos-p1);
	double y=ey.dot(pos-p1);
	if(x<0){
		double angle=atan2(-y,-x);
		return angle*rad2phase;
	}else if(x>distance){
		double angle=atan2(x-distance,-y);
		return anchors[1]+angle*rad2phase;
	}else{
		if(y<0){
			return anchors[0]+x/course_length;
		}else{
			return anchors[2]+(distance-x)/course_length;
		}
	}
}

void CapsuleCoursePatrolAgent::control(){
	MotionState originalMyMotion(parent->observables.at("motion"));
	MotionState myMotion=originalMyMotion.transformTo(getLocalCRS());
	Eigen::Vector3d pos=myMotion.pos();
	Eigen::Vector3d vel=myMotion.vel();

	Eigen::Vector3d dstDir;
	if(chkMWS()){
		std::vector<std::pair<double,Eigen::Vector3d>> tmp;
		for(auto& m:parent->observables.at("/sensor/mws/track"_json_pointer)){
			Eigen::Vector3d dir=m.get<Track2D>().dir(getLocalCRS());
			tmp.push_back(std::make_pair(-dir.dot(myMotion.dirBtoP(Eigen::Vector3d(1,0,0))),-dir));
		}
		Eigen::Vector3d dr=(*std::min_element(tmp.begin(),tmp.end(),
			[](const std::pair<double,Eigen::Vector3d> &lhs,const std::pair<double,Eigen::Vector3d> &rhs){
			return lhs.first<rhs.first;
		})).second;
		dstDir=dr;
		dstDir(2)=0;
		dstDir.normalize();
		double dAz=atan2(dstDir(1),dstDir(0))-myMotion.getAZ();
        while(dAz>M_PI){dAz-=2*M_PI;}
        while(dAz<=-M_PI){dAz+=2*M_PI;}
		if(dAz > deltaAZLimit){
			dstDir << cos(myMotion.getAZ()+deltaAZLimit), sin(myMotion.getAZ()+deltaAZLimit), 0;
		}else if(dAz < -deltaAZLimit){
			dstDir << cos(myMotion.getAZ()-deltaAZLimit), sin(myMotion.getAZ()-deltaAZLimit), 0;
		}
        altitudeKeeper.minPitch=minPitchForEvasion;
        altitudeKeeper.maxPitch=maxPitchForEvasion;
        dstDir=altitudeKeeper(myMotion,dstDir,minAltForEvasion);
	}else{
		double currentPhase=getCoursePhase(pos);
        double dt=interval[SimPhase::CONTROL]*manager->getBaseTimeStep()*10;
		double delta = velocity*dt/course_length;
		auto [targetPoint, targetHeading] = getCoursePoint(currentPhase+delta);
		dstDir=targetPoint-pos;
		dstDir(2)=0;
		dstDir.normalize();
		double dAz=atan2(dstDir(1),dstDir(0))-myMotion.getAZ();
        while(dAz>M_PI){dAz-=2*M_PI;}
        while(dAz<=-M_PI){dAz+=2*M_PI;}
		if(dAz > deltaAZLimit){
			dstDir << cos(myMotion.getAZ()+deltaAZLimit), sin(myMotion.getAZ()+deltaAZLimit), 0;
		}else if(dAz < -deltaAZLimit){
			dstDir << cos(myMotion.getAZ()-deltaAZLimit), sin(myMotion.getAZ()-deltaAZLimit), 0;
		}
        altitudeKeeper.minPitch=minPitchForNormal;
        altitudeKeeper.maxPitch=maxPitchForNormal;
        dstDir=altitudeKeeper(myMotion,dstDir,-targetPoint(2));
	}

	commands[parent->getFullName()]={
		{"motion",{
			{"dstDir",originalMyMotion.dirAtoP(dstDir,myMotion.pos(),getLocalCRS())}//元のparent座標系に戻す
		}},
		{"weapon",{
			{"launch",false},
			{"target",Track3D()}
		}}
	};
	commands[parent->getFullName()]["motion"]["dstV"]=velocity;
}

bool CapsuleCoursePatrolAgent::chkMWS(){
    if(parent->observables.contains("/sensor/mws"_json_pointer)){
        return parent->observables.at("/sensor/mws/track"_json_pointer).get<std::vector<Track2D>>().size()>0;
    }else{
        return false;
    }
}

void exportCapsuleCoursePatrolAgent(py::module &m, const std::shared_ptr<asrc::core::FactoryHelper>& factoryHelper)
{
	using namespace pybind11::literals;
    expose_entity_subclass<CapsuleCoursePatrolAgent>(m,"CapsuleCoursePatrolAgent")
	;
    FACTORY_ADD_CLASS(Agent,CapsuleCoursePatrolAgent)
}

ASRC_PLUGIN_NAMESPACE_END
