// Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#include "EkkerMissile.h"
#include <boost/math/tools/roots.hpp>
#include "MathUtility.h"
#include "Units.h"
#include "SimulationManager.h"
#include "EkkerMDT.h"

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)
using namespace util;

void EkkerMissile::initialize(){
    BaseType::initialize();
    tMax=getValueFromJsonKR(modelConfig,"tMax",randomGen);
    tBurn=getValueFromJsonKR(modelConfig,"tBurn",randomGen);
    hitD=getValueFromJsonKR(modelConfig,"hitD",randomGen);
    minV=getValueFromJsonKR(modelConfig,"minV",randomGen);
    massI=getValueFromJsonKR(modelConfig,"massI",randomGen);
    massF=getValueFromJsonKR(modelConfig,"massF",randomGen);
    thrust=getValueFromJsonKR(modelConfig,"thrust",randomGen);
    maxLoadG=getValueFromJsonKR(modelConfig,"maxLoadG",randomGen);
    maxA=deg2rad(getValueFromJsonKR(modelConfig,"maxA",randomGen));
    maxD=deg2rad(getValueFromJsonKR(modelConfig,"maxD",randomGen));
    l=getValueFromJsonKR(modelConfig,"length",randomGen);
    d=getValueFromJsonKR(modelConfig,"diameter",randomGen);
    ln=getValueFromJsonKR(modelConfig,"lengthN",randomGen);
    lcgi=getValueFromJsonKR(modelConfig,"lcgi",randomGen);
    lcgf=getValueFromJsonKR(modelConfig,"lcgf",randomGen);
    lw=getValueFromJsonKR(modelConfig,"locationW",randomGen);
    bw=getValueFromJsonKR(modelConfig,"spanW",randomGen);
    bt=getValueFromJsonKR(modelConfig,"spanT",randomGen);
    thicknessRatio=getValueFromJsonKR(modelConfig,"thicknessRatio",randomGen);
    frictionFactor=getValueFromJsonKRD(modelConfig,"frictionFactor",randomGen,1.1);
    Sw=getValueFromJsonKR(modelConfig,"areaW",randomGen);
    St=getValueFromJsonKR(modelConfig,"areaT",randomGen);
    Sref=M_PI*d*d/4;
    mass=massI;
    lcg=lcgi;
    observables["spec"]["tMax"]=tMax;
    observables["spec"]["tBurn"]=tBurn;
    observables["spec"]["hitD"]=hitD;
    observables["spec"]["thrust"]=thrust;
}
void EkkerMissile::serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full){
    BaseType::serializeInternalState(archive,full);

    if(full){
        ASRC_SERIALIZE_NVP(archive
            ,tMax
            ,tBurn
            ,hitD
            ,minV
            ,massI
            ,massF
            ,thrust
            ,maxLoadG
            ,maxA
            ,maxD
            ,l
            ,d
            ,ln
            ,lcgi
            ,lcgf
            ,lw
            ,bw
            ,bt
            ,thicknessRatio
            ,frictionFactor
            ,Sw
            ,St
            ,Sref
        )
    }
    ASRC_SERIALIZE_NVP(archive
        ,mass
        ,lcg
    )
}
void EkkerMissile::calcMotion(double tAftLaunch,double dt){
    motion_prev=motion;
    double V=motion.vel.norm();
    auto propNav=getShared<PropNav>(controllers["Navigator"]);
    Eigen::Vector3d accelCmd=propNav->commands["accel"];
    Eigen::Vector3d omegaV=propNav->commands["omega"];
    double omega=omegaV.norm();
    double omegaMax=maxLoadG*gravity/V;
    if(omega>omegaMax){
        omegaV*=(omegaMax/omega);
        omega=omegaMax;
        accelCmd=omegaV.cross(motion.vel());
    }
    double alt=motion.getHeight();
    bool pwroff=tAftLaunch>=tBurn;
    double thrust_;
    if(pwroff){
        thrust_ = 0.0;
        mass = massF;
        lcg = lcgf;
    }else{
        thrust_ = thrust;
        mass = massI+(massF-massI)*(tAftLaunch/tBurn-1);
        lcg = lcgi+(lcgf-lcgi)*(tAftLaunch/tBurn-1);
    }
    std::map<std::string,double> atm=atmosphere(alt);
    double M=V/atm["a"];
    double qd=0.5*atm["rho"]*V*V;
    Eigen::Vector3d ex=motion.vel()/V;
    Eigen::Vector3d gravityAccelVector=motion.dirHtoP(Eigen::Vector3d(0,0,1),"NED")*gravity; //TODO 慣性力モデルに置換
    Eigen::Vector3d ey=(accelCmd-gravityAccelVector).cross(ex)*mass;
    double desiredSideForce=ey.norm();
    Eigen::Vector3d sideAx;
    double aoa,CL,CD;
    if(desiredSideForce>0){
        ey/=desiredSideForce;
        sideAx=ex.cross(ey);
        std::pair<double,double> tmp=EkkerMDT::cldcmd(d,l,ln,lcg,bt,St,M);
        double cld=tmp.first;
        double cmd=tmp.second;
        auto func=[&](double aoa_){
            std::pair<double,double> tmp=EkkerMDT::clacma(d,l,ln,lcg,bw,Sw,lw,bt,St,M,aoa_);
            double cla=tmp.first;
            double cma=tmp.second;
            double delta=std::clamp<double>(-cma/cmd*aoa_,-maxD,maxD);
            return (cla*aoa_+cld*delta)*(qd*Sref)+thrust_*sin(aoa_)-desiredSideForce;
        };
        if(func(0.0)>=0){
            aoa=0.0;
        }else if(func(maxA)<=0){
            aoa=maxA;
        }else{
            try{
                boost::math::tools::eps_tolerance<double> tol(16);
                boost::uintmax_t max_iter=10;
                auto result=boost::math::tools::toms748_solve(func,0.0,maxA,tol,max_iter);
                aoa=(result.first+result.second)/2;
            }catch(std::exception& e){
                std::cout<<"exception at EkkerMissile::calcMotion"<<std::endl;
                DEBUG_PRINT_EXPRESSION(func(0.0))
                DEBUG_PRINT_EXPRESSION(func(maxA))
                DEBUG_PRINT_EXPRESSION(alt)
                DEBUG_PRINT_EXPRESSION(M)
                DEBUG_PRINT_EXPRESSION(desiredSideForce)
                DEBUG_PRINT_EXPRESSION(thrust_)
                throw e;
            }
        }
        tmp=EkkerMDT::clacma(d,l,ln,lcg,bw,Sw,lw,bt,St,M,aoa);
        double cla=tmp.first;
        double cma=tmp.second;
        double delta=std::clamp<double>(-cma/cmd*aoa,-maxD,maxD);
        //CL=(cla-cld*cma/cmd)*aoa;
        CL=cla*aoa+cld*delta;
        CD=EkkerMDT::cdtrim(d,l,ln,bw,Sw,thicknessRatio,bt,St,thicknessRatio,M,alt,aoa,delta,frictionFactor,pwroff);
    }else{
        sideAx<<0,0,0;
        CL=0.0;
        CD=EkkerMDT::cdtrim(d,l,ln,bw,Sw,thicknessRatio,bt,St,thicknessRatio,M,alt,0,0,frictionFactor,pwroff);
        aoa=0.0;
    }
    Eigen::Vector3d sideAccel=sideAx*(CL*qd*Sref+thrust_*sin(aoa))/mass+(gravityAccelVector-ex*(gravityAccelVector.dot(ex)));
	//attitude and deflection are assumed to be immidiately adjustable as desired.
	accelScalar=(-CD*Sref*qd+thrust_*cos(aoa))/mass+gravityAccelVector.dot(ex);
	Eigen::Vector3d accel=ex*accelScalar;
    omegaV=motion.vel().cross(sideAccel)/(V*V);
    omega=omegaV.norm();
    if(omega*dt<1e-8){//ほぼ無回転
        motion.setPos(motion.pos()+motion.vel()*dt+accel*dt*dt/2.);
        motion.setVel(motion.vel()+accel*dt);
    }else{
        Eigen::Vector3d ex0=motion.vel()/V;
        Eigen::Vector3d ey0=omegaV.cross(ex0).normalized();
        double wt=omega*dt;
        double acc=accelScalar;
        Eigen::Vector3d vn=(ex0*cos(wt)+ey0*sin(wt)).normalized();
        motion.setVel((V+acc*dt)*vn);
        Eigen::Vector3d dr=V*dt*(ex0*sinc(wt)+ey0*oneMinusCos_x2(wt)*wt)+acc*dt*dt*(ex0*(sinc(wt)-oneMinusCos_x2(wt))+ey0*wt*(sincMinusOne_x2(wt)+oneMinusCos_x2(wt)));
        motion.setPos(motion.pos()+dr);
        accel=(motion.vel()-vel_prev())/dt;
    }
    motion.setOmega(omegaV);
    calcQ();
}
bool EkkerMissile::hitCheck(const Eigen::Vector3d &tpos,const Eigen::Vector3d &tpos_prev){
    Eigen::Vector3d r1=pos_prev();
    Eigen::Vector3d r2=tpos_prev;
    Eigen::Vector3d d1=pos()-r1;
    Eigen::Vector3d d2=tpos-r2;
    Eigen::Vector3d A=r1-r2;double a=A.norm();
    Eigen::Vector3d B=d1-d2;double b=B.norm();
    if(a<hitD){//初期位置であたり
        return true;
    }else if(b<1e-8){//相対速度がほぼ0}
        return a<hitD;
    }
    double tMin=std::min(1.,std::max(0.,-2*A.dot(B)/(b*b)));
    return (A+B*tMin).norm()<hitD;
}
bool EkkerMissile::checkDeactivateCondition(double tAftLaunch){
    //最大飛翔時間超過or地面に激突or燃焼後に速度が規定値未満
    bool ret=(tAftLaunch>=tMax ||
            motion.getHeight()<0 ||
            (tAftLaunch>=tBurn && motion.vel.norm()<minV)
    );
    return ret;
}

void exportEkkerMissile(py::module &m, const std::shared_ptr<asrc::core::FactoryHelper>& factoryHelper)
{
    using namespace pybind11::literals;
    auto cls=expose_entity_subclass<EkkerMissile>(m,"EkkerMissile");
    cls
    DEF_READWRITE(EkkerMissile,tMax)
    DEF_READWRITE(EkkerMissile,tBurn)
    DEF_READWRITE(EkkerMissile,hitD)
    DEF_READWRITE(EkkerMissile,minV)
    DEF_READWRITE(EkkerMissile,mass)
    DEF_READWRITE(EkkerMissile,massI)
    DEF_READWRITE(EkkerMissile,massF)
    DEF_READWRITE(EkkerMissile,thrust)
    DEF_READWRITE(EkkerMissile,maxLoadG)
    DEF_READWRITE(EkkerMissile,Sref)
    DEF_READWRITE(EkkerMissile,maxA)
    DEF_READWRITE(EkkerMissile,l)
    DEF_READWRITE(EkkerMissile,d)
    DEF_READWRITE(EkkerMissile,ln)
    DEF_READWRITE(EkkerMissile,lcg)
    DEF_READWRITE(EkkerMissile,lcgi)
    DEF_READWRITE(EkkerMissile,lcgf)
    DEF_READWRITE(EkkerMissile,lw)
    DEF_READWRITE(EkkerMissile,bw)
    DEF_READWRITE(EkkerMissile,bt)
    DEF_READWRITE(EkkerMissile,thicknessRatio)
    DEF_READWRITE(EkkerMissile,frictionFactor)
    DEF_READWRITE(EkkerMissile,Sw)
    DEF_READWRITE(EkkerMissile,St)
    ;
    FACTORY_ADD_CLASS(PhysicalAsset,EkkerMissile)
}

ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
