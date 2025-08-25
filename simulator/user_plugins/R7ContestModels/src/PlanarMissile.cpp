// Copyright (c) 2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#include "PlanarMissile.h"
#include <boost/math/tools/roots.hpp>
#include <ASRCAISim1/Utility.h>
#include <ASRCAISim1/Units.h>
#include <ASRCAISim1/SimulationManager.h>
#include <ASRCAISim1/MassPointFighter.h>
#include <ASRCAISim1/EkkerMDT.h>

ASRC_PLUGIN_NAMESPACE_BEGIN

using namespace asrc::core;
using namespace util;

void PlanarMissile::calcMotion(double tAftLaunch,double dt){
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
                std::cout<<"exception at PlanarMissile::calcMotion"<<std::endl;
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
    sideAccel=motion.relPtoH(sideAccel,motion.pos(),"NED",true);
    sideAccel(2)=0;
    sideAccel=motion.relHtoP(sideAccel,motion.pos(),"NED",true);
	//attitude and deflection are assumed to be immidiately adjustable as desired.
	accelScalar=(-CD*Sref*qd+thrust_*cos(aoa))/mass+gravityAccelVector.dot(ex);
	Eigen::Vector3d accel=ex*accelScalar;
    omegaV=motion.vel().cross(sideAccel)/(V*V);
    omega=omegaV.norm();
    double wt=omega*dt;
    if(wt<1e-8){//ほぼ無回転
        auto accelVector=motion.relBtoP(Eigen::Vector3d(accelScalar,0,0));
        motion.setPos(motion.pos()+motion.vel()*dt+accelVector*dt*dt/2.);
        auto posH=motion.absPtoH(motion.pos(),"NED",true);
        posH(2)=-motion_prev.getHeight();
        motion.setPos(motion.absHtoP(posH,"NED",true));
        Eigen::Vector3d vn=motion.vel().normalized();
        motion.setVel((V+accelScalar*dt)*vn);
        auto velH=motion.velPtoH(motion.vel(),motion.pos(),"NED",true);
        velH(2)=0;
        motion.setVel(motion.velHtoP(velH,motion.pos(),"NED",true));
    }else{
        Eigen::Vector3d ex0=motion.vel()/V;
        Eigen::Vector3d ey0=omegaV.cross(ex0).normalized();
        Eigen::Vector3d dr=V*dt*(ex0*sinc(wt)+ey0*oneMinusCos_x2(wt)*wt)+accelScalar*dt*dt*(ex0*(sinc(wt)-oneMinusCos_x2(wt))+ey0*wt*(sincMinusOne_x2(wt)+oneMinusCos_x2(wt)));
        motion.setPos(motion.pos()+dr);
        auto posH=motion.absPtoH(motion.pos(),"NED",true);
        posH(2)=-motion_prev.getHeight();
        motion.setPos(motion.absHtoP(posH,"NED",true));
        Eigen::Vector3d vn=(ex0*cos(wt)+ey0*sin(wt)).normalized();
        motion.setVel((V+accelScalar*dt)*vn);
        auto velH=motion.velPtoH(motion.vel(),motion.pos(),"NED",true);
        velH(2)=0;
        motion.setVel(motion.velHtoP(velH,motion.pos(),"NED",true));
    }
    motion.setOmega(omegaV);
    calcQ();
}

bool PlanarMissile::calcRangeSub(double vs,double hs,double vt,double ht,double obs,double aa,double r){
    Eigen::Vector3d posBef=motion.pos();
    Eigen::Vector3d velBef=motion.vel();
    Eigen::Vector3d omegaBef=motion.omega();
    Quaternion qBef=motion.q;
    Time timeBef=motion.time;
    MotionState motion_prevBef=motion_prev;
    bool isBoundToParentBef=isBoundToParent;
    if(isBoundToParentBef){
        unbindFromParent();
    }
    double dt=interval[SimPhase::BEHAVE]*manager->getBaseTimeStep();
    motion.setPos(motion.absHtoP(Eigen::Vector3d(0,0,-hs),"NED",true));
    motion.setVel(motion.velHtoP(Eigen::Vector3d(cos(obs),-sin(obs),0)*std::max(vs,1e-3),Eigen::Vector3d(-r,0,-hs),"NED",true));
    motion.setOmega(Eigen::Vector3d(0,0,0));
    calcQ();
    Eigen::Vector3d tpos=motion.absHtoP(Eigen::Vector3d(r,0,-hs),"NED",true);
    Eigen::Vector3d tvel=motion.velHtoP(Eigen::Vector3d(cos(aa),sin(aa),0)*vt,Eigen::Vector3d(0,0,-hs),"NED",true);
    Eigen::Vector3d tpos_prev=tpos-dt*tvel;
    motion_prev.setPos(motion.pos()-dt*motion.vel());
    motion_prev.setVel(motion.vel());
    double t=0;
    bool finished=false;
    bool hit=false;
    auto propNav=getShared<PropNav>(controllers["Navigator"]);
    bool hasLaunchedBef=hasLaunched;
    hasLaunched=true;
    while(!finished){
        commands={
            {"Navigator",{
                {"rs",motion.pos()},
                {"vs",motion.vel()},
                {"rt",tpos},
                {"vt",tvel}
            }}
        };
        propNav->control();
        calcMotion(t,dt);
        tpos_prev=tpos;
        tpos+=dt*tvel;
        t+=dt;
        if(hitCheck(tpos,tpos_prev)){
            finished=true;
            hit=true;
        }else if(checkDeactivateCondition(t)){
            finished=true;
            hit=false;
        }
    }
    if(isBoundToParentBef){
        bindToParent();
    }
    motion.setPos(posBef);
    motion.setVel(velBef);
    motion.setOmega(omegaBef);
    motion.setTime(timeBef);
    motion.setQ(qBef);
    motion_prev=motion_prevBef;
    hasLaunched=hasLaunchedBef;
    return hit;
}

void PlanarMissile::makeRangeTable(const std::string& dstPath){
    py::gil_scoped_acquire acquire;
    std::cout<<"makeRangeTable start"<<std::endl;
    int nvs=7;
    int nhs=6;
    int nvt=7;
    int nht=2;//
    int nobs=7;
    int naa=13;
    Eigen::VectorXd vs=Eigen::VectorXd::LinSpaced(nvs,0.0,600.0);
    Eigen::VectorXd hs=Eigen::VectorXd::LinSpaced(nhs,0.0,20000.0);
    Eigen::VectorXd vt=Eigen::VectorXd::LinSpaced(nvt,0.0,600.0);
    Eigen::VectorXd ht=Eigen::VectorXd::LinSpaced(nht,0.0,20000.0);
    Eigen::VectorXd obs=Eigen::VectorXd::LinSpaced(nobs,0.0,M_PI);
    Eigen::VectorXd aa=Eigen::VectorXd::LinSpaced(naa,-M_PI,M_PI);
    int numDataPoints=nvs*nhs*nht*nvt*nobs*naa;
    Eigen::MatrixXd indices=Eigen::MatrixXd::Zero(6,numDataPoints);//ColMajor
    Eigen::MatrixXd args=Eigen::MatrixXd::Zero(6,numDataPoints);//ColMajor
    std::cout<<"create indices..."<<std::endl;
    int idx=0;
    for(int aa_i=0;aa_i<naa;aa_i++){
        for(int obs_i=0;obs_i<nobs;obs_i++){
            for(int ht_i=0;ht_i<nht;ht_i++){
                for(int vt_i=0;vt_i<nvt;vt_i++){
                    for(int hs_i=0;hs_i<nhs;hs_i++){
                        for(int vs_i=0;vs_i<nvs;++vs_i){
                            args.block(0,idx,6,1)<<vs(vs_i),hs(hs_i),vt(vt_i),ht(ht_i),obs(obs_i),aa(aa_i);
                            idx++;
                        }
                    }
                }
            }
        }
    }
    std::cout<<"create indices done."<<std::endl;
    int numProcess=std::thread::hardware_concurrency();
    std::vector<std::thread> th;
    std::vector<std::future<Eigen::VectorXd>> f;
    std::vector<int> begins;
    int numPerProc=numDataPoints/numProcess;
    std::cout<<"run in "<<numProcess<<" processes. total="<<numDataPoints<<", perProc="<<numPerProc<<std::endl;
    std::mutex mtx;
    std::vector<std::shared_ptr<Missile>> msls;
    for(int i=0;i<numProcess;++i){
        nl::json dummy_ic=instanceConfig;
        dummy_ic["entityFullName"]=getFullName()+"/for_range_table_"+std::to_string(i+1);
        msls.push_back(createUnmanagedEntityByClassName<Missile>(isEpisodic(),"PhysicalAsset",getFactoryClassName(),modelConfig,dummy_ic));
        std::promise<Eigen::VectorXd> p;
        f.emplace_back(p.get_future());
        Eigen::MatrixXd subargs=args.block(0,numPerProc*i,6,(i!=numProcess-1 ? numPerProc : numDataPoints-numPerProc*i));
        th.emplace_back(makeRangeTableSub,std::ref(mtx),std::move(p),msls[i],subargs);
    }
    std::cout<<"waiting calculation"<<std::endl;
    std::vector<Eigen::VectorXd> rangeTablePoints_={vs,hs,vt,ht,obs,aa};
    Eigen::VectorXd returned(numDataPoints);
    for(int i=0;i<numProcess;++i){
        try{
            returned.block(numPerProc*i,0,(i!=numProcess-1 ? numPerProc : numDataPoints-numPerProc*i),1)=f[i].get();
        }catch(std::exception& ex){
            {
                std::lock_guard<std::mutex> lock(mtx);
                std::cout<<"exception in proc("<<i<<"): "<<ex.what()<<std::endl;
            }
        }
    }
    Eigen::Tensor<double,6> rangeTable_=Eigen::TensorMap<Eigen::Tensor<double,6>>(returned.data(),nvs,nhs,nvt,nht,nobs,naa);
    for(int i=0;i<numProcess;++i){
        th[i].join();
    }
    std::cout<<"exporting to npz file"<<std::endl;
    auto np=py::module_::import("numpy");
    np.attr("savez")(dstPath,py::arg("vs")=vs,py::arg("hs")=hs,py::arg("vt")=vt,py::arg("ht")=ht,py::arg("obs")=obs,py::arg("aa")=aa,py::arg("ranges")=rangeTable_);
    std::cout<<"makeRangeTable done."<<std::endl;
}

void exportPlanarMissile(py::module &m, const std::shared_ptr<asrc::core::FactoryHelper>& factoryHelper)
{
    using namespace pybind11::literals;

    expose_entity_subclass<PlanarMissile>(m,"PlanarMissile")
    ;
    FACTORY_ADD_CLASS(PhysicalAsset,PlanarMissile)

}

ASRC_PLUGIN_NAMESPACE_END
