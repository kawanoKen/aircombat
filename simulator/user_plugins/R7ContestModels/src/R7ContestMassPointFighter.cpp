// Copyright (c) 2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#include "R7ContestMassPointFighter.h"
#include <ASRCAISim1/Utility.h>
#include <ASRCAISim1/Units.h>
#include <ASRCAISim1/SimulationManager.h>
#include <ASRCAISim1/MassPointFighter.h>

ASRC_PLUGIN_NAMESPACE_BEGIN

using namespace asrc::core;
using namespace util;

void R7ContestMassPointFighter::initialize(){
    BaseType::initialize();
    if(modelConfig.at("dynamics").contains("table")){
        nl::json& j_table=modelConfig.at("dynamics").at("table");
        if(j_table.is_string()){
            // .npz file
            py::gil_scoped_acquire acquire;
            std::string moduleName=py::cast<std::string>(py::cast(this->shared_from_this()).attr("__module__"));
            auto np=py::module_::import("numpy");
            py::object loaded;
            std::filesystem::path filePath=getFactoryHelperChain().resolveFilePath(modelConfig.at("dynamics").at("table").get<std::string>());
            if(std::filesystem::exists(filePath)){
                loaded=np.attr("load")(filePath.generic_string());
            }else{
                throw std::runtime_error("The dynamics table file \""+filePath.generic_string()+"\"is not found.");
            }
            altTable=py::cast<Eigen::VectorXd>(loaded["altitude"]);
            assert(altTable.size()>1);
            vMinTable=py::cast<Eigen::Tensor<double,1>>(loaded["vMin"]);
            assert(vMinTable.size()==altTable.size());
            vMaxTable=py::cast<Eigen::Tensor<double,1>>(loaded["vMax"]);
            assert(vMaxTable.size()==altTable.size());
            std::vector<std::string> keys{
                "accelMin",
                "accelMax",
                "rollMin",
                "rollMax",
                "pitchMin",
                "pitchMax",
                "yawMin",
                "yawMax"
            };
            int nvs=-1;
            for(auto&& key : keys){
                twoDimTables[key]=py::cast<Eigen::Tensor<double,2>>(loaded[key.c_str()]);
                if(nvs<0){
                    nvs=twoDimTables[key].dimension(1);
                }
                assert(twoDimTables[key].dimension(0)==altTable.size() && twoDimTables[key].dimension(1)==nvs);
            }
        }else if(j_table.is_object()){
            loadTableFromJsonObject(j_table);
        }else{
            throw std::runtime_error("The value of 'table' should be string or object.");
        }
    }else{
        loadTableFromJsonObject(modelConfig.at("dynamics"));
    }

    {
        std::vector<std::string> keys{
            "vMin",
            "vMax",
            "accelMin",
            "accelMax",
            "rollMin",
            "rollMax",
            "pitchMin",
            "pitchMax",
            "yawMin",
            "yawMax"
        };
        if(modelConfig.at("dynamics").contains("scale")){
            const nl::json& j=modelConfig.at("dynamics").at("scale");
            for(auto&& key : keys){
                if(j.contains(key)){
                    scaleFactor[key]=getValueFromJsonKR(j,key,randomGen);
                }else{
                    scaleFactor[key]=1.0;
                }
            }
        }else{
            for(auto&& key : keys){
                scaleFactor[key]=1.0;
            }
        }
    }

    observables["spec"]["dynamics"]={
        {"scale",scaleFactor}
    };
}

void R7ContestMassPointFighter::serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full){
    BaseType::serializeInternalState(archive,full);

    if(full){
        ASRC_SERIALIZE_NVP(archive
            ,altTable
            ,vMinTable
            ,vMaxTable
            ,twoDimTables
            ,scaleFactor
        )
    }
}

double R7ContestMassPointFighter::getThrust(){
    auto [accelMin,accelMax] = getAccelRange(motion);
    nl::json ctrl=controllers["FlightController"].lock()->commands["motion"];
    double pCmd=ctrl.at("pCmd");

    // 重力の影響を反映
    double V=motion.vel.norm();
    Eigen::Vector3d vn=motion.vel()/V;
    Eigen::Vector3d gravityAxis=motion.dirHtoP(Eigen::Vector3d(0,0,1),Eigen::Vector3d(0,0,0),"NED",false);
    double gOfs=vn.dot(gravityAxis)*gravity;

    return m*(accelMin+pCmd*(accelMax-accelMin)+gOfs);
}

void R7ContestMassPointFighter::calcMotion(double dt){
    motion_prev=motion;
    double V=motion.vel.norm();
    nl::json ctrl=controllers["FlightController"].lock()->commands["motion"];
    Eigen::Vector3d omegaB=ctrl.at("omegaB");
    auto extra=getExtraDynamicsState();
    auto [vMin,vMax]=getVelocityRange(motion,extra);
    auto [rollMin,rollMax]=getRollRateRange(motion,extra);
    auto [pitchMin,pitchMax]=getPitchRateRange(motion,extra);
    auto [yawMin,yawMax]=getYawRateRange(motion,extra);
    omegaB(0)=std::clamp(omegaB(0),rollMin,rollMax);
    omegaB(1)=std::clamp(omegaB(1),pitchMin,pitchMax);
    omegaB(2)=std::clamp(omegaB(2),yawMin,yawMax);
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

nl::json R7ContestMassPointFighter::getExtraDynamicsState(){
    return nl::json::object();
}

std::pair<double,double> R7ContestMassPointFighter::getVelocityRange(const MotionState& m,const nl::json& extraArgs){
    double alt=m.getHeight();
    Eigen::MatrixXd arg(1,1);
    arg<<alt;
    return {
        interpn({altTable},vMinTable,arg)(0)*scaleFactor["vMin"],
        interpn({altTable},vMaxTable,arg)(0)*scaleFactor["vMax"]
    };
}
double R7ContestMassPointFighter::getTwoDimTableValue(const std::string& key,const MotionState& m,const nl::json& extraArgs){
    double alt=m.getHeight();
    double V=m.vel.norm();
    int alt_i=0;
    Eigen::VectorXd vs;
    Eigen::MatrixXd arg(1,1);
    double vMin,vMax;
    Eigen::DSizes<Eigen::Index, 2> startIndices{0,0};
    Eigen::DSizes<Eigen::Index, 2> sliceSizes=twoDimTables[key].dimensions();
    Eigen::DSizes<Eigen::Index, 1> squeezedSizes{twoDimTables[key].dimensions()[1]};
    sliceSizes[0]=1;
    if(alt<altTable(0) || alt>=altTable(altTable.size()-1)){
        if(alt<altTable(0)){
            alt_i=0;
        }else{
            alt_i=altTable.size()-1;
        }
        vMin=vMinTable(alt_i)*scaleFactor["vMin"];
        vMax=vMaxTable(alt_i)*scaleFactor["vMax"];
        if(vMin==vMax){
            return twoDimTables[key](alt_i,0)*scaleFactor[key];
        }
        vs=Eigen::VectorXd::LinSpaced(twoDimTables[key].dimensions()[1],vMin,vMax);
        arg<<std::clamp(V,vMin,vMax);
        startIndices[0]=alt_i;
        return interpn({vs},twoDimTables[key].slice(startIndices,sliceSizes).reshape(squeezedSizes),arg)(0);
    }else{
        for(alt_i=0;alt_i<altTable.size()-1;++alt_i){
            if(altTable(alt_i)<=alt && alt<altTable(alt_i+1)){
                double ret1,ret2;
                vMin=vMinTable(alt_i)*scaleFactor["vMin"];
                vMax=vMaxTable(alt_i)*scaleFactor["vMax"];
                if(V<=vMin){
                    ret1=twoDimTables[key](alt_i,0);
                }else if(V>=vMax){
                    ret1=twoDimTables[key](alt_i,altTable.size()-1);
                }else{
                    vs=Eigen::VectorXd::LinSpaced(twoDimTables[key].dimensions()[1],vMin,vMax);
                    arg<<std::clamp(V,vMin,vMax);
                    startIndices[0]=alt_i;
                    ret1=interpn({vs},twoDimTables[key].slice(startIndices,sliceSizes).reshape(squeezedSizes),arg)(0);
                }
                vMin=vMinTable(alt_i+1)*scaleFactor["vMin"];
                vMax=vMaxTable(alt_i+1)*scaleFactor["vMax"];
                if(V<=vMin){
                    ret2=twoDimTables[key](alt_i+1,0);
                }else if(V>=vMax){
                    ret2=twoDimTables[key](alt_i+1,altTable.size()-1);
                }else{
                    vs=Eigen::VectorXd::LinSpaced(twoDimTables[key].dimensions()[1],vMin,vMax);
                    arg<<std::clamp(V,vMin,vMax);
                    startIndices[0]=alt_i+1;
                    ret2=interpn({vs},twoDimTables[key].slice(startIndices,sliceSizes).reshape(squeezedSizes),arg)(0);
                }
                return (ret1+(ret2-ret1)*(alt-altTable(alt_i))/(altTable(alt_i+1)-altTable(alt_i)))*scaleFactor[key];
            }
        }
        throw std::runtime_error("Something went wrong.");
    }
}

std::pair<double,double> R7ContestMassPointFighter::getAccelRange(const MotionState& m,const nl::json& extraArgs){
    return {getTwoDimTableValue("accelMin",m,extraArgs),getTwoDimTableValue("accelMax",m,extraArgs)};
}
std::pair<double,double> R7ContestMassPointFighter::getRollRateRange(const MotionState& m,const nl::json& extraArgs){
    return {getTwoDimTableValue("rollMin",m,extraArgs),getTwoDimTableValue("rollMax",m,extraArgs)};
}
std::pair<double,double> R7ContestMassPointFighter::getPitchRateRange(const MotionState& m,const nl::json& extraArgs){
    return {getTwoDimTableValue("pitchMin",m,extraArgs),getTwoDimTableValue("pitchMax",m,extraArgs)};
}
std::pair<double,double> R7ContestMassPointFighter::getYawRateRange(const MotionState& m,const nl::json& extraArgs){
    return {getTwoDimTableValue("yawMin",m,extraArgs),getTwoDimTableValue("yawMax",m,extraArgs)};
}

void R7ContestMassPointFighter::loadTableFromJsonObject(const nl::json& j_table){
    //
    bool altitude_table;
    if(j_table.contains("altitude")){
        const nl::json& j=j_table.at("altitude");
        if(j.is_array()){
            j.get_to(altTable);
            assert(altTable.size()>1);
        }else{
            throw std::runtime_error("The value of 'altitude' should be 1-dim array.");
        }
    }else{
        altTable=Eigen::Vector2d(0.0,20000.0);
    }
    if(j_table.contains("vMin")){
        const nl::json& j=j_table.at("vMin");
        if(j.is_array()){
            j.get_to(vMinTable);
            if(vMinTable.size()==1){
                vMinTable=Eigen::Tensor<double,1>(altTable.size()).setConstant(vMinTable(0));
            }
            assert(vMinTable.size()==altTable.size());
        }else if(j.is_number()){
             vMinTable=Eigen::Tensor<double,1>(altTable.size()).setConstant(j.get<double>());
        }else{
            throw std::runtime_error("The value of 'vMin' should be 1-dim array or number.");
        }
    }else{
        vMinTable=Eigen::Tensor<double,1>(altTable.size()).setConstant(150.0);
    }
    if(j_table.contains("vMax")){
        const nl::json& j=j_table.at("vMax");
        if(j.is_array()){
            j.get_to(vMaxTable);
            if(vMaxTable.size()==1){
                vMaxTable=Eigen::Tensor<double,1>(altTable.size()).setConstant(vMaxTable(0));
            }
            assert(vMaxTable.size()==altTable.size());
        }else if(j.is_number()){
             vMaxTable=Eigen::Tensor<double,1>(altTable.size()).setConstant(j.get<double>());
        }else{
            throw std::runtime_error("The value of 'vMax' should be 1-dim array or number.");
        }
    }else{
        vMaxTable=Eigen::Tensor<double,1>(altTable.size()).setConstant(450.0);
    }
    std::vector<std::string> keys{
        "accelMin",
        "accelMax",
        "rollMin",
        "rollMax",
        "pitchMin",
        "pitchMax",
        "yawMin",
        "yawMax"
    };
    std::map<std::string,double> defaultValues{
        {"accelMin",-gravity},
        {"accelMax",gravity},
        {"rollMin",-deg2rad(180.0)},
        {"rollMax",deg2rad(180.0)},
        {"pitchMin",-deg2rad(30.0)},
        {"pitchMax",deg2rad(30.0)},
        {"yawMin",-deg2rad(30.0)},
        {"yawMax",deg2rad(30.0)}
    };
    int nvs=-1;
    for(auto&& key : keys){
        if(j_table.contains(key)){
            const nl::json& j=j_table.at(key);
            if(j.is_array()){
                assert(j.size()>0);
                if(j[0].is_array()){
                    // 2-dim array
                    j.get_to(twoDimTables[key]);
                }else if(j[0].is_number()){
                    // 1-dim array
                    Eigen::Tensor<double,1> tmp=j;
                    if(tmp.size()==1){
                        if(nvs<0){
                            nvs=2;
                        }
                        tmp=Eigen::Tensor<double,1>(nvs).setConstant(tmp(0));
                    }else{
                        if(nvs<0){
                            nvs=tmp.size();
                        }
                    }
                    twoDimTables[key]=tmp.reshape(Eigen::DSizes<Eigen::Index,2>{1,nvs});
                }else{
                    throw std::runtime_error("The value of '"+key+"' should be 2-dim array, 1-dim array or number.");
                }
                if(twoDimTables[key].size()==1){
                    if(nvs<0){
                        nvs=2;
                    }
                    twoDimTables[key]=Eigen::Tensor<double,2>(altTable.size(),nvs).setConstant(defaultValues[key]);
                }else if(twoDimTables[key].dimension(0)==1){
                    if(nvs<0){
                        nvs=twoDimTables[key].dimension(1);
                    }
                    Eigen::Tensor<double,2> tmp=twoDimTables[key];
                    twoDimTables[key]=Eigen::Tensor<double,2>(altTable.size(),nvs);
                    for(int i=0;i<altTable.size();++i){
                        twoDimTables[key].slice(Eigen::DSizes<Eigen::Index,2>{i,0},tmp.dimensions())=tmp;
                    }
                }else if(twoDimTables[key].dimension(1)==1){
                    if(nvs<0){
                        nvs=2;
                    }
                    Eigen::Tensor<double,2> tmp=twoDimTables[key];
                    twoDimTables[key]=Eigen::Tensor<double,2>(tmp.dimension(0),nvs);
                    for(int i=0;i<nvs;++i){
                        twoDimTables[key].slice(Eigen::DSizes<Eigen::Index,2>{0,i},tmp.dimensions())=tmp;
                    }
                }
                assert(twoDimTables[key].dimension(0)==altTable.size() && twoDimTables[key].dimension(1)==nvs);
            }else if(j.is_number()){
                if(nvs<0){
                    nvs=2;
                }
                twoDimTables[key]=Eigen::Tensor<double,2>(altTable.size(),nvs).setConstant(j.get<double>());
            }else{
                throw std::runtime_error("The value of '"+key+"' should be 2-dim array, 1-dim array or number.");
            }
        }else{
            if(nvs<0){
                nvs=2;
            }
            twoDimTables[key]=Eigen::Tensor<double,2>(altTable.size(),nvs).setConstant(defaultValues[key]);
        }
    }
}

std::shared_ptr<EntityAccessor> R7ContestMassPointFighter::getAccessorImpl(){
    return std::make_shared<R7ContestMassPointFighterAccessor>(getShared<R7ContestMassPointFighter>(this->shared_from_this()));
}

nl::json R7ContestMassPointFlightController::calcDirect(const nl::json &cmd){
    auto p=getShared<R7ContestMassPointFighter>(parent);
    auto extra=p->getExtraDynamicsState();
    auto [rollMin,rollMax]=p->getRollRateRange(p->motion,extra);
    auto [pitchMin,pitchMax]=p->getPitchRateRange(p->motion,extra);
    auto [yawMin,yawMax]=p->getYawRateRange(p->motion,extra);
    auto roll=std::clamp(cmd.at("roll").get<double>(),-1.,1.);
    if(roll>=0){
        if(rollMax>0){
            roll=std::max(rollMin,roll*abs(rollMax));
        }else{
            roll=rollMax;
        }
    }else{
        if(rollMin<0){
            roll=std::min(rollMax,roll*abs(rollMin));
        }else{
            roll=rollMin;
        }
    }
    auto pitch=std::clamp(cmd.at("pitch").get<double>(),-1.,1.);
    if(pitch>=0){
        if(pitchMax>0){
            pitch=std::max(pitchMin,pitch*abs(pitchMax));
        }else{
            pitch=pitchMax;
        }
    }else{
        if(pitchMin<0){
            pitch=std::min(pitchMax,pitch*abs(pitchMin));
        }else{
            pitch=pitchMin;
        }
    }
    auto yaw=std::clamp(cmd.at("yaw").get<double>(),-1.,1.);
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
        auto [accelMin,accelMax]=p->getRollRateRange(p->motion,extra);
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
        {"omegaB",Eigen::Vector3d(roll,pitch,yaw)},
        {"pCmd",pCmd}
    };
}
nl::json R7ContestMassPointFlightController::calcFromDirAndVel(const nl::json &cmd){
    auto p=getShared<R7ContestMassPointFighter>(parent);
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
        if(roll>M_PI_2){
            roll=M_PI-roll;
        }else if(roll<-M_PI_2){
            roll=M_PI+roll;
        }
        auto [rollMin,rollMax]=p->getRollRateRange(p->motion,extra);
        roll=std::clamp(roll,rollMin,rollMax);
        dq=Quaternion::fromAngle(ex,roll)*dq;
        if(dq.w()<0){
            dq=-dq;
        }
        double cosRot=dq.w();
        double sinRot=dq.vec().norm();
        Eigen::Vector3d ax=dq.vec().normalized();
        omegaB=p->relPtoB(ax*atan2(sinRot,cosRot)*2/dt);
        auto [pitchMin,pitchMax]=p->getPitchRateRange(p->motion,extra);
        auto [yawMin,yawMax]=p->getYawRateRange(p->motion,extra);
        omegaB*=std::min<double>({
            1.0,
            omegaB(0)==0 ? 1.0 : abs((omegaB(0)>0 ? rollMax : rollMin)/omegaB(0)),
            omegaB(1)==0 ? 1.0 : abs((omegaB(1)>0 ? pitchMax : pitchMin)/omegaB(1)),
            omegaB(2)==0 ? 1.0 : abs((omegaB(2)>0 ? yawMax : yawMin)/omegaB(2)),
        });
    }else if(cmd.contains("dstTurnRate")){
        omegaB=cmd.at("dstTurnRate");
		throw std::runtime_error("Only one of dstDir, or dstTurnRate is acceptable.");
    }

    return {
        {"omegaB",omegaB},
        {"pCmd",pCmd}
    };
}

R7ContestMassPointFighterAccessor::R7ContestMassPointFighterAccessor(const std::shared_ptr<R7ContestMassPointFighter>& f)
:FighterAccessor(f)
,fighter(f)
{
}
nl::json R7ContestMassPointFighterAccessor::getExtraDynamicsState(){
    return fighter.lock()->getExtraDynamicsState();
}
std::pair<double,double> R7ContestMassPointFighterAccessor::getVelocityRange(const MotionState& m,const nl::json& extraArgs){
    return fighter.lock()->getVelocityRange(m,extraArgs);
}
std::pair<double,double> R7ContestMassPointFighterAccessor::getAccelRange(const MotionState& m,const nl::json& extraArgs){
    return fighter.lock()->getAccelRange(m,extraArgs);
}
std::pair<double,double> R7ContestMassPointFighterAccessor::getRollRateRange(const MotionState& m,const nl::json& extraArgs){
    return fighter.lock()->getRollRateRange(m,extraArgs);
}
std::pair<double,double> R7ContestMassPointFighterAccessor::getPitchRateRange(const MotionState& m,const nl::json& extraArgs){
    return fighter.lock()->getPitchRateRange(m,extraArgs);
}
std::pair<double,double> R7ContestMassPointFighterAccessor::getYawRateRange(const MotionState& m,const nl::json& extraArgs){
    return fighter.lock()->getYawRateRange(m,extraArgs);
}

void exportR7ContestMassPointFighter(py::module &m, const std::shared_ptr<asrc::core::FactoryHelper>& factoryHelper)
{
    using namespace pybind11::literals;

    expose_entity_subclass<R7ContestMassPointFighter>(m,"R7ContestMassPointFighter")
    ;
    FACTORY_ADD_CLASS(PhysicalAsset,R7ContestMassPointFighter)

    expose_entity_subclass<R7ContestMassPointFlightController>(m,"R7ContestMassPointFlightController")
    ;
    FACTORY_ADD_CLASS(Controller,R7ContestMassPointFlightController)

    expose_common_class<R7ContestMassPointFighterAccessor>(m,"R7ContestMassPointFighterAccessor")
    .def(py_init<const std::shared_ptr<R7ContestMassPointFighter>&>())
    DEF_FUNC(R7ContestMassPointFighterAccessor,getExtraDynamicsState)
    DEF_FUNC(R7ContestMassPointFighterAccessor,getVelocityRange)
    DEF_FUNC(R7ContestMassPointFighterAccessor,getAccelRange)
    DEF_FUNC(R7ContestMassPointFighterAccessor,getRollRateRange)
    DEF_FUNC(R7ContestMassPointFighterAccessor,getPitchRateRange)
    DEF_FUNC(R7ContestMassPointFighterAccessor,getYawRateRange)
    ;
}

ASRC_PLUGIN_NAMESPACE_END
