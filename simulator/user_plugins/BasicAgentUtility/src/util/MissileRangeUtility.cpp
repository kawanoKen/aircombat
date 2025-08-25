// Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#include "util/MissileRangeUtility.h"
#include <ASRCAISim1/Fighter.h>

ASRC_PLUGIN_NAMESPACE_BEGIN

namespace util{

using namespace asrc::core;
using namespace util;

double calcRHead(const std::shared_ptr<PhysicalAssetAccessor>& parent,const MotionState& motion,const Track3D& track, bool shooter_head_on){
    std::shared_ptr<FighterAccessor> f=std::dynamic_pointer_cast<FighterAccessor>(parent);
	auto crs=motion.getCRS();
	Eigen::Vector3d rt=track.pos(crs);
	Eigen::Vector3d vt=track.vel(crs);
	Eigen::Vector3d rs=motion.pos(crs);
	Eigen::Vector3d vs=motion.vel(crs);
    if(shooter_head_on){
        double Vs=vs.norm();
        vs=(rt-rs).normalized()*Vs;
    }
    return f->getRmax(rs,vs,rt,vt,M_PI,crs);
}
double calcRTail(const std::shared_ptr<PhysicalAssetAccessor>& parent,const MotionState& motion,const Track3D& track, bool shooter_head_on){
    std::shared_ptr<FighterAccessor> f=std::dynamic_pointer_cast<FighterAccessor>(parent);
	auto crs=motion.getCRS();
	Eigen::Vector3d rt=track.pos(crs);
	Eigen::Vector3d vt=track.vel(crs);
	Eigen::Vector3d rs=motion.pos(crs);
	Eigen::Vector3d vs=motion.vel(crs);
    if(shooter_head_on){
        double Vs=vs.norm();
        vs=(rt-rs).normalized()*Vs;
    }
    return f->getRmax(rs,vs,rt,vt,0.0,crs);
}
double calcRNorm(const std::shared_ptr<PhysicalAssetAccessor>& parent,const MotionState& motion,const Track3D& track, bool shooter_head_on){
    double RHead=calcRHead(parent,motion,track,shooter_head_on);
    double RTail=calcRTail(parent,motion,track,shooter_head_on);
	auto crs=motion.getCRS();
	Eigen::Vector3d rs=motion.pos(crs);
	Eigen::Vector3d rt=track.pos(crs);
    double r=(rs-rt).norm()-RTail;
    double delta=RHead-RTail;
    double outRangeScale=100000.0;
    if(delta==0){
        if(r<0){
            r=r/outRangeScale;
        }else if(r>0){
            r=1+r/outRangeScale;
        }else{
            r=0;
        }
    }else{
        if(r<0){
            r=r/outRangeScale;
        }else if(r>delta){
            r=1+(r-delta)/outRangeScale;
        }else{
            r/=delta;
        }
    }
    return r;
}
double calcRHeadE(const std::shared_ptr<PhysicalAssetAccessor>& parent,const MotionState& motion,const Track3D& track, bool shooter_head_on){
    std::shared_ptr<FighterAccessor> f=std::dynamic_pointer_cast<FighterAccessor>(parent);
	auto crs=motion.getCRS();
    Eigen::Vector3d rs=track.pos(crs);
    Eigen::Vector3d vs=track.vel(crs);
    Eigen::Vector3d rt=motion.pos(crs);
    Eigen::Vector3d vt=motion.vel(crs);
    if(shooter_head_on){
        double Vs=vs.norm();
        vs=(rt-rs).normalized()*Vs;
    }
    return f->getRmax(rs,vs,rt,vt,0.0,crs);
}
double calcRTailE(const std::shared_ptr<PhysicalAssetAccessor>& parent,const MotionState& motion,const Track3D& track, bool shooter_head_on){
    std::shared_ptr<FighterAccessor> f=std::dynamic_pointer_cast<FighterAccessor>(parent);
	auto crs=motion.getCRS();
    Eigen::Vector3d rs=track.pos(crs);
    Eigen::Vector3d vs=track.vel(crs);
    Eigen::Vector3d rt=motion.pos(crs);
    Eigen::Vector3d vt=motion.vel(crs);
    if(shooter_head_on){
        double Vs=vs.norm();
        vs=(rt-rs).normalized()*Vs;
    }
    return f->getRmax(rs,vs,rt,vt,M_PI,crs);
}
double calcRNormE(const std::shared_ptr<PhysicalAssetAccessor>& parent,const MotionState& motion,const Track3D& track, bool shooter_head_on){
    double RHead=calcRHeadE(parent,motion,track,shooter_head_on);
    double RTail=calcRTailE(parent,motion,track,shooter_head_on);
	auto crs=motion.getCRS();
	Eigen::Vector3d rt=motion.pos(crs);
	Eigen::Vector3d rs=track.pos(crs);
    double r=(rs-rt).norm()-RTail;
    double delta=RHead-RTail;
    double outRangeScale=100000.0;
    if(delta==0){
        if(r<0){
            r=r/outRangeScale;
        }else if(r>0){
            r=1+r/outRangeScale;
        }else{
            r=0;
        }
    }else{
        if(r<0){
            r=r/outRangeScale;
        }else if(r>delta){
            r=1+(r-delta)/outRangeScale;
        }else{
            r/=delta;
        }
    }
    return r;
}

void exportMissileRangeUtility(py::module &m)
{
    using namespace pybind11::literals;

    m.def("calcRHead",&calcRHead);
    m.def("calcRTail",&calcRTail);
    m.def("calcRNorm",&calcRNorm);
    m.def("calcRHeadE",&calcRHeadE);
    m.def("calcRTailE",&calcRTailE);
    m.def("calcRNormE",&calcRNormE);
}

}

ASRC_PLUGIN_NAMESPACE_END
