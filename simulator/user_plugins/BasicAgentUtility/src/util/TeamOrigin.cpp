// Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#include "util/TeamOrigin.h"
#include <ASRCAISim1/Utility.h>

ASRC_PLUGIN_NAMESPACE_BEGIN

namespace util{

using namespace asrc::core;
using namespace util;

TeamOrigin::TeamOrigin(){
    isEastSider=true;
    pos<<0,0,0;
}
TeamOrigin::TeamOrigin(bool isEastSider_,double dLine){
    isEastSider=isEastSider_;
    if(isEastSider){
        pos=Eigen::Vector3d(0,dLine,0);
    }else{
        pos=Eigen::Vector3d(0,-dLine,0);
    }
}
TeamOrigin::~TeamOrigin(){}
Eigen::Vector3d TeamOrigin::relBtoP(const Eigen::Vector3d& v) const{//陣営座標系⇛慣性座標系
    if(isEastSider){
        return Eigen::Vector3d(v(1),-v(0),v(2));
    }else{
        return Eigen::Vector3d(-v(1),v(0),v(2));
    }
}
Eigen::Vector3d TeamOrigin::relPtoB(const Eigen::Vector3d& v) const{//慣性座標系⇛陣営座標系
    if(isEastSider){
        return Eigen::Vector3d(-v(1),v(0),v(2));
    }else{
        return Eigen::Vector3d(v(1),-v(0),v(2));
    }
}
Eigen::Vector3d TeamOrigin::absBtoP(const Eigen::Vector3d& v) const{//陣営座標系⇛慣性座標系
    return relBtoP(v)+pos;
}
Eigen::Vector3d TeamOrigin::absPtoB(const Eigen::Vector3d& v) const{//慣性座標系⇛陣営座標系
    return relPtoB(v-pos);
}

void exportTeamOrigin(py::module &m)
{
    using namespace pybind11::literals;
    py::class_<TeamOrigin,std::shared_ptr<TeamOrigin>>(m,"TeamOrigin")
    .def(py::init<>())
    .def(py::init<bool,double>())
    DEF_FUNC(TeamOrigin,relBtoP)
    DEF_FUNC(TeamOrigin,relPtoB)
    DEF_FUNC(TeamOrigin,absBtoP)
    DEF_FUNC(TeamOrigin,absPtoB)
    DEF_READWRITE(TeamOrigin,isEastSider)
    DEF_READWRITE(TeamOrigin,pos)
    ;
}

}

ASRC_PLUGIN_NAMESPACE_END
