// Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#include "StaticCollisionAvoider2D.h"
#include "MathUtility.h"
#include "Utility.h"
#include "Units.h"

namespace py=pybind11;

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)
ASRC_INLINE_NAMESPACE_BEGIN(util)

double asAngle(double v){
    //-π〜+π
    double ret=v;
    while(ret>=M_PI){ret-=2*M_PI;}
    while(ret<-M_PI){ret+=2*M_PI;}
    return ret;
}
double asNonNegativeAngle(double v){
    //0〜+2π
    double ret=v;
    while(ret>=2*M_PI){ret-=2*M_PI;}
    while(ret<0){ret+=2*M_PI;}
    return ret;
}
double getAZ(const Eigen::VectorXd& v){
    return atan2(v(1),v(0));
}
bool isInsideSector(const std::pair<double,double>& sector,double angle){
    double width=asNonNegativeAngle(sector.second-sector.first);
    double d=asNonNegativeAngle(angle-sector.first);
    return 0<d && d<width;
}
std::vector<std::pair<double,double>> mergeSectors(const std::pair<double,double>& s1,const std::pair<double,double>& s2){
    std::vector<std::pair<double,double>> ret;
    if(isInsideSector(s1,s2.first)){
        double e1=asNonNegativeAngle(s1.second-s1.first);
        double e2=asNonNegativeAngle(s2.second-s1.first);
        ret.push_back(std::make_pair(s1.first,asAngle(s1.first+std::max(e1,e2))));
    }else if(isInsideSector(s2,s1.first)){
        double e1=asNonNegativeAngle(s1.second-s2.first);
        double e2=asNonNegativeAngle(s2.second-s2.first);
        ret.push_back(std::make_pair(s2.first,asAngle(s2.first+std::max(e1,e2))));
    }else{
        ret.push_back(s1);
        ret.push_back(s2);
    }
    return std::move(ret);
}

BorderGeometry::BorderGeometry(const nl::json& config){
    limit=getValueFromJsonKD(config,"limit",3000.0);
    threshold=getValueFromJsonKD(config,"threshold",7500.0);
    adjustStrength=getValueFromJsonKD(config,"adjustStrength",1.0e-3);
}
BorderGeometry::~BorderGeometry(){
}

LinearSegment::LinearSegment(const nl::json& config):BorderGeometry(config){
    p1=getValueFromJsonKD<Eigen::VectorXd>(config,"p1",Eigen::Vector2d(0,0)).block<2,1>(0,0);
    p2=getValueFromJsonKD<Eigen::VectorXd>(config,"p2",Eigen::Vector2d(0,0)).block<2,1>(0,0);
    infinite_p1=getValueFromJsonKD(config,"infinite_p1",false);
    infinite_p2=getValueFromJsonKD(config,"infinite_p2",false);
    isOneSide=getValueFromJsonKD(config,"isOneSide",false);
    if(isOneSide){
        Eigen::Vector2d inner=getValueFromJsonKD<Eigen::VectorXd>(config,"inner",inner).block<2,1>(0,0);
        double L=(p2-p1).norm();
        double t = L>0 ? (inner-p1).dot(p2-p1)/(L*L) : (inner-p1).norm();
        Eigen::Vector2d h=p1*(1-t)+p2*t;
        normal=(inner-h).normalized();
    }else{
        normal=Eigen::Vector2d(0,0);        
    }
}
LinearSegment::~LinearSegment(){
}
std::pair<double,double> LinearSegment::calcUnacceptableRegion(const MotionState& motion, const Eigen::VectorXd& dstDir){
    Eigen::Vector2d r(motion.pos(0),motion.pos(1));
    double L=(p2-p1).norm();
    double t = L>0 ? (r-p1).dot(p2-p1)/(L*L) : (r-p1).norm();
    Eigen::Vector2d h=p1*(1-t)+p2*t;
    double d,ah;
    if(isOneSide){
        d=normal.dot(r-h);
        ah=getAZ(-normal);
    }else{
        d=(r-h).norm();
        ah=getAZ(h-r);
    }
    double dAZ=M_PI/2.0-atan(adjustStrength*(d-limit));
    double a1,a2;
    if(t<0 && !infinite_p1){
        bool inside = (r-p1).norm()<threshold;
        double a0=getAZ(p1-r);
        if(inside){
            double dAZ1=asAngle(a0-ah);
            if(dAZ1>=0){
                if(dAZ1<dAZ){
                    a1=a0;
                    a2=asAngle(ah+dAZ);
                }else{
                    a1=a0;
                    a2=a0;
                }
            }else{
                if(-dAZ<dAZ1){
                    a1=asAngle(ah-dAZ);
                    a2=a0;
                }else{
                    a1=a0;
                    a2=a0;
                }
            }
        }else{
            a1=a0;
            a2=a0;
        }
    }else if(t>1 && !infinite_p2){
        bool inside = (r-p2).norm()<threshold;
        double a0=getAZ(p2-r);
        if(inside){
            double dAZ2=asAngle(a0-ah);
            if(dAZ2>=0){
                if(dAZ2<dAZ){
                    a1=a0;
                    a2=asAngle(ah+dAZ);
                }else{
                    a1=a0;
                    a2=a0;
                }
            }else{
                if(-dAZ<dAZ2){
                    a1=asAngle(ah-dAZ);
                    a2=a0;
                }else{
                    a1=a0;
                    a2=a0;
                }
            }
        }else{
            a1=a0;
            a2=a0;
        }
    }else{// if(t>=0 && t<=1){
        bool inside = d<threshold;
        if(inside){
            a1=asAngle(ah-dAZ);
            a2=asAngle(ah+dAZ);
        }else{
            a1=ah;
            a2=ah;
        }
    }
    return std::make_pair(a1,a2);
}
StaticCollisionAvoider2D::StaticCollisionAvoider2D(){
}
StaticCollisionAvoider2D::~StaticCollisionAvoider2D(){
}
Eigen::VectorXd StaticCollisionAvoider2D::operator()(const MotionState& motion, const Eigen::VectorXd& dstDir){
    std::vector<std::pair<double,double>> ret;
    for(auto&& b:borders){
       std::pair<double,double> s=b->calcUnacceptableRegion(motion,dstDir);
        if(s.first==s.second){
            continue;
        }
        if(ret.size()==0){
            ret.push_back(s);
        }else{
            std::vector<std::pair<double,double>> merged;
            for(auto&& r:ret){
                std::vector<std::pair<double,double>> m=mergeSectors(r,s);
                if(m.size()==1){//merged
                    s=m[0];
                }//else{s=m[1];}
                merged.push_back(m[0]);
            }
            ret=std::move(merged);
        }
    }
    double dstAZ=getAZ(dstDir);
    for(auto&& r:ret){
        if(isInsideSector(r,dstAZ)){
            double d1=asAngle(r.first-dstAZ);
            double d2=asAngle(r.second-dstAZ);
            if(abs(d1)<abs(d2)){
                dstAZ=r.first;
            }else{
                dstAZ=r.second;
            }
            break;
        }
    }
    if(dstDir.size()>2){
        double factor=sqrt(1.0-dstDir(2)*dstDir(2));
        return Eigen::Vector3d(cos(dstAZ)*factor,sin(dstAZ)*factor,dstDir(2));
    }else{
        return Eigen::Vector2d(cos(dstAZ),sin(dstAZ));
    }
}

void exportStaticCollisionAvoider2D(py::module &m)
{
    using namespace pybind11::literals;
    m.def("asAngle",&asAngle);
    m.def("asNonNegativeAngle",&asNonNegativeAngle);
    m.def("getAZ",&getAZ);
    m.def("isInsideSector",&isInsideSector);
    m.def("mergeSectors",&mergeSectors);

    bind_stl_container<std::vector<std::shared_ptr<BorderGeometry>>>(m,py::module_local(false));

    expose_common_class<BorderGeometry>(m,"BorderGeometry")
    .def(py_init<const nl::json&>())
    DEF_FUNC(BorderGeometry,calcUnacceptableRegion)
    DEF_READWRITE(BorderGeometry,limit)
    DEF_READWRITE(BorderGeometry,threshold)
    DEF_READWRITE(BorderGeometry,adjustStrength)
    ;

    expose_common_class<LinearSegment>(m,"LinearSegment")
    .def(py_init<const nl::json&>())
    DEF_FUNC(LinearSegment,calcUnacceptableRegion)
    DEF_READWRITE(LinearSegment,p1)
    DEF_READWRITE(LinearSegment,p2)
    DEF_READWRITE(LinearSegment,isOneSide)
    DEF_READWRITE(LinearSegment,normal)
    ;

    py::class_<StaticCollisionAvoider2D>(m,"StaticCollisionAvoider2D")
    .def(py::init<>())
    .def("__call__",&StaticCollisionAvoider2D::operator())
    DEF_READWRITE(StaticCollisionAvoider2D,borders)
    ;
}

ASRC_NAMESPACE_END(util)
ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
