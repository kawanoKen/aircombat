// Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#include "Coordinate.h"
#include "crs/CoordinateReferenceSystemBase.h"
#include "crs/ProjectedCRS.h"
#include "Units.h"

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)

using namespace util;

Coordinate::Coordinate()
:crs(nullptr),time(),value(0,0,0),location(0,0,0),type(CoordinateType::POSITION_ABS){
}
Coordinate::Coordinate(const Eigen::Vector3d& value_, const Eigen::Vector3d& location_, const std::shared_ptr<CoordinateReferenceSystem>& crs_, const Time& time_, const CoordinateType& type_)
:crs(crs_),time(time_),value(value_),location((type_==CoordinateType::POSITION_ABS) ? value_ : location_),type(type_){
}
Coordinate::Coordinate(const Eigen::Vector3d& value_, const std::shared_ptr<CoordinateReferenceSystem>& crs_, const Time& time_, const CoordinateType& type_)
:crs(crs_),time(time_),value(value_),location((type_==CoordinateType::POSITION_ABS) ? value_ : Eigen::Vector3d(0,0,0)),type(type_){
}
Coordinate::Coordinate(const nl::json& j_):Coordinate(){
    load_from_json(j_);
}

// getter,setter
std::shared_ptr<CoordinateReferenceSystem> Coordinate::getCRS() const noexcept{
    return crs;
}
Coordinate& Coordinate::setCRS(const std::shared_ptr<CoordinateReferenceSystem>& newCRS,bool transform) &{
    //CRSを再設定する。座標変換の有無を第2引数で指定する。
    if(!newCRS){
        throw std::runtime_error("Do not give nullptr as a new CRS of Coordinate!");
    }
    if(transform){
        value=crs->transformTo(value,location,time,newCRS,type)();
        location=crs->transformTo(location,location,time,newCRS,CoordinateType::POSITION_ABS)();
    }
    crs=newCRS;
    return *this;
}
Coordinate Coordinate::setCRS(const std::shared_ptr<CoordinateReferenceSystem>& newCRS,bool transform) &&{
    //CRSを再設定する。座標変換の有無を第2引数で指定する。
    if(!newCRS){
        throw std::runtime_error("Do not give nullptr as a new CRS of Coordinate!");
    }
    if(transform){
        value=crs->transformTo(value,location,time,newCRS,type)();
        location=crs->transformTo(location,location,time,newCRS,CoordinateType::POSITION_ABS)();
    }
    crs=newCRS;
    return std::move(*this);
}
Coordinate Coordinate::asTypeOf(const CoordinateType& newType) const{
    return Coordinate(
        value,
        location,
        crs,
        time,
        newType
    );
}
Time Coordinate::getTime() const{
    return time;
}
Coordinate& Coordinate::setTime(const Time& time_) &{
    time=time_;
    return *this;
}
Coordinate Coordinate::setTime(const Time& time_) &&{
    time=time_;
    return std::move(*this);
}
CoordinateType Coordinate::getType() const{
    return type;
}
Coordinate& Coordinate::setType(const CoordinateType& type_) &{
    type=type_;
    return *this;
}
Coordinate Coordinate::setType(const CoordinateType& type_) &&{
    type=type_;
    return std::move(*this);
}
Eigen::Vector3d Coordinate::operator()() const noexcept{
    return getValue();
}
double Coordinate::operator()(const Eigen::Index& i) const{
    return getValue(i);
}
Eigen::Vector3d Coordinate::operator()(const std::shared_ptr<CoordinateReferenceSystem>& newCRS) const{
    return getValue(newCRS);
}
Eigen::Vector3d Coordinate::getValue() const noexcept{
    return value;
}
double Coordinate::getValue(const Eigen::Index& i) const{
    return value(i);//これ自身で例外は投げずEigen::Vector側に委ねる
}
Eigen::Vector3d Coordinate::getValue(const std::shared_ptr<CoordinateReferenceSystem>& newCRS) const{
    if(crs==newCRS){
        //同一ならバイパス
        return value;
    }
    if(!crs){
        throw std::runtime_error("Coordinate without CRS cannot be converted to another CRS.");
    }
    if(!newCRS){
        throw std::runtime_error("Do not give nullptr as new parent CRS of Coordinate!");
    }
    return crs->transformTo(value,location,time,newCRS,type)();
}
Coordinate& Coordinate::setValue(const Eigen::Vector3d& value_) &{
    value=value_;
    if(type==CoordinateType::POSITION_ABS){
        location=value_;
    }
    return *this;
}
Coordinate Coordinate::setValue(const Eigen::Vector3d& value_) &&{
    value=value_;
    if(type==CoordinateType::POSITION_ABS){
        location=value_;
    }
    return std::move(*this);
}
Eigen::Vector3d Coordinate::getLocation() const noexcept{
    return location;
}
double Coordinate::getLocation(const Eigen::Index& i) const{
    return location(i);//これ自身で例外は投げずEigen::Vector側に委ねる
}
Eigen::Vector3d Coordinate::getLocation(const std::shared_ptr<CoordinateReferenceSystem>& newCRS) const{
    if(crs==newCRS){
        //同一ならバイパス
        return location;
    }
    if(!crs){
        throw std::runtime_error("Coordinate without CRS cannot be converted to another CRS.");
    }
    if(!newCRS){
        throw std::runtime_error("Do not give nullptr as new parent CRS of Coordinate!");
    }
    return crs->transformTo(location,location,time,newCRS,CoordinateType::POSITION_ABS)();
}
Coordinate& Coordinate::setLocation(const Eigen::Vector3d& location_) &{
    location=location_;
    if(type==CoordinateType::POSITION_ABS){
        value=location_;
    }
    return *this;
}
Coordinate Coordinate::setLocation(const Eigen::Vector3d& location_) &&{
    location=location_;
    if(type==CoordinateType::POSITION_ABS){
        value=location_;
    }
    return std::move(*this);
}

// ノルム・単位ベクトル
double Coordinate::norm() const{
    return value.norm();
}
void Coordinate::normalize(){
    value.normalize();
}
Coordinate Coordinate::normalized() const{
    return std::move(Coordinate(value.normalized(),crs,time,type));
}

// 座標変換
Coordinate Coordinate::transformTo(const std::shared_ptr<CoordinateReferenceSystem>& newCRS) const{
    if(crs==newCRS){
        //同一ならバイパス
        return *this;
    }
    if(!crs){
        throw std::runtime_error("Coordinate without CRS cannot be converted to another CRS.");
    }
    if(!newCRS){
        throw std::runtime_error("Do not give nullptr as new parent CRS of Coordinate!");
    }
    return Coordinate(
        crs->transformTo(value,location,time,newCRS,type)(),
        crs->transformTo(location,location,time,newCRS,CoordinateType::POSITION_ABS)(),
        newCRS,
        time,
        type
    );
}

// 座標軸の変換 (axis conversion in the parent crs)
Eigen::Vector3d Coordinate::toCartesian() const{
    return crs ? crs->toCartesian(value) : value;
}
Eigen::Vector3d Coordinate::toSpherical() const{
    if(crs){
        return crs->toSpherical(value);
    }else{
        //親CRSの指定が無い場合はNEDのTopocentricとみなす。
        //NED -> (Azimuth Elevation Range)
        double xy=sqrt(value(0)*value(0)+value(1)*value(1));
        if(xy==0){
            return Eigen::Vector3d(
                0,
                value(2)>0 ? -M_PI_2 : M_PI_2,
                abs(value(2))
            );
        }else{
            double el=atan2(-value(2),xy);
            return Eigen::Vector3d(
                atan2(value(1),value(0)),
                el,
                sqrt(xy*xy+value(2)*value(2))
            );
        }
    }
}
Eigen::Vector3d Coordinate::toEllipsoidal() const{
    if(crs){
        return crs->toEllipsoidal(value);
    }else{
        throw std::runtime_error("axis conversion to ellipsoidal axis is not defined for Coordinate.");
    }
}

void exportCoordinate(py::module &m){
    using namespace pybind11::literals;

    expose_enum_value_helper(
        expose_enum_class<CoordinateType>(m,"CoordinateType")
        ,"POSITION_ABS"
        ,"POSITION_REL"
        ,"DIRECTION"
        ,"VELOCITY"
        ,"ANGULAR_VELOCITY"
    );

    expose_common_class<Coordinate>(m,"Coordinate")
    .def(py_init<const Eigen::Vector3d&,const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&,const Time&,const CoordinateType&>())
    .def(py_init<const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&,const Time&,const CoordinateType&>())
    .def(py_init<const nl::json&>())
    // getter,setter
    DEF_FUNC(Coordinate,getCRS)
    .def("setCRS",[](Coordinate& v,const std::shared_ptr<CoordinateReferenceSystem>& newCRS,bool transform){return v.setCRS(newCRS,transform);},py::return_value_policy::reference)
    DEF_FUNC(Coordinate,transformTo)
    DEF_FUNC(Coordinate,getTime)
    .def("setTime",[](Coordinate& v,const Time& time_){return v.setTime(time_);},py::return_value_policy::reference)
    DEF_FUNC(Coordinate,getType)
    .def("setType",[](Coordinate& v,const CoordinateType& type_){return v.setType(type_);},py::return_value_policy::reference)
    DEF_FUNC(Coordinate,asTypeOf)
    .def("__call__",py::overload_cast<>(&Coordinate::operator(),py::const_))
    .def("__call__",py::overload_cast<const Eigen::Index&>(&Coordinate::operator(),py::const_))
    .def("__call__",py::overload_cast<const std::shared_ptr<CoordinateReferenceSystem>&>(&Coordinate::operator(),py::const_))
    .def("getValue",py::overload_cast<>(&Coordinate::getValue,py::const_))
    .def("getValue",py::overload_cast<const Eigen::Index&>(&Coordinate::getValue,py::const_))
    .def("getValue",py::overload_cast<const std::shared_ptr<CoordinateReferenceSystem>&>(&Coordinate::getValue,py::const_))
    .def("setValue",[](Coordinate& v,const Eigen::Vector3d value_){return v.setValue(value_);},py::return_value_policy::reference)
    .def("getLocation",py::overload_cast<>(&Coordinate::getLocation,py::const_))
    .def("getLocation",py::overload_cast<const Eigen::Index&>(&Coordinate::getLocation,py::const_))
    .def("getLocation",py::overload_cast<const std::shared_ptr<CoordinateReferenceSystem>&>(&Coordinate::getLocation,py::const_))
    .def("setLocation",[](Coordinate& v,const Eigen::Vector3d location_){return v.setLocation(location_);},py::return_value_policy::reference)
    // ノルム・単位ベクトル
    DEF_FUNC(Coordinate,norm)
    DEF_FUNC(Coordinate,normalize)
    DEF_FUNC(Coordinate,normalized)
    // 座標変換
    DEF_FUNC(Coordinate,transformTo)
    // 座標軸の変換 (axis conversion in the parent crs)
    DEF_FUNC(Coordinate,toCartesian)
    DEF_FUNC(Coordinate,toSpherical)
    DEF_FUNC(Coordinate,toEllipsoidal)
    ;

}

ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
