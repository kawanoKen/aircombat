// Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#include "Sensor.h"
#include "Utility.h"
#include "Units.h"
#include "SimulationManager.h"
#include "Agent.h"
#include "Fighter.h"
#include "Missile.h"
#include <boost/uuid/nil_generator.hpp>

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)
using namespace util;

void Sensor::initialize(){
    BaseType::initialize();
    assert(!parent.expired() && isBoundToParent);
}
std::pair<bool,std::vector<Track3D>> Sensor3D::isTrackingAny(){
    return std::make_pair(false,std::vector<Track3D>());
}
std::pair<bool,Track3D> Sensor3D::isTracking(const std::weak_ptr<PhysicalAsset>& target_){
    return std::make_pair(false,Track3D());
}
std::pair<bool,Track3D> Sensor3D::isTracking(const Track3D& target_){
    return std::make_pair(false,Track3D());
}
std::pair<bool,Track3D> Sensor3D::isTracking(const boost::uuids::uuid& target_){
    return std::make_pair(false,Track3D());
}

std::pair<bool,std::vector<Track2D>> Sensor2D::isTrackingAny(){
    return std::make_pair(false,std::vector<Track2D>());
}
std::pair<bool,Track2D> Sensor2D::isTracking(const std::weak_ptr<PhysicalAsset>& target_){
    return std::make_pair(false,Track2D());
}
std::pair<bool,Track2D> Sensor2D::isTracking(const Track2D& target_){
    return std::make_pair(false,Track2D());
}
std::pair<bool,Track2D> Sensor2D::isTracking(const boost::uuids::uuid& target_){
    return std::make_pair(false,Track2D());
}

void AircraftRadar::initialize(){
    BaseType::initialize();
    Lref=getValueFromJsonKR(modelConfig,"Lref",randomGen);
    thetaFOR=deg2rad(getValueFromJsonKR(modelConfig,"thetaFOR",randomGen));
    observables={
        {"spec",{
            {"Lref",Lref},
            {"thetaFOR",thetaFOR}
        }}
    };
}
void AircraftRadar::serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full){
    BaseType::serializeInternalState(archive,full);

    if(full){
        ASRC_SERIALIZE_NVP(archive
            ,Lref
            ,thetaFOR
        )
    }
    ASRC_SERIALIZE_NVP(archive
        ,track
    )
}
void AircraftRadar::setDependency(){
    //no dependency
}
void AircraftRadar::perceive(bool inReset){
    //PhysicalAsset::perceive(inReset);//Sensors don't need motion and isAlive as observable.
    track.clear();
    auto crs=getParentCRS();// equals to parent.lock()->getBodyCRS() because this is bound to parent.
    if(!crs){
        throw std::runtime_error("crs is nullptr!");
    }
    for(auto&& e:manager->getAssets([&](const std::shared_ptr<const Asset>& asset)->bool {
        return asset->getTeam()!=getTeam() && isinstance<Fighter>(asset);
    })){
        auto f=getShared<Fighter>(e);
        if(f->isAlive()){
            Eigen::Vector3d fpos=f->pos(crs);
            Eigen::Vector3d rpos=absPtoB(fpos);
            double L=rpos.norm();
            if(rpos(0)>L*cos(thetaFOR) && (Lref<0 || L<=Lref*pow(f->rcsScale,0.25))){
                track.push_back(Track3D(f,crs,manager->getTime(),fpos,f->vel(crs)));
            }
        }
    }
    observables["track"]=track;
}
std::pair<bool,std::vector<Track3D>> AircraftRadar::isTrackingAny(){
    if(track.size()>0){
        std::vector<Track3D> ret;
        for(auto& e:track){
            ret.push_back(e);
        }
        return std::make_pair(true,std::move(ret));
    }else{
        return std::make_pair(false,std::vector<Track3D>());
    }
}
std::pair<bool,Track3D> AircraftRadar::isTracking(const std::weak_ptr<PhysicalAsset>& target_){
    if(target_.expired()){
        return std::make_pair(false,Track3D());
    }else{
        return isTracking(target_.lock()->getUUIDForTrack());
    }
}
std::pair<bool,Track3D> AircraftRadar::isTracking(const Track3D& target_){
    if(target_.is_none()){
        return std::make_pair(false,Track3D());
    }else{
        return isTracking(target_.truth);
    }
}
std::pair<bool,Track3D> AircraftRadar::isTracking(const boost::uuids::uuid& target_){
    if(target_==boost::uuids::nil_uuid()){
        return std::make_pair(false,Track3D());
    }
    for(auto& e:track){
        if(e.isSame(target_)){
            return std::make_pair(true,e);
        }
    }
    return std::make_pair(false,Track3D());
}

void MWS::initialize(){
    BaseType::initialize();
    isEsmIsh=getValueFromJsonKRD(modelConfig,"isEsmIsh",randomGen,false);
    Lref=getValueFromJsonKRD(modelConfig,"Lref",randomGen,20000.0);
    thetaFOR=deg2rad(getValueFromJsonKRD(modelConfig,"thetaFOR",randomGen,90.0));
    observables={
        {"spec",{
            {"isEsmIsh",isEsmIsh},
            {"Lref",Lref},
            {"thetaFOR",thetaFOR}
        }}
    };
}
void MWS::serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full){
    BaseType::serializeInternalState(archive,full);

    if(full){
        ASRC_SERIALIZE_NVP(archive
            ,isEsmIsh
            ,Lref
            ,thetaFOR
        )
    }
    ASRC_SERIALIZE_NVP(archive
        ,track
    )
}
void MWS::setDependency(){
    manager->addDependencyGenerator([manager=manager,wPtr=weak_from_this()](const std::shared_ptr<Asset>& asset){
        if(!wPtr.expired()){
            auto ptr=getShared<Type>(wPtr);
            if(
                asset->getTeam()!=ptr->getTeam()
                && isinstance<Missile>(asset)
            ){
                auto m=getShared<const Missile>(asset);
                manager->addDependency(SimPhase::PERCEIVE,m->sensor.lock(),ptr);
            }
        }
    });
}
void MWS::perceive(bool inReset){
    //PhysicalAsset::perceive(inReset);//Sensors don't need motion and isAlive as observable.
    track.clear();
    auto crs=getParentCRS();
    Eigen::Vector3d myPos=pos(crs);
    for(auto&& e:manager->getAssets([&](const std::shared_ptr<const Asset>& asset)->bool {
        return asset->getTeam()!=getTeam() && isinstance<Missile>(asset);
    })){
        auto m=getShared<Missile>(e);
        Eigen::Vector3d rpos=absPtoB(m->pos(crs));
        double L=rpos.norm();
        if(isEsmIsh){
            if(m->hasLaunched && m->isAlive() && m->sensor.lock()->isActive){
                std::pair<bool,Track3D> r=m->sensor.lock()->isTracking(Track3D(parent,crs));
                if(r.first && rpos(0)>L*cos(thetaFOR) && (Lref<0 || L<=Lref)){
                    track.push_back(Track2D(m,crs,myPos));//真値を入れる
                }
            }
        }else{
            if(m->hasLaunched && m->isAlive() && rpos(0)>L*cos(thetaFOR) && (Lref<0 || L<=Lref)){
                track.push_back(Track2D(m,crs,myPos));//真値を入れる
            }
        }
    }
    observables["track"]=track;
}
std::pair<bool,std::vector<Track2D>> MWS::isTrackingAny(){
    if(track.size()>0){
       std::vector<Track2D> ret;
       for(auto& e:track){
           ret.push_back(e);
       }
        return std::make_pair(true,std::move(ret));
    }else{
        return std::make_pair(false,std::vector<Track2D>());
    }
}
std::pair<bool,Track2D> MWS::isTracking(const std::weak_ptr<PhysicalAsset>& target_){
    if(target_.expired()){
        return std::make_pair(false,Track2D());
    }else{
        return isTracking(target_.lock()->getUUIDForTrack());
    }
}
std::pair<bool,Track2D> MWS::isTracking(const Track2D& target_){
    if(target_.is_none()){
        return std::make_pair(false,Track2D());
    }else{
        return isTracking(target_.truth);
    }
}
std::pair<bool,Track2D> MWS::isTracking(const boost::uuids::uuid& target_){
    if(target_==boost::uuids::nil_uuid()){
        return std::make_pair(false,Track2D());
    }
    for(auto& e:track){
        if(e.isSame(target_)){
            return std::make_pair(true,e);
        }
    }
    return std::make_pair(false,Track2D());
}

void MissileSensor::initialize(){
    BaseType::initialize();
    Lref=getValueFromJsonKR(modelConfig,"Lref",randomGen);
    thetaFOR=deg2rad(getValueFromJsonKR(modelConfig,"thetaFOR",randomGen));
    thetaFOV=deg2rad(getValueFromJsonKR(modelConfig,"thetaFOV",randomGen));
    isActive=false;
    estTPos=Coordinate(Eigen::Vector3d::Zero(),getParentCRS(),manager->getTime(),CoordinateType::POSITION_ABS);
    observables={
        {"spec",{
            {"Lref",Lref},
            {"thetaFOR",thetaFOR},
            {"thetaFOV",thetaFOV}
        }}
    };
}
void MissileSensor::serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full){
    BaseType::serializeInternalState(archive,full);

    if(full){
        ASRC_SERIALIZE_NVP(archive
            ,Lref
            ,thetaFOR
            ,thetaFOV
        )
    }
    ASRC_SERIALIZE_NVP(archive
        ,track
        ,target
        ,isActive
        ,estTPos
    )
}
void MissileSensor::setDependency(){
    manager->addDependency(SimPhase::CONTROL,parent.lock(),shared_from_this());
}
void MissileSensor::perceive(bool inReset){
    //PhysicalAsset::perceive(inReset);//Sensors don't need motion and isAlive as observable.
    track.clear();
    if(isActive){
        auto crs=getParentCRS();
        Eigen::Vector3d steer=absPtoB(estTPos(crs)).normalized();
        for(auto&& e:manager->getAssets([&](const std::shared_ptr<const Asset>& asset)->bool {
            return asset->getTeam()!=getTeam() && isinstance<Fighter>(asset);
        })){
            auto f=getShared<Fighter>(e);
            if(f->isAlive()){
                Eigen::Vector3d fpos=f->pos(crs);
                Eigen::Vector3d rpos=absPtoB(fpos);
                double L=rpos.norm();
                if(rpos(0)>L*cos(thetaFOR) && (Lref<0 || L<=Lref) && rpos.dot(steer)>L*cos(thetaFOV)){
                    track.push_back(Track3D(f,crs,manager->getTime(),fpos,f->vel(crs)));
                }else{
                }
            }
        }
    }
    observables["track"]=track;
}
void MissileSensor::control(){
    nl::json cmd=parent.lock()->commands["Sensor"];
    for(auto&& elem:cmd){
        std::string cmdName=elem.at("name");
        if(cmdName=="steering"){
            auto crs=getParentCRS();
            Coordinate p(elem.at("estTPos"));
            Coordinate v(elem.at("estTVel"));
            //次のperceiveまでの時間を計算して速度を加算
            double dt=(getNextTick(SimPhase::PERCEIVE,manager->getTickCount())-manager->getTickCountAt(p.getTime()))*manager->getBaseTimeStep();
            p.setValue(p()+v(p.getCRS())*dt);
            estTPos=p.transformTo(getParentCRS());
        }else if(cmdName=="activate"){
            if(!isActive){
                target=elem.at("target");
                isActive=true;
            }
        }
    }
}
void MissileSensor::kill(){
    target=Track3D();
    isActive=false;
    this->PhysicalAsset::kill();
}
std::pair<bool,std::vector<Track3D>> MissileSensor::isTrackingAny(){
    if(track.size()>0){
        std::vector<Track3D> ret;
        for(auto& e:track){
            ret.push_back(e);
        }
        return std::make_pair(true,std::move(ret));
    }else{
        return std::make_pair(false,std::vector<Track3D>());
    }
}
std::pair<bool,Track3D> MissileSensor::isTracking(const std::weak_ptr<PhysicalAsset>& target_){
    if(target_.expired()){
        return std::make_pair(false,Track3D());
    }else{
        return isTracking(target_.lock()->getUUIDForTrack());
    }
}
std::pair<bool,Track3D> MissileSensor::isTracking(const Track3D& target_){
    if(target_.is_none()){
        return std::make_pair(false,Track3D());
    }else{
        return isTracking(target_.truth);
    }
}
std::pair<bool,Track3D> MissileSensor::isTracking(const boost::uuids::uuid& target_){
    if(target_==boost::uuids::nil_uuid()){
        return std::make_pair(false,Track3D());
    }
    if(isActive){
        for(auto& e:track){
           if(e.isSame(target_)){
               return std::make_pair(true,e);
            }
        }
    }
    return std::make_pair(false,Track3D());
}

void exportSensor(py::module &m, const std::shared_ptr<asrc::core::FactoryHelper>& factoryHelper)
{
    using namespace pybind11::literals;
    expose_entity_subclass<Sensor>(m,"Sensor")
    ;
    //FACTORY_ADD_CLASS(PhysicalAsset,Sensor) //Do not register to Factory because Sensor is abstract class.

    expose_entity_subclass<Sensor3D>(m,"Sensor3D")
    DEF_FUNC(Sensor3D,isTrackingAny)
    .def("isTracking",py::overload_cast<const std::weak_ptr<PhysicalAsset>&>(&Sensor3D::isTracking))
    .def("isTracking",py::overload_cast<const Track3D&>(&Sensor3D::isTracking))
    .def("isTracking",py::overload_cast<const boost::uuids::uuid&>(&Sensor3D::isTracking))
    ;
    //FACTORY_ADD_CLASS(PhysicalAsset,Sensor3D) //Do not register to Factory because Sensor3D is abstract class.

    expose_entity_subclass<Sensor2D>(m,"Sensor2D")
    DEF_FUNC(Sensor2D,isTrackingAny)
    .def("isTracking",py::overload_cast<const std::weak_ptr<PhysicalAsset>&>(&Sensor2D::isTracking))
    .def("isTracking",py::overload_cast<const Track2D&>(&Sensor2D::isTracking))
    .def("isTracking",py::overload_cast<const boost::uuids::uuid&>(&Sensor2D::isTracking))
    ;
    //FACTORY_ADD_CLASS(PhysicalAsset,Sensor2D) //Do not register to Factory because Sensor2D is abstract class.

    expose_entity_subclass<AircraftRadar>(m,"AircraftRadar")
    DEF_READWRITE(AircraftRadar,Lref)
    DEF_READWRITE(AircraftRadar,thetaFOR)
    DEF_READWRITE(AircraftRadar,track)
    ;
    FACTORY_ADD_CLASS(PhysicalAsset,AircraftRadar)

    expose_entity_subclass<MWS>(m,"MWS")
    DEF_READWRITE(MWS,track)
    ;
    FACTORY_ADD_CLASS(PhysicalAsset,MWS)

    expose_entity_subclass<MissileSensor>(m,"MissileSensor")
    DEF_READWRITE(MissileSensor,Lref)
    DEF_READWRITE(MissileSensor,thetaFOR)
    DEF_READWRITE(MissileSensor,thetaFOV)
    DEF_READWRITE(MissileSensor,isActive)
    DEF_READWRITE(MissileSensor,estTPos)
    DEF_READWRITE(MissileSensor,track)
    DEF_READWRITE(MissileSensor,target)
    ;
    FACTORY_ADD_CLASS(PhysicalAsset,MissileSensor)
}

ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
