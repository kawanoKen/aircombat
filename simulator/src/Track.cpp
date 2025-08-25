// Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#include "Track.h"
#include "Utility.h"
#include "Units.h"
#include "crs/CoordinateReferenceSystemBase.h"
#include "SimulationManager.h"
#include <boost/uuid/nil_generator.hpp>

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)

ASRC_DEFINE_BASE_DATA_CLASS(TrackBase)

TrackBase::TrackBase()
:crs(nullptr),_time(),time(_time){
    truth=boost::uuids::nil_uuid();
}
TrackBase::TrackBase(const std::weak_ptr<PhysicalAsset>& truth_,const std::shared_ptr<CoordinateReferenceSystem>& crs_)
:TrackBase(){
    if(!truth_.expired()){
        truth=truth_.lock()->getUUIDForTrack();
        crs=crs_;
        if(!crs){
            crs=truth_.lock()->motion.getCRS();
        }
        if(truth==boost::uuids::nil_uuid()){
            _time=truth_.lock()->manager->getTime();
        }else{
            _time=truth_.lock()->motion.time;
        }
    }
}
TrackBase::TrackBase(const std::weak_ptr<PhysicalAsset>& truth_,const std::shared_ptr<CoordinateReferenceSystem>& crs_,const Time& time_)
:crs(crs_),_time(time_),time(_time){
    if(truth_.expired()){
        truth=boost::uuids::nil_uuid();
    }else{
        truth=truth_.lock()->getUUIDForTrack();
    }
}
TrackBase::TrackBase(const boost::uuids::uuid& truth_,const std::shared_ptr<CoordinateReferenceSystem>& crs_,const Time& time_)
:crs(crs_),_time(time_),time(_time){
    truth=truth_;
}
TrackBase::TrackBase(const nl::json& j_):TrackBase(){
    load_from_json(j_);
}
TrackBase::TrackBase(const TrackBase &other)
:crs(other.crs),_time(other.time),time(_time),truth(other.truth){
}
TrackBase::TrackBase(TrackBase &&other)
:crs(std::forward<TrackBase>(other).crs),_time(other.time),time(_time),truth(other.truth){
}
void TrackBase::operator=(const TrackBase& other){
    crs=other.crs;
    _time=other.time;
    truth=other.truth;
}
void TrackBase::operator=(TrackBase&& other){
    crs=other.crs;
    _time=other.time;
    truth=other.truth;
}
std::shared_ptr<CoordinateReferenceSystem> TrackBase::getCRS() const {
    return crs;
}
void TrackBase::setCRS(const std::shared_ptr<CoordinateReferenceSystem>& newCRS,bool transform){
    //CRSを再設定する。座標変換の有無を第2引数で指定する。
    if(!newCRS){
        throw std::runtime_error("Do not give nullptr as a new CRS of TrackBase!");
    }
    crs=newCRS;
}
void TrackBase::setTime(const Time& time_){
    _time=time_;
}
bool TrackBase::is_none() const{
    return truth==boost::uuids::nil_uuid();
}
bool TrackBase::isSame(const TrackBase& other) const{
    if(this->is_none() || other.is_none()){
        return false;
    }
    return truth==other.truth;
}
bool TrackBase::isSame(const boost::uuids::uuid& other) const{
    if(this->is_none() || other==boost::uuids::nil_uuid()){
        return false;
    }
    return truth==other;
}
bool TrackBase::isSame(const std::weak_ptr<Asset>& other) const{
    if(this->is_none() || other.expired()){
        return false;
    }
    return truth==(other.lock()->getUUIDForTrack());
}
void TrackBase::polymorphic_assign(const Track3D::PtrBaseType& other){
    crs=other.crs;
    _time=other.time;
    truth=other.truth;
}
void TrackBase::serialize_impl(asrc::core::util::AvailableArchiveTypes& archive){
    ASRC_SERIALIZE_NVP(archive
        ,crs
        ,truth
    );
    call_operator(archive
        ,cereal::make_nvp("time",_time)
    );
}

Track3D::Track3D()
:TrackBase(),
 _pos(Eigen::Vector3d::Zero(),nullptr,Time(),CoordinateType::POSITION_ABS),
 _vel(Eigen::Vector3d::Zero(),nullptr,Time(),CoordinateType::VELOCITY),
 pos(_pos),vel(_vel){
}
Track3D::Track3D(const std::weak_ptr<PhysicalAsset>& truth_,const std::shared_ptr<CoordinateReferenceSystem>& crs_)
:TrackBase(truth_,crs_),
 _pos(Eigen::Vector3d::Zero(),crs,time,CoordinateType::POSITION_ABS),
 _vel(Eigen::Vector3d::Zero(),crs,time,CoordinateType::VELOCITY),
 pos(_pos),vel(_vel){
    if(truth!=boost::uuids::nil_uuid()){
        if(crs){
            _pos=truth_.lock()->pos.transformTo(crs);
            _vel=truth_.lock()->vel.transformTo(crs);
        }else{
            _pos=truth_.lock()->pos;
            _vel=truth_.lock()->vel;
        }
    }
}
Track3D::Track3D(const std::weak_ptr<PhysicalAsset>& truth_,const std::shared_ptr<CoordinateReferenceSystem>& crs_,const Time& time_,const Eigen::Vector3d& pos_,const Eigen::Vector3d& vel_)
:TrackBase(truth_,crs_,time_),
 _pos(pos_,crs,time,CoordinateType::POSITION_ABS),
 _vel(vel_,pos_,crs,time,CoordinateType::VELOCITY),
 pos(_pos),vel(_vel){
}
Track3D::Track3D(const boost::uuids::uuid& truth_,const std::shared_ptr<CoordinateReferenceSystem>& crs_,const Time& time_,const Eigen::Vector3d &pos_,const Eigen::Vector3d  &vel_)
:TrackBase(truth_,crs_,time_),
 _pos(pos_,crs,time,CoordinateType::POSITION_ABS),
 _vel(vel_,pos_,crs,time,CoordinateType::VELOCITY),
 pos(_pos),vel(_vel){
}
Track3D::Track3D(const nl::json& j_):Track3D(){
    load_from_json(j_);
}
Track3D::Track3D(const Track3D &other)
:TrackBase(other),_pos(other.pos),_vel(other.vel),
 pos(_pos),vel(_vel),
 buffer(other.buffer){
}
Track3D::Track3D(Track3D &&other)
:TrackBase(other),_pos(other.pos),_vel(other.vel),
 pos(_pos),vel(_vel),
 buffer(other.buffer){
}
void Track3D::operator=(const Track3D& other){
    BaseType::operator=(other);
    _pos=other.pos;
    _vel=other.vel;
    buffer.clear();
    for(auto& e:other.buffer){
        buffer.push_back(e);
    }
}
void Track3D::operator=(Track3D&& other){
    BaseType::operator=(other);
    _pos=other.pos;
    _vel=other.vel;
    buffer.clear();
    for(auto& e:other.buffer){
        buffer.push_back(e);
    }
}
void Track3D::setCRS(const std::shared_ptr<CoordinateReferenceSystem>& newCRS,bool transform){
    //CRSを再設定する。座標変換の有無を第2引数で指定する。
    BaseType::setCRS(newCRS,transform);
    _pos.setCRS(newCRS,transform);
    _vel.setCRS(newCRS,transform);
}
Track3D Track3D::transformTo(const std::shared_ptr<CoordinateReferenceSystem>& newCRS) const{
    if(!crs){
        throw std::runtime_error("Track3D without CRS cannot be converted to another CRS.");
    }
    if(!newCRS){
        throw std::runtime_error("Do not give nullptr as new parent CRS of Track3D!");
    }
    if(crs==newCRS){
        //同一ならバイパス
        return *this;
    }
    return std::move(Track3D(
        truth,
        newCRS,
        time,
        pos(newCRS),
        vel(newCRS)
    ));
}
void Track3D::setTime(const Time& time_){
    BaseType::setTime(time_);
    _pos.setTime(time);
    _vel.setTime(time);
}
void Track3D::setPos(const Eigen::Vector3d& pos_){
    _pos.setValue(pos_);
    _vel.setLocation(pos_);
}
void Track3D::setPos(const Coordinate& pos_){
    setPos(pos_(crs));
}
void Track3D::setVel(const Eigen::Vector3d& vel_){
    _vel.setValue(vel_);
}
void Track3D::setVel(const Coordinate& vel_){
    setVel(vel_(crs));
}
Track3D Track3D::copy() const{
    return Track3D(truth,crs,time,pos(),vel());
}
void Track3D::clearBuffer(){
    buffer.clear();
}
void Track3D::addBuffer(const Track3D& other){
    buffer.push_back(other);
}
void Track3D::merge(){
    // このクラスでは算術平均でマージするが、必要に応じて派生クラスでオーバーライドすること。
    if(buffer.size()>0){
        Eigen::Vector3d meanPos(0,0,0);
        Eigen::Vector3d meanVel(0,0,0);
        for(auto& e:buffer){
            Track3D extrapolated=e.transformTo(crs);
            extrapolated.updateByExtrapolation(time-e.time);
            meanPos+=extrapolated.pos();
            meanVel+=extrapolated.vel();
        }
        meanPos/=(buffer.size()+1.0);
        meanVel/=(buffer.size()+1.0);
        _pos.setValue(meanPos);
        _vel.setLocation(meanPos);
        _vel.setValue(meanVel);
        clearBuffer();
    }
}
void Track3D::update(const Track3D& other){
    truth=other.truth;
    crs=other.crs;
    _time=other.time;
    _pos=other.pos;
    _vel=other.vel;
    buffer.clear();
}
void Track3D::updateByExtrapolation(const double& dt){
    // このクラスではcrs上の見かけの位置、速度で外挿するが、必要に応じて派生クラスでオーバーライドすること。
    // あるいは、慣性座標系等の外挿しても問題ないcrsに変換してから外挿すること
    _pos.setValue(pos()+vel()*dt);
    _vel.setLocation(pos());
}
Track3D Track3D::extrapolate(const double& dt){
    // このクラスではcrs上の見かけの位置、速度で外挿するが、必要に応じて派生クラスでオーバーライドすること。
    // あるいは、慣性座標系等の外挿しても問題ないcrsに変換してから外挿すること
    return Track3D(truth,crs,time+dt,pos()+vel()*dt,vel());
}
Track3D Track3D::extrapolateTo(const Time& dstTime){
    return extrapolate(dstTime-time);
}
void Track3D::polymorphic_assign(const Track3D::PtrBaseType& other_){
    BaseType::polymorphic_assign(other_);
    try{
        const Track3D& other=static_cast<const Track3D&>(other_);
        _pos=other.pos;
        _vel=other.vel;
        buffer.clear();
        for(auto& e:other.buffer){
            buffer.push_back(e);
        }
    }catch(std::bad_cast){
        throw std::runtime_error("Track3D::polymorphic_assign failed. src="+other_.getDemangledName());
    }
}
void Track3D::serialize_impl(asrc::core::util::AvailableArchiveTypes& archive){
    BaseType::serialize_impl(archive);

    if(isOutputArchive(archive)){
        call_operator(archive
            ,cereal::make_nvp("pos",pos())
            ,cereal::make_nvp("vel",vel())
        );
    }else{
        Eigen::Vector3d pos_,vel_;
        call_operator(archive
            ,cereal::make_nvp("pos",pos_)
            ,cereal::make_nvp("vel",vel_)
        );
        _pos=Coordinate(pos_,crs,time,CoordinateType::POSITION_ABS);
        _vel=Coordinate(vel_,pos(),crs,time,CoordinateType::VELOCITY);
    }

    ASRC_SERIALIZE_NVP(archive
        ,buffer
    );
}

Track2D::Track2D()
:TrackBase(),
 _origin(Eigen::Vector3d::Zero(),nullptr,Time(),CoordinateType::POSITION_ABS),
 _dir(Eigen::Vector3d(1,0,0),nullptr,Time(),CoordinateType::DIRECTION),
 _omega(Eigen::Vector3d::Zero(),nullptr,Time(),CoordinateType::ANGULAR_VELOCITY),
 origin(_origin),dir(_dir),omega(_omega){
}
Track2D::Track2D(const std::weak_ptr<PhysicalAsset>& truth_,const std::shared_ptr<CoordinateReferenceSystem>& crs_,const Eigen::Vector3d  &origin_)
:TrackBase(truth_,crs_),
 _origin(origin_,crs,time,CoordinateType::POSITION_ABS),
 _dir(Eigen::Vector3d(1,0,0),origin_,crs,time,CoordinateType::DIRECTION),
 _omega(Eigen::Vector3d::Zero(),origin_,crs,time,CoordinateType::ANGULAR_VELOCITY),
 origin(_origin),dir(_dir),omega(_omega){
    if(truth!=boost::uuids::nil_uuid()){
        if(crs){
            _origin=Coordinate(origin_,crs,time,CoordinateType::POSITION_ABS);
            Eigen::Vector3d dr=truth_.lock()->pos(crs)-origin_;
            double R=dr.norm();
            _dir=Coordinate(dr/R,origin_,crs,time,CoordinateType::DIRECTION);
            _omega=Coordinate(dir().cross(truth_.lock()->vel(crs))/R,origin_,crs,time,CoordinateType::ANGULAR_VELOCITY);
        }else{
            _origin=Coordinate(origin_,crs,time,CoordinateType::POSITION_ABS);
            Eigen::Vector3d dr=truth_.lock()->pos()-origin_;
            double R=dr.norm();
            _dir=Coordinate(dr/R,origin_,crs,time,CoordinateType::DIRECTION);
            _omega=Coordinate(dir().cross(truth_.lock()->vel())/R,origin_,crs,time,CoordinateType::ANGULAR_VELOCITY);
        }
    }
}
Track2D::Track2D(const std::weak_ptr<PhysicalAsset>& truth_,const std::shared_ptr<CoordinateReferenceSystem>& crs_,const Time& time_,const Eigen::Vector3d &dir_,const Eigen::Vector3d  &origin_,const Eigen::Vector3d  &omega_)
:TrackBase(truth_,crs_,time_),
 _origin(origin_,crs,time,CoordinateType::POSITION_ABS),
 _dir(dir_,origin_,crs,time,CoordinateType::DIRECTION),
 _omega(omega_,origin_,crs,time,CoordinateType::ANGULAR_VELOCITY),
 origin(_origin),dir(_dir),omega(_omega){
}
Track2D::Track2D(const boost::uuids::uuid& truth_,const std::shared_ptr<CoordinateReferenceSystem>& crs_,const Time& time_,const Eigen::Vector3d &dir_,const Eigen::Vector3d  &origin_,const Eigen::Vector3d  &omega_)
:TrackBase(truth_,crs_,time_),
 _origin(origin_,crs,time_,CoordinateType::POSITION_ABS),
 _dir(dir_,origin_,crs,time_,CoordinateType::DIRECTION),
 _omega(omega_,origin_,crs,time_,CoordinateType::ANGULAR_VELOCITY),
 origin(_origin),dir(_dir),omega(_omega){
}
Track2D::Track2D(const nl::json& j_):Track2D(){
    load_from_json(j_);
}
Track2D::Track2D(const Track2D &other)
:TrackBase(other),_dir(other.dir),_origin(other.origin),_omega(other.omega),
 dir(_dir),origin(_origin),omega(_omega),
 buffer(other.buffer){
}
Track2D::Track2D(Track2D &&other)
:TrackBase(other),_dir(other.dir),_origin(other.origin),_omega(other.omega),
 dir(_dir),origin(_origin),omega(_omega),
 buffer(other.buffer){
}
void Track2D::operator=(const Track2D& other){
    BaseType::operator=(other);
    _origin=other.origin;
    _dir=other.dir;
    _omega=other.omega;
    buffer.clear();
    for(auto& e:other.buffer){
        buffer.push_back(e);
    }
}
void Track2D::operator=(Track2D&& other){
    BaseType::operator=(other);
    _origin=other.origin;
    _dir=other.dir;
    _omega=other.omega;
    buffer.clear();
    for(auto& e:other.buffer){
        buffer.push_back(e);
    }
}
void Track2D::setCRS(const std::shared_ptr<CoordinateReferenceSystem>& newCRS,bool transform){
    //CRSを再設定する。座標変換の有無を第2引数で指定する。
    BaseType::setCRS(newCRS,transform);
    _origin.setCRS(newCRS,transform);
    _dir.setCRS(newCRS,transform);
    _omega.setCRS(newCRS,transform);
}
Track2D Track2D::transformTo(const std::shared_ptr<CoordinateReferenceSystem>& newCRS) const{
    if(!crs){
        throw std::runtime_error("Track2D without CRS cannot be converted to another CRS.");
    }
    if(!newCRS){
        throw std::runtime_error("Do not give nullptr as new parent CRS of Track2D!");
    }
    if(crs==newCRS){
        //同一ならバイパス
        return *this;
    }
    return std::move(Track2D(
        truth,
        newCRS,
        time,
        dir(newCRS),
        origin(newCRS),
        omega(newCRS)
    ));
}
void Track2D::setTime(const Time& time_){
    BaseType::setTime(time_);
    _origin.setTime(time);
    _dir.setTime(time);
    _omega.setTime(time);
}
void Track2D::setOrigin(const Eigen::Vector3d& origin_){
    _origin.setValue(origin_);
    _dir.setLocation(origin_);
    _omega.setLocation(origin_);
}
void Track2D::setOrigin(const Coordinate& origin_){
    setOrigin(origin_(crs));
}
void Track2D::setDir(const Eigen::Vector3d& dir_){
    _dir.setValue(dir_);
}
void Track2D::setDir(const Coordinate& dir_){
    setDir(dir_(crs));
}
void Track2D::setOmega(const Eigen::Vector3d& omega_){
    _omega.setValue(omega_);
}
void Track2D::setOmega(const Coordinate& omega_){
    setOmega(omega_(crs));
}
Track2D Track2D::copy() const{
    return Track2D(truth,crs,time,dir(),origin(),omega());
}
void Track2D::clearBuffer(){
    buffer.clear();
}
void Track2D::addBuffer(const Track2D& other){
    buffer.push_back(other);
}
void Track2D::merge(){
    // このクラスでは算術平均でマージするが、必要に応じて派生クラスでオーバーライドすること。
    // 例えば、観測点が複数ヶ所にわたる場合には明らかに不適切である。
    if(buffer.size()>0){
        Eigen::Vector3d meanOrigin(0,0,0);
        Eigen::Vector3d meanDir(0,0,0);
        Eigen::Vector3d meanOmega(0,0,0);
        for(auto& e:buffer){
            Track2D extrapolated=e.transformTo(crs);
            extrapolated.updateByExtrapolation(time-e.time);
            meanDir+=extrapolated.dir();
            meanOrigin+=extrapolated.origin();
            meanOmega+=extrapolated.omega();
        }
        meanDir=(meanDir/(buffer.size()+1.0)).normalized();
        meanOrigin/=(buffer.size()+1.0);
        meanOmega/=(buffer.size()+1.0);
        _origin.setValue(meanOrigin);
        _dir.setLocation(meanOrigin);
        _omega.setLocation(meanOrigin);
        _dir.setValue(meanDir);
        _omega.setValue(meanOmega);
        clearBuffer();
    }
}
void Track2D::update(const Track2D& other){
    truth=other.truth;
    crs=other.crs;
    _time=other.time;
    _dir=other.dir;
    _origin=other.origin;
    _omega=other.omega;
    buffer.clear();
}
void Track2D::updateByExtrapolation(const double& dt){
    // このクラスではcrs上の見かけの位置、角速度で外挿するが、必要に応じて派生クラスでオーバーライドすること。
    // あるいは、慣性座標系等の外挿しても問題ないcrsに変換してから外挿すること
    double w=omega.norm();   
    if(w>0){
        Quaternion q=Quaternion::fromAngle(omega()/w,w*dt);
        _dir.setValue(q.transformVector(dir()));
    }
}
Track2D Track2D::extrapolate(const double& dt){
    // このクラスではcrs上の見かけの位置、角速度で外挿するが、必要に応じて派生クラスでオーバーライドすること。
    // あるいは、慣性座標系等の外挿しても問題ないcrsに変換してから外挿すること
    double w=omega.norm();
    if(w>0){
        Quaternion q=Quaternion::fromAngle(omega()/w,w*dt);
        return Track2D(truth,crs,time+dt,q.transformVector(dir()),origin(),omega());
    }else{
        return Track2D(truth,crs,time+dt,dir(),origin(),omega());
    }
}
Track2D Track2D::extrapolateTo(const Time& dstTime){
    return extrapolate(dstTime-time);
}
void Track2D::polymorphic_assign(const Track2D::PtrBaseType& other_){
    BaseType::polymorphic_assign(other_);
    try{
        const Track2D& other=static_cast<const Track2D&>(other_);
        _origin=other.origin;
        _dir=other.dir;
        _omega=other.omega;
        buffer.clear();
        for(auto& e:other.buffer){
            buffer.push_back(e);
        }
    }catch(std::bad_cast){
        throw std::runtime_error("Track2D::polymorphic_assign failed. src="+other_.getDemangledName());
    }
}
void Track2D::serialize_impl(asrc::core::util::AvailableArchiveTypes& archive){
    BaseType::serialize_impl(archive);

    if(isOutputArchive(archive)){
        call_operator(archive
            ,cereal::make_nvp("origin",origin())
            ,cereal::make_nvp("dir",dir())
            ,cereal::make_nvp("omega",omega())
        );
    }else{
        Eigen::Vector3d origin_,dir_,omega_;
        call_operator(archive
            ,cereal::make_nvp("origin",origin_)
            ,cereal::make_nvp("dir",dir_)
            ,cereal::make_nvp("omega",omega_)
        );
        _origin=Coordinate(origin_,crs,time,CoordinateType::POSITION_ABS);
        _dir=Coordinate(dir_,origin(),crs,time,CoordinateType::DIRECTION);
        _omega=Coordinate(omega_,origin(),crs,time,CoordinateType::ANGULAR_VELOCITY);
    }

    ASRC_SERIALIZE_NVP(archive
        ,buffer
    );
}

void exportTrack(py::module &m)
{
    Track3D::PtrBaseType::addDerivedCppClass<Track3D>();
    Track2D::PtrBaseType::addDerivedCppClass<Track2D>();

    using namespace pybind11::literals;

    bind_stl_container<std::vector<Track3D>>(m,py::module_local(false));
    bind_stl_container<std::vector<Track2D>>(m,py::module_local(false));
    bind_stl_container<std::map<std::string,Track3D>>(m,py::module_local(false));
    bind_stl_container<std::map<std::string,Track2D>>(m,py::module_local(false));
    bind_stl_container<std::vector<std::pair<Track3D,bool>>>(m,py::module_local(false));
    bind_stl_container<std::vector<std::pair<Track2D,bool>>>(m,py::module_local(false));

    expose_common_class<TrackBase>(m,"TrackBase")
    .def(py_init<>())
    .def(py_init<const std::weak_ptr<PhysicalAsset>&,const std::shared_ptr<CoordinateReferenceSystem>&>())
    .def(py_init<const std::weak_ptr<PhysicalAsset>&,const std::shared_ptr<CoordinateReferenceSystem>&,const Time&>())
    .def(py_init<const boost::uuids::uuid&,const std::shared_ptr<CoordinateReferenceSystem>&,const Time&>())
    .def(py_init<const nl::json&>())
    DEF_FUNC(TrackBase,getCRS)
    DEF_FUNC(TrackBase,setCRS)
    DEF_FUNC(TrackBase,setTime)
    DEF_FUNC(TrackBase,is_none)
    .def("isSame",py::overload_cast<const TrackBase&>(&TrackBase::isSame,py::const_))
    .def("isSame",py::overload_cast<const boost::uuids::uuid&>(&TrackBase::isSame,py::const_))
    .def("isSame",py::overload_cast<const std::weak_ptr<Asset>&>(&TrackBase::isSame,py::const_))
    DEF_READWRITE(TrackBase,truth)
    .def_property_readonly("time",[](const TrackBase& v){return v.time;})
    ;
    expose_common_class<Track3D>(m,"Track3D")
    .def(py_init<>())
    .def(py_init<const std::weak_ptr<PhysicalAsset>&,const std::shared_ptr<CoordinateReferenceSystem>&>())
    .def(py_init<const std::weak_ptr<PhysicalAsset>&,const std::shared_ptr<CoordinateReferenceSystem>&,const Time&,const Eigen::Vector3d&,const Eigen::Vector3d&>())
    .def(py_init<const boost::uuids::uuid&,const std::shared_ptr<CoordinateReferenceSystem>&,const Time&,const Eigen::Vector3d&,const Eigen::Vector3d&>())
    .def(py_init<const nl::json&>())
    DEF_FUNC(Track3D,transformTo)
    .def("setPos",py::overload_cast<const Eigen::Vector3d&>(&Track3D::setPos))
    .def("setPos",py::overload_cast<const Coordinate&>(&Track3D::setPos))
    .def("setVel",py::overload_cast<const Eigen::Vector3d&>(&Track3D::setVel))
    .def("setVel",py::overload_cast<const Coordinate&>(&Track3D::setVel))
    DEF_FUNC(Track3D,copy)
    DEF_FUNC(Track3D,clearBuffer)
    DEF_FUNC(Track3D,addBuffer)
    DEF_FUNC(Track3D,merge)
    DEF_FUNC(Track3D,update)
    DEF_FUNC(Track3D,updateByExtrapolation)
    DEF_FUNC(Track3D,extrapolate)
    DEF_FUNC(Track3D,extrapolateTo)
    .def_property_readonly("pos",[](const Track3D& v){return v.pos;})
    .def_property_readonly("vel",[](const Track3D& v){return v.vel;})
    DEF_READWRITE(Track3D,buffer)
    ;
    expose_common_class<Track2D>(m,"Track2D")
    .def(py_init<>())
    .def(py_init<const std::weak_ptr<PhysicalAsset>&,const std::shared_ptr<CoordinateReferenceSystem>&,const Eigen::Vector3d&>())
    .def(py_init<const std::weak_ptr<PhysicalAsset>&,const std::shared_ptr<CoordinateReferenceSystem>&,const Time&,const Eigen::Vector3d&,const Eigen::Vector3d&,const Eigen::Vector3d&>())
    .def(py_init<const boost::uuids::uuid&,const std::shared_ptr<CoordinateReferenceSystem>&,const Time&,const Eigen::Vector3d&,const Eigen::Vector3d&,const Eigen::Vector3d&>())
    .def(py_init<const nl::json&>())
    DEF_FUNC(Track2D,transformTo)
    .def("setOrigin",py::overload_cast<const Eigen::Vector3d&>(&Track2D::setOrigin))
    .def("setOrigin",py::overload_cast<const Coordinate&>(&Track2D::setOrigin))
    .def("setDir",py::overload_cast<const Eigen::Vector3d&>(&Track2D::setDir))
    .def("setDir",py::overload_cast<const Coordinate&>(&Track2D::setDir))
    .def("setOmega",py::overload_cast<const Eigen::Vector3d&>(&Track2D::setOmega))
    .def("setOmega",py::overload_cast<const Coordinate&>(&Track2D::setOmega))
    DEF_FUNC(Track2D,copy)
    DEF_FUNC(Track2D,clearBuffer)
    DEF_FUNC(Track2D,addBuffer)
    DEF_FUNC(Track2D,merge)
    DEF_FUNC(Track2D,update)
    DEF_FUNC(Track2D,updateByExtrapolation)
    DEF_FUNC(Track2D,extrapolate)
    DEF_FUNC(Track2D,extrapolateTo)
    .def_property_readonly("origin",[](const Track2D& v){return v.origin;})
    .def_property_readonly("dir",[](const Track2D& v){return v.dir;})
    .def_property_readonly("omega",[](const Track2D& v){return v.omega;})
    DEF_READWRITE(Track2D,buffer)
    ;
}

ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
