// Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#include "crs/DerivedCRS.h"
#include "crs/ProjectedCRS.h"
#include "Units.h"
#include <algorithm>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <iomanip>

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)

using namespace util;

void DerivedCRS::initialize(){
    BaseType::initialize();

    if(instanceConfig.contains("base")){
        _base_j=instanceConfig.at("base");
    }else if(modelConfig.contains("base")){
        _base_j=modelConfig.at("base");
    }else{
        _base_j=nl::json::object();
        //throw std::runtime_error("DerivedCRS must have a base CRS."); //異なる形での指定を認める。
    }
    assert(_base_j.is_object());
    _base_j["baseName"]="CoordinateReferenceSystem";
}
// 内部状態のシリアライゼーション
void DerivedCRS::serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full){
    BaseType::serializeInternalState(archive,full);

    if(full){
        if(asrc::core::util::isInputArchive(archive)){
            if(instanceConfig.contains("base")){
                _base_j=instanceConfig.at("base");
            }else if(modelConfig.contains("base")){
                _base_j=modelConfig.at("base");
            }else{
                _base_j=nl::json::object();
                //throw std::runtime_error("DerivedCRS must have a base CRS."); //異なる形での指定を認める。
            }
            assert(_base_j.is_object());
            _base_j["baseName"]="CoordinateReferenceSystem";
        }
    }

    ASRC_SERIALIZE_NVP(archive
        ,_base
        ,_nonDerivedBase
        ,_baseProjected
    )

    if(asrc::core::util::isInputArchive(archive)){
        if(_base){
            addDerivedCRS(_base,non_const_this());
        }
    }
}
bool DerivedCRS::isValid() const{
    if(!getBaseCRS()->isValid()){
        return false;
    }
    return this->CoordinateReferenceSystem::isValid();
}
void DerivedCRS::validateImpl(bool isFirstTime) const{
    auto base=getBaseCRS();
    if(!base->isValid()){
        CoordinateReferenceSystem::validateImpl(base,!base->hasBeenValidated());
    }
    this->CoordinateReferenceSystem::validateImpl(isFirstTime);
}
//座標系の種類
bool DerivedCRS::isDerived() const{
    return true;
}
bool DerivedCRS::isProjected() const{
    return getBaseCRS()->isProjected();
}
//親座標系の取得
std::shared_ptr<CoordinateReferenceSystem> DerivedCRS::getBaseCRS() const{
    if(!_base){
        _base=createOrGetEntity<CoordinateReferenceSystem>(_base_j);
        if(!_base){
            throw std::runtime_error("The given base CRS is not a valid object. j="+_base_j.dump());
        }
        if(auto derived = std::dynamic_pointer_cast<DerivedCRS>(_base)){
            _nonDerivedBase=derived->getNonDerivedBaseCRS();
            _baseProjected=derived->getBaseProjectedCRS();
        }else{
            _nonDerivedBase=_base;
            if(isProjected()){
                _baseProjected=non_const_this<ProjectedCRS>();
            }else{
                _baseProjected=nullptr;
            }
        }
        addDerivedCRS(_base,non_const_this());
    }
    return _base;
}
std::shared_ptr<CoordinateReferenceSystem> DerivedCRS::getNonDerivedBaseCRS() const{
    if(!_nonDerivedBase){
        getBaseCRS();
    }
    return _nonDerivedBase;
}
std::shared_ptr<ProjectedCRS> DerivedCRS::getBaseProjectedCRS() const{
    //親座標系を辿ってProjectedCRSに当たった場合、そのProjectedCRS。なければnullptr。自分自身がProjectedCRSの場合は自分自身。
    if(isProjected()){
        if(auto projected=non_const_this<ProjectedCRS>()){
            return projected;
        }else{
            if(!_base){
                getBaseCRS();
            }
            if(_base->isDerived()){// AffineCRS derived from ProjectedCRS
                return _base->getBaseProjectedCRS();
            }else{
                //ここには到達しない
                return non_const_this<ProjectedCRS>();//nullptr
            }
        }
    }else{
        return nullptr;
    }
}
//中間座標系の取得
std::shared_ptr<CoordinateReferenceSystem> DerivedCRS::getIntermediateCRS() const{
    return getBaseCRS();
}
//「高度」の取得
double DerivedCRS::getHeight(const Eigen::Vector3d& location, const Time& time) const{
    //楕円体高度(幾何高度)を返す。ellipsoid (geometric) height
    return getNonDerivedBaseCRS()->getHeight(
        transformToNonDerivedBaseCRS(location,time,CoordinateType::POSITION_ABS)
    );
}
double DerivedCRS::getGeoidHeight(const Eigen::Vector3d& location, const Time& time) const{
    //標高(ジオイド高度)を返す。geoid height (elevation)
    return getNonDerivedBaseCRS()->getGeoidHeight(
        transformToNonDerivedBaseCRS(location,time,CoordinateType::POSITION_ABS)
    );
}
// CRSインスタンス間の変換可否
bool DerivedCRS::isTransformableTo(const std::shared_ptr<CoordinateReferenceSystem>& other) const{
    return getNonDerivedBaseCRS()->isTransformableTo(other);
}
bool DerivedCRS::isTransformableFrom(const std::shared_ptr<CoordinateReferenceSystem>& other) const{
    return getNonDerivedBaseCRS()->isTransformableFrom(other);
}
// 座標変換
Coordinate DerivedCRS::transformTo(const Eigen::Vector3d& value, const Eigen::Vector3d& location, const Time& time, const std::shared_ptr<CoordinateReferenceSystem>& dstCRS, const CoordinateType& coordinateType) const{
    if(!isValid()){validate();}
    if(!dstCRS){
        throw std::runtime_error("dstCRS is nullptr!");
    }
    if(this==std::dynamic_pointer_cast<DerivedCRS>(dstCRS).get()){
        return std::move(Coordinate(
            value,
            location,
            dstCRS,
            time,
            coordinateType
        ));
    }
    if(!isTransformableTo(dstCRS)){
        throw std::runtime_error(getFactoryClassName()+" can't be transformed to "+dstCRS->getFactoryClassName());
    }
    if(dstCRS==getBaseCRS()){
        return transformToBaseCRS(value,location,time,coordinateType);
    }else if(dstCRS==getNonDerivedBaseCRS()){
        return transformToNonDerivedBaseCRS(value,location,time,coordinateType);
    }else if(dstCRS==getIntermediateCRS()){
        return transformToIntermediateCRS(value,location,time,coordinateType);
    }else{
        auto base=getBaseCRS();
        auto valueInBase=transformToBaseCRS(value,location,time,coordinateType);
        return dstCRS->transformFrom(
            valueInBase(),
            valueInBase.getLocation(),
            time,
            base,
            coordinateType
        );
    }
}
Coordinate DerivedCRS::transformFrom(const Eigen::Vector3d& value, const Eigen::Vector3d& location, const Time& time, const std::shared_ptr<CoordinateReferenceSystem>& srcCRS, const CoordinateType& coordinateType) const{
    if(!isValid()){validate();}
    if(!srcCRS){
        throw std::runtime_error("srcCRS is nullptr!");
    }
    if(this==std::dynamic_pointer_cast<DerivedCRS>(srcCRS).get()){
        return std::move(Coordinate(
            value,
            location,
            srcCRS,
            time,
            coordinateType
        ));
    }
    if(!isTransformableFrom(srcCRS)){
        throw std::runtime_error(getFactoryClassName()+" can't be transformed from "+srcCRS->getFactoryClassName());
    }
    if(srcCRS==getBaseCRS()){
        return transformFromBaseCRS(value,location,time,coordinateType);
    }else if(srcCRS==getNonDerivedBaseCRS()){
        return transformFromNonDerivedBaseCRS(value,location,time,coordinateType);
    }else if(srcCRS==getIntermediateCRS()){
        return transformFromIntermediateCRS(value,location,time,coordinateType);
    }else{
        auto base=getBaseCRS();
        auto valueInBase=base->transformFrom(value,location,time,srcCRS,coordinateType);
        return transformFromBaseCRS(
            valueInBase(),
            valueInBase.getLocation(),
            time,
            coordinateType
        );
    }
}
//親座標系との座標変換
Coordinate DerivedCRS::transformToBaseCRS(const Coordinate& value,const Coordinate& location) const{
    if(location.getType()!=CoordinateType::POSITION_ABS){
        throw std::runtime_error("Given value is not a 'absolute location' coordinate.");
    }
    return transformToBaseCRS(
        value(non_const_this()),
        location(non_const_this()),
        value.getTime(),
        value.getType()
    );
}
Coordinate DerivedCRS::transformFromBaseCRS(const Coordinate& value,const Coordinate& location) const{
    if(location.getType()!=CoordinateType::POSITION_ABS){
        throw std::runtime_error("Given value is not a 'absolute location' coordinate.");
    }
    return transformToBaseCRS(
        value(getBaseCRS()),
        location(getBaseCRS()),
        value.getTime(),
        value.getType()
    );
}
Coordinate DerivedCRS::transformToBaseCRS(const Coordinate& value) const{
    Coordinate valueInSelf=value.transformTo(non_const_this());
    return transformToBaseCRS(
        valueInSelf(),
        valueInSelf.getLocation(),
        value.getTime(),
        value.getType()
    );
}
Coordinate DerivedCRS::transformFromBaseCRS(const Coordinate& value) const{
    Coordinate valueInBase=value.transformTo(getBaseCRS());
    return transformFromBaseCRS(
        valueInBase(),
        valueInBase.getLocation(),
        value.getTime(),
        value.getType()
    );
}
Coordinate DerivedCRS::transformToBaseCRS(const Eigen::Vector3d& value, const Time& time, const CoordinateType& coordinateType) const{
    return transformToBaseCRS(value,Eigen::Vector3d::Zero(),time,coordinateType);
}
Coordinate DerivedCRS::transformFromBaseCRS(const Eigen::Vector3d& value, const Time& time, const CoordinateType& coordinateType) const{
    return transformFromBaseCRS(value,Eigen::Vector3d::Zero(),time,coordinateType);
}
Coordinate DerivedCRS::transformToNonDerivedBaseCRS(const Coordinate& value,const Coordinate& location) const{
    if(location.getType()!=CoordinateType::POSITION_ABS){
        throw std::runtime_error("Given value is not a 'absolute location' coordinate.");
    }
    return transformToNonDerivedBaseCRS(
        value(non_const_this()),
        location(non_const_this()),
        value.getTime(),
        value.getType()
    );
}
Coordinate DerivedCRS::transformFromNonDerivedBaseCRS(const Coordinate& value,const Coordinate& location) const{
    if(location.getType()!=CoordinateType::POSITION_ABS){
        throw std::runtime_error("Given value is not a 'absolute location' coordinate.");
    }
    return transformToNonDerivedBaseCRS(
        value(getNonDerivedBaseCRS()),
        location(getNonDerivedBaseCRS()),
        value.getTime(),
        value.getType()
    );
}
Coordinate DerivedCRS::transformToNonDerivedBaseCRS(const Coordinate& value) const{
    Coordinate valueInSelf=value.transformTo(non_const_this());
    return transformToNonDerivedBaseCRS(
        valueInSelf(),
        valueInSelf.getLocation(),
        value.getTime(),
        value.getType()
    );
}
Coordinate DerivedCRS::transformFromNonDerivedBaseCRS(const Coordinate& value) const{
    Coordinate valueInNonDerivedBase=value.transformTo(getNonDerivedBaseCRS());
    return transformFromNonDerivedBaseCRS(
        valueInNonDerivedBase(),
        valueInNonDerivedBase.getLocation(),
        value.getTime(),
        value.getType()
    );
}
Coordinate DerivedCRS::transformToNonDerivedBaseCRS(const Eigen::Vector3d& value, const Time& time, const CoordinateType& coordinateType) const{
    return transformToNonDerivedBaseCRS(value,Eigen::Vector3d::Zero(),time,coordinateType);
}
Coordinate DerivedCRS::transformFromNonDerivedBaseCRS(const Eigen::Vector3d& value, const Time& time, const CoordinateType& coordinateType) const{
    return transformFromNonDerivedBaseCRS(value,Eigen::Vector3d::Zero(),time,coordinateType);
}
Coordinate DerivedCRS::transformToIntermediateCRS(const Coordinate& value,const Coordinate& location) const{
    if(location.getType()!=CoordinateType::POSITION_ABS){
        throw std::runtime_error("Given value is not a 'absolute location' coordinate.");
    }
    return transformToIntermediateCRS(
        value(non_const_this()),
        location(non_const_this()),
        value.getTime(),
        value.getType()
    );
}
Coordinate DerivedCRS::transformFromIntermediateCRS(const Coordinate& value,const Coordinate& location) const{
    if(location.getType()!=CoordinateType::POSITION_ABS){
        throw std::runtime_error("Given value is not a 'absolute location' coordinate.");
    }
    return transformToIntermediateCRS(
        value(getIntermediateCRS()),
        location(getIntermediateCRS()),
        value.getTime(),
        value.getType()
    );
}
Coordinate DerivedCRS::transformToIntermediateCRS(const Coordinate& value) const{
    Coordinate valueInSelf=value.transformTo(non_const_this());
    return transformToIntermediateCRS(
        valueInSelf(),
        valueInSelf.getLocation(),
        value.getTime(),
        value.getType()
    );
}
Coordinate DerivedCRS::transformFromIntermediateCRS(const Coordinate& value) const{
    Coordinate valueInIntermediate=value.transformTo(getIntermediateCRS());
    return transformFromIntermediateCRS(
        valueInIntermediate(),
        valueInIntermediate.getLocation(),
        value.getTime(),
        value.getType()
    );
}
Coordinate DerivedCRS::transformToIntermediateCRS(const Eigen::Vector3d& value, const Time& time, const CoordinateType& coordinateType) const{
    return transformToIntermediateCRS(value,Eigen::Vector3d::Zero(),time,coordinateType);
}
Coordinate DerivedCRS::transformFromIntermediateCRS(const Eigen::Vector3d& value, const Time& time, const CoordinateType& coordinateType) const{
    return transformFromIntermediateCRS(value,Eigen::Vector3d::Zero(),time,coordinateType);
}
void DerivedCRS::setBaseCRS(const std::shared_ptr<CoordinateReferenceSystem>& newBaseCRS) const{
    if(!isValid()){validate();}
    if(_base==newBaseCRS){
        return;
    }
    if(_base){
        removeDerivedCRS(_base,non_const_this());
    }
    if(!newBaseCRS){
        throw std::runtime_error("The given base CRS is not a valid object.");
    }
    _base=newBaseCRS;
    if(auto derived = std::dynamic_pointer_cast<DerivedCRS>(_base)){
        _nonDerivedBase=derived->getNonDerivedBaseCRS();
    }else{
        _nonDerivedBase=_base;
    }
    addDerivedCRS(_base,non_const_this());
    invalidate();
}


void exportDerivedCRS(py::module &m, const std::shared_ptr<asrc::core::FactoryHelper>& factoryHelper){
    using namespace pybind11::literals;

    expose_entity_subclass<DerivedCRS>(m,"DerivedCRS")
    .def("transformToBaseCRS",py::overload_cast<const Coordinate&,const Coordinate&>(&DerivedCRS::transformToBaseCRS,py::const_))
    .def("transformToBaseCRS",py::overload_cast<const Coordinate&>(&DerivedCRS::transformToBaseCRS,py::const_))
    .def("transformToBaseCRS",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const Time&,const CoordinateType&>(&DerivedCRS::transformToBaseCRS,py::const_))
    .def("transformToBaseCRS",py::overload_cast<const Eigen::Vector3d&,const Time&,const CoordinateType&>(&DerivedCRS::transformToBaseCRS,py::const_))
    .def("transformFromBaseCRS",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const Time&,const CoordinateType&>(&DerivedCRS::transformFromBaseCRS,py::const_))
    .def("transformFromBaseCRS",py::overload_cast<const Coordinate&,const Coordinate&>(&DerivedCRS::transformFromBaseCRS,py::const_))
    .def("transformFromBaseCRS",py::overload_cast<const Coordinate&>(&DerivedCRS::transformFromBaseCRS,py::const_))
    .def("transformFromBaseCRS",py::overload_cast<const Eigen::Vector3d&,const Time&,const CoordinateType&>(&DerivedCRS::transformFromBaseCRS,py::const_))
    .def("transformToNonDerivedBaseCRS",py::overload_cast<const Coordinate&,const Coordinate&>(&DerivedCRS::transformToNonDerivedBaseCRS,py::const_))
    .def("transformToNonDerivedBaseCRS",py::overload_cast<const Coordinate&>(&DerivedCRS::transformToNonDerivedBaseCRS,py::const_))
    .def("transformToNonDerivedBaseCRS",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const Time&,const CoordinateType&>(&DerivedCRS::transformToNonDerivedBaseCRS,py::const_))
    .def("transformToNonDerivedBaseCRS",py::overload_cast<const Eigen::Vector3d&,const Time&,const CoordinateType&>(&DerivedCRS::transformToNonDerivedBaseCRS,py::const_))
    .def("transformFromNonDerivedBaseCRS",py::overload_cast<const Coordinate&,const Coordinate&>(&DerivedCRS::transformFromNonDerivedBaseCRS,py::const_))
    .def("transformFromNonDerivedBaseCRS",py::overload_cast<const Coordinate&>(&DerivedCRS::transformFromNonDerivedBaseCRS,py::const_))
    .def("transformFromNonDerivedBaseCRS",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const Time&,const CoordinateType&>(&DerivedCRS::transformFromNonDerivedBaseCRS,py::const_))
    .def("transformFromNonDerivedBaseCRS",py::overload_cast<const Eigen::Vector3d&,const Time&,const CoordinateType&>(&DerivedCRS::transformFromNonDerivedBaseCRS,py::const_))
    .def("transformToIntermediateCRS",py::overload_cast<const Coordinate&,const Coordinate&>(&DerivedCRS::transformToIntermediateCRS,py::const_))
    .def("transformToIntermediateCRS",py::overload_cast<const Coordinate&>(&DerivedCRS::transformToIntermediateCRS,py::const_))
    .def("transformToIntermediateCRS",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const Time&,const CoordinateType&>(&DerivedCRS::transformToIntermediateCRS,py::const_))
    .def("transformToIntermediateCRS",py::overload_cast<const Eigen::Vector3d&,const Time&,const CoordinateType&>(&DerivedCRS::transformToIntermediateCRS,py::const_))
    .def("transformFromIntermediateCRS",py::overload_cast<const Coordinate&,const Coordinate&>(&DerivedCRS::transformFromIntermediateCRS,py::const_))
    .def("transformFromIntermediateCRS",py::overload_cast<const Coordinate&>(&DerivedCRS::transformFromIntermediateCRS,py::const_))
    .def("transformFromIntermediateCRS",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const Time&,const CoordinateType&>(&DerivedCRS::transformFromIntermediateCRS,py::const_))
    .def("transformFromIntermediateCRS",py::overload_cast<const Eigen::Vector3d&,const Time&,const CoordinateType&>(&DerivedCRS::transformFromIntermediateCRS,py::const_))
    ;
}

ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
