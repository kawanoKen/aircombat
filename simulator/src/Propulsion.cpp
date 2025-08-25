// Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#include "Propulsion.h"
#include "Fighter.h"

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)

using namespace util;

void Propulsion::initialize(){
    BaseType::initialize();
    assert(!parent.expired() && isBoundToParent);
}
void Propulsion::control(){
    auto p=getShared<Fighter>(parent);
    if(p->isAlive()){
        nl::json ctrl=p->controllers["FlightController"].lock()->commands["motion"];
        setPowerCommand(ctrl.at("pCmd"));
    }
}

void exportPropulsion(py::module &m, const std::shared_ptr<asrc::core::FactoryHelper>& factoryHelper)
{
    using namespace pybind11::literals;
    expose_entity_subclass<Propulsion>(m,"Propulsion")
    DEF_FUNC(Propulsion,getFuelFlowRate)
    DEF_FUNC(Propulsion,getThrust)
    DEF_FUNC(Propulsion,calcFuelFlowRate)
    DEF_FUNC(Propulsion,calcThrust)
    DEF_FUNC(Propulsion,setPowerCommand)
    ;
    //FACTORY_ADD_CLASS(PhysicalAsset,Propulsion) //Do not register to Factory because Propulsion is abstract class.
}

ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
