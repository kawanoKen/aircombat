// Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#include "Common.h"
#include "EntityIdentifier.h"
#include <cmath>
#include <iostream>
#include <pybind11/pybind11.h>
#include "Units.h"
#include "Utility.h"
#include "util/serialization/serialization.h"
#include "TimeSystem.h"
#include "JsonMutableFromPython.h"
#include "ConfigDispatcher.h"
#include "Quaternion.h"
#include "MathUtility.h"
#include "Coordinate.h"
#include "MotionState.h"
#include "crs/CoordinateReferenceSystem.h"
#include "CommunicationBuffer.h"
#include "Entity.h"
#include "Asset.h"
#include "StaticCollisionAvoider2D.h"
#include "PhysicalAsset.h"
#include "Controller.h"
#include "Track.h"
#include "Propulsion.h"
#include "FlightControllerUtility.h"
#include "Fighter.h"
#include "MassPointFighter.h"
#include "CoordinatedFighter.h"
#include "SixDoFFighter.h"
#include "Missile.h"
#include "EkkerMDT.h"
#include "EkkerMissile.h"
#include "Sensor.h"
#include "Agent.h"
#include "Callback.h"
#include "ObservationModifierBase.h"
#include "Ruler.h"
#include "Reward.h"
#include "WinLoseReward.h"
#include "GUIDataFrameCreator.h"
#include "Viewer.h"
#include "SimulationManager.h"
#include "Factory.h"
namespace py=pybind11;

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)

ASRC_INLINE_NAMESPACE_BEGIN(util)

void util_init(py::module &m_parent, std::shared_ptr<asrc::core::FactoryHelper> factoryHelper)
{
	using namespace pybind11::literals;
	auto m=m_parent.def_submodule("util");
    exportJsonMutableFromPython(m);
    exportUnits(m);
    exportUtility(m);
    exportMathUtility(m);
    exportSerialization(m);
    exportCommunicationBuffer(m);
    exportStaticCollisionAvoider2D(m);
    exportFlightControllerUtility(m);
    exportGUIDataFrameCreator(m);
}

ASRC_NAMESPACE_END(util)

void init(py::module &m_parent, std::shared_ptr<asrc::core::FactoryHelper> factoryHelper)
{
	using namespace pybind11::literals;
	auto m=m_parent;//.def_submodule("core");

    bindSTLContainer(m);
    util::util_init(m,factoryHelper);
    exportQuaternion(m);
    exportEntityIdentifier(m);
    exportTimeSystem(m);
    exportCoordinate(m);
    exportMotionState(m);
    exportFactory(m);
    exportEntity(m);
    exportCoordinateReferenceSystem(m,factoryHelper);
    exportAsset(m,factoryHelper);
    exportPhysicalAsset(m,factoryHelper);
    exportController(m,factoryHelper);
    exportTrack(m);
    exportPropulsion(m,factoryHelper);
    exportFighter(m,factoryHelper);
    exportMassPointFighter(m,factoryHelper);
    exportCoordinatedFighter(m,factoryHelper);
    exportSixDoFFighter(m,factoryHelper);
    exportMissile(m,factoryHelper);
    exportEkkerMDT(m);
    exportEkkerMissile(m,factoryHelper);
    exportSensor(m,factoryHelper);
    exportAgent(m,factoryHelper);
    exportCallback(m,factoryHelper);
    exportObservationModifierBase(m,factoryHelper);
    exportRuler(m,factoryHelper);
    exportReward(m,factoryHelper);
    exportWinLoseReward(m,factoryHelper);
    exportViewer(m,factoryHelper);
    exportSimulationManager(m);
}

std::shared_ptr<FactoryHelper> getModuleFactoryHelper(){
    static std::shared_ptr<FactoryHelper> _internal=std::make_shared<FactoryHelper>("core");
    return _internal;
}
PYBIND11_MODULE(libCore,m)
{
    using namespace pybind11::literals;
    m.doc()="ASRCAISim1";
    auto factoryHelper=getModuleFactoryHelper();
    ::asrc::core::init(m,factoryHelper);
    m.attr("factoryHelper") = factoryHelper;
}

ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
