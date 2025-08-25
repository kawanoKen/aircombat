// Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#include "crs/CoordinateReferenceSystem.h"

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)

using namespace util;

void exportCoordinateReferenceSystem(py::module &m, const std::shared_ptr<asrc::core::FactoryHelper>& factoryHelper){
    exportCoordinateReferenceSystemBase(m,factoryHelper);
    exportGeodeticCRS(m,factoryHelper);
    exportPureFlatCRS(m,factoryHelper);
    exportDerivedCRS(m,factoryHelper);
    exportAffineCRS(m,factoryHelper);
    exportTopocentricCRS(m,factoryHelper);
    exportProjectedCRS(m,factoryHelper);
}

ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
