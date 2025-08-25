// Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#pragma once
#include "../Common.h"
#include "CoordinateReferenceSystemBase.h"
#include "GeodeticCRS.h"
#include "PureFlatCRS.h"
#include "DerivedCRS.h"
#include "AffineCRS.h"
#include "TopocentricCRS.h"
#include "ProjectedCRS.h"

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)

void exportCoordinateReferenceSystem(py::module &m, const std::shared_ptr<asrc::core::FactoryHelper>& factoryHelper);

ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
