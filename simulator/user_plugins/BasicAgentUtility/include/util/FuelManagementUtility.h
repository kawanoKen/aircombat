// Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#pragma once
#include "../Common.h"
#include <ASRCAISim1/MotionState.h>

ASRC_PLUGIN_NAMESPACE_BEGIN

namespace util{

//余剰燃料に関する関数
double PYBIND11_EXPORT calcDistanceMargin(const std::string& team, const asrc::core::MotionState& parentMotion, const nl::json& parentObservables, const nl::json& rulerObservables);

void exportFuelManagementUtility(py::module &m);

}

ASRC_PLUGIN_NAMESPACE_END
