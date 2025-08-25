// Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#include "util/FuelManagementUtility.h"
#include <limits>

ASRC_PLUGIN_NAMESPACE_BEGIN

namespace util{

using namespace asrc::core;
using namespace util;

double calcDistanceMargin(const std::string& team, const MotionState& parentMotion, const nl::json& parentObservables, const nl::json& rulerObservables){
    //正規化した余剰燃料(距離換算)の計算
    double optCruiseFuelFlowRatePerDistance=parentObservables.at("/spec/propulsion/optCruiseFuelFlowRatePerDistance"_json_pointer);
    double fuelRemaining=parentObservables.at("/propulsion/fuelRemaining"_json_pointer);
    double maxReachableRange=std::numeric_limits<double>::infinity();
    if(optCruiseFuelFlowRatePerDistance>0.0){
        maxReachableRange=fuelRemaining/optCruiseFuelFlowRatePerDistance;
    }
    Eigen::Vector2d forwardAx=rulerObservables.at("forwardAx").at(team);
    double dLine=rulerObservables.at("dLine");
    double distanceFromLine=forwardAx.dot(parentMotion.pos().block<2,1>(0,0,2,1))+dLine;
    double distanceFromBase=rulerObservables.at("distanceFromBase").at(team);
    double fuelMargin=rulerObservables.at("fuelMargin");
    return std::clamp<double>((maxReachableRange/(1+fuelMargin)-(distanceFromLine+distanceFromBase))/(2*dLine),-1.0,1.0);
}

void exportFuelManagementUtility(py::module &m)
{
    using namespace pybind11::literals;

    m.def("calcDistanceMargin",&calcDistanceMargin);
}

}

ASRC_PLUGIN_NAMESPACE_END
