// Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#pragma once
#include "../Common.h"

ASRC_PLUGIN_NAMESPACE_BEGIN

namespace util{

//MultiDiscrete⇔Discreteの相互変換
int PYBIND11_EXPORT multi_discrete_to_discrete(const Eigen::VectorXi& multi_discrete_action,const Eigen::VectorXi& multi_discrete_shape);
Eigen::VectorXi PYBIND11_EXPORT discrete_to_multi_discrete(const int& discrete_action,const Eigen::VectorXi& multi_discrete_shape);

void exportDiscreteMultiDiscreteConversion(py::module &m);

}

ASRC_PLUGIN_NAMESPACE_END
