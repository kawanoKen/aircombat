// Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#include "util/init.h"

ASRC_PLUGIN_NAMESPACE_BEGIN

namespace util{

void init(py::module &m_parent)
{
	using namespace pybind11::literals;
	auto m=m_parent.def_submodule("util");
	exportDiscreteMultiDiscreteConversion(m);
	exportFuelManagementUtility(m);
	exportMissileRangeUtility(m);
	exportTeamOrigin(m);
	exportSortTrack2DByAngle(m);
	exportSortTrack3DByDistance(m);
}

}

ASRC_PLUGIN_NAMESPACE_END
