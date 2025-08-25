// Copyright (c) 2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#include <ASRCAISim1/Factory.h>
#include "Common.h"
#include "VirtualSimulatorSample.h"
#include "R7ContestAgentSample01.h"
#include "R7ContestRewardSample01.h"
#include "R7ContestRewardSample02.h"

ASRC_PLUGIN_NAMESPACE_BEGIN

ASRC_EXPORT_PYTHON_MODULE(m,factoryHelper)
    bindSTLContainer(m); // py::module_local(true)なSTLコンテナのバインドを行う。

	exportVirtualSimulatorSample(m);
    exportR7ContestAgentSample01(m,factoryHelper);
    exportR7ContestRewardSample01(m,factoryHelper);
    exportR7ContestRewardSample02(m,factoryHelper);

}

ASRC_PLUGIN_NAMESPACE_END
