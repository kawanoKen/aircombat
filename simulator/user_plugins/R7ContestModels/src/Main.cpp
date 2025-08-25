// Copyright (c) 2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#include <ASRCAISim1/Common.h>
#include "Common.h"
#include "R7ContestMassPointFighter.h"
#include "PlanarFighter.h"
#include "PlanarMissile.h"
#include "CapsuleCoursePatrolAgent.h"

ASRC_PLUGIN_NAMESPACE_BEGIN

ASRC_EXPORT_PYTHON_MODULE(m,factoryHelper)
    bindSTLContainer(m); // py::module_local(true)なSTLコンテナのバインドを行う。

    exportR7ContestMassPointFighter(m,factoryHelper);
    exportPlanarFighter(m,factoryHelper);
    exportPlanarMissile(m,factoryHelper);
    exportCapsuleCoursePatrolAgent(m,factoryHelper);

}

ASRC_PLUGIN_NAMESPACE_END
