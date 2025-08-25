// Copyright (c) 2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#include <ASRCAISim1/Common.h>
#include <ASRCAISim1/Factory.h>
#include "Common.h"
#include "BasicAACRuler01.h"
#include "BasicAACReward01.h"
#include "BasicAACRuleBasedAgent01.h"

ASRC_PLUGIN_NAMESPACE_BEGIN

ASRC_EXPORT_PYTHON_MODULE(m,factoryHelper)
    bindSTLContainer(m); // py::module_local(true)なSTLコンテナのバインドを行う。

    exportBasicAACRuler01(m,factoryHelper);
    exportBasicAACReward01(m,factoryHelper);
    exportBasicAACRuleBasedAgent01(m,factoryHelper);
}

ASRC_PLUGIN_NAMESPACE_END
