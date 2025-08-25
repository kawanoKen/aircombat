// Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#include <ASRCAISim1/Factory.h>
#include <ASRCAISim1/PhysicalAsset.h>
#include <ASRCAISim1/Missile.h>
#include <ASRCAISim1/Track.h>
#include "SupervisedTaskMixer.h"
#include "SimpleSupervisedTaskSample.h"
#include <iostream>
namespace py=pybind11;

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)

ASRC_EXPORT_PYTHON_MODULE(m,factoryHelper)
    using namespace pybind11::literals;
    bindSTLContainer(m); //bind standard STL containers from <ASRCAISim1/Common.h>
    exportSupervisedTaskMixer(m,factoryHelper);
    exportSimpleSupervisedTaskSample(m,factoryHelper);
}

ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)

