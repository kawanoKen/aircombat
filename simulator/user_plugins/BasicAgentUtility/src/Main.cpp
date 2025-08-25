// Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#include "Common.h"
#include "util/init.h"

namespace py=pybind11;

ASRC_PLUGIN_NAMESPACE_BEGIN

ASRC_EXPORT_PYTHON_MODULE(m,factoryHelper)

    bindSTLContainer(m); // py::module_local(true)なSTLコンテナのバインドを行う。

    util::init(m);

}

ASRC_PLUGIN_NAMESPACE_END
