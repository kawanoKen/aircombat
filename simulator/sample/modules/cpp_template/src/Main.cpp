// Copyright (c) 2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#include <ASRCAISim1/Common.h>
#include "Common.h"
#include "your_class.h"

ASRC_PLUGIN_NAMESPACE_BEGIN

void test(){
    std::cout<<"test!"<<std::endl;
}

ASRC_EXPORT_PYTHON_MODULE(m,factoryHelper)
    bindSTLContainer(m); // py::module_local(true)なSTLコンテナのバインドを行う。

    m.def("test",&test);
    export_your_class(m,factoryHelper);
}

ASRC_PLUGIN_NAMESPACE_END
