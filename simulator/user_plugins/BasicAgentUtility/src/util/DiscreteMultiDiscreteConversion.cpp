// Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#include "util/DiscreteMultiDiscreteConversion.h"

ASRC_PLUGIN_NAMESPACE_BEGIN

namespace util{

using namespace asrc::core;
using namespace util;

//MultiDiscrete⇔Discreteの相互変換
int multi_discrete_to_discrete(const Eigen::VectorXi& multi_discrete_action,const Eigen::VectorXi& multi_discrete_shape){
    //MultiDiscretionなactionをDiscreteなactionに変換する。
    int ret=0;
    for(int i=0;i<multi_discrete_shape.size();++i){
        ret*=multi_discrete_shape(i);
        ret+=multi_discrete_action(i);
    }
    return ret;
}
Eigen::VectorXi discrete_to_multi_discrete(const int& discrete_action,const Eigen::VectorXi& multi_discrete_shape){
    //Discrete化されたactionを元のMultiDiscreteなactionに変換する。
    Eigen::VectorXi ret=Eigen::VectorXi::Zero(multi_discrete_shape.size());
    int idx=discrete_action;
    for(int i=multi_discrete_shape.size()-1;i>=0;--i){
        ret(i)=idx%multi_discrete_shape(i);
        idx-=ret(i);
        idx/=multi_discrete_shape(i);
    }
    return ret;
}

void exportDiscreteMultiDiscreteConversion(py::module &m)
{
    using namespace pybind11::literals;

    m.def("multi_discrete_to_discrete",&multi_discrete_to_discrete);
    m.def("discrete_to_multi_discrete",&discrete_to_multi_discrete);
}

}

ASRC_PLUGIN_NAMESPACE_END
