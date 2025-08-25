// Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#include "util/serialization/mersenne_twister.h"

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)
ASRC_INLINE_NAMESPACE_BEGIN(util)

void exportStdMersenneTwister(py::module &m){
    py::class_<std::mt19937>(m,"std_mt19937")
    .def(py::init([](){return std::mt19937();}))
    .def("seed",[](std::mt19937& gen,const unsigned int& seed_){gen.seed(seed_);})
    .def("__call__",[](std::mt19937& gen){return gen();})
    .def("discard",[](std::mt19937& gen,const unsigned long long& z){gen.discard(z);})
    ;
}

ASRC_NAMESPACE_END(util)
ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
