// Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#include "SimpleSupervisedTaskSample.h"
#include <ASRCAISim1/Agent.h>
#include <random>
#include <regex>

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)
using namespace util;

void SimpleSupervisedTaskSample01::initialize(){
    BaseType::initialize();
    taskName=getValueFromJsonKRD<std::string>(modelConfig,"taskName",randomGen,"Classification:multiples_of_3");
    lower=getValueFromJsonKRD(modelConfig,"lower",randomGen,1);
    upper=getValueFromJsonKRD(modelConfig,"upper",randomGen,100);
}
void SimpleSupervisedTaskSample01::serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full){
    BaseType::serializeInternalState(archive,full);

    if(full){
        ASRC_SERIALIZE_NVP(archive
            ,taskName
            ,lower
            ,upper
        )
    }
    ASRC_SERIALIZE_NVP(archive
        ,inputs
    )
}
void SimpleSupervisedTaskSample01::onEpisodeEnd(){
    inputs.clear();
}
py::object SimpleSupervisedTaskSample01::calcTaskInputSpace(const std::shared_ptr<Agent>& agent){
    py::module_ spaces=py::module_::import("gymnasium.spaces");
    Eigen::VectorXi shape(1);
    shape<<1;
    return spaces.attr("Box")(lower,upper,shape,py::dtype::of<float>());
}
py::object SimpleSupervisedTaskSample01::calcTaskTruthSpace(const std::shared_ptr<Agent>& agent){
    py::module_ spaces=py::module_::import("gymnasium.spaces");
    return spaces.attr("Discrete")(2);
}
py::object SimpleSupervisedTaskSample01::calcTaskInput(const std::shared_ptr<Agent>& agent){
    std::uniform_int_distribution<std::uint64_t> i_dist(1,100);
    inputs[agent->getFullName()]=i_dist(randomGen);
    Eigen::VectorXf obs(1);
    obs<<(float)inputs[agent->getFullName()];
    return py::cast(obs);
}
std::pair<bool,py::object> SimpleSupervisedTaskSample01::calcTaskTruth(const std::shared_ptr<Agent>& agent){
    Eigen::VectorXi obs(1);
    obs<<(inputs[agent->getFullName()]%3==0 ? 1 : 0);
    return std::make_pair<bool,py::object>(true,py::cast(obs));
}

void SimpleSupervisedTaskSample02::initialize(){
    BaseType::initialize();
    taskName=getValueFromJsonKRD<std::string>(modelConfig,"taskName",randomGen,"Regression:product_of_2_values");
    lower=getValueFromJsonKRD(modelConfig,"lower",randomGen,-10.0);
    upper=getValueFromJsonKRD(modelConfig,"upper",randomGen,10.0);
    assert(lower<upper);
}
void SimpleSupervisedTaskSample02::serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full){
    BaseType::serializeInternalState(archive,full);

    if(full){
        ASRC_SERIALIZE_NVP(archive
            ,taskName
            ,lower
            ,upper
        )
    }
    ASRC_SERIALIZE_NVP(archive
        ,inputs
    )
}
void SimpleSupervisedTaskSample02::onEpisodeEnd(){
    inputs.clear();
}
py::object SimpleSupervisedTaskSample02::calcTaskInputSpace(const std::shared_ptr<Agent>& agent){
    py::module_ spaces=py::module_::import("gymnasium.spaces");
    Eigen::VectorXi shape(1);
    shape<<2;
    return spaces.attr("Box")(lower,upper,shape,py::dtype::of<float>());
}
py::object SimpleSupervisedTaskSample02::calcTaskTruthSpace(const std::shared_ptr<Agent>& agent){
    py::module_ spaces=py::module_::import("gymnasium.spaces");
    float vMin,vMax;
    if(lower*upper<=0.0){//入力の変域に0を含む
        vMin=lower*upper;
        vMax=std::max<float>(abs(lower*lower),abs(upper*upper));
    }else{//入力の変域に0を含まない
        vMin=std::min<float>(abs(lower*lower),abs(upper*upper));
        vMax=std::max<float>(abs(lower*lower),abs(upper*upper));
    }
    Eigen::VectorXi shape(1);
    shape<<1;
    return spaces.attr("Box")(vMin,vMax,shape,py::dtype::of<float>());
}
py::object SimpleSupervisedTaskSample02::calcTaskInput(const std::shared_ptr<Agent>& agent){
    Eigen::VectorXf obs(2);
    std::uniform_real_distribution<float> d_dist(lower,upper);
    obs<<d_dist(randomGen),d_dist(randomGen);
    inputs[agent->getFullName()]=obs;
    return py::cast(obs);
}
std::pair<bool,py::object> SimpleSupervisedTaskSample02::calcTaskTruth(const std::shared_ptr<Agent>& agent){
    Eigen::VectorXf obs(1);
    obs<<inputs[agent->getFullName()](0)*inputs[agent->getFullName()](1);
    return std::make_pair<bool,py::object>(true,py::cast(obs));
}

void exportSimpleSupervisedTaskSample(py::module &m, const std::shared_ptr<asrc::core::FactoryHelper>& factoryHelper)
{
    using namespace pybind11::literals;
    expose_entity_subclass<SimpleSupervisedTaskSample01>(m,"SimpleSupervisedTaskSample01")
    DEF_READWRITE(SimpleSupervisedTaskSample01,lower)
    DEF_READWRITE(SimpleSupervisedTaskSample01,upper)
    ;
    FACTORY_ADD_CLASS(Callback,SimpleSupervisedTaskSample01)

    expose_entity_subclass<SimpleSupervisedTaskSample02>(m,"SimpleSupervisedTaskSample02")
    DEF_READWRITE(SimpleSupervisedTaskSample02,lower)
    DEF_READWRITE(SimpleSupervisedTaskSample02,upper)
    ;
    FACTORY_ADD_CLASS(Callback,SimpleSupervisedTaskSample02)
}

ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
