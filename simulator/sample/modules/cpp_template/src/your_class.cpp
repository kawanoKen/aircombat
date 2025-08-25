// Copyright (c) 2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#include "your_class.h"
ASRC_PLUGIN_NAMESPACE_BEGIN

using namespace asrc::core;
using namespace util;

void your_class::initialize(){
    BaseType::initialize();
}
void your_class::serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full){
    BaseType::serializeInternalState(archive,full);
}

void your_class::validate(){
    BaseType::validate();
}
py::object your_class::observation_space(){
    py::module_ spaces=py::module_::import("gymnasium.spaces");
    return spaces.attr("Discrete")(1);
}
py::object your_class::makeObs(){
    return py::int_(1);
}

py::object your_class::action_space(){
    py::module_ spaces=py::module_::import("gymnasium.spaces");
    return spaces.attr("Discrete")(1);
}

void your_class::deploy(py::object action_){
    int action=py::cast<int>(action_);
}
void your_class::control(){
}
py::object your_class::convertActionFromAnother(const nl::json& decision,const nl::json& command){
    return py::int_(1);
}
void your_class::controlWithAnotherAgent(const nl::json& decision,const nl::json& command){
	//基本的にはオーバーライド不要だが、模倣時にActionと異なる行動を取らせたい場合に使用する。
	control();
	//例えば、以下のようにcommandを置換すると射撃のみexpertと同タイミングで行うように変更可能。
	//commands[parent->getFullName()]["weapon"]=command.at(parent->getFullName()).at("weapon");
}

void export_your_class(py::module &m, const std::shared_ptr<asrc::core::FactoryHelper>& factoryHelper)
{
    using namespace pybind11::literals;
    expose_entity_subclass<your_class>(m,"your_class",py::module_local()) // 投稿時はpy::module_local()を引数に加えてビルドしたものを用いること！
    ;
    FACTORY_ADD_CLASS(Agent,your_class)
}

ASRC_PLUGIN_NAMESPACE_END
