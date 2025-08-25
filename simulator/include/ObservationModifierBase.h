// Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#pragma once
#include <pybind11/pybind11.h>
#include "Utility.h"
#include "Callback.h"
#include "Agent.h"
namespace py=pybind11;
namespace nl=nlohmann;

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)

ASRC_DECLARE_DERIVED_CLASS_WITH_TRAMPOLINE(ObservationModifierBase,Callback)
	/*observationを上書きするCallbackの基底クラス。
    基底クラスでは、ExpertWrapperとSimpleMultiPortCombinerに対して子Agentに対する再帰的な上書き処理を提供しており、
    具象クラスでは通常のAgentに対する上書き処理と、上書き対象となるAgentの判定方法を実装すればよい。
	*/
    public:
    //constructors & destructor
    using BaseType::BaseType;
    //functions
    virtual void onGetObservationSpace() override;
    virtual void onMakeObs() override;
    virtual py::object overrideObservationSpace(const std::shared_ptr<Agent>& agent,py::object oldSpace);
    virtual py::object overrideObservationSpaceImpl(const std::shared_ptr<Agent>& agent,py::object oldSpace)=0;
    virtual py::object overrideObservation(const std::shared_ptr<Agent>& agent,py::object oldObs);
    virtual py::object overrideObservationImpl(const std::shared_ptr<Agent>& agent,py::object oldObs)=0;
    virtual bool isTarget(const std::shared_ptr<Agent>& agent)=0;
};

ASRC_DECLARE_DERIVED_TRAMPOLINE(ObservationModifierBase)
    virtual py::object overrideObservationSpace(const std::shared_ptr<Agent>& agent,py::object oldSpace) override{
        PYBIND11_OVERRIDE(py::object,Base,overrideObservationSpace,agent,oldSpace);
    }
    virtual py::object overrideObservationSpaceImpl(const std::shared_ptr<Agent>& agent,py::object oldSpace) override{
        PYBIND11_OVERRIDE_PURE(py::object,Base,overrideObservationSpaceImpl,agent,oldSpace);
    }
    virtual py::object overrideObservation(const std::shared_ptr<Agent>& agent,py::object oldObs) override{
        PYBIND11_OVERRIDE(py::object,Base,overrideObservation,agent,oldObs);
    }
    virtual py::object overrideObservationImpl(const std::shared_ptr<Agent>& agent,py::object oldObs) override{
        PYBIND11_OVERRIDE_PURE(py::object,Base,overrideObservationImpl,agent,oldObs);
    }
    virtual bool isTarget(const std::shared_ptr<Agent>& agent) override{
        PYBIND11_OVERRIDE_PURE(bool,Base,isTarget,agent);
    }
};

void exportObservationModifierBase(py::module &m, const std::shared_ptr<asrc::core::FactoryHelper>& factoryHelper);

ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
