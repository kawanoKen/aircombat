// Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#pragma once
#include "PhysicalAsset.h"
namespace py=pybind11;
namespace nl=nlohmann;

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)

ASRC_DECLARE_DERIVED_CLASS_WITH_TRAMPOLINE(Propulsion,PhysicalAsset)
	/*
	Generic Propulsion base class.
    */
    public:
    //constructors & destructor
    using BaseType::BaseType;
    //functions
    virtual void initialize() override;
    virtual double getFuelFlowRate()=0;//In [kg/s]
    virtual double getThrust()=0;// In [N]
    virtual double calcFuelFlowRate(const nl::json& args)=0;//as generic interface
    virtual double calcThrust(const nl::json& args)=0;//as generic interface
    virtual void setPowerCommand(const double& pCmd_)=0;//Normalized command in [0,1]
    virtual void control() override;
};

ASRC_DECLARE_DERIVED_TRAMPOLINE(Propulsion)
    virtual double getFuelFlowRate() override{
        PYBIND11_OVERRIDE_PURE(double,Base,getFuelFlowRate);
    }
    virtual double getThrust() override{
        PYBIND11_OVERRIDE_PURE(double,Base,getThrust);
    }
    virtual double calcFuelFlowRate(const nl::json& args) override{
        PYBIND11_OVERRIDE_PURE(double,Base,calcFuelFlowRate,args);
    }
    virtual double calcThrust(const nl::json& args) override{
        PYBIND11_OVERRIDE_PURE(double,Base,calcThrust,args);
    }
    virtual void setPowerCommand(const double& pCmd_) override{
        PYBIND11_OVERRIDE_PURE(void,Base,setPowerCommand,pCmd_);
    }
};

void exportPropulsion(py::module &m, const std::shared_ptr<asrc::core::FactoryHelper>& factoryHelper);

ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
