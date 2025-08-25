// Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#pragma once
#include <vector>
#include <functional>
#include <future>
#include <mutex>
#include <pybind11/pybind11.h>
#include <Eigen/Core>
#include <Eigen/CXX11/Tensor>
#include "MathUtility.h"
#include "Missile.h"
namespace py=pybind11;
namespace nl=nlohmann;

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)

ASRC_DECLARE_DERIVED_CLASS_WITH_TRAMPOLINE(EkkerMissile,Missile)
    public:
    //configで指定するもの
    double tMax,tBurn,hitD,minV;
    double massI,massF,thrust,maxLoadG;
    double Sref,maxA,maxD,l,d,ln,lcgi,lcgf,lw,bw,bt,thicknessRatio,Sw,St;
    double frictionFactor;
    //内部変数
    double mass;//質量
    double lcg;//重心位置
    //constructors & destructor
    using BaseType::BaseType;
    //functions
    virtual void initialize() override;
    virtual void serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full) override;
    virtual void calcMotion(double tAftLaunch,double dt) override;
    virtual bool hitCheck(const Eigen::Vector3d &tpos,const Eigen::Vector3d &tpos_prev) override;
    virtual bool checkDeactivateCondition(double tAftLaunch) override;
};

ASRC_DECLARE_DERIVED_TRAMPOLINE(EkkerMissile)
    virtual void calcMotion(double tAftLaunch,double dt) override{
        PYBIND11_OVERRIDE(void,Base,calcMotion,tAftLaunch,dt);
    }
    virtual bool hitCheck(const Eigen::Vector3d &tpos,const Eigen::Vector3d &tpos_prev) override{
        PYBIND11_OVERRIDE(bool,Base,hitCheck,tpos,tpos_prev);
    }
    virtual bool checkDeactivateCondition(double tAftLaunch) override{
        PYBIND11_OVERRIDE(bool,Base,checkDeactivateCondition,tAftLaunch);
    }
};

void exportEkkerMissile(py::module &m, const std::shared_ptr<asrc::core::FactoryHelper>& factoryHelper);

ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
