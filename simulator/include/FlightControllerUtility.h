// Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#pragma once
#include "Common.h"
#include <cmath>
#include <random>
#include <iostream>
#include <pybind11/pybind11.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/CXX11/Tensor>
#include "MotionState.h"

namespace py=pybind11;

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)
ASRC_INLINE_NAMESPACE_BEGIN(util)

Coordinate PYBIND11_EXPORT pitchLimitter(const MotionState& motion, const Coordinate& dstDir,const double& pitchLimit);
Eigen::Vector3d PYBIND11_EXPORT pitchLimitter(const MotionState& motion, const Eigen::Vector3d& dstDir,const double& pitchLimit);//dstDirをEigen::Vector3dで与えた場合はmotionと同じCRSと仮定する。

ASRC_DECLARE_NON_POLYMORPHIC_DATA_CLASS(AltitudeKeeper)
    public:
    AltitudeKeeper();
    AltitudeKeeper(const nl::json& config);
    Coordinate operator()(const MotionState& motion, const Coordinate& dstDir, const double& dstAlt);
    Eigen::Vector3d operator()(const MotionState& motion, const Eigen::Vector3d& dstDir, const double& dstAlt);//dstDirをEigen::Vector3dで与えた場合はmotionと同じCRSと仮定する。
    double getDstPitch(const MotionState& motion, const double& dstAlt);
    double inverse(const MotionState& motion, const double& dstPitch);
    template<class Archive>
    void serialize(Archive & archive) {
        if constexpr (traits::output_archive<Archive>){
            ASRC_SERIALIZE_NVP(archive
                ,pGain
                ,dGain
                ,minPitch
                ,maxPitch
            )
        }else{
            util::try_load(archive,"pGain",pGain);
            util::try_load(archive,"dGain",dGain);
            util::try_load(archive,"minPitch",minPitch);
            util::try_load(archive,"maxPitch",maxPitch);
        }
    }
    public:
    double pGain;//目標高度との差に対するゲイン。負の値。
    double dGain;//高度変化に対するゲイン。正の値。
    double minPitch;//ピッチ角の下限
    double maxPitch;//ピッチ角の上限
};

void exportFlightControllerUtility(py::module &m);

ASRC_NAMESPACE_END(util)
ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
