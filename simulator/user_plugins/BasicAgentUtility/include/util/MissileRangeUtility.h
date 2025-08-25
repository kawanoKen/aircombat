// Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#pragma once
#include "../Common.h"
#include <ASRCAISim1/PhysicalAsset.h>
#include <ASRCAISim1/MotionState.h>
#include <ASRCAISim1/Track.h>

ASRC_PLUGIN_NAMESPACE_BEGIN

namespace util{

//射程計算に関する関数
double PYBIND11_EXPORT calcRHead(const std::shared_ptr<asrc::core::PhysicalAssetAccessor>& parent,const asrc::core::MotionState& motion,const asrc::core::Track3D& track, bool shooter_head_on);//相手が現在の位置、速度で直ちに正面を向いて水平飛行になった場合の射程
double PYBIND11_EXPORT calcRTail(const std::shared_ptr<asrc::core::PhysicalAssetAccessor>& parent,const asrc::core::MotionState& motion,const asrc::core::Track3D& track, bool shooter_head_on);//相手が現在の位置、速度で直ちに背後を向いて水平飛行になった場合の射程
double PYBIND11_EXPORT calcRNorm(const std::shared_ptr<asrc::core::PhysicalAssetAccessor>& parent,const asrc::core::MotionState& motion,const asrc::core::Track3D& track, bool shooter_head_on);//正規化した射程
double PYBIND11_EXPORT calcRHeadE(const std::shared_ptr<asrc::core::PhysicalAssetAccessor>& parent,const asrc::core::MotionState& motion,const asrc::core::Track3D& track, bool shooter_head_on);//自身が現在の位置、速度で直ちに正面を向いて水平飛行になった場合の相手の射程
double PYBIND11_EXPORT calcRTailE(const std::shared_ptr<asrc::core::PhysicalAssetAccessor>& parent,const asrc::core::MotionState& motion,const asrc::core::Track3D& track, bool shooter_head_on);//自身が現在の位置、速度で直ちに背後を向いて水平飛行になった場合の相手の射程
double PYBIND11_EXPORT calcRNormE(const std::shared_ptr<asrc::core::PhysicalAssetAccessor>& parent,const asrc::core::MotionState& motion,const asrc::core::Track3D& track, bool shooter_head_on);//正規化した相手の射程

void exportMissileRangeUtility(py::module &m);

}

ASRC_PLUGIN_NAMESPACE_END
