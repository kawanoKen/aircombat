// Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#pragma once
#include "../Common.h"
#include <ASRCAISim1/Track.h>
#include <ASRCAISim1/MotionState.h>

ASRC_PLUGIN_NAMESPACE_BEGIN

namespace util{

/**
 * @brief Track2Dを特定方向との角度差の順でソートする関数
 * 
 * @param [in,out] tracks ソート対象のTrack2Dリスト。無効なTrack2Dを含んではならない。
 * @param [in] motions 角度差を計算する対象のMotionState
 * @param [in] direction 角度差の基準となる方向ベクトル(motionsのBody座標系での値)
 * @param [in] smallestFirst trueならば小さい順、falseならば大きい順
 */
void PYBIND11_EXPORT sortTrack2DByAngle(
    std::vector<asrc::core::Track2D>& tracks,
    const asrc::core::MotionState& motion,
	const Eigen::Vector3d& direction,
    bool smallestFirst
);

/**
 * @brief Track2Dを距離順でソートする関数
 * 
 * @param [in,out] tracks ソート対象のTrack2Dリスト。無効なTrack2Dを含んではならない。
 * @param [in] motions 角度差を計算する対象のMotionState
 * @param [in] direction 角度差の基準となる方向ベクトル(motionsのBody座標系での値)
 * @param [in] smallestFirst trueならば小さい順、falseならば大きい順
 */
void PYBIND11_EXPORT sortTrack2DByAngle(
    py::list& tracks,
    const asrc::core::MotionState& motion,
	const Eigen::Vector3d& direction,
    bool smallestFirst
);

void exportSortTrack2DByAngle(py::module &m);

}

ASRC_PLUGIN_NAMESPACE_END
