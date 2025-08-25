// Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#pragma once
#include "../Common.h"
#include <ASRCAISim1/Track.h>
#include <ASRCAISim1/MotionState.h>

ASRC_PLUGIN_NAMESPACE_BEGIN

namespace util{

/**
 * @brief Track3Dを距離順でソートする関数(C++用)
 * 
 * @param [in,out] tracks ソート対象のTrack3Dのリスト。無効なTrack3Dを含んではならない。
 * @param [in] motions 距離を計算する対象のMotionStateのリスト。無効なMotionStateはスキップされる。
 * @param [in] nearestFirst trueならば近い順、falseならば遠い順
 */
void PYBIND11_EXPORT sortTrack3DByDistance(
    std::vector<asrc::core::Track3D>& tracks,
    const std::vector<asrc::core::MotionState>& motions,
    bool nearestFirst
);

/**
 * @brief Track3Dを距離順でソートする関数(Python用)
 * 
 * @param [in,out] tracks ソート対象のTrack3Dのリスト無効なTrack3Dを含んではならない。
 * @param [in] motions 距離を計算する対象のMotionStateのリスト。無効なMotionStateはスキップされる。
 * @param [in] nearestFirst trueならば近い順、falseならば遠い順
 */
void PYBIND11_EXPORT sortTrack3DByDistance(
    py::list& tracks,
    const py::list& motions,
    bool nearestFirst
);

void exportSortTrack3DByDistance(py::module &m);

}

ASRC_PLUGIN_NAMESPACE_END
