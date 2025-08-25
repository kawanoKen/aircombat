// Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#include "util/sortTrack2DByAngle.h"
#include <limits>
#include <functional>
#include <pybind11/functional.h>

ASRC_PLUGIN_NAMESPACE_BEGIN

namespace util{

using namespace asrc::core;
using namespace util;

void sortTrack2DByAngle(
    std::vector<Track2D>& tracks,
    const MotionState& motion,
	const Eigen::Vector3d& direction,
    bool smallestFirst
){
    if(tracks.size()==0){
        return;
    }
    auto crs=motion.getCRS();
    if(smallestFirst){
        std::sort(tracks.begin(),tracks.end(),
        [&](Track2D& lhs,Track2D& rhs)-> bool {
            return -lhs.dir(crs).dot(motion.dirBtoP(direction))
                <-rhs.dir(crs).dot(motion.dirBtoP(direction));
        });
    }else{
        std::sort(tracks.begin(),tracks.end(),
        [&](Track2D& lhs,Track2D& rhs)-> bool {
            return lhs.dir(crs).dot(motion.dirBtoP(direction))
                <rhs.dir(crs).dot(motion.dirBtoP(direction));
        });
    }
}

void sortTrack2DByAngle(
    py::list& tracks,
    const MotionState& motion,
	const Eigen::Vector3d& direction,
    bool smallestFirst
){
    using namespace pybind11::literals;
    if(py::len(tracks)==0){
        return;
    }
    auto crs=motion.getCRS();
    if(smallestFirst){
        std::function<double(std::reference_wrapper<Track2D>)> comparer=[&](std::reference_wrapper<Track2D> t){
            return -t.get().dir(crs).dot(motion.dirBtoP(direction));
        };
        tracks.attr("sort")("key"_a=comparer);
    }else{
        std::function<double(std::reference_wrapper<Track2D>)> comparer=[&](std::reference_wrapper<Track2D> t){
            return t.get().dir(crs).dot(motion.dirBtoP(direction));
        };
        tracks.attr("sort")("key"_a=comparer);
    }
}

void exportSortTrack2DByAngle(py::module &m)
{
    using namespace pybind11::literals;

    m.def("sortTrack2DByAngle",py::overload_cast<py::list&,const MotionState&,const Eigen::Vector3d&,bool>(&sortTrack2DByAngle));
    m.def("sortTrack2DByAngle",py::overload_cast<std::vector<Track2D>&,const MotionState&,const Eigen::Vector3d&,bool>(&sortTrack2DByAngle));
}

}

ASRC_PLUGIN_NAMESPACE_END
