// Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#include "util/sortTrack3DByDistance.h"
#include <limits>
#include <functional>
#include <pybind11/functional.h>

ASRC_PLUGIN_NAMESPACE_BEGIN

namespace util{

using namespace asrc::core;
using namespace util;

void sortTrack3DByDistance(
    std::vector<Track3D>& tracks,
    const std::vector<asrc::core::MotionState>& motions,
    bool nearestFirst
){
    if(tracks.size()==0 || motions.size()==0){
        return;
    }
    auto crs=tracks[0].getCRS();
    if(nearestFirst){
        std::sort(tracks.begin(),tracks.end(),
        [&](Track3D& lhs,Track3D& rhs)-> bool {
            double lMin=-1,rMin=-1;
            for(auto&& motion : motions){
                if(!motion.getCRS()){
                    continue;
                }
                double tmp=(lhs.pos(crs)-motion.pos(crs)).norm();
                if(lMin<0 || lMin<tmp){
                    lMin=tmp;
                }
                tmp=(rhs.pos(crs)-motion.pos(crs)).norm();
                if(rMin<0 || rMin<tmp){
                    rMin=tmp;
                }
            }
            return lMin<rMin;
        });
    }else{
        std::sort(tracks.begin(),tracks.end(),
        [&](Track3D& lhs,Track3D& rhs)-> bool {
            double lMax=-1,rMax=-1;
            for(auto&& motion : motions){
                if(!motion.getCRS()){
                    continue;
                }
                double tmp=(lhs.pos(crs)-motion.pos(crs)).norm();
                if(lMax<0 || lMax>tmp){
                    lMax=tmp;
                }
                tmp=(rhs.pos(crs)-motion.pos(crs)).norm();
                if(rMax<0 || rMax>tmp){
                    rMax=tmp;
                }
            }
            return lMax>rMax;
        });
    }
}

void sortTrack3DByDistance(
    py::list& tracks,
    const py::list& motions,
    bool nearestFirst
){
    using namespace pybind11::literals;
    if(py::len(tracks)==0 || py::len(motions)==0){
        return;
    }
    auto crs=tracks[0].cast<std::reference_wrapper<Track3D>>().get().getCRS();
    if(nearestFirst){
        std::function<double(std::reference_wrapper<Track3D>)> comparer=[&](std::reference_wrapper<Track3D> t){
            double rMin=-1;
            for(auto&& motion : motions){
                MotionState& motionR=motion.cast<std::reference_wrapper<MotionState>>().get();
                if(!motionR.getCRS()){
                    continue;
                }
                double tmp=(t.get().pos(crs)-motionR.pos(crs)).norm();
                if(rMin<0 || rMin<tmp){
                    rMin=tmp;
                }
            }
            return rMin;
        };
        tracks.attr("sort")("key"_a=comparer);
    }else{
        std::function<double(std::reference_wrapper<Track3D>)> comparer=[&](std::reference_wrapper<Track3D> t){
            double rMax=-1;
            for(auto&& motion : motions){
                MotionState& motionR=motion.cast<std::reference_wrapper<MotionState>>().get();
                if(!motionR.getCRS()){
                    continue;
                }
                double tmp=(t.get().pos(crs)-motionR.pos(crs)).norm();
                if(rMax<0 || rMax>tmp){
                    rMax=tmp;
                }
            }
            return -rMax;
        };
        tracks.attr("sort")("key"_a=comparer);
    }
}

void exportSortTrack3DByDistance(py::module &m)
{
    using namespace pybind11::literals;

    m.def("sortTrack3DByDistance",py::overload_cast<py::list&,const py::list&,bool>(&sortTrack3DByDistance));
    m.def("sortTrack3DByDistance",py::overload_cast<std::vector<Track3D>&,const std::vector<MotionState>&,bool>(&sortTrack3DByDistance));
}

}

ASRC_PLUGIN_NAMESPACE_END
