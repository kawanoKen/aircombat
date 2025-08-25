// Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#pragma once
#include <memory>
#include <vector>
#include <pybind11/pybind11.h>
#include <Eigen/Core>
#include "MathUtility.h"
#include "PhysicalAsset.h"
#include "Track.h"
namespace py=pybind11;
namespace nl=nlohmann;

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)

ASRC_DECLARE_DERIVED_CLASS_WITHOUT_TRAMPOLINE(Sensor,PhysicalAsset)
    public:
    //constructors & destructor
    using BaseType::BaseType;
    //functions
    virtual void initialize() override;
};
ASRC_DECLARE_DERIVED_CLASS_WITH_TRAMPOLINE(Sensor3D,Sensor)
    public:
    //constructors & destructor
    using BaseType::BaseType;
    //functions
    virtual std::pair<bool,std::vector<Track3D>> isTrackingAny();
    virtual std::pair<bool,Track3D> isTracking(const std::weak_ptr<PhysicalAsset>& target_);
    virtual std::pair<bool,Track3D> isTracking(const Track3D& target_);
    virtual std::pair<bool,Track3D> isTracking(const boost::uuids::uuid& target_);
};
ASRC_DECLARE_DERIVED_TRAMPOLINE(Sensor3D)
    virtual std::pair<bool,std::vector<Track3D>> isTrackingAny() override{
        typedef  std::pair<bool,std::vector<Track3D>> retType;
        PYBIND11_OVERRIDE(retType,Base,isTrackingAny);
    }
    virtual std::pair<bool,Track3D> isTracking(const std::weak_ptr<PhysicalAsset>& target_) override{
        typedef  std::pair<bool,Track3D> retType;
        PYBIND11_OVERRIDE(retType,Base,isTracking,target_);
    }
    virtual std::pair<bool,Track3D> isTracking(const Track3D& target_) override{
        typedef  std::pair<bool,Track3D> retType;
        PYBIND11_OVERRIDE(retType,Base,isTracking,target_);
    }
    virtual std::pair<bool,Track3D> isTracking(const boost::uuids::uuid& target_) override{
        typedef  std::pair<bool,Track3D> retType;
        PYBIND11_OVERRIDE(retType,Base,isTracking,target_);
    }
};
ASRC_DECLARE_DERIVED_CLASS_WITH_TRAMPOLINE(Sensor2D,Sensor)
    public:
    //constructors & destructor
    using BaseType::BaseType;
    //functions
    virtual std::pair<bool,std::vector<Track2D>> isTrackingAny();
    virtual std::pair<bool,Track2D> isTracking(const std::weak_ptr<PhysicalAsset>& target_);
    virtual std::pair<bool,Track2D> isTracking(const Track2D& target_);
    virtual std::pair<bool,Track2D> isTracking(const boost::uuids::uuid& target_);
};
ASRC_DECLARE_DERIVED_TRAMPOLINE(Sensor2D)
    virtual std::pair<bool,std::vector<Track2D>> isTrackingAny() override{
        typedef  std::pair<bool,std::vector<Track2D>> retType;
        PYBIND11_OVERRIDE(retType,Base,isTrackingAny);
    }
    virtual std::pair<bool,Track2D> isTracking(const std::weak_ptr<PhysicalAsset>& target_) override{
        typedef  std::pair<bool,Track2D> retType;
        PYBIND11_OVERRIDE(retType,Base,isTracking,target_);
    }
    virtual std::pair<bool,Track2D> isTracking(const Track2D& target_) override{
        typedef  std::pair<bool,Track2D> retType;
        PYBIND11_OVERRIDE(retType,Base,isTracking,target_);
    }
    virtual std::pair<bool,Track2D> isTracking(const boost::uuids::uuid& target_) override{
        typedef  std::pair<bool,Track2D> retType;
        PYBIND11_OVERRIDE(retType,Base,isTracking,target_);
    }
};


ASRC_DECLARE_DERIVED_CLASS_WITHOUT_TRAMPOLINE(AircraftRadar,Sensor3D)
    public:
    //parameters
    double Lref,thetaFOR;
    //internal variables
    std::vector<Track3D> track;
    public:
    //constructors & destructor
    using BaseType::BaseType;
    //functions
    virtual void initialize() override;
    virtual void serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full) override;
    virtual void setDependency() override;
    virtual void perceive(bool inReset) override;
    virtual std::pair<bool,std::vector<Track3D>> isTrackingAny() override;
    virtual std::pair<bool,Track3D> isTracking(const std::weak_ptr<PhysicalAsset>& target_) override;
    virtual std::pair<bool,Track3D> isTracking(const Track3D& target_) override;
    virtual std::pair<bool,Track3D> isTracking(const boost::uuids::uuid& target_) override;
};

ASRC_DECLARE_DERIVED_CLASS_WITHOUT_TRAMPOLINE(MWS,Sensor2D)
    public:
    //parameters
    bool isEsmIsh;
    double Lref,thetaFOR;
    //internal variables
    std::vector<Track2D> track;
    public:
    //constructors & destructor
    using BaseType::BaseType;
    //functions
    virtual void initialize() override;
    virtual void serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full) override;
    virtual void setDependency() override;
    virtual void perceive(bool inReset) override;
    virtual std::pair<bool,std::vector<Track2D>> isTrackingAny() override;
    virtual std::pair<bool,Track2D> isTracking(const std::weak_ptr<PhysicalAsset>& target_) override;
    virtual std::pair<bool,Track2D> isTracking(const Track2D& target_) override;
    virtual std::pair<bool,Track2D> isTracking(const boost::uuids::uuid& target_) override;
};

ASRC_DECLARE_DERIVED_CLASS_WITHOUT_TRAMPOLINE(MissileSensor,Sensor3D)
    public:
    //parameters
    double Lref,thetaFOR,thetaFOV;
    //internal variables
    std::vector<Track3D> track;
    Track3D target;
    bool isActive;
    Coordinate estTPos;
    public:
    //constructors & destructor
    using BaseType::BaseType;
    //functions
    virtual void initialize() override;
    virtual void serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full) override;
    virtual void setDependency() override;
    virtual void perceive(bool inReset) override;
    virtual void control() override;
    virtual void kill() override;
    virtual std::pair<bool,std::vector<Track3D>> isTrackingAny() override;
    virtual std::pair<bool,Track3D> isTracking(const std::weak_ptr<PhysicalAsset>& target_) override;
    virtual std::pair<bool,Track3D> isTracking(const Track3D& target_) override;
    virtual std::pair<bool,Track3D> isTracking(const boost::uuids::uuid& target_) override;
};

void exportSensor(py::module &m, const std::shared_ptr<asrc::core::FactoryHelper>& factoryHelper);

ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
