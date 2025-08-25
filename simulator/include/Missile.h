// Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#pragma once
#include <vector>
#include <functional>
#include <future>
#include <mutex>
#include <pybind11/pybind11.h>
#include <Eigen/Core>
#include <Eigen/CXX11/Tensor>
#include "TimeSystem.h"
#include "MathUtility.h"
#include "Controller.h"
#include "PhysicalAsset.h"
#include "Track.h"
#include "Sensor.h"
namespace py=pybind11;
namespace nl=nlohmann;

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)

ASRC_DECLARE_DERIVED_CLASS_WITHOUT_TRAMPOLINE(PropNav,Controller)
    public:
    double gain;
    //constructors & destructor
    using BaseType::BaseType;
    //functions
    virtual void initialize() override;
    virtual void serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full) override;
    virtual void control() override;
    std::pair<Eigen::Vector3d,Eigen::Vector3d> calc(const Eigen::Vector3d &rs,const Eigen::Vector3d &vs,const Eigen::Vector3d &rt,const Eigen::Vector3d &vt);
};

ASRC_DECLARE_DERIVED_CLASS_WITH_TRAMPOLINE(Missile,PhysicalAsset)
    friend void makeRangeTableSub(std::mutex& m,std::promise<Eigen::VectorXd> p,const nl::json& modelConfig_,const nl::json& instanceConfig_,const Eigen::MatrixXd& args);
    public:
    enum class Mode{
        GUIDED,
        SELF,
        MEMORY
    };
    //configで指定するもの
    //基底クラスでは何も使わない

    //位置、姿勢等の運動状態に関する追加変数
    Eigen::Vector3d accel;
    double accelScalar;
    //その他の内部変数
    std::vector<Eigen::VectorXd> rangeTablePoints;
    Eigen::Tensor<double,6> rangeTable;
    Track3D target;
    Time targetUpdatedTime;
    bool hasLaunched;
    Mode mode;
    Time launchedT;
    Coordinate estTPos,estTVel;
    //子要素
    std::weak_ptr<MissileSensor> sensor;
    public:
    //constructors & destructor
    using BaseType::BaseType;
    //functions
    virtual void initialize() override;
    virtual void serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full) override;
    virtual void makeChildren() override;
    virtual void validate() override;
    virtual void setDependency() override;
    virtual void perceive(bool inReset) override;
    virtual void control() override;
    virtual void behave() override;
    virtual void kill() override;
    virtual void calcMotion(double tAftLaunch,double dt)=0;
    virtual bool hitCheck(const Eigen::Vector3d &tpos,const Eigen::Vector3d &tpos_prev)=0;
    virtual bool checkDeactivateCondition(double tAftLaunch)=0;
    virtual void calcQ();
    virtual double getRmax(const Eigen::Vector3d &rs,const Eigen::Vector3d &vs,const Eigen::Vector3d &rt,const Eigen::Vector3d &vt);
    virtual double getRmax(const Eigen::Vector3d &rs,const Eigen::Vector3d &vs,const Eigen::Vector3d &rt,const Eigen::Vector3d &vt,const std::shared_ptr<CoordinateReferenceSystem>& crs);
    virtual double getRmax(const Eigen::Vector3d &rs,const Eigen::Vector3d &vs,const Eigen::Vector3d &rt,const Eigen::Vector3d &vt,const double& aa);
    virtual double getRmax(const Eigen::Vector3d &rs,const Eigen::Vector3d &vs,const Eigen::Vector3d &rt,const Eigen::Vector3d &vt,const double& aa,const std::shared_ptr<CoordinateReferenceSystem>& crs);
    virtual double calcRange(double vs,double hs,double vt,double ht,double obs,double aa);
    virtual bool calcRangeSub(double vs,double hs,double vt,double ht,double obs,double aa,double r);
    virtual void makeRangeTable(const std::string& dstPath);
};
DEFINE_SERIALIZE_ENUM_AS_STRING(Missile::Mode)

ASRC_DECLARE_DERIVED_TRAMPOLINE(Missile)
    virtual void calcMotion(double tAftLaunch,double dt) override{
        PYBIND11_OVERRIDE_PURE(void,Base,calcMotion,tAftLaunch,dt);
    }
    virtual bool hitCheck(const Eigen::Vector3d &tpos,const Eigen::Vector3d &tpos_prev) override{
        PYBIND11_OVERRIDE_PURE(bool,Base,hitCheck,tpos,tpos_prev);
    }
    virtual bool checkDeactivateCondition(double tAftLaunch) override{
        PYBIND11_OVERRIDE_PURE(bool,Base,checkDeactivateCondition,tAftLaunch);
    }
    virtual void calcQ() override{
        PYBIND11_OVERRIDE(void,Base,calcQ);
    }
    virtual double getRmax(const Eigen::Vector3d &rs,const Eigen::Vector3d &vs,const Eigen::Vector3d &rt,const Eigen::Vector3d &vt) override{
        PYBIND11_OVERRIDE(double,Base,getRmax,rs,vs,rt,vt);
    }
    virtual double getRmax(const Eigen::Vector3d &rs,const Eigen::Vector3d &vs,const Eigen::Vector3d &rt,const Eigen::Vector3d &vt,const std::shared_ptr<CoordinateReferenceSystem>& crs) override{
        PYBIND11_OVERRIDE(double,Base,getRmax,rs,vs,rt,vt,crs);
    }
    virtual double getRmax(const Eigen::Vector3d &rs,const Eigen::Vector3d &vs,const Eigen::Vector3d &rt,const Eigen::Vector3d &vt,const double& aa) override{
        PYBIND11_OVERRIDE(double,Base,getRmax,rs,vs,rt,vt,aa);
    }
    virtual double getRmax(const Eigen::Vector3d &rs,const Eigen::Vector3d &vs,const Eigen::Vector3d &rt,const Eigen::Vector3d &vt,const double& aa,const std::shared_ptr<CoordinateReferenceSystem>& crs) override{
        PYBIND11_OVERRIDE(double,Base,getRmax,rs,vs,rt,vt,aa,crs);
    }
    virtual double calcRange(double vs,double hs,double vt,double ht,double obs,double aa) override{
        PYBIND11_OVERRIDE(double,Base,calcRange,vs,hs,vt,ht,obs,aa);
    }
    virtual bool calcRangeSub(double vs,double hs,double vt,double ht,double obs,double aa,double r) override{
        PYBIND11_OVERRIDE(bool,Base,calcRangeSub,vs,hs,vt,ht,obs,aa,r);
    }
    virtual void makeRangeTable(const std::string& dstPath) override{
        PYBIND11_OVERRIDE(void,Base,makeRangeTable,dstPath);
    }
};
void PYBIND11_EXPORT makeRangeTableSub(std::mutex& m,std::promise<Eigen::VectorXd> p,const std::shared_ptr<Missile>& msl,const Eigen::MatrixXd& args);

void exportMissile(py::module &m, const std::shared_ptr<asrc::core::FactoryHelper>& factoryHelper);

ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)

ASRC_PYBIND11_MAKE_OPAQUE(std::vector<std::weak_ptr<::asrc::core::Missile>>);
