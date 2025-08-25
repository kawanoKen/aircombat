/**
 * Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
 * @file
 * 
 * @brief 戦闘機と戦闘機の構成要素を表すクラスを定義する。
 */
#pragma once
#include <deque>
#include <vector>
#include <functional>
#include <pybind11/pybind11.h>
#include <Eigen/Core>
#include "TimeSystem.h"
#include "MathUtility.h"
#include "PhysicalAsset.h"
#include "Controller.h"
#include "Track.h"
#include "Propulsion.h"
#include "Sensor.h"
#include "Missile.h"
namespace py=pybind11;
namespace nl=nlohmann;

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)

/**
 * @class Fighter
 * @brief 戦闘機を表すクラス
 */
ASRC_DECLARE_DERIVED_CLASS_WITH_TRAMPOLINE(Fighter,PhysicalAsset)
    public:
    double m;//Mass in [kg]
    double rcsScale;//as a dimensionless value
    double fuelCapacity;//In [kg]
    std::weak_ptr<Propulsion> engine;
    std::weak_ptr<AircraftRadar> radar;
    std::weak_ptr<MWS> mws;
    std::vector<std::weak_ptr<Missile>> missiles;
    std::weak_ptr<Missile> dummyMissile;
    std::vector<std::pair<Track3D,bool>> missileTargets;
    int nextMsl,numMsls,remMsls;
    bool isDatalinkEnabled;
    std::vector<Track3D> track;
    std::vector<std::vector<std::string>> trackSource;
    std::string datalinkName;
    Track3D target;int targetID;
    bool enableThrustAfterFuelEmpty;
    double fuelRemaining;
    double optCruiseFuelFlowRatePerDistance;
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
    virtual double getThrust();// In [N]
    virtual double calcThrust(const nl::json& args);//as generic interface
    virtual double calcOptimumCruiseFuelFlowRatePerDistance()=0;//Minimum fuel flow rate (per distance) in level steady flight
    virtual double getMaxReachableRange();//In optimized level steady flight
    virtual std::pair<bool,Track3D> isTracking(const std::weak_ptr<PhysicalAsset>& target_);
    virtual std::pair<bool,Track3D> isTracking(const Track3D& target_);
    virtual std::pair<bool,Track3D> isTracking(const boost::uuids::uuid& target_);
    virtual bool isLaunchable();
    virtual bool isLaunchableAt(const Track3D& target_);
    virtual std::shared_ptr<Missile> launchInVirtual(const std::shared_ptr<SimulationManager>& dstManager,const nl::json& launchCommand);
    virtual void setFlightControllerMode(const std::string& ctrlName);
    virtual void calcMotion(double dt)=0;
    virtual Eigen::Vector3d toEulerAngle();
    virtual double getRmax(const Eigen::Vector3d &rs,const Eigen::Vector3d &vs,const Eigen::Vector3d &rt,const Eigen::Vector3d &vt);
    virtual double getRmax(const Eigen::Vector3d &rs,const Eigen::Vector3d &vs,const Eigen::Vector3d &rt,const Eigen::Vector3d &vt,const std::shared_ptr<CoordinateReferenceSystem>& crs);
    virtual double getRmax(const Eigen::Vector3d &rs,const Eigen::Vector3d &vs,const Eigen::Vector3d &rt,const Eigen::Vector3d &vt,const double& aa);
    virtual double getRmax(const Eigen::Vector3d &rs,const Eigen::Vector3d &vs,const Eigen::Vector3d &rt,const Eigen::Vector3d &vt,const double& aa,const std::shared_ptr<CoordinateReferenceSystem>& crs);
    virtual std::shared_ptr<EntityAccessor> getAccessorImpl() override;
};

ASRC_DECLARE_DERIVED_TRAMPOLINE(Fighter)
    virtual double getThrust() override{
        PYBIND11_OVERRIDE(double,Base,getThrust);
    }
    virtual double calcThrust(const nl::json& args) override{
        PYBIND11_OVERRIDE(double,Base,calcThrust,args);
    }
    virtual double calcOptimumCruiseFuelFlowRatePerDistance() override{
        PYBIND11_OVERRIDE_PURE(double,Base,calcOptimumCruiseFuelFlowRatePerDistance);
    }
    virtual std::pair<bool,Track3D> isTracking(const std::weak_ptr<PhysicalAsset>& target_) override{
        typedef std::pair<bool,Track3D> retType;
        PYBIND11_OVERRIDE(retType,Base,isTracking,target_);
    }
    virtual std::pair<bool,Track3D> isTracking(const Track3D&  target_) override{
        typedef std::pair<bool,Track3D> retType;
        PYBIND11_OVERRIDE(retType,Base,isTracking,target_);
    }
    virtual bool isLaunchable() override{
        PYBIND11_OVERRIDE(bool,Base,isLaunchable);
    }
    virtual bool isLaunchableAt(const Track3D& target_) override{
        PYBIND11_OVERRIDE(bool,Base,isLaunchableAt,target_);
    }
    virtual std::shared_ptr<Missile> launchInVirtual(const std::shared_ptr<SimulationManager>& dstManager,const nl::json& launchCommand) override{
        PYBIND11_OVERRIDE(std::shared_ptr<Missile>,Base,launchInVirtual,dstManager,launchCommand);
    }
    virtual void setFlightControllerMode(const std::string& ctrlName) override{
        PYBIND11_OVERRIDE(void,Base,setFlightControllerMode,ctrlName);
    }
    virtual void calcMotion(double dt) override{
        PYBIND11_OVERRIDE_PURE(void,Base,calcMotion,dt);
    }
    virtual Eigen::Vector3d toEulerAngle() override{
        PYBIND11_OVERRIDE(Eigen::Vector3d,Base,toEulerAngle);
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
};

/**
 * @class FighterAccessor
 * @brief Fighter とその派生クラスのメンバに対するアクセス制限を実現するためのAccessorクラス。
 */
ASRC_DECLARE_DERIVED_CLASS_WITH_TRAMPOLINE(FighterAccessor,PhysicalAssetAccessor)
    public:
    FighterAccessor(const std::shared_ptr<Fighter>& f);
    virtual void setFlightControllerMode(const std::string& ctrlName=""); //!< 飛行制御モードを変更する。
    virtual double getMaxReachableRange(); //!< 最適巡航で到達可能な最大距離[m]を返す。

    /**
     * @brief 自身の持つ誘導弾の推定射程[m]を返す。アスペクト角はvsとvtの向きから自動計算される。
     * 
     * @param [in] rs 射撃する側の位置。自身の parentCRS 上での値とする。
     * @param [in] vs 射撃する側の速度。自身の parentCRS 上での値とする。
     * @param [in] rt 射撃される側の位置。自身の parentCRS 上での値とする。
     * @param [in] vt 射撃される側の速度。自身の parentCRS 上での値とする。
     */
    virtual double getRmax(const Eigen::Vector3d &rs,const Eigen::Vector3d &vs,const Eigen::Vector3d &rt,const Eigen::Vector3d &vt);

    /**
     * @brief 自身の持つ誘導弾の推定射程[m]を返す。アスペクト角はvsとvtの向きから自動計算される。
     * 
     * @param [in] rs 射撃する側の位置。 crs 上での値とする。
     * @param [in] vs 射撃する側の速度。 crs 上での値とする。
     * @param [in] rt 射撃される側の位置。 crs 上での値とする。
     * @param [in] vt 射撃される側の速度。 crs 上での値とする。
     * @param [in] crs 引数で与えたベクトルの座標系
     */
    virtual double getRmax(const Eigen::Vector3d &rs,const Eigen::Vector3d &vs,const Eigen::Vector3d &rt,const Eigen::Vector3d &vt,const std::shared_ptr<CoordinateReferenceSystem>& crs);
    
    /**
     * @brief 自身の持つ誘導弾の推定射程[m]を返す。vtの向きをアスペクト角aaから自動計算して回転させる。
     * 
     * @param [in] rs 射撃する側の位置。自身の parentCRS 上での値とする。
     * @param [in] vs 射撃する側の速度。自身の parentCRS 上での値とする。
     * @param [in] rt 射撃される側の位置。自身の parentCRS 上での値とする。
     * @param [in] vt 射撃される側の速度。自身の parentCRS 上での値とする。
     * @param [in] aa アスペクト角。
     */

    virtual double getRmax(const Eigen::Vector3d &rs,const Eigen::Vector3d &vs,const Eigen::Vector3d &rt,const Eigen::Vector3d &vt,const double& aa);
    /**
     * @brief 自身の持つ誘導弾の推定射程[m]を返す。vtの向きをアスペクト角aaから自動計算して回転させる。
     * 
     * @param [in] rs 射撃する側の位置。 crs 上での値とする。
     * @param [in] vs 射撃する側の速度。 crs 上での値とする。
     * @param [in] rt 射撃される側の位置。 crs 上での値とする。
     * @param [in] vt 射撃される側の速度。 crs 上での値とする。
     * @param [in] aa アスペクト角。
     * @param [in] crs 引数で与えたベクトルの座標系
     */
    virtual double getRmax(const Eigen::Vector3d &rs,const Eigen::Vector3d &vs,const Eigen::Vector3d &rt,const Eigen::Vector3d &vt,const double& aa,const std::shared_ptr<CoordinateReferenceSystem>& crs);
    virtual bool isLaunchable(); //!< 自身が射撃可能な状況にあるかどうかを返す。
    virtual bool isLaunchableAt(const Track3D& target_); //!< 自身が target_ に対して射撃可能な状況にあるかどうかを返す。
    /**
     * @brief 仮想シミュレータ dstManager 上で 仮想誘導弾を発射する。 launchCommand には目標となる Track3D と 発射時点の運動状態を表す MotionState を与える。
     */
    virtual std::shared_ptr<Missile> launchInVirtual(const std::shared_ptr<SimulationManager>& dstManager,const nl::json& launchCommand);
    protected:
    std::weak_ptr<Fighter> fighter;
};
ASRC_DECLARE_DERIVED_TRAMPOLINE(FighterAccessor)
    virtual void setFlightControllerMode(const std::string& ctrlName="") override{
        PYBIND11_OVERRIDE(void,Base,setFlightControllerMode,ctrlName);
    }
    virtual double getMaxReachableRange() override{
        PYBIND11_OVERRIDE(double,Base,getMaxReachableRange);
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
    virtual bool isLaunchable() override{
        PYBIND11_OVERRIDE(bool,Base,isLaunchable);
    }
    virtual bool isLaunchableAt(const Track3D& target_) override{
        PYBIND11_OVERRIDE(bool,Base,isLaunchableAt,target_);
    }
    virtual std::shared_ptr<Missile> launchInVirtual(const std::shared_ptr<SimulationManager>& dstManager,const nl::json& launchCommand) override{
        PYBIND11_OVERRIDE(std::shared_ptr<Missile>,Base,launchInVirtual,dstManager,launchCommand);
    }
};

ASRC_DECLARE_DERIVED_CLASS_WITHOUT_TRAMPOLINE(SensorDataSharer,Controller)
    public:
    //constructors & destructor
    using BaseType::BaseType;
    //functions
    virtual void perceive(bool inReset) override;
};

ASRC_DECLARE_DERIVED_CLASS_WITHOUT_TRAMPOLINE(SensorDataSanitizer,Controller)
    public:
    std::map<std::string,Time> lastSharedTime;
    //constructors & destructor
    using BaseType::BaseType;
    //functions
    virtual void serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full) override;
    virtual void perceive(bool inReset) override;
};

ASRC_DECLARE_DERIVED_CLASS_WITHOUT_TRAMPOLINE(OtherDataSharer,Controller)
    public:
    //constructors & destructor
    using BaseType::BaseType;
    //functions
    virtual void perceive(bool inReset) override;
    virtual void control() override;
};

ASRC_DECLARE_DERIVED_CLASS_WITHOUT_TRAMPOLINE(OtherDataSanitizer,Controller)
    public:
    std::map<std::string,Time> lastSharedTimeOfAgentObservable;
    std::map<std::string,Time> lastSharedTimeOfFighterObservable;
    Time lastSharedTimeOfAgentCommand;
    //constructors & destructor
    using BaseType::BaseType;
    //functions
    virtual void serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full) override;
    virtual void perceive(bool inReset) override;
    virtual void control() override;
};

ASRC_DECLARE_DERIVED_CLASS_WITH_TRAMPOLINE(HumanIntervention,Controller)
    //delay for shot command in order to simulate approval by human operator.
    public:
    int capacity;
    double delay,cooldown;//in seconds.
    Time lastShotApprovalTime;//in seconds.
    std::deque<std::pair<Time,nl::json>> recognizedShotCommands;//in seconds.
    //constructors & destructor
    using BaseType::BaseType;
    //functions
    virtual void initialize() override;
    virtual void serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full) override;
    virtual bool isLaunchable();
    virtual bool isLaunchableAt(const Track3D& target_);
    virtual void control() override;
};
ASRC_DECLARE_DERIVED_TRAMPOLINE(HumanIntervention)
    virtual bool isLaunchable() override{
        PYBIND11_OVERRIDE(bool,Base,isLaunchable);
    }
    virtual bool isLaunchableAt(const Track3D& target_) override{
        PYBIND11_OVERRIDE(bool,Base,isLaunchableAt,target_);
    }
};

ASRC_DECLARE_DERIVED_CLASS_WITH_TRAMPOLINE(WeaponController,Controller)
    public:
    bool enable_loal;
    double launchRangeLimit;// in meters
    double offBoresightAngleLimit;// in degrees
    bool enable_utdc;
    bool enable_takeover_guidance;
    //constructors & destructor
    using BaseType::BaseType;
    //functions
    virtual void initialize() override;
    virtual void serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full) override;
    virtual bool isLaunchable();
    virtual bool isLaunchableAt(const Track3D& target_);
    virtual void control() override;
};
ASRC_DECLARE_DERIVED_TRAMPOLINE(WeaponController)
    virtual bool isLaunchable() override{
        PYBIND11_OVERRIDE(bool,Base,isLaunchable);
    }
    virtual bool isLaunchableAt(const Track3D& target_) override{
        PYBIND11_OVERRIDE(bool,Base,isLaunchableAt,target_);
    }
};

ASRC_DECLARE_DERIVED_CLASS_WITH_TRAMPOLINE(FlightController,Controller)
    public:
    //constructors & destructor
    using BaseType::BaseType;
    //functions
    virtual void initialize() override;
    virtual void serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full) override;
    virtual void control() override;
    virtual nl::json getDefaultCommand();
    virtual nl::json calc(const nl::json &cmd);
    virtual void setMode(const std::string& mode_);
    protected:
    std::string mode;
};
ASRC_DECLARE_DERIVED_TRAMPOLINE(FlightController)
    virtual nl::json getDefaultCommand() override{
        PYBIND11_OVERRIDE(nl::json,Base,getDefaultCommand);
    }
    virtual nl::json calc(const nl::json &cmd) override{
        PYBIND11_OVERRIDE(nl::json,Base,calc,cmd);
    }
    virtual void setMode(const std::string& mode_) override{
        PYBIND11_OVERRIDE(void,Base,setMode,mode_);
    }
};

void exportFighter(py::module &m, const std::shared_ptr<asrc::core::FactoryHelper>& factoryHelper);

ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
