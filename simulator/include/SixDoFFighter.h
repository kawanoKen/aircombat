// Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#pragma once
#include <functional>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/CXX11/Tensor>
#include "MathUtility.h"
#include "FlightControllerUtility.h"
#include "CoordinatedFighter.h"
#include "Utility.h"
namespace py=pybind11;
namespace nl=nlohmann;

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)

ASRC_DECLARE_DERIVED_CLASS_WITH_TRAMPOLINE(SixDoFFighter,Fighter)
	/*
	6 degree-of-freedom aircraft model.
	Aerodynamic coefficients and engine characteristics are based on public released F-16 information listed below.
    Most of the model is based on [1], wave drag is based on [3], and supersonic CD0 is based on [4].
    Flight controller model uses a simple combination of dynamic inversion and LQR.
    Control variables for attitude control are based on [5].
	References:
	[1] Stevens, Brian L., et al. "Aircraft control and simulation: dynamics, controls design, and autonomous systems." John Wiley & Sons, 2015.
	[3] Krus, P., et al. "Modelling of Transonic and Supersonic Aerodynamics for Conceptual Design and Flight Simulation." Proceedings of the 10th Aerospace Technology Congress, 2019.
    [4] Webb, T. S., et al. "Correlation of F-16 Aerodynamics and Performance Predictions with Early Flight Test Results." Agard Conference Proceedings, N242, 1977.
    [5] Heidlauf, P., et al. "Verification Challenges in F-16 Ground Collision Avoidance and Other Automated Maneuvers." ARCH18. 5th International Workshop on Applied Verification of Countinuous and Hybrid Systems, 2018.
	*/
    //protected:
    public:
    //model parameters
    double S,Le,b,mac,XcgR,Xcg;
    Eigen::Matrix3d I,Iinv;//慣性テンソル
    double deLimit,daLimit,drLimit,deMaxRate,daMaxRate,drMaxRate,deTimeConstant,daTimeConstant,drTimeConstant;
    Eigen::VectorXd cdwTable;
    double de,da,dr;
    public:
    //constructors & destructor
    using BaseType::BaseType;
    //functions
    virtual void initialize() override;
    virtual void makeChildren() override;
    virtual void serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full) override;
    virtual void validate() override;
    virtual double calcOptimumCruiseFuelFlowRatePerDistance() override;//Minimum fuel flow rate (per distance) in level steady flight
    virtual Eigen::VectorXd trim6(double alt,double V,double tgtBank);
    virtual std::map<std::string,double> calcAeroV(double alpha_,double beta_,double de_,double da_,double dr_,double mach_,bool withDerivatives);
    virtual Eigen::VectorXd calcDerivative(const Eigen::VectorXd& x,const Time& t,const Eigen::VectorXd& u);
    virtual void calcMotion(double dt) override;
};

ASRC_DECLARE_DERIVED_TRAMPOLINE(SixDoFFighter)
    virtual double calcOptimumCruiseFuelFlowRatePerDistance() override{
        PYBIND11_OVERRIDE(double,Base,calcOptimumCruiseFuelFlowRatePerDistance);
    }
    virtual void calcMotion(double dt) override{
        PYBIND11_OVERRIDE(void,Base,calcMotion,dt);
    }
    virtual std::map<std::string,double> calcAeroV(double alpha_,double beta_,double de_,double da_,double dr_,double mach_,bool withDerivatives) override{
        typedef std::map<std::string,double> retType;
        PYBIND11_OVERRIDE(retType,Base,calcAeroV,alpha_,beta_,de_,da_,dr_,mach_,withDerivatives);
    }
};

ASRC_DECLARE_DERIVED_CLASS_WITHOUT_TRAMPOLINE(SixDoFFlightController,FlightController)
    public:
    double PositiveNzLimit,NegativeNzLimit;
    double maxNyrCmd,maxPsCmd;
    double kPsCmdP,kPsCmdD,kNzCmdP,kNzCmdD,kNyrCmdP,kNyrCmdD;
    AltitudeKeeper altitudeKeeper;
    //constructors & destructor
    using BaseType::BaseType;
    //functions
    virtual void initialize() override;
    virtual void serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full) override;
    virtual nl::json getDefaultCommand() override;
    virtual nl::json calc(const nl::json &cmd) override;
    nl::json calcDirect(const nl::json &cmd);
    nl::json calcFromManualInput(const nl::json &cmd);
    nl::json calcFromDirAndVel(const nl::json &cmd);
    Eigen::VectorXd calcU(const nl::json &cmd,const Eigen::VectorXd& x);
};

ASRC_DECLARE_DERIVED_CLASS_WITHOUT_TRAMPOLINE(StevensFighter,SixDoFFighter)
	/*
	6 degree-of-freedom aircraft model with aerodynamic coefficients written in [1].
	References:
	[1] Stevens, Brian L., et al. "Aircraft control and simulation: dynamics, controls design, and autonomous systems." John Wiley & Sons, 2015.
	*/
    Eigen::Tensor<double,2> dampTable,cxTable,clTable,cmTable,cnTable,dldaTable,dldrTable,dndaTable,dndrTable;
    Eigen::Tensor<double,1> czTable;
    Eigen::VectorXd alphaTable,betaSymTable,betaAsymTable,deTable;
    double dCydB,dCydda,dCyddr,dCzdde;
    //constructors & destructor
    using BaseType::BaseType;
    //functions
    virtual void initialize() override;
    virtual void serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full) override;
    virtual std::map<std::string,double> calcAeroV(double alpha_,double beta_,double de_,double da_,double dr_,double mach_,bool withDerivatives) override;
};

ASRC_DECLARE_DERIVED_CLASS_WITHOUT_TRAMPOLINE(MorelliFighter,SixDoFFighter)
	/*
	6 degree-of-freedom aircraft model variant with aerodynamic coefficients written in [5].
	References:
	[5] Morelli, E. A. "Global Nonlinear Parametric Modeling with Application to F-16 Aerodynamics." Proceedings of the 1998 American Control Conference, 1998.
	*/
    std::map<std::string,Eigen::VectorXd> aeroC;
    //constructors & destructor
    using BaseType::BaseType;
    //functions
    virtual void initialize() override;
    virtual void serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full) override;
    virtual std::map<std::string,double> calcAeroV(double alpha_,double beta_,double de_,double da_,double dr_,double mach_,bool withDerivatives) override;
};

void exportSixDoFFighter(py::module &m, const std::shared_ptr<asrc::core::FactoryHelper>& factoryHelper);

ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
