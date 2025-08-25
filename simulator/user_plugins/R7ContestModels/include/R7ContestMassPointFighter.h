/**
 * Copyright (c) 2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
 * @file
 * 
 * @brief R7年度コンテスト用の簡略化した機体運動モデルを用いる戦闘機クラス
 */
#pragma once
#include <deque>
#include <vector>
#include <functional>
#include <pybind11/pybind11.h>
#include <Eigen/Core>
#include <ASRCAISim1/MathUtility.h>
#include <ASRCAISim1/MassPointFighter.h>
#include <ASRCAISim1/Controller.h>
#include "Common.h"
namespace py=pybind11;
namespace nl=nlohmann;

ASRC_PLUGIN_NAMESPACE_BEGIN

/**
 * @class R7ContestMassPointFighter
 * @brief R7年度コンテスト用の簡略化した機体運動モデルを用いる戦闘機クラス
 * 
 * @details 迎角と横滑り角を無視し、機体のX軸と速度方向が常に一致するものとして速度ベクトルを直接操作する。
        速度、加速度、角速度(速度ベクトルの変化率として)を高度依存のテーブルデータとして与える。
 */
ASRC_DECLARE_DERIVED_CLASS_WITH_TRAMPOLINE(R7ContestMassPointFighter,asrc::core::MassPointFighter)
    public:
    using MotionState=asrc::core::MotionState;
    //constructors & destructor
    using BaseType::BaseType;
    //functions
    virtual void initialize() override;
    virtual void serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full) override;
    virtual double getThrust() override;// In [N]
    virtual void calcMotion(double dt) override;
    virtual nl::json getExtraDynamicsState(); //!< 派生クラスで定義された、追加の状態量を返す。このクラスでは使用しない。
    virtual std::pair<double,double> getVelocityRange(const MotionState& m,const nl::json& extraArgs=nl::json::object()); //!< 指定した運動状態 m で実現可能な速度範囲を返す。 extraArgs は派生クラス用の追加状態量であり、このクラスでは使用しない。
    double getTwoDimTableValue(const std::string& key,const MotionState& m,const nl::json& extraArgs=nl::json::object());
    virtual std::pair<double,double> getAccelRange(const MotionState& m,const nl::json& extraArgs=nl::json::object()); //!< 指定した運動状態 m で実現可能な加速度範囲を返す。 extraArgs は派生クラス用の追加状態量であり、このクラスでは使用しない。
    virtual std::pair<double,double> getRollRateRange(const MotionState& m,const nl::json& extraArgs=nl::json::object()); //!< 指定した運動状態 m で実現可能なロール角速度範囲を返す。 extraArgs は派生クラス用の追加状態量であり、このクラスでは使用しない。
    virtual std::pair<double,double> getPitchRateRange(const MotionState& m,const nl::json& extraArgs=nl::json::object()); //!< 指定した運動状態 m で実現可能なピッチ角速度範囲を返す。 extraArgs は派生クラス用の追加状態量であり、このクラスでは使用しない。
    virtual std::pair<double,double> getYawRateRange(const MotionState& m,const nl::json& extraArgs=nl::json::object()); //!< 指定した運動状態 m で実現可能なヨー角速度範囲を返す。 extraArgs は派生クラス用の追加状態量であり、このクラスでは使用しない。
    virtual void loadTableFromJsonObject(const nl::json& j_table);
    virtual std::shared_ptr<asrc::core::EntityAccessor> getAccessorImpl() override;
    // テーブルデータ
    Eigen::VectorXd altTable; //!< 高度のデータ点
    Eigen::Tensor<double,1> vMinTable; //!< 最低速度テーブル。高度に依存
    Eigen::Tensor<double,1> vMaxTable; //!< 最高速度テーブル。高度に依存
    std::map<std::string,Eigen::Tensor<double,2>> twoDimTables; //!< 高度と速度に依存する2次元テーブル。速度はその高度の[vMin,vMax]を均等に分割した点のデータとする。

    /**
     * @brief テーブルデータにかける係数
     */
    std::map<std::string,double> scaleFactor;
};

ASRC_DECLARE_DERIVED_TRAMPOLINE(R7ContestMassPointFighter)
    using MotionState=asrc::core::MotionState;
    virtual nl::json getExtraDynamicsState() override{
        PYBIND11_OVERRIDE(nl::json,Base,getExtraDynamicsState);
    }
    virtual std::pair<double,double> getVelocityRange(const MotionState& m,const nl::json& extraArgs=nl::json::object()) override{
        using _double_pair=std::pair<double,double>;
        PYBIND11_OVERRIDE(_double_pair,Base,getVelocityRange,m,extraArgs);
    }
    virtual std::pair<double,double> getAccelRange(const MotionState& m,const nl::json& extraArgs=nl::json::object()) override{
        using _double_pair=std::pair<double,double>;
        PYBIND11_OVERRIDE(_double_pair,Base,getAccelRange,m,extraArgs);
    }
    virtual std::pair<double,double> getRollRateRange(const MotionState& m,const nl::json& extraArgs=nl::json::object()) override{
        using _double_pair=std::pair<double,double>;
        PYBIND11_OVERRIDE(_double_pair,Base,getRollRateRange,m,extraArgs);
    }
    virtual std::pair<double,double> getPitchRateRange(const MotionState& m,const nl::json& extraArgs=nl::json::object()) override{
        using _double_pair=std::pair<double,double>;
        PYBIND11_OVERRIDE(_double_pair,Base,getPitchRateRange,m,extraArgs);
    }
    virtual std::pair<double,double> getYawRateRange(const MotionState& m,const nl::json& extraArgs=nl::json::object()) override{
        using _double_pair=std::pair<double,double>;
        PYBIND11_OVERRIDE(_double_pair,Base,getYawRateRange,m,extraArgs);
    }
    virtual void loadTableFromJsonObject(const nl::json& j_table){
         PYBIND11_OVERRIDE(void,Base,loadTableFromJsonObject,j_table);
    }
};

/**
 * @class R7ContestMassPointFighterAccessor
 * @brief R7ContestMassPointFighter とその派生クラスのメンバに対するアクセス制限を実現するためのAccessorクラス。
 */
ASRC_DECLARE_DERIVED_CLASS_WITH_TRAMPOLINE(R7ContestMassPointFighterAccessor,asrc::core::FighterAccessor)
    using MotionState=asrc::core::MotionState;
    public:
    R7ContestMassPointFighterAccessor(const std::shared_ptr<R7ContestMassPointFighter>& f);
    virtual nl::json getExtraDynamicsState(); //!< 派生クラスで定義された、追加の状態量を返す。このクラスでは使用しない。
    virtual std::pair<double,double> getVelocityRange(const MotionState& m,const nl::json& extraArgs=nl::json::object()); //!< 指定した運動状態 m で実現可能な速度範囲を返す。 extraArgs は派生クラス用の追加状態量であり、このクラスでは使用しない。
    virtual std::pair<double,double> getAccelRange(const MotionState& m,const nl::json& extraArgs=nl::json::object()); //!< 指定した運動状態 m で実現可能な加速度範囲を返す。 extraArgs は派生クラス用の追加状態量であり、このクラスでは使用しない。
    virtual std::pair<double,double> getRollRateRange(const MotionState& m,const nl::json& extraArgs=nl::json::object()); //!< 指定した運動状態 m で実現可能なロール角速度範囲を返す。 extraArgs は派生クラス用の追加状態量であり、このクラスでは使用しない。
    virtual std::pair<double,double> getPitchRateRange(const MotionState& m,const nl::json& extraArgs=nl::json::object()); //!< 指定した運動状態 m で実現可能なピッチ角速度範囲を返す。 extraArgs は派生クラス用の追加状態量であり、このクラスでは使用しない。
    virtual std::pair<double,double> getYawRateRange(const MotionState& m,const nl::json& extraArgs=nl::json::object()); //!< 指定した運動状態 m で実現可能なヨー角速度範囲を返す。 extraArgs は派生クラス用の追加状態量であり、このクラスでは使用しない。
    protected:
    std::weak_ptr<R7ContestMassPointFighter> fighter;
};

ASRC_DECLARE_DERIVED_TRAMPOLINE(R7ContestMassPointFighterAccessor)
    using MotionState=asrc::core::MotionState;
    virtual nl::json getExtraDynamicsState() override{
        PYBIND11_OVERRIDE(nl::json,Base,getExtraDynamicsState);
    }
    virtual std::pair<double,double> getVelocityRange(const MotionState& m,const nl::json& extraArgs=nl::json::object()) override{
        using _double_pair=std::pair<double,double>;
        PYBIND11_OVERRIDE(_double_pair,Base,getVelocityRange,m,extraArgs);
    }
    virtual std::pair<double,double> getAccelRange(const MotionState& m,const nl::json& extraArgs=nl::json::object()) override{
        using _double_pair=std::pair<double,double>;
        PYBIND11_OVERRIDE(_double_pair,Base,getAccelRange,m,extraArgs);
    }
    virtual std::pair<double,double> getRollRateRange(const MotionState& m,const nl::json& extraArgs=nl::json::object()) override{
        using _double_pair=std::pair<double,double>;
        PYBIND11_OVERRIDE(_double_pair,Base,getRollRateRange,m,extraArgs);
    }
    virtual std::pair<double,double> getPitchRateRange(const MotionState& m,const nl::json& extraArgs=nl::json::object()) override{
        using _double_pair=std::pair<double,double>;
        PYBIND11_OVERRIDE(_double_pair,Base,getPitchRateRange,m,extraArgs);
    }
    virtual std::pair<double,double> getYawRateRange(const MotionState& m,const nl::json& extraArgs=nl::json::object()) override{
        using _double_pair=std::pair<double,double>;
        PYBIND11_OVERRIDE(_double_pair,Base,getYawRateRange,m,extraArgs);
    }
};

ASRC_DECLARE_DERIVED_CLASS_WITHOUT_TRAMPOLINE(R7ContestMassPointFlightController,asrc::core::MassPointFlightController)
    public:
    //constructors & destructor
    using BaseType::BaseType;
    //functions
    virtual nl::json calcDirect(const nl::json &cmd) override;
    virtual nl::json calcFromDirAndVel(const nl::json &cmd) override;
};

void exportR7ContestMassPointFighter(py::module &m, const std::shared_ptr<asrc::core::FactoryHelper>& factoryHelper);

ASRC_PLUGIN_NAMESPACE_END
