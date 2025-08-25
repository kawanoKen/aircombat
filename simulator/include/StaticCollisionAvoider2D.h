// Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
// 2次元の固定された境界に対し簡易的な衝突回避を行うためのクラス。
// 衝突回避は進行方向の制限によって行い、現在の状態と進みたい方向を与え、許容される最も近い進行方向を返すことによって行う。
// 各境界は単一の禁止方位区間(扇形のイメージ)を返すものとし、それら全てに抵触しない方位を本クラスで計算する。
// なお、現バージョンでは複数の境界の干渉で許容方位が存在しなくなってしまうような状況には対応していない。
#pragma once
#include "Common.h"
#include <cmath>
#include <random>
#include <iostream>
#include <pybind11/pybind11.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/CXX11/Tensor>
#include "MotionState.h"
#include "Factory.h"

namespace py=pybind11;

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)
ASRC_INLINE_NAMESPACE_BEGIN(util)

ASRC_DECLARE_BASE_REF_CLASS(BorderGeometry)
    /*2次元の固定された境界を表す基底クラス
    */
    public:
    double limit;//接近限界距離
    double threshold;//補正開始距離
    double adjustStrength;//補正の強さ
    BorderGeometry(const nl::json& config);
    virtual ~BorderGeometry();
    virtual std::pair<double,double> calcUnacceptableRegion(const MotionState& motion, const Eigen::VectorXd& dstDir)=0;
};
ASRC_DECLARE_BASE_REF_TRAMPOLINE(BorderGeometry)
    virtual std::pair<double,double> calcUnacceptableRegion(const MotionState& motion, const Eigen::VectorXd& dstDir) override{
        typedef std::pair<double,double> retType;
        PYBIND11_OVERRIDE_PURE(retType,Base,calcUnacceptableRegion,motion,dstDir);
    }
};

ASRC_DECLARE_DERIVED_CLASS_WITH_TRAMPOLINE(LinearSegment,BorderGeometry)
    /*2点p1、p2間を結んだ線分状の境界を表現するクラス。
    線分から距離threshold以内の場合に進行方向の制限を行う。
    また、isOneSideをTrueとした場合、線分の片側のみを進入可能領域として扱い、反対側に出ているときに強い補正をかけて「あるべき領域に戻るように」試みる。
    また、infinite_pxフラグをTrueとすることで、有限長の線分でなく半直線又は直線として扱うことも可能である。
    config={
        "p1": [1000.0, -1000.0], #始点
        "p2": [1000.0, 1000.0], #終点
        "infinite_p1": False, #p1側を無限遠方まで伸ばすかどうか
        "infinite_p2": False, #p2側を無限遠方まで伸ばすかどうか
        "isOneSide": True, #片側のみ進入可能とするかどうか
        "inner": [0.0, 0.0], #片側のみ進入可能とする場合の「領域内」となる1点
        "limit": 100.0, #接近限界距離。この距離のときに進行可能方向が線分と平行になる。
        "threshold": 200.0, #補正開始距離
        "adjustStrength": 200.0, #補正の強さ
    }
    */
    public:
    Eigen::Vector2d p1,p2;//始点と終点
    bool infinite_p1, infinite_p2; //無限遠方まで伸ばすかどうか
    bool isOneSide;//片側のみ進入可能とするかどうか
    Eigen::Vector2d normal;//片側のみ進入可能とする場合の進入可能領域側を向いた法線
    LinearSegment(const nl::json& config);
    virtual ~LinearSegment();
    virtual std::pair<double,double> calcUnacceptableRegion(const MotionState& motion, const Eigen::VectorXd& dstDir);
};
ASRC_DECLARE_DERIVED_TRAMPOLINE(LinearSegment)
    virtual std::pair<double,double> calcUnacceptableRegion(const MotionState& motion, const Eigen::VectorXd& dstDir) override{
        typedef std::pair<double,double> retType;
        PYBIND11_OVERRIDE(retType,Base,calcUnacceptableRegion,motion,dstDir);
    }
};

class  PYBIND11_EXPORT StaticCollisionAvoider2D{
    /*2次元の固定された境界に対し簡易的な衝突回避を行うためのクラス。
    衝突回避は進行方向の制限によって行い、現在の状態と進みたい方向を与え、許容される最も近い進行方向を返すことによって行う。
    各境界は単一の禁止方位区間(扇形のイメージ)を返すものとし、それら全てに抵触しない方位を本クラスで計算する。
    なお、現バージョンでは複数の境界の干渉で許容方位が存在しなくなってしまうような状況には対応していない。
    */
    public:
    std::vector<std::shared_ptr<BorderGeometry>> borders;
    StaticCollisionAvoider2D();
    ~StaticCollisionAvoider2D();
    Eigen::VectorXd operator()(const MotionState& motion, const Eigen::VectorXd& dstDir);
};

void exportStaticCollisionAvoider2D(py::module &m);

ASRC_NAMESPACE_END(util)
ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)

ASRC_PYBIND11_MAKE_OPAQUE(std::vector<std::shared_ptr<::asrc::core::util::BorderGeometry>>);
