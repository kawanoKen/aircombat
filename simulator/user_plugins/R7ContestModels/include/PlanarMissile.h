// Copyright (c) 2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#pragma once
#include <ASRCAISim1/EkkerMissile.h>
#include "Common.h"

ASRC_PLUGIN_NAMESPACE_BEGIN

ASRC_DECLARE_DERIVED_CLASS_WITHOUT_TRAMPOLINE(PlanarMissile,asrc::core::EkkerMissile)
    //運動を2次元平面上に拘束した航空機モデル。
    public:
    //constructors & destructor
    using BaseType::BaseType;
    //functions
    virtual void calcMotion(double tAftLaunch,double dt) override;

    //射程テーブルの計算(自身と目標が同一平面上に存在する場合しか計算できないため、htを無視して彼我ともにhsを用いる)
    virtual bool calcRangeSub(double vs,double hs,double vt,double ht,double obs,double aa,double r) override;
    virtual void makeRangeTable(const std::string& dstPath) override;
};

void exportPlanarMissile(py::module &m, const std::shared_ptr<asrc::core::FactoryHelper>& factoryHelper);

ASRC_PLUGIN_NAMESPACE_END
