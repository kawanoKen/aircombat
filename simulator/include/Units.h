/**
 * Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
 * @file
 * 
 * @brief 単位系の変換や物理定数等を集めたファイル
 */
#pragma once
#include "Common.h"
#include <cmath>
#include <iostream>
#include <pybind11/pybind11.h>
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace py=pybind11;

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)
ASRC_INLINE_NAMESPACE_BEGIN(util)

    const double Req=6178137.0; //!< 地球の赤道半径
    const double gravity=9.80665; //!< 地球の標準重力加速度
    double PYBIND11_EXPORT ft2m(const double &bef); //!< ftからmへの変換
    double PYBIND11_EXPORT m2ft(const double &bef); //!< mからftへの変換
    double PYBIND11_EXPORT deg2rad(const double &bef); //!< degreeからradianへの変換
    double PYBIND11_EXPORT rad2deg(const double &bef); //!< radianからdegreeへの変換
    double PYBIND11_EXPORT N2kgf(const double &bef); //!< Nからkgfへの変換
    double PYBIND11_EXPORT kgf2N(const double &bef); //!< kgfからNへの変換
    double PYBIND11_EXPORT lb2kg(const double &bef); //!< lbからkgへの変換
    double PYBIND11_EXPORT kg2lb(const double &bef); //!< kgからlbへの変換
    double PYBIND11_EXPORT N2lbf(const double &bef); //!< Nからlbfへの変換
    double PYBIND11_EXPORT lbf2N(const double &bef); //!< lbfからNへの変換
    double PYBIND11_EXPORT psi2Pa(const double &bef); //!< psiからPaへの変換
    double PYBIND11_EXPORT Pa2psi(const double &bef); //!< Paからpsiへの変換
    double PYBIND11_EXPORT slug2N(const double &bef); //!< slugからNへの変換
    double PYBIND11_EXPORT N2slug(const double &bef); //!< Nからslugへの変換
    double PYBIND11_EXPORT slugft22kgm2(const double &bef); //!< slug・ft^2からkg・m2への変換
    /**
     * @brief 高度hにおけるISO 2533の標準大気モデルの状態を返す。
     * 
     * @returns
     *      - T 温度 [K]
     *      - a 音速 [m/s]
     *      - P 気圧 [Pa]
     *      - rho 密度 [kg/m^3]
     *      - nu 動粘性係数 [m^2/s]
     *      - dRhodH 密度勾配 [kg/m^4]
     *      - dTdH 温度勾配 [K/m]
     *      - dPdH 気圧勾配 [Pa/m]
     *      - dadH 音速勾配 [1/s]
     */
    std::map<std::string,double> PYBIND11_EXPORT atmosphere(const double &h);
    double PYBIND11_EXPORT horizon(const double &h); //!< 地球を真球と仮定した場合の高度hにおける水平線の見通し距離[m]

void exportUnits(py::module &m);

ASRC_NAMESPACE_END(util)
ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
