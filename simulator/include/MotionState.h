/**
 * Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
 * @file
 * 
 * @brief シミュレーション登場物の運動状態を管理するクラス MotionState を定義する。
 */
#pragma once
#include "Common.h"
#include <pybind11/pybind11.h>
#include <nlohmann/json.hpp>
#include <Eigen/Core>
#include "TimeSystem.h"
#include "Quaternion.h"
#include "MathUtility.h"
#include "Coordinate.h"
#include "crs/CoordinateReferenceSystemBase.h"
namespace py=pybind11;
namespace nl=nlohmann;

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)

/**
 * @class MotionState
 * @brief あるPhysicalAssetの運動状態を保持するクラス
 * 
 * @details PhysicalAsset自身に固定された座標系(CRS)としても振る舞う。
 *          PhysicalAssetはMotionStateのpos,vel,omega,q,timeの5つのメンバ変数を直接編集することで運動状態を更新する。
 *          それ以外のMotionStateのメンバ変数は上記の編集後にMotionState::validate()を呼ぶことで更新する。
 * 
 *          @par 座標変換関数名における座標系の種類を表す頭文字
 *          - B Body: 自身の位置を原点とし、自身の姿勢に紐付いた座標軸を持つ座標系
 *          - P Parent: 親となる座標系で、基本的にはSimulationManagerのrootCRSとするが、異なるCRSを用いても良い
 *          - H Horizontal: 自身の鉛直下方の楕円体表面上の点を原点とする局所水平座標系
 *          - A Another: 引数として与えられる変換先/変換元の座標系
 */
ASRC_DECLARE_NON_POLYMORPHIC_DATA_CLASS(MotionState)
    private:
    std::shared_ptr<CoordinateReferenceSystem> crs; //!< [For internal use] 親となる座標系
    mutable std::uint64_t crsUpdateCount; //!< [For internal use] 最後のvalidateを呼んだ時点でのcrs->getUpdateCount()
    mutable bool _isValid; //!< [For internal use] 全ての内部変数が適切な状態か否か
    Time _time; //!< [For internal use] 時刻
    Coordinate _pos; //!< [For internal use] parent系での位置
    Coordinate _vel; //!< [For internal use] parent系での速度 (parentがgeographic系なら対応するgeocentric系(ECEF)での速度)
    Quaternion _q; //!< [For internal use] parent系での姿勢 (parentがgeographic系なら対応するgeocentric系(ECEF)での姿勢)
    Coordinate _omega; //!< [For internal use] parent系での角速度 (parentがgeographic系なら対応するgeocentric系(ECEF)での角速度)
    std::string _axisOrder; //!< [For internal use] 自身の座標軸の順序を表す。省略時はFSD(FWD-STBD-DOWN)の順とする。
    mutable Quaternion qToFSDFromNED; //!< [For internal use] body座標系(FSD)から局所水平座標系(NED)への変換クォータニオン。 validate で更新。
    mutable double _az; //!< [For internal use] 自身の+X軸方向の(NED座標系における)方位角。 validate で更新。
    mutable double _el; //!< [For internal use] 自身の+X軸方向の(NED座標系における)仰角。 validate で更新。
    mutable double alt; //!< [For internal use] 楕円体高度。 validate で更新。
    public:
    const Time& time; //!< 時刻
    const Coordinate& pos; //!< parent系での位置
    const Coordinate& vel; //!< parent系での速度 (parentがgeographic系なら対応するgeocentric系(ECEF)での速度)
    const Quaternion& q; //!< parent系での姿勢 (parentがgeographic系なら対応するgeocentric系(ECEF)での姿勢)
    const Coordinate& omega; //!< parent系での角速度 (parentがgeographic系なら対応するgeocentric系(ECEF)での角速度)
    const std::string& axisOrder; //!< 自身の座標軸の順序を表す。省略時はFSD(FWD-STBD-DOWN)の順とする。
    //constructors & destructor
    MotionState();
    MotionState(const std::shared_ptr<CoordinateReferenceSystem>& crs_,const Time& time_,const Eigen::Vector3d& pos_,const Eigen::Vector3d& vel_,const Eigen::Vector3d& omega_,const Quaternion& q_,const std::string& axisOrder_="FSD");
    MotionState(const nl::json& j_);
    MotionState(const MotionState& other);
    MotionState(MotionState&& other);
    void operator=(const MotionState& other);
    void operator=(MotionState&& other);
    ~MotionState();
    MotionState copy() const; //!< Python向け。自身のコピーを返す。
    void setPos(const Eigen::Vector3d& pos_); //!< 位置情報を変更する。
    void setPos(const Coordinate& pos_); //!< 位置情報を変更する。座標変換を伴う。
    void setVel(const Eigen::Vector3d& vel_); //!< 速度情報を変更する。
    void setVel(const Coordinate& vel_); //!< 速度情報を変更する。座標変換を伴う。
    void setQ(const Quaternion& q_); //!< 姿勢情報を変更する。
    void setOmega(const Eigen::Vector3d&omega_); //!< 角速度情報を変更する。
    void setOmega(const Coordinate& omega_); //!< 角速度情報を変更する。座標変換を伴う。
    void setTime(const Time& time_); //!< 時刻情報を変更する。
    void setAxisOrder(const std::string& axisOrder_); //!< body座標系の座標軸の順番を変更する。指定可能な文字は checkBodyCartesianAxisOrder() を参照のこと。
    bool isValid() const; //!< [For internal use] 全ての内部変数が適切な状態か否かを返す。
    void invalidate() const; //!< [For internal use] 内部変数の変更を記録し、更新が必要なタイミングで自動的に validate が呼ばれるようにする。
    void validate() const; //!< [For internal use] 全ての内部変数を適切な状態に更新する。

    ///@name (1)座標軸の変換 (axis conversion in the parent crs)
    ///@{
    Eigen::Vector3d toCartesian(const Eigen::Vector3d& v) const; //!< 親座標系(crs)の本来の座標軸で表された引数を、cartesianな座標軸での値に変換する
    Eigen::Vector3d toSpherical(const Eigen::Vector3d& v) const; //!< 親座標系(crs)の本来の座標軸で表された引数を、sphericalな座標軸での値に変換する
    Eigen::Vector3d toEllipsoidal(const Eigen::Vector3d& v) const; //!< 親座標系(crs)の本来の座標軸で表された引数を、ellipsoidalな座標軸での値に変換する
    Eigen::Vector3d fromCartesian(const Eigen::Vector3d& v) const; //!< cartesianな座標軸で表された引数を、親座標系(crs)の本来の座標軸での値に変換する
    Eigen::Vector3d fromSpherical(const Eigen::Vector3d& v) const; //!< sphericalな座標軸で表された引数を、親座標系(crs)の本来の座標軸での値に変換する
    Eigen::Vector3d fromEllipsoidal(const Eigen::Vector3d& v) const; //!< ellipsoidalな座標軸で表された引数を、親座標系(crs)の本来の座標軸での値に変換する
    ///@}

    ///@name (2)相対位置ベクトルの変換 (transformation of "relative" position)
    ///@{
    Eigen::Vector3d relBtoP(const Eigen::Vector3d &v) const; //!< self系⇛parent系
    Eigen::Vector3d relPtoB(const Eigen::Vector3d &v) const; //!< paernt系⇛self系
    Eigen::Vector3d relHtoP(const Eigen::Vector3d &v,const std::string& dstAxisOrder="FSD",bool onSurface=false) const; //!< horizontal系⇛parent系
    Eigen::Vector3d relPtoH(const Eigen::Vector3d &v,const std::string& dstAxisOrder="FSD",bool onSurface=false) const; //!< parent系⇛horizontal系
    Eigen::Vector3d relHtoB(const Eigen::Vector3d &v,const std::string& dstAxisOrder="FSD",bool onSurface=false) const; //!< horizontal系⇛self系
    Eigen::Vector3d relBtoH(const Eigen::Vector3d &v,const std::string& dstAxisOrder="FSD",bool onSurface=false) const; //!< self系⇛horizontal系
    Eigen::Vector3d relBtoA(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another) const; //!< self系⇛another系
    Eigen::Vector3d relAtoB(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another) const; //!< another系⇛self系
    Eigen::Vector3d relPtoA(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another) const; //!< parent系⇛another系
    Eigen::Vector3d relAtoP(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another) const; //!< another系⇛parent系
    Eigen::Vector3d relHtoA(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another,const std::string& dstAxisOrder="FSD",bool onSurface=false) const; //!< horizontal系⇛another系
    Eigen::Vector3d relAtoH(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another,const std::string& dstAxisOrder="FSD",bool onSurface=false) const; //!< another系⇛horizontal系

    Eigen::Vector3d relBtoP(const Eigen::Vector3d &v,const Eigen::Vector3d &r) const; //!< self系⇛parent系
    Eigen::Vector3d relPtoB(const Eigen::Vector3d &v,const Eigen::Vector3d &r) const; //!< parent系⇛self系
    Eigen::Vector3d relHtoP(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::string& dstAxisOrder="FSD",bool onSurface=false) const; //!< horizontal系⇛parent系
    Eigen::Vector3d relPtoH(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::string& dstAxisOrder="FSD",bool onSurface=false) const; //!< parent系⇛horizontal系
    Eigen::Vector3d relHtoB(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::string& dstAxisOrder="FSD",bool onSurface=false) const; //!< horizontal系⇛self系
    Eigen::Vector3d relBtoH(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::string& dstAxisOrder="FSD",bool onSurface=false) const; //!< self系⇛horizontal系
    Eigen::Vector3d relBtoA(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another) const; //!< self系⇛another系
    Eigen::Vector3d relAtoB(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another) const; //!< another系⇛self系
    Eigen::Vector3d relPtoA(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another) const; //!< parent系⇛another系
    Eigen::Vector3d relAtoP(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another) const; //!< another系⇛parent系
    Eigen::Vector3d relHtoA(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another,const std::string& dstAxisOrder="FSD",bool onSurface=false) const; //!< horizontal系⇛another系
    Eigen::Vector3d relAtoH(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another,const std::string& dstAxisOrder="FSD",bool onSurface=false) const; //!< another系⇛horizontal系
    ///@}

    ///@name (3)絶対位置ベクトルの変換 (transformation of "absolute" position)
    ///@{
    Eigen::Vector3d absBtoP(const Eigen::Vector3d &v) const; //!< self系⇛parent系
    Eigen::Vector3d absPtoB(const Eigen::Vector3d &v) const; //!< parent系⇛self系
    Eigen::Vector3d absHtoP(const Eigen::Vector3d &v,const std::string& dstAxisOrder="FSD",bool onSurface=false) const; //!< horizontal系⇛parent系
    Eigen::Vector3d absPtoH(const Eigen::Vector3d &v,const std::string& dstAxisOrder="FSD",bool onSurface=false) const; //!< parent系⇛horizontal系
    Eigen::Vector3d absHtoB(const Eigen::Vector3d &v,const std::string& dstAxisOrder="FSD",bool onSurface=false) const; //!< horizontal系⇛self系
    Eigen::Vector3d absBtoH(const Eigen::Vector3d &v,const std::string& dstAxisOrder="FSD",bool onSurface=false) const; //!< self系⇛horizontal系
    Eigen::Vector3d absBtoA(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another) const; //!< self系⇛another系
    Eigen::Vector3d absAtoB(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another) const; //!< another系⇛self系
    Eigen::Vector3d absPtoA(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another) const; //!< parent系⇛another系
    Eigen::Vector3d absAtoP(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another) const; //!< another系⇛parent系
    Eigen::Vector3d absHtoA(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another,const std::string& dstAxisOrder="FSD",bool onSurface=false) const; //!< horizontal系⇛another系
    Eigen::Vector3d absAtoH(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another,const std::string& dstAxisOrder="FSD",bool onSurface=false) const; //!< another系⇛horizontal系

    Eigen::Vector3d absBtoP(const Eigen::Vector3d &v,const Eigen::Vector3d &r) const; //!< self系⇛parent系
    Eigen::Vector3d absPtoB(const Eigen::Vector3d &v,const Eigen::Vector3d &r) const; //!< parent系⇛self系
    Eigen::Vector3d absHtoP(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::string& dstAxisOrder="FSD",bool onSurface=false) const; //!< horizontal系⇛parent系
    Eigen::Vector3d absPtoH(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::string& dstAxisOrder="FSD",bool onSurface=false) const; //!< parent系⇛horizontal系
    Eigen::Vector3d absHtoB(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::string& dstAxisOrder="FSD",bool onSurface=false) const; //!< horizontal系⇛self系
    Eigen::Vector3d absBtoH(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::string& dstAxisOrder="FSD",bool onSurface=false) const; //!< self系⇛horizontal系
    Eigen::Vector3d absBtoA(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another) const; //!< self系⇛another系
    Eigen::Vector3d absAtoB(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another) const; //!< another系⇛self系
    Eigen::Vector3d absPtoA(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another) const; //!< parent系⇛another系
    Eigen::Vector3d absAtoP(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another) const; //!< another系⇛parent系
    Eigen::Vector3d absHtoA(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another,const std::string& dstAxisOrder="FSD",bool onSurface=false) const; //!< horizontal系⇛another系
    Eigen::Vector3d absAtoH(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another,const std::string& dstAxisOrder="FSD",bool onSurface=false) const; //!< another系⇛horizontal系
    ///@}

    ///@name (4)速度ベクトルの変換 (transformation of "observed" velocity)
    ///@{
    Eigen::Vector3d velBtoP(const Eigen::Vector3d &v) const; //!< self系⇛parent系
    Eigen::Vector3d velPtoB(const Eigen::Vector3d &v) const; //!< parent系⇛self系
    Eigen::Vector3d velHtoP(const Eigen::Vector3d &v,const std::string& dstAxisOrder="FSD",bool onSurface=false) const; //!< horizontal系⇛parent系
    Eigen::Vector3d velPtoH(const Eigen::Vector3d &v,const std::string& dstAxisOrder="FSD",bool onSurface=false) const; //!< parent系⇛horizontal系
    Eigen::Vector3d velHtoB(const Eigen::Vector3d &v,const std::string& dstAxisOrder="FSD",bool onSurface=false) const; //!< horizontal系⇛self系
    Eigen::Vector3d velBtoH(const Eigen::Vector3d &v,const std::string& dstAxisOrder="FSD",bool onSurface=false) const; //!< self系⇛horizontal系
    Eigen::Vector3d velBtoA(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another) const; //!< self系⇛another系
    Eigen::Vector3d velAtoB(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another) const; //!< another系⇛self系
    Eigen::Vector3d velPtoA(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another) const; //!< parent系⇛another系
    Eigen::Vector3d velAtoP(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another) const; //!< another系⇛parent系
    Eigen::Vector3d velHtoA(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another,const std::string& dstAxisOrder="FSD",bool onSurface=false) const; //!< horizontal系⇛another系
    Eigen::Vector3d velAtoH(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another,const std::string& dstAxisOrder="FSD",bool onSurface=false) const; //!< another系⇛horizontal系

    Eigen::Vector3d velBtoP(const Eigen::Vector3d &v,const Eigen::Vector3d &r) const; //!< self系⇛parent系
    Eigen::Vector3d velPtoB(const Eigen::Vector3d &v,const Eigen::Vector3d &r) const; //!< parent系⇛self系
    Eigen::Vector3d velHtoP(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::string& dstAxisOrder="FSD",bool onSurface=false) const; //!< horizontal系⇛parent系
    Eigen::Vector3d velPtoH(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::string& dstAxisOrder="FSD",bool onSurface=false) const; //!< parent系⇛horizontal系
    Eigen::Vector3d velHtoB(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::string& dstAxisOrder="FSD",bool onSurface=false) const; //!< horizontal系⇛self系
    Eigen::Vector3d velBtoH(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::string& dstAxisOrder="FSD",bool onSurface=false) const; //!< self系⇛horizontal系
    Eigen::Vector3d velBtoA(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another) const; //!< self系⇛another系
    Eigen::Vector3d velAtoB(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another) const; //!< another系⇛self系
    Eigen::Vector3d velPtoA(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another) const; //!< parent系⇛another系
    Eigen::Vector3d velAtoP(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another) const; //!< another系⇛parent系
    Eigen::Vector3d velHtoA(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another,const std::string& dstAxisOrder="FSD",bool onSurface=false) const; //!< horizontal系⇛another系
    Eigen::Vector3d velAtoH(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another,const std::string& dstAxisOrder="FSD",bool onSurface=false) const; //!< another系⇛horizontal系
    ///@}

    ///@name (5)角速度ベクトルの変換 (transformation of "observed" angular velocity)
    ///@{
    Eigen::Vector3d omegaBtoP(const Eigen::Vector3d &v) const; //!< self系⇛parent系
    Eigen::Vector3d omegaPtoB(const Eigen::Vector3d &v) const; //!< parent系⇛self系
    Eigen::Vector3d omegaHtoP(const Eigen::Vector3d &v,const std::string& dstAxisOrder="FSD",bool onSurface=false) const; //!< horizontal系⇛parent系
    Eigen::Vector3d omegaPtoH(const Eigen::Vector3d &v,const std::string& dstAxisOrder="FSD",bool onSurface=false) const; //!< parent系⇛horizontal系
    Eigen::Vector3d omegaHtoB(const Eigen::Vector3d &v,const std::string& dstAxisOrder="FSD",bool onSurface=false) const; //!< horizontal系⇛self系
    Eigen::Vector3d omegaBtoH(const Eigen::Vector3d &v,const std::string& dstAxisOrder="FSD",bool onSurface=false) const; //!< self系⇛horizontal系
    Eigen::Vector3d omegaBtoA(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another) const; //!< self系⇛another系
    Eigen::Vector3d omegaAtoB(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another) const; //!< another系⇛self系
    Eigen::Vector3d omegaPtoA(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another) const; //!< parent系⇛another系
    Eigen::Vector3d omegaAtoP(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another) const; //!< another系⇛parent系
    Eigen::Vector3d omegaHtoA(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another,const std::string& dstAxisOrder="FSD",bool onSurface=false) const; //!< horizontal系⇛another系
    Eigen::Vector3d omegaAtoH(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another,const std::string& dstAxisOrder="FSD",bool onSurface=false) const; //!< another系⇛horizontal系

    Eigen::Vector3d omegaBtoP(const Eigen::Vector3d &v,const Eigen::Vector3d &r) const; //!< self系⇛parent系
    Eigen::Vector3d omegaPtoB(const Eigen::Vector3d &v,const Eigen::Vector3d &r) const; //!< parent系⇛self系
    Eigen::Vector3d omegaHtoP(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::string& dstAxisOrder="FSD",bool onSurface=false) const; //!< horizontal系⇛parent系
    Eigen::Vector3d omegaPtoH(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::string& dstAxisOrder="FSD",bool onSurface=false) const; //!< parent系⇛horizontal系
    Eigen::Vector3d omegaHtoB(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::string& dstAxisOrder="FSD",bool onSurface=false) const; //!< horizontal系⇛self系
    Eigen::Vector3d omegaBtoH(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::string& dstAxisOrder="FSD",bool onSurface=false) const; //!< self系⇛horizontal系
    Eigen::Vector3d omegaBtoA(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another) const; //!< self系⇛another系
    Eigen::Vector3d omegaAtoB(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another) const; //!< another系⇛self系
    Eigen::Vector3d omegaPtoA(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another) const; //!< parent系⇛another系
    Eigen::Vector3d omegaAtoP(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another) const; //!< another系⇛parent系
    Eigen::Vector3d omegaHtoA(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another,const std::string& dstAxisOrder="FSD",bool onSurface=false) const; //!< horizontal系⇛another系
    Eigen::Vector3d omegaAtoH(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another,const std::string& dstAxisOrder="FSD",bool onSurface=false) const; //!< another系⇛horizontal系
    ///@}

    ///@name (6)方向ベクトルの変換 (transformation of direction)
    ///@{
    Eigen::Vector3d dirBtoP(const Eigen::Vector3d &v) const; //!< self系⇛parent系
    Eigen::Vector3d dirPtoB(const Eigen::Vector3d &v) const; //!< paernt系⇛self系
    Eigen::Vector3d dirHtoP(const Eigen::Vector3d &v,const std::string& dstAxisOrder="FSD",bool onSurface=false) const; //!< horizontal系⇛parent系
    Eigen::Vector3d dirPtoH(const Eigen::Vector3d &v,const std::string& dstAxisOrder="FSD",bool onSurface=false) const; //!< parent系⇛horizontal系
    Eigen::Vector3d dirHtoB(const Eigen::Vector3d &v,const std::string& dstAxisOrder="FSD",bool onSurface=false) const; //!< horizontal系⇛self系
    Eigen::Vector3d dirBtoH(const Eigen::Vector3d &v,const std::string& dstAxisOrder="FSD",bool onSurface=false) const; //!< self系⇛horizontal系
    Eigen::Vector3d dirBtoA(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another) const; //!< self系⇛another系
    Eigen::Vector3d dirAtoB(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another) const; //!< another系⇛self系
    Eigen::Vector3d dirPtoA(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another) const; //!< parent系⇛another系
    Eigen::Vector3d dirAtoP(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another) const; //!< another系⇛parent系
    Eigen::Vector3d dirHtoA(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another,const std::string& dstAxisOrder="FSD",bool onSurface=false) const; //!< horizontal系⇛another系
    Eigen::Vector3d dirAtoH(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another,const std::string& dstAxisOrder="FSD",bool onSurface=false) const; //!< another系⇛horizontal系

    Eigen::Vector3d dirBtoP(const Eigen::Vector3d &v,const Eigen::Vector3d &r) const; //!< self系⇛parent系
    Eigen::Vector3d dirPtoB(const Eigen::Vector3d &v,const Eigen::Vector3d &r) const; //!< parent系⇛self系
    Eigen::Vector3d dirHtoP(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::string& dstAxisOrder="FSD",bool onSurface=false) const; //!< horizontal系⇛parent系
    Eigen::Vector3d dirPtoH(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::string& dstAxisOrder="FSD",bool onSurface=false) const; //!< parent系⇛horizontal系
    Eigen::Vector3d dirHtoB(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::string& dstAxisOrder="FSD",bool onSurface=false) const; //!< horizontal系⇛self系
    Eigen::Vector3d dirBtoH(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::string& dstAxisOrder="FSD",bool onSurface=false) const; //!< self系⇛horizontal系
    Eigen::Vector3d dirBtoA(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another) const; //!< self系⇛another系
    Eigen::Vector3d dirAtoB(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another) const; //!< another系⇛self系
    Eigen::Vector3d dirPtoA(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another) const; //!< parent系⇛another系
    Eigen::Vector3d dirAtoP(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another) const; //!< another系⇛parent系
    Eigen::Vector3d dirHtoA(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another,const std::string& dstAxisOrder="FSD",bool onSurface=false) const; //!< horizontal系⇛another系
    Eigen::Vector3d dirAtoH(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another,const std::string& dstAxisOrder="FSD",bool onSurface=false) const; //!< another系⇛horizontal系
    ///@}

    ///@name (7)他のCRSへの変換
    ///@{
    std::shared_ptr<CoordinateReferenceSystem> getCRS() const; //!< 現在のCRSを返す。
    void setCRS(const std::shared_ptr<CoordinateReferenceSystem>& newCRS,bool transform); //!< CRSを再設定する。座標変換の有無を第2引数で指定する。
    MotionState transformTo(const std::shared_ptr<CoordinateReferenceSystem>& newCRS) const; //!< newCRSに座標変換した新たなMotionStateを返す。
    ///@}
    ///@name (8)方位角、仰角、高度の取得
    ///@{
    double getAZ() const; //!< 方位角(真北を0、東向きを正)を返す
    double getEL() const; //!< 仰角(水平を0、鉛直上向きを正)を返す
    double getHeight() const; //<! 楕円体高度(幾何高度)を返す。ellipsoid (geometric) height
    double getGeoidHeight() const; //!< 標高(ジオイド高度)を返す。geoid height (elevation)
    ///@}
    ///@name (9)時刻情報を用いた外挿
    ///@{
    MotionState extrapolate(const double& dt) const; //!< dt秒だけ進めるように外挿する。
    MotionState extrapolateTo(const Time& dstTime) const; //!< dstTimeまで進めるように外挿する。
    ///@}
    ///@name (10)シリアライゼーション
    ///@{
    template<class Archive>
    void save(Archive & archive) const{
        if(!isValid()){validate();}
        archive(
            CEREAL_NVP(crs),
            CEREAL_NVP(time),
            cereal::make_nvp<Archive>("pos",pos()),
            cereal::make_nvp<Archive>("vel",vel()),
            cereal::make_nvp<Archive>("omega",omega()),
            CEREAL_NVP(q),
            CEREAL_NVP(axisOrder),
            cereal::make_nvp<Archive>("az",_az),
            cereal::make_nvp<Archive>("el",_el),
            CEREAL_NVP(alt),
            CEREAL_NVP(qToFSDFromNED)
        );
    }
    template<class Archive>
    void load(Archive & archive) {
        if constexpr (traits::same_archive_as<Archive,util::NLJSONInputArchive>){
            const nl::json& j=archive.getCurrentNode();
            if(!j.is_object()){
                throw std::runtime_error("MotionState only accepts json object.");
            }
        }
        archive(
            CEREAL_NVP(crs),
            cereal::make_nvp<Archive>("time",_time)
        );
        Eigen::Vector3d pos_,vel_,omega_;
        archive(
            cereal::make_nvp<Archive>("pos",pos_),
            cereal::make_nvp<Archive>("vel",vel_),
            cereal::make_nvp<Archive>("omega",omega_)
        );
        _pos=Coordinate(pos_,crs,time,CoordinateType::POSITION_ABS);
        _vel=Coordinate(vel_,pos(),crs,time,CoordinateType::VELOCITY);
        _omega=Coordinate(omega_,pos(),crs,time,CoordinateType::ANGULAR_VELOCITY);
        archive(
            cereal::make_nvp<Archive>("q",_q)
        );
        if(!util::try_load(archive,"axisOrder",_axisOrder)){
            _axisOrder="FSD";
        }
        if(util::try_load(archive,"az",_az)){
            if(
                util::try_load(archive,"el",_el)
                && util::try_load(archive,"alt",alt)
                && util::try_load(archive,"qToFSDFromNED",qToFSDFromNED)
            ){
                _isValid=true;
            }else{
                validate();
            }
        }else{
            Quaternion qToFSDFromNED_;
            if(util::try_load(archive,"qToFSDFromNED",qToFSDFromNED)){
                Eigen::Vector3d ex=qToFSDFromNED_.transformVector(Eigen::Vector3d(1,0,0));
                _az=atan2(ex(1),ex(0));
                validate();
            }else{
                _az=0.0;
                validate();
            }
        }
    }
    ///@}
};

void exportMotionState(py::module &m);

ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)

ASRC_PYBIND11_MAKE_OPAQUE(std::vector<::asrc::core::MotionState>);
ASRC_PYBIND11_MAKE_OPAQUE(std::map<std::string,::asrc::core::MotionState>);
