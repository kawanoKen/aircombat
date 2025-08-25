/**
 * Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
 * @file
 * 
 * @brief Asset のうち物理的実体を持つものを表すクラス
 */
#pragma once
//#define EIGEN_MPL2_ONLY
#include <map>
//#include <memory>
#include <pybind11/pybind11.h>
#include <Eigen/Core>
#include "Quaternion.h"
#include "MotionState.h"
#include "crs/CoordinateReferenceSystem.h"
#include "MathUtility.h"
#include "Asset.h"
#include "SimulationManager.h"

namespace py=pybind11;
namespace nl=nlohmann;

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)

class Agent;
class Controller;

/**
 * @class PhysicalAsset
 * @brief Asset のうち物理的実体を持つものを表すクラス
 */
ASRC_DECLARE_DERIVED_CLASS_WITH_TRAMPOLINE(PhysicalAsset,Asset)
    private:
    friend class SimulationManager;
    virtual void setAgent(const std::weak_ptr<Agent>& agent,const std::string &port);
    public:
    static const std::string baseName; // baseNameを上書き
    std::shared_ptr<SimulationManagerAccessorForPhysicalAsset> manager; //!< SimulationManager の Accessor
    std::weak_ptr<PhysicalAsset> parent; //!< 親となる PhysicalAsset 。親が存在しない場合はnullptr
    std::string team,group,name;
    bool isAlive_;
    //Agent関係
    std::weak_ptr<Agent> agent; //!< 自身を操作する Agent が 存在する場合、そのポインタ
    std::map<std::string,std::weak_ptr<PhysicalAsset>> children; //!< 自身の子となる PhysicalAsset
    std::map<std::string,std::weak_ptr<Controller>> controllers; //!< 自身の子となる Controller
    std::string port; //!< 自身を操作する Agent が 存在する場合、その Agent における自身と紐づけられたportの名前
    //座標系関係
    bool isBoundToParent; //!< 親となる PhysicalAsset に固定されているかどうか。
    std::string bodyAxisOrder; //!< body座標系の座標軸の順序。デフォルトは"FSD"
    MotionState motion;//!< 自身の運動状態。基本的には manager->getRootCRS() で得られる基準座標系における値として保持する。ただし、親に固定されている場合は親のBody座標系とする。
    MotionState motion_prev;//!< 自身の直前の運動場多い。基本的には manager->getRootCRS() で得られる基準座標系における値として保持する。ただし、親に固定されている場合は親のBody座標系とする。
    /**
     * @brief 自身に固定された座標系(AffineCRS)。
     * 
     * @details 必要がなければCRSインスタンスとしては生成せず、メンバ変数で同等の変換機能を提供する。
     *          PhysicalAsset::getBodyCRS()を呼び出した場合のみ生成し、自身のメンバ変数の更新と同時にこのCRSも更新する。
     * 
     *          用途としては、親となるPhysicalAssetに固定されたPhysicalAssetのMotionStateのcrsとして使用する場合が挙げられる。
     */
    std::shared_ptr<AffineCRS> bodyCRS; 
    const Coordinate &pos; //!< motion.pos を透過的に参照するためのconst参照変数
    const Coordinate &vel; //!< motion.vel を透過的に参照するためのconst参照変数
    const Coordinate &omega; //!< motion.omega を透過的に参照するためのconst参照変数
    const Coordinate &pos_prev; //!< motion_prev.pos を透過的に参照するためのconst参照変数
    const Coordinate &vel_prev; //!< motion_prev.vel を透過的に参照するためのconst参照変数
    const Coordinate &omega_prev; //!< motion_prev.omega を透過的に参照するためのconst参照変数
    const Quaternion &q; //!< motion.q を透過的に参照するためのconst参照変数
    const Quaternion &q_prev; //!< motion_prev.q を透過的に参照するためのconst参照変数
    public:
    //constructors & destructor
    PhysicalAsset(const nl::json& modelConfig_,const nl::json& instanceConfig_);
    //functions
    virtual void initialize() override;
    virtual void serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full) override;
    virtual bool isAlive() const override;
    virtual std::string getTeam() const override;
    virtual std::string getGroup() const override;
    virtual std::string getName() const override;
    virtual void perceive(bool inReset) override;
    virtual void control() override;
    virtual void behave() override;
    virtual void kill() override;
    protected:
    virtual void setManagerAccessor(const std::shared_ptr<EntityManagerAccessor>& ema) override;
    public:
    virtual void setInitialMotionStateFromJson(const nl::json& j); //!< instanceConfigとして与えられた情報を用いてMotionStateを初期化する。派生クラスごとに異なる初期化処理を記述してよい。

    //座標系の取り扱いに関すること (基本的にはMotionStateのメンバ関数を呼ぶだけ)
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
    Eigen::Vector3d relBtoP(const Eigen::Vector3d &v) const;//!< self系⇛parent系
    Eigen::Vector3d relPtoB(const Eigen::Vector3d &v) const;//!< paernt系⇛self系
    Eigen::Vector3d relHtoP(const Eigen::Vector3d &v,const std::string& dstAxisOrder="FSD",bool onSurface=false) const;//!< horizontal系⇛parent系
    Eigen::Vector3d relPtoH(const Eigen::Vector3d &v,const std::string& dstAxisOrder="FSD",bool onSurface=false) const;//!< parent系⇛horizontal系
    Eigen::Vector3d relHtoB(const Eigen::Vector3d &v,const std::string& dstAxisOrder="FSD",bool onSurface=false) const;//!< horizontal系⇛self系
    Eigen::Vector3d relBtoH(const Eigen::Vector3d &v,const std::string& dstAxisOrder="FSD",bool onSurface=false) const;//!< self系⇛horizontal系
    Eigen::Vector3d relBtoA(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another) const;//!< self系⇛another系
    Eigen::Vector3d relAtoB(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another) const;//!< another系⇛self系
    Eigen::Vector3d relPtoA(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another) const;//!< parent系⇛another系
    Eigen::Vector3d relAtoP(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another) const;//!< another系⇛parent系
    Eigen::Vector3d relHtoA(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another,const std::string& dstAxisOrder="FSD",bool onSurface=false) const;//!< horizontal系⇛another系
    Eigen::Vector3d relAtoH(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another,const std::string& dstAxisOrder="FSD",bool onSurface=false) const;//!< another系⇛horizontal系

    Eigen::Vector3d relBtoP(const Eigen::Vector3d &v,const Eigen::Vector3d &r) const;//!< self系⇛parent系
    Eigen::Vector3d relPtoB(const Eigen::Vector3d &v,const Eigen::Vector3d &r) const;//!< parent系⇛self系
    Eigen::Vector3d relHtoP(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::string& dstAxisOrder="FSD",bool onSurface=false) const;//!< horizontal系⇛parent系
    Eigen::Vector3d relPtoH(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::string& dstAxisOrder="FSD",bool onSurface=false) const;//!< parent系⇛horizontal系
    Eigen::Vector3d relHtoB(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::string& dstAxisOrder="FSD",bool onSurface=false) const;//!< horizontal系⇛self系
    Eigen::Vector3d relBtoH(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::string& dstAxisOrder="FSD",bool onSurface=false) const;//!< self系⇛horizontal系
    Eigen::Vector3d relBtoA(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another) const;//!< self系⇛another系
    Eigen::Vector3d relAtoB(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another) const;//!< another系⇛self系
    Eigen::Vector3d relPtoA(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another) const;//!< parent系⇛another系
    Eigen::Vector3d relAtoP(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another) const;//!< another系⇛parent系
    Eigen::Vector3d relHtoA(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another,const std::string& dstAxisOrder="FSD",bool onSurface=false) const;//!< horizontal系⇛another系
    Eigen::Vector3d relAtoH(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another,const std::string& dstAxisOrder="FSD",bool onSurface=false) const;//!< another系⇛horizontal系
    ///@}

    ///@name (3)絶対位置ベクトルの変換 (transformation of "absolute" position)
    ///@{
    Eigen::Vector3d absBtoP(const Eigen::Vector3d &v) const;//!< self系⇛parent系
    Eigen::Vector3d absPtoB(const Eigen::Vector3d &v) const;//!< parent系⇛self系
    Eigen::Vector3d absHtoP(const Eigen::Vector3d &v,const std::string& dstAxisOrder="FSD",bool onSurface=false) const;//!< horizontal系⇛parent系
    Eigen::Vector3d absPtoH(const Eigen::Vector3d &v,const std::string& dstAxisOrder="FSD",bool onSurface=false) const;//!< parent系⇛horizontal系
    Eigen::Vector3d absHtoB(const Eigen::Vector3d &v,const std::string& dstAxisOrder="FSD",bool onSurface=false) const;//!< horizontal系⇛self系
    Eigen::Vector3d absBtoH(const Eigen::Vector3d &v,const std::string& dstAxisOrder="FSD",bool onSurface=false) const;//!< self系⇛horizontal系
    Eigen::Vector3d absBtoA(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another) const;//!< self系⇛another系
    Eigen::Vector3d absAtoB(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another) const;//!< another系⇛self系
    Eigen::Vector3d absPtoA(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another) const;//!< parent系⇛another系
    Eigen::Vector3d absAtoP(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another) const;//!< another系⇛parent系
    Eigen::Vector3d absHtoA(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another,const std::string& dstAxisOrder="FSD",bool onSurface=false) const;//!< horizontal系⇛another系
    Eigen::Vector3d absAtoH(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another,const std::string& dstAxisOrder="FSD",bool onSurface=false) const;//!< another系⇛horizontal系

    Eigen::Vector3d absBtoP(const Eigen::Vector3d &v,const Eigen::Vector3d &r) const;//!< self系⇛parent系
    Eigen::Vector3d absPtoB(const Eigen::Vector3d &v,const Eigen::Vector3d &r) const;//!< parent系⇛self系
    Eigen::Vector3d absHtoP(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::string& dstAxisOrder="FSD",bool onSurface=false) const;//!< horizontal系⇛parent系
    Eigen::Vector3d absPtoH(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::string& dstAxisOrder="FSD",bool onSurface=false) const;//!< parent系⇛horizontal系
    Eigen::Vector3d absHtoB(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::string& dstAxisOrder="FSD",bool onSurface=false) const;//!< horizontal系⇛self系
    Eigen::Vector3d absBtoH(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::string& dstAxisOrder="FSD",bool onSurface=false) const;//!< self系⇛horizontal系
    Eigen::Vector3d absBtoA(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another) const;//!< self系⇛another系
    Eigen::Vector3d absAtoB(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another) const;//!< another系⇛self系
    Eigen::Vector3d absPtoA(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another) const;//!< parent系⇛another系
    Eigen::Vector3d absAtoP(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another) const;//!< another系⇛parent系
    Eigen::Vector3d absHtoA(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another,const std::string& dstAxisOrder="FSD",bool onSurface=false) const;//!< horizontal系⇛another系
    Eigen::Vector3d absAtoH(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another,const std::string& dstAxisOrder="FSD",bool onSurface=false) const;//!< another系⇛horizontal系
    ///@}

    ///@name (4)速度ベクトルの変換 (transformation of "observed" velocity)
    ///@{
    Eigen::Vector3d velBtoP(const Eigen::Vector3d &v) const;//!< self系⇛parent系
    Eigen::Vector3d velPtoB(const Eigen::Vector3d &v) const;//!< parent系⇛self系
    Eigen::Vector3d velHtoP(const Eigen::Vector3d &v,const std::string& dstAxisOrder="FSD",bool onSurface=false) const;//!< horizontal系⇛parent系
    Eigen::Vector3d velPtoH(const Eigen::Vector3d &v,const std::string& dstAxisOrder="FSD",bool onSurface=false) const;//!< parent系⇛horizontal系
    Eigen::Vector3d velHtoB(const Eigen::Vector3d &v,const std::string& dstAxisOrder="FSD",bool onSurface=false) const;//!< horizontal系⇛self系
    Eigen::Vector3d velBtoH(const Eigen::Vector3d &v,const std::string& dstAxisOrder="FSD",bool onSurface=false) const;//!< self系⇛horizontal系
    Eigen::Vector3d velBtoA(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another) const;//!< self系⇛another系
    Eigen::Vector3d velAtoB(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another) const;//!< another系⇛self系
    Eigen::Vector3d velPtoA(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another) const;//!< parent系⇛another系
    Eigen::Vector3d velAtoP(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another) const;//!< another系⇛parent系
    Eigen::Vector3d velHtoA(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another,const std::string& dstAxisOrder="FSD",bool onSurface=false) const;//!< horizontal系⇛another系
    Eigen::Vector3d velAtoH(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another,const std::string& dstAxisOrder="FSD",bool onSurface=false) const;//!< another系⇛horizontal系

    Eigen::Vector3d velBtoP(const Eigen::Vector3d &v,const Eigen::Vector3d &r) const;//!< self系⇛parent系
    Eigen::Vector3d velPtoB(const Eigen::Vector3d &v,const Eigen::Vector3d &r) const;//!< parent系⇛self系
    Eigen::Vector3d velHtoP(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::string& dstAxisOrder="FSD",bool onSurface=false) const;//!< horizontal系⇛parent系
    Eigen::Vector3d velPtoH(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::string& dstAxisOrder="FSD",bool onSurface=false) const;//!< parent系⇛horizontal系
    Eigen::Vector3d velHtoB(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::string& dstAxisOrder="FSD",bool onSurface=false) const;//!< horizontal系⇛self系
    Eigen::Vector3d velBtoH(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::string& dstAxisOrder="FSD",bool onSurface=false) const;//!< self系⇛horizontal系
    Eigen::Vector3d velBtoA(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another) const;//!< self系⇛another系
    Eigen::Vector3d velAtoB(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another) const;//!< another系⇛self系
    Eigen::Vector3d velPtoA(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another) const;//!< parent系⇛another系
    Eigen::Vector3d velAtoP(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another) const;//!< another系⇛parent系
    Eigen::Vector3d velHtoA(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another,const std::string& dstAxisOrder="FSD",bool onSurface=false) const;//!< horizontal系⇛another系
    Eigen::Vector3d velAtoH(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another,const std::string& dstAxisOrder="FSD",bool onSurface=false) const;//!< another系⇛horizontal系
    ///@}

    ///@name (5)角速度ベクトルの変換 (transformation of "observed" angular velocity)
    ///@{
    Eigen::Vector3d omegaBtoP(const Eigen::Vector3d &v) const;//!< self系⇛parent系
    Eigen::Vector3d omegaPtoB(const Eigen::Vector3d &v) const;//!< parent系⇛self系
    Eigen::Vector3d omegaHtoP(const Eigen::Vector3d &v,const std::string& dstAxisOrder="FSD",bool onSurface=false) const;//!< horizontal系⇛parent系
    Eigen::Vector3d omegaPtoH(const Eigen::Vector3d &v,const std::string& dstAxisOrder="FSD",bool onSurface=false) const;//!< parent系⇛horizontal系
    Eigen::Vector3d omegaHtoB(const Eigen::Vector3d &v,const std::string& dstAxisOrder="FSD",bool onSurface=false) const;//!< horizontal系⇛self系
    Eigen::Vector3d omegaBtoH(const Eigen::Vector3d &v,const std::string& dstAxisOrder="FSD",bool onSurface=false) const;//!< self系⇛horizontal系
    Eigen::Vector3d omegaBtoA(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another) const;//!< self系⇛another系
    Eigen::Vector3d omegaAtoB(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another) const;//!< another系⇛self系
    Eigen::Vector3d omegaPtoA(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another) const;//!< parent系⇛another系
    Eigen::Vector3d omegaAtoP(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another) const;//!< another系⇛parent系
    Eigen::Vector3d omegaHtoA(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another,const std::string& dstAxisOrder="FSD",bool onSurface=false) const;//!< horizontal系⇛another系
    Eigen::Vector3d omegaAtoH(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another,const std::string& dstAxisOrder="FSD",bool onSurface=false) const;//!< another系⇛horizontal系

    Eigen::Vector3d omegaBtoP(const Eigen::Vector3d &v,const Eigen::Vector3d &r) const;//!< self系⇛parent系
    Eigen::Vector3d omegaPtoB(const Eigen::Vector3d &v,const Eigen::Vector3d &r) const;//!< parent系⇛self系
    Eigen::Vector3d omegaHtoP(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::string& dstAxisOrder="FSD",bool onSurface=false) const;//!< horizontal系⇛parent系
    Eigen::Vector3d omegaPtoH(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::string& dstAxisOrder="FSD",bool onSurface=false) const;//!< parent系⇛horizontal系
    Eigen::Vector3d omegaHtoB(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::string& dstAxisOrder="FSD",bool onSurface=false) const;//!< horizontal系⇛self系
    Eigen::Vector3d omegaBtoH(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::string& dstAxisOrder="FSD",bool onSurface=false) const;//!< self系⇛horizontal系
    Eigen::Vector3d omegaBtoA(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another) const;//!< self系⇛another系
    Eigen::Vector3d omegaAtoB(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another) const;//!< another系⇛self系
    Eigen::Vector3d omegaPtoA(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another) const;//!< parent系⇛another系
    Eigen::Vector3d omegaAtoP(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another) const;//!< another系⇛parent系
    Eigen::Vector3d omegaHtoA(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another,const std::string& dstAxisOrder="FSD",bool onSurface=false) const;//!< horizontal系⇛another系
    Eigen::Vector3d omegaAtoH(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another,const std::string& dstAxisOrder="FSD",bool onSurface=false) const;//!< another系⇛horizontal系
    ///@}

    ///@name (6)方向ベクトルの変換 (transformation of direction)
    ///@{
    Eigen::Vector3d dirBtoP(const Eigen::Vector3d &v) const;//!< self系⇛parent系
    Eigen::Vector3d dirPtoB(const Eigen::Vector3d &v) const;//!< paernt系⇛self系
    Eigen::Vector3d dirHtoP(const Eigen::Vector3d &v,const std::string& dstAxisOrder="FSD",bool onSurface=false) const;//!< horizontal系⇛parent系
    Eigen::Vector3d dirPtoH(const Eigen::Vector3d &v,const std::string& dstAxisOrder="FSD",bool onSurface=false) const;//!< parent系⇛horizontal系
    Eigen::Vector3d dirHtoB(const Eigen::Vector3d &v,const std::string& dstAxisOrder="FSD",bool onSurface=false) const;//!< horizontal系⇛self系
    Eigen::Vector3d dirBtoH(const Eigen::Vector3d &v,const std::string& dstAxisOrder="FSD",bool onSurface=false) const;//!< self系⇛horizontal系
    Eigen::Vector3d dirBtoA(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another) const;//!< self系⇛another系
    Eigen::Vector3d dirAtoB(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another) const;//!< another系⇛self系
    Eigen::Vector3d dirPtoA(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another) const;//!< parent系⇛another系
    Eigen::Vector3d dirAtoP(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another) const;//!< another系⇛parent系
    Eigen::Vector3d dirHtoA(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another,const std::string& dstAxisOrder="FSD",bool onSurface=false) const;//!< horizontal系⇛another系
    Eigen::Vector3d dirAtoH(const Eigen::Vector3d &v,const std::shared_ptr<CoordinateReferenceSystem>& another,const std::string& dstAxisOrder="FSD",bool onSurface=false) const;//!< another系⇛horizontal系

    Eigen::Vector3d dirBtoP(const Eigen::Vector3d &v,const Eigen::Vector3d &r) const;//!< self系⇛parent系
    Eigen::Vector3d dirPtoB(const Eigen::Vector3d &v,const Eigen::Vector3d &r) const;//!< parent系⇛self系
    Eigen::Vector3d dirHtoP(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::string& dstAxisOrder="FSD",bool onSurface=false) const;//!< horizontal系⇛parent系
    Eigen::Vector3d dirPtoH(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::string& dstAxisOrder="FSD",bool onSurface=false) const;//!< parent系⇛horizontal系
    Eigen::Vector3d dirHtoB(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::string& dstAxisOrder="FSD",bool onSurface=false) const;//!< horizontal系⇛self系
    Eigen::Vector3d dirBtoH(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::string& dstAxisOrder="FSD",bool onSurface=false) const;//!< self系⇛horizontal系
    Eigen::Vector3d dirBtoA(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another) const;//!< self系⇛another系
    Eigen::Vector3d dirAtoB(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another) const;//!< another系⇛self系
    Eigen::Vector3d dirPtoA(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another) const;//!< parent系⇛another系
    Eigen::Vector3d dirAtoP(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another) const;//!< another系⇛parent系
    Eigen::Vector3d dirHtoA(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another,const std::string& dstAxisOrder="FSD",bool onSurface=false) const;//!< horizontal系⇛another系
    Eigen::Vector3d dirAtoH(const Eigen::Vector3d &v,const Eigen::Vector3d &r,const std::shared_ptr<CoordinateReferenceSystem>& another,const std::string& dstAxisOrder="FSD",bool onSurface=false) const;//!< another系⇛horizontal系
    ///@}

    ///@name (7)方位角、仰角、高度の取得
    ///@{
    double getAZ() const; //!< 方位角(真北を0、東向きを正)を返す
    double getEL() const; //!< 仰角(水平を0、鉛直上向きを正)を返す
    double getHeight() const; //!< 楕円体高度(幾何高度)を返す。ellipsoid (geometric) height
    double getGeoidHeight() const; //!< 標高(ジオイド高度)を返す。geoid height (elevation)
    ///@}

    ///@name 自身の座標情報
    ///@{
    std::shared_ptr<CoordinateReferenceSystem> getParentCRS(); //!< motion.getCRS()と同等。
    std::shared_ptr<AffineCRS> getBodyCRS();
    ///@}
    ///@name 親との固定状態の変更
    ///@{
    virtual void bindToParent(); //!< 親に固定する。
    virtual void unbindFromParent(); //!< 親への固定を解除する。
    ///@}

    virtual std::shared_ptr<EntityAccessor> getAccessorImpl() override;
    /**
     * @brief CommunicationBufferを生成する。
     * 
     * @param [in] name_ CommunicationBuffer を識別する名前
     * @param [in] participants_ 無条件で参加する Asset のリスト
     * @param [in] inviteOnRequest_ Asset 側から要求があった場合のみ参加させる Asset のリスト
     * 
     * @sa [CommunicationBuffer による Asset 間通信の表現](\ref page_simulation_CommunicationBuffer)
     */
    bool generateCommunicationBuffer(const std::string& name_,const nl::json& participants_,const nl::json& inviteOnRequest_);
};

ASRC_DECLARE_DERIVED_TRAMPOLINE(PhysicalAsset)
    virtual bool isAlive() const override{
        PYBIND11_OVERRIDE(bool,Base,isAlive);
    }
    virtual std::string getTeam() const override{
        PYBIND11_OVERRIDE(std::string,Base,getTeam);
    }
    virtual std::string getGroup() const override{
        PYBIND11_OVERRIDE(std::string,Base,getGroup);
    }
    virtual std::string getName() const override{
        PYBIND11_OVERRIDE(std::string,Base,getName);
    }
    virtual void setInitialMotionStateFromJson(const nl::json& j) override{
        PYBIND11_OVERRIDE(void,Base,setInitialMotionStateFromJson,j);
    }
    //親との固定状態の変更
    virtual void bindToParent() override{
        PYBIND11_OVERRIDE(void,Base,bindToParent);
    }
    virtual void unbindFromParent() override{
        PYBIND11_OVERRIDE(void,Base,unbindFromParent);
    }
};

/**
 * @class PhysicalAssetAccessor
 * @brief PhysicalAsset とその派生クラスのメンバに対するアクセス制限を実現するためのAccessorクラス。
 */
ASRC_DECLARE_DERIVED_CLASS_WITHOUT_TRAMPOLINE(PhysicalAssetAccessor,AssetAccessor)
    public:
    PhysicalAssetAccessor(const std::shared_ptr<PhysicalAsset>& pa);
    protected:
    std::weak_ptr<PhysicalAsset> physicalAsset;
};

void exportPhysicalAsset(py::module &m, const std::shared_ptr<asrc::core::FactoryHelper>& factoryHelper);

ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)

ASRC_PYBIND11_MAKE_OPAQUE(std::map<std::string,std::shared_ptr<::asrc::core::PhysicalAssetAccessor>>);
