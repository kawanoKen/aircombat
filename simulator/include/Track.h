/**
 * Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
 * @file
 * 
 * @brief シミュレーション登場物の航跡を表すクラス
 */
#pragma once
#include "Common.h"
#include <memory>
#include <vector>
#include <pybind11/pybind11.h>
#include <Eigen/Core>
#include <boost/uuid/uuid.hpp>
#include "TimeSystem.h"
#include "MathUtility.h"
#include "Coordinate.h"
#include "PhysicalAsset.h"
namespace py=pybind11;
namespace nl=nlohmann;

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)

class CoordinateReferenceSystem;

/**
 * @class TrackBase
 * @brief 航跡情報を表すための基底クラス
 * 
 * @details
 *      基底クラスでは座標系、時刻、PhysicalAsset識別子(truth)の情報のみを保持し、
 *      実際には航跡の種類に応じてTrack3D又はTrack2Dを使い分けるものとする。
 */
ASRC_DECLARE_BASE_DATA_CLASS(TrackBase)
    protected:
    std::shared_ptr<CoordinateReferenceSystem> crs; //!< CRS
    Time _time; //!< 時刻情報
    public:
    const Time& time; //!< 時刻情報
    boost::uuids::uuid truth; //!< 航跡が指す対象をPhysicalAssetを表すUUID。この情報を取得可能とするかどうかはシミュレーションモデルの作成者が決めてよい。
    //constructors & destructor
    TrackBase();
    TrackBase(const std::weak_ptr<PhysicalAsset>& truth_,const std::shared_ptr<CoordinateReferenceSystem>& crs_);
    TrackBase(const std::weak_ptr<PhysicalAsset>& truth_,const std::shared_ptr<CoordinateReferenceSystem>& crs_,const Time& time_);
    TrackBase(const boost::uuids::uuid& truth_,const std::shared_ptr<CoordinateReferenceSystem>& crs_,const Time& time_);
    TrackBase(const nl::json& j_);
    TrackBase(const TrackBase& other);
    TrackBase(TrackBase&& other);
    void operator=(const TrackBase& other);
    void operator=(TrackBase&& other);
    //functions
    virtual std::shared_ptr<CoordinateReferenceSystem> getCRS() const; //!< CRSを返す。
    virtual void setCRS(const std::shared_ptr<CoordinateReferenceSystem>& newCRS,bool transform); //!< CRSを再設定する。座標変換の有無を第2引数で指定する。
    virtual void setTime(const Time& time_); //!< 時刻を再設定する。
    virtual bool is_none() const; //!< 空の航跡かどうかを返す。
    virtual bool isSame(const TrackBase& other) const; //!< other と同じ PhysicalAssetを指しているかどうかを返す。
    virtual bool isSame(const boost::uuids::uuid& other) const; //!< other と同じ PhysicalAssetを指しているかどうかを返す。
    virtual bool isSame(const std::weak_ptr<Asset>& other) const; //!< other と同じ PhysicalAssetを指しているかどうかを返す。
    virtual void polymorphic_assign(const PtrBaseType& other) override;
    virtual void serialize_impl(asrc::core::util::AvailableArchiveTypes& archive) override;
};

ASRC_DECLARE_BASE_DATA_TRAMPOLINE(TrackBase)
    virtual std::shared_ptr<CoordinateReferenceSystem> getCRS() const override{
        PYBIND11_OVERRIDE(std::shared_ptr<CoordinateReferenceSystem>,Base,getCRS);
    }
    virtual void setCRS(const std::shared_ptr<CoordinateReferenceSystem>& newCRS,bool transform) override{
        PYBIND11_OVERRIDE(void,Base,setCRS,newCRS,transform);
    }
    virtual void setTime(const Time& time_) override{
        PYBIND11_OVERRIDE(void,Base,setTime,time_);
    }
    virtual bool is_none() const override{
        PYBIND11_OVERRIDE(bool,Base,is_none);
    }
    virtual bool isSame(const TrackBase& other) const override{
        PYBIND11_OVERRIDE(bool,Base,isSame,other);
    }
    virtual bool isSame(const boost::uuids::uuid& other) const override{
        PYBIND11_OVERRIDE(bool,Base,isSame,other);
    }
    virtual bool isSame(const std::weak_ptr<Asset>& other) const override{
        PYBIND11_OVERRIDE(bool,Base,isSame,other);
    }
};

/**
 * @class Track3D
 * @brief 3次元航跡情報を表すための基底クラス
 * 
 * @details
 *      このクラスでは位置と速度の情報のみを持つ。
 *      派生クラスにおいて誤差共分散行列等の付加情報を持たせてもよい。
 */
ASRC_DECLARE_DERIVED_CLASS_WITH_TRAMPOLINE(Track3D,TrackBase)
    protected:
    Coordinate _pos,_vel;
    public:
    const Coordinate& pos; //!< 位置情報
    const Coordinate& vel; //!< 速度情報
    std::vector<Track3D> buffer; //!<　複数の Track3D をマージするためのバッファ
    public:
    //constructors & destructor
    Track3D();
    Track3D(const std::weak_ptr<PhysicalAsset>& truth_,const std::shared_ptr<CoordinateReferenceSystem>& crs_);
    Track3D(const std::weak_ptr<PhysicalAsset>& truth_,const std::shared_ptr<CoordinateReferenceSystem>& crs_,const Time& time_,const Eigen::Vector3d& pos_,const Eigen::Vector3d& vel_);
    Track3D(const boost::uuids::uuid& truth_,const std::shared_ptr<CoordinateReferenceSystem>& crs_,const Time& time_,const Eigen::Vector3d& pos_,const Eigen::Vector3d& vel_);
    Track3D(const nl::json& j_);
    Track3D(const Track3D& other);
    Track3D(Track3D&& other);
    void operator=(const Track3D& other);
    void operator=(Track3D&& other);
    //functions
    virtual void setCRS(const std::shared_ptr<CoordinateReferenceSystem>& newCRS,bool transform) override;
    virtual Track3D transformTo(const std::shared_ptr<CoordinateReferenceSystem>& newCRS) const; //!< newCRS に座標変換した新たな Track3D を返す。
    virtual void setTime(const Time& time_) override;
    void setPos(const Eigen::Vector3d& pos_); //!< 位置を変更する。
    void setPos(const Coordinate& pos_); //!< 位置を変更する。
    void setVel(const Eigen::Vector3d& vel_); //!< 速度を変更する。
    void setVel(const Coordinate& vel_); //!< 速度を変更する。
    Track3D copy() const; //!< 自身のコピーを返す。
    virtual void clearBuffer(); //!< マージ用のbufferを消去する。
    virtual void addBuffer(const Track3D& other); //!< マージ用のbufferに航跡を追加する。
    virtual void merge(); //!< マージ用のbufferをマージして自身の値として設定する。
    virtual void update(const Track3D& other); //!< otherの値を自身の値として設定する。
    virtual void updateByExtrapolation(const double& dt); //!< 自身の状態をdt秒だけ外挿する。
    virtual Track3D extrapolate(const double& dt); //!< 自身の状態をdt秒だけ外挿した新たなTrack3Dを返す。
    virtual Track3D extrapolateTo(const Time& dstTime); //!< 自身の状態をdstTimeまで外挿した新たなTrack3Dを返す。
    virtual void polymorphic_assign(const PtrBaseType& other) override;
    virtual void serialize_impl(asrc::core::util::AvailableArchiveTypes& archive) override;
};

ASRC_DECLARE_DERIVED_TRAMPOLINE(Track3D)
    virtual Track3D transformTo(const std::shared_ptr<CoordinateReferenceSystem>& newCRS) const override{
        PYBIND11_OVERRIDE(Track3D,Base,transformTo,newCRS);
    }
    virtual void clearBuffer() override{
        PYBIND11_OVERRIDE(void,Base,clearBuffer);
    }
    virtual void addBuffer(const Track3D& other) override{
        PYBIND11_OVERRIDE(void,Base,addBuffer,other);
    }
    virtual void merge() override{
        PYBIND11_OVERRIDE(void,Base,merge);
    }
    virtual void update(const Track3D& other) override{
        PYBIND11_OVERRIDE(void,Base,update,other);
    }
    virtual void updateByExtrapolation(const double& dt) override{
        PYBIND11_OVERRIDE(void,Base,updateByExtrapolation,dt);
    }
    virtual Track3D extrapolate(const double& dt) override{
        PYBIND11_OVERRIDE(Track3D,Base,extrapolate,dt);
    }
    virtual Track3D extrapolateTo(const Time& dstTime) override{
        PYBIND11_OVERRIDE(Track3D,Base,extrapolateTo,dstTime);
    }
};

/**
 * @class Track2D
 * @brief 2次元航跡情報を表すための基底クラス
 * 
 * @details
 *      このクラスでは方向、方向変化率、観測点位置の情報のみを持つ。
 *      派生クラスにおいて誤差共分散行列等の付加情報を持たせてもよい。
 */
ASRC_DECLARE_DERIVED_CLASS_WITH_TRAMPOLINE(Track2D,TrackBase)
    protected:
    Coordinate _origin,_dir,_omega;
    public:
    const Coordinate& origin; //!< 観測点の位置
    const Coordinate& dir; //!< 方向
    const Coordinate& omega; //!< 方向変化率
    std::vector<Track2D> buffer; //!<　複数の Track2D をマージするためのバッファ
    public:
    //constructors & destructor
    Track2D();
    Track2D(const std::weak_ptr<PhysicalAsset>& truth_,const std::shared_ptr<CoordinateReferenceSystem>& crs_,const Eigen::Vector3d& origin_);
    Track2D(const std::weak_ptr<PhysicalAsset>& truth_,const std::shared_ptr<CoordinateReferenceSystem>& crs_,const Time& time_,const Eigen::Vector3d& dir_,const Eigen::Vector3d& origin_,const Eigen::Vector3d& omega_);
    Track2D(const boost::uuids::uuid& truth_,const std::shared_ptr<CoordinateReferenceSystem>& crs_,const Time& time_,const Eigen::Vector3d& dir_,const Eigen::Vector3d& origin_,const Eigen::Vector3d& omega_);
    Track2D(const nl::json& j_);
    Track2D(const Track2D& other);
    Track2D(Track2D&& other);
    void operator=(const Track2D& other);
    void operator=(Track2D&& other);
    //functions
    virtual void setCRS(const std::shared_ptr<CoordinateReferenceSystem>& newCRS,bool transform) override; //CRSを再設定する。座標変換の有無を第2引数で指定する。
    virtual Track2D transformTo(const std::shared_ptr<CoordinateReferenceSystem>& newCRS) const; //!< newCRS に座標変換した新たな Track2D を返す。
    virtual void setTime(const Time& time_) override;
    void setOrigin(const Eigen::Vector3d& origin_); //!< 観測点位置を変更する。
    void setOrigin(const Coordinate& origin_); //!< 観測点位置を変更する。
    void setDir(const Eigen::Vector3d& dir_); //!< 方向を変更する。
    void setDir(const Coordinate& dir_); //!< 方向を変更する。
    void setOmega(const Eigen::Vector3d& omega_); //!< 方向変化率を変更する。
    void setOmega(const Coordinate& omega_); //!< 方向変化率を変更する。
    Track2D copy() const; //!< 自身のコピーを返す。
    virtual void clearBuffer(); //!< マージ用のbufferを消去する。
    virtual void addBuffer(const Track2D& other); //!< マージ用のbufferに航跡を追加する。
    virtual void merge(); //!< マージ用のbufferをマージして自身の値として設定する。
    virtual void update(const Track2D& other); //!< otherの値を自身の値として設定する。
    virtual void updateByExtrapolation(const double& dt); //!< 自身の状態をdt秒だけ外挿する。
    virtual Track2D extrapolate(const double& dt); //!< 自身の状態をdt秒だけ外挿した新たなTrack2Dを返す。
    virtual Track2D extrapolateTo(const Time& dstTime); //!< 自身の状態をdstTimeまで外挿した新たなTrack2Dを返す。
    virtual void polymorphic_assign(const PtrBaseType& other) override;
    virtual void serialize_impl(asrc::core::util::AvailableArchiveTypes& archive) override;
};

ASRC_DECLARE_DERIVED_TRAMPOLINE(Track2D)
    virtual Track2D transformTo(const std::shared_ptr<CoordinateReferenceSystem>& newCRS) const override{
        PYBIND11_OVERRIDE(Track2D,Base,transformTo,newCRS);
    }
    virtual void clearBuffer() override{
        PYBIND11_OVERRIDE(void,Base,clearBuffer);
    }
    virtual void addBuffer(const Track2D& other) override{
        PYBIND11_OVERRIDE(void,Base,addBuffer,other);
    }
    virtual void merge() override{
        PYBIND11_OVERRIDE(void,Base,merge);
    }
    virtual void update(const Track2D& other) override{
        PYBIND11_OVERRIDE(void,Base,update,other);
    }
    virtual void updateByExtrapolation(const double& dt) override{
        PYBIND11_OVERRIDE(void,Base,updateByExtrapolation,dt);
    }
    virtual Track2D extrapolate(const double& dt) override{
        PYBIND11_OVERRIDE(Track2D,Base,extrapolate,dt);
    }
    virtual Track2D extrapolateTo(const Time& dstTime) override{
        PYBIND11_OVERRIDE(Track2D,Base,extrapolateTo,dstTime);
    }
};

void exportTrack(py::module &m);

ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)

ASRC_PYBIND11_MAKE_OPAQUE(std::vector<::asrc::core::Track3D>);
ASRC_PYBIND11_MAKE_OPAQUE(std::vector<::asrc::core::Track2D>);
ASRC_PYBIND11_MAKE_OPAQUE(std::map<std::string,::asrc::core::Track3D>);
ASRC_PYBIND11_MAKE_OPAQUE(std::map<std::string,::asrc::core::Track2D>);
ASRC_PYBIND11_MAKE_OPAQUE(std::vector<std::pair<::asrc::core::Track3D,bool>>);
ASRC_PYBIND11_MAKE_OPAQUE(std::vector<std::pair<::asrc::core::Track2D,bool>>);
