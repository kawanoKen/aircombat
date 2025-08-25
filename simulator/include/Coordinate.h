/**
 * Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
 * @file
 * 
 * @brief ある座標系上の特定時刻における座標値を管理するクラス
 */
#pragma once
#include "Common.h"
#include "TimeSystem.h"
#include <pybind11/pybind11.h>
#include <nlohmann/json.hpp>
#include <Eigen/Core>
#include "Pythonable.h"
namespace py=pybind11;
namespace nl=nlohmann;

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)

/**
 * @brief 座標の種類を表す列挙体。
 * 
 * @attention 座標系が互いに固定されていない場合の座標変換は、この種類に基づいて異なる変換式が使用される。
 */
enum class CoordinateType{
    POSITION_ABS, //!< 絶対位置
    POSITION_REL, //!< 相対位置
    DIRECTION, //!< 方向
    VELOCITY, //!< 並進速度
    ANGULAR_VELOCITY, //!< 角速度
};
DEFINE_SERIALIZE_ENUM_AS_STRING(CoordinateType)

class CoordinateReferenceSystem;
/**
 * @brief ある座標系上の特定時刻における座標値を管理するクラス
 */
ASRC_DECLARE_NON_POLYMORPHIC_DATA_CLASS(Coordinate)
    protected:
    std::shared_ptr<CoordinateReferenceSystem> crs; //!< [For internal use] 親となる座標系
    Time time; //!< 時刻
    Eigen::Vector3d value; //!< 座標値
    Eigen::Vector3d location; //!< この座標値を持つ物体が観測された点の座標(絶対位置)。座標変換式が場所に依存するもの(回転座標系上の速度等)を取り扱うための変数。
    CoordinateType type; //!< 座標の種類
    public:
    Coordinate();
    /**
     * @brief フルコンストラクタ
     * @param [in] value_    座標値
     * @param [in] location_ 観測された点の座標値(絶対位置)
     * @param [in] crs_      座標系
     * @param [in] time_     時刻
     * @param [in] type_     座標の種類
     */
    Coordinate(const Eigen::Vector3d& value_, const Eigen::Vector3d& location_, const std::shared_ptr<CoordinateReferenceSystem>& crs_, const Time& time_, const CoordinateType& type_);
    /**
     * @brief locationを省略したコンストラクタ。原点上にlocationがあると解釈する。
     */
    Coordinate(const Eigen::Vector3d& value_, const std::shared_ptr<CoordinateReferenceSystem>& crs_, const Time& time_, const CoordinateType& type_);
    Coordinate(const nl::json& j_); //!< jsonから値を設定する。

    ///@name getter,setter
    ///@{
    std::shared_ptr<CoordinateReferenceSystem> getCRS() const noexcept;
    Coordinate& setCRS(const std::shared_ptr<CoordinateReferenceSystem>& newCRS,bool transform) &; //!< CRSを再設定する。座標変換の有無を第2引数で指定する。
    Coordinate setCRS(const std::shared_ptr<CoordinateReferenceSystem>& newCRS,bool transform) &&; //!< CRSを再設定する。座標変換の有無を第2引数で指定する。
    Time getTime() const; //!< 時刻を返す。
    Coordinate& setTime(const Time& time_) &; //!< 時刻を再設定する。変換は行わない。
    Coordinate setTime(const Time& time_) &&; //!< 時刻を再設定する。変換は行わない。
    CoordinateType getType() const; //!< 座標の種類を返す。
    Coordinate& setType(const CoordinateType& type_) &; //!< 座標の種類を再設定する。変換は行わない。
    Coordinate setType(const CoordinateType& type_) &&; //!< 座標の種類を再設定する。変換は行わない。
    Coordinate asTypeOf(const CoordinateType& newType) const; //!< 他の種類の座標として再解釈したものを新インスタンスとして得る。変換は行わない。座標系の運動を考慮せずに回転と原点の並行移動のみ行いたい場合に用いる。
    Eigen::Vector3d operator()() const noexcept; //!< 座標値を返す。
    double operator()(const Eigen::Index& i) const; //!< 座標値のi成分を返す。
    Eigen::Vector3d operator()(const std::shared_ptr<CoordinateReferenceSystem>& newCRS) const; //!< 座標値をnewCRSにおける座標値に変換し、Eigen::Vector3dとして返す。this->transformTo(newCRS)()と等価。
    Eigen::Vector3d getValue() const noexcept; //!< 座標値を返す。
    double getValue(const Eigen::Index& i) const; //!< valueのi成分を返す。
    Eigen::Vector3d getValue(const std::shared_ptr<CoordinateReferenceSystem>& newCRS) const; //!< 座標値をnewCRSにおける座標値に変換し、Eigen::Vector3dとして返す。this->transformTo(newCRS)()と等価。
    Coordinate& setValue(const Eigen::Vector3d& value_) &; //!< 座標値を再設定する。変換は行わない。
    Coordinate setValue(const Eigen::Vector3d& value_) &&; //!< 座標値を再設定する。変換は行わない。
    Eigen::Vector3d getLocation() const noexcept; //!< 観測された点の座標値(絶対位置)を返す。
    double getLocation(const Eigen::Index& i) const; //!< 観測された点の座標値(絶対位置)のi成分を返す。
    Eigen::Vector3d getLocation(const std::shared_ptr<CoordinateReferenceSystem>& newCRS) const; //!< 観測された点の座標値(絶対位置)をnewCRSにおける座標値に変換し、Eigen::Vector3dとして返す。
    Coordinate& setLocation(const Eigen::Vector3d& location_) &; //!< 観測された点の座標値(絶対位置)を再設定する。変換は行わない。
    Coordinate setLocation(const Eigen::Vector3d& location_) &&; //!< 観測された点の座標値(絶対位置)を再設定する。変換は行わない。
    ///@}

    ///@name ノルム・単位ベクトル
    ///@{
    double norm() const; //!< 自身の座標値のノルムを返す。
    void normalize(); //!< 自身の座標値を正規化する。
    Coordinate normalized() const; //!< 正規化された座標値を持つ新たな Coordinate を返す。
    ///@}

    ///@name 座標変換
    ///@{
    Coordinate transformTo(const std::shared_ptr<CoordinateReferenceSystem>& newCRS) const; //!< 他のCRSにおける座標値を新インスタンスとして得る。
    ///@}

    ///@name 座標軸の変換 (axis conversion in the current crs)
    ///@{
    Eigen::Vector3d toCartesian() const; //!< cartesianな座標軸での値に変換されたvalueを返す
    Eigen::Vector3d toSpherical() const; //!< sphericalな座標軸での値に変換されたvalueを返す
    Eigen::Vector3d toEllipsoidal() const; //!< ellipsoidalな座標軸での値に変換されたvalueを返す
    ///@}

    ///@name シリアライゼーション
    ///@{
    template<class Archive>
    void serialize(Archive & archive) {
        archive(
            CEREAL_NVP(value),
            CEREAL_NVP(location),
            CEREAL_NVP(crs),
            CEREAL_NVP(time),
            CEREAL_NVP(type)
        );
    }
    ///@}
};

void exportCoordinate(py::module &m);

ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
