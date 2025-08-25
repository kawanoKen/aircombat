/**
 * Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
 * @file
 * 
 * @brief ある親座標系基準の相対的な座標系を表すCRS
 */
#pragma once
#include "../Common.h"
#include "CoordinateReferenceSystemBase.h"

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)

class ProjectedCRS;

/**
 * @class DerivedCRS
 * @brief ある親座標系基準の相対的な座標系を表すCRS
 * 
 * @details ISO 19111ではDerivedCRSとして定義されているものに相当する。
 */
ASRC_DECLARE_DERIVED_CLASS_WITH_TRAMPOLINE(DerivedCRS,CoordinateReferenceSystem)
    public:
    using BaseType::BaseType;
    virtual void initialize() override;
    // 内部状態のシリアライゼーション
    virtual void serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full) override;
    virtual bool isValid() const override;
    protected:
    virtual void validateImpl(bool isFirstTime) const override;
    public:
    //座標系の種類
    virtual bool isDerived() const override; //DerivedCRSかどうか(他の座標系から派生したものかどうか)
    virtual bool isProjected() const override; //投影された座標系かどうか(ProjectedCRS又はProjectedCRSをbaseとしたDerivedCRSが該当)
    //親座標系の取得
	virtual std::shared_ptr<CoordinateReferenceSystem> getBaseCRS() const override; //すぐ上位の親座標系。親がいなければ自分自身。
	virtual std::shared_ptr<CoordinateReferenceSystem> getNonDerivedBaseCRS() const override; //最上位の親座標系。親がいなければ自分自身。
	virtual std::shared_ptr<ProjectedCRS> getBaseProjectedCRS() const override; //親座標系を辿ってProjectedCRSに当たった場合、そのProjectedCRS。なければnullptr。自分自身がProjectedCRSの場合は自分自身。
    //中間座標系の取得
	virtual std::shared_ptr<CoordinateReferenceSystem> getIntermediateCRS() const override; //親子関係にない他のCRSとの間で座標変換を行う際に経由する中間座標系。DerivedCRSにとってはbaseCRS若しくはnonDerivedBaseCRS、又はそのintermediateCRS。
    //「高度」の取得
    public:
    using BaseType::getHeight;
    using BaseType::getGeoidHeight;
    virtual double getHeight(const Eigen::Vector3d& location, const Time& time) const override; //楕円体高度(幾何高度)を返す。ellipsoid (geometric) height
    virtual double getGeoidHeight(const Eigen::Vector3d& location, const Time& time) const override; //標高(ジオイド高度)を返す。geoid height (elevation)
    // CRSインスタンス間の変換可否
    public:
    virtual bool isTransformableTo(const std::shared_ptr<CoordinateReferenceSystem>& other) const override;
    virtual bool isTransformableFrom(const std::shared_ptr<CoordinateReferenceSystem>& other) const override;
    // 座標変換
    using BaseType::transformTo;
    using BaseType::transformFrom;
    virtual Coordinate transformTo(const Eigen::Vector3d& value, const Eigen::Vector3d& location, const Time& time, const std::shared_ptr<CoordinateReferenceSystem>& dstCRS, const CoordinateType& coordinateType) const override;
    virtual Coordinate transformFrom(const Eigen::Vector3d& value, const Eigen::Vector3d& location, const Time& time, const std::shared_ptr<CoordinateReferenceSystem>& srcCRS, const CoordinateType& coordinateType) const override;

    //
    // 以下は本クラスで追加定義するメンバ
    //
    protected:
    virtual void setBaseCRS(const std::shared_ptr<CoordinateReferenceSystem>& newBaseCRS) const; //!< 親座標系を変更する。
    public:
    /** @name 親座標系との座標変換
     */
    ///@{
    /**
     * @brief 位置locationで観測された値valueを、親座標系における座標に変換する。
     * 
     * @param [in] value    変換対象の座標
     * @param [in] location valueが観測された位置
     */
    Coordinate transformToBaseCRS(const Coordinate& value,const Coordinate& location) const;

    /**
     * @brief 親座標系における位置locationで観測された値valueを、このCRSにおける座標に変換する。
     * 
     * @param [in] value    変換対象の座標
     * @param [in] location valueが観測された位置
     * 
     * @attention 引数がCoordinateの場合はその値自身がcrsを持っておりこのCRSへの直接変換が可能なのでこの関数はあまり意味がない。
     */
    Coordinate transformFromBaseCRS(const Coordinate& value,const Coordinate& location) const;

    /**
     * @brief 位置locationで観測された値valueを、親座標系における座標に変換する。
     * 
     * @param [in] value    変換対象の座標
     * 
     * @details value自身が保持しているlocationの情報を使用する。
     */
    Coordinate transformToBaseCRS(const Coordinate& value) const;

    /**
     * @brief 親座標系において観測された値valueを、このCRSにおける座標に変換する。
     * 
     * @param [in] value    変換対象の座標
     * 
     * @details value自身が保持しているlocationの情報を使用する。
     * 
     * @attention 引数がCoordinateの場合はその値自身がcrsを持っておりこのCRSへの直接変換が可能なのでこの関数はあまり意味がない。
     */
    Coordinate transformFromBaseCRS(const Coordinate& value) const;

    /**
     * @brief このCRSにおいて位置locationで観測された値valueを、親座標系における座標に変換する
     * @param [in] value          変換対象の座標
     * @param [in] location       valueが観測された位置
     * @param [in] time           valueが観測された時刻
     * @param [in] coordinateType 座標の種類
     * 
     * @details 派生クラスはこれをオーバーライドして実際の座標変換処理を記述すること。
     */
    virtual Coordinate transformToBaseCRS(const Eigen::Vector3d& value, const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType) const=0;

    /**
     * @brief 親座標系において位置locationで観測された値valueを、このCRSにおける座標に変換する
     * @param [in] value          変換対象の座標
     * @param [in] location       valueが観測された位置
     * @param [in] time           valueが観測された時刻
     * @param [in] coordinateType 座標の種類
     * 
     * @details 派生クラスはこれをオーバーライドして実際の座標変換処理を記述すること。
     */
    virtual Coordinate transformFromBaseCRS(const Eigen::Vector3d& value, const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType) const=0;

    /**
     * @brief このCRSにおいて観測された値valueを、親座標系における座標に変換する
     * @param [in] value          変換対象の座標
     * @param [in] time           valueが観測された時刻
     * @param [in] coordinateType 座標の種類
     * 
     * @details 派生クラスはこれをオーバーライドして実際の座標変換処理を記述すること。
     * 
     *          locationは原点(0,0,0)とみなす。
     */
    virtual Coordinate transformToBaseCRS(const Eigen::Vector3d& value, const Time& time, const CoordinateType& coordinateType) const;

    /**
     * @brief 親座標系において位置locationで観測された値valueを、このCRSにおける座標に変換する
     * @param [in] value          変換対象の座標
     * @param [in] time           valueが観測された時刻
     * @param [in] coordinateType 座標の種類
     * 
     * @details 派生クラスはこれをオーバーライドして実際の座標変換処理を記述すること。
     * 
     *          locationは原点(0,0,0)とみなす。
     */
    virtual Coordinate transformFromBaseCRS(const Eigen::Vector3d& value, const Time& time, const CoordinateType& coordinateType) const;
    ///@}

    /** @name 親を辿った最上位の(=Derivedでない)座標系との座標変換
     */
    ///@{
    /**
     * @brief 位置locationで観測された値valueを、親を辿った最上位の座標系における座標に変換する。
     * 
     * @param [in] value    変換対象の座標
     * @param [in] location valueが観測された位置
     */
    Coordinate transformToNonDerivedBaseCRS(const Coordinate& value,const Coordinate& location) const;

    /**
     * @brief 親を辿った最上位の座標系における位置locationで観測された値valueを、このCRSにおける座標に変換する。
     * 
     * @param [in] value    変換対象の座標
     * @param [in] location valueが観測された位置
     * 
     * @attention 引数がCoordinateの場合はその値自身がcrsを持っておりこのCRSへの直接変換が可能なのでこの関数はあまり意味がない。
     */
    Coordinate transformFromNonDerivedBaseCRS(const Coordinate& value,const Coordinate& location) const;

    /**
     * @brief 位置locationで観測された値valueを、親を辿った最上位の座標系における座標に変換する。
     * 
     * @param [in] value    変換対象の座標
     * 
     * @details value自身が保持しているlocationの情報を使用する。
     */
    Coordinate transformToNonDerivedBaseCRS(const Coordinate& value) const;

    /**
     * @brief 親を辿った最上位の座標系において観測された値valueを、このCRSにおける座標に変換する。
     * 
     * @param [in] value    変換対象の座標
     * 
     * @details value自身が保持しているlocationの情報を使用する。
     * 
     * @attention 引数がCoordinateの場合はその値自身がcrsを持っておりこのCRSへの直接変換が可能なのでこの関数はあまり意味がない。
     */
    Coordinate transformFromNonDerivedBaseCRS(const Coordinate& value) const;

    /**
     * @brief このCRSにおいて位置locationで観測された値valueを、親を辿った最上位の座標系における座標に変換する
     * @param [in] value          変換対象の座標
     * @param [in] location       valueが観測された位置
     * @param [in] time           valueが観測された時刻
     * @param [in] coordinateType 座標の種類
     * 
     * @details 派生クラスはこれをオーバーライドして実際の座標変換処理を記述すること。
     */
    virtual Coordinate transformToNonDerivedBaseCRS(const Eigen::Vector3d& value,const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType) const=0;

    /**
     * @brief 親を辿った最上位の座標系において位置locationで観測された値valueを、このCRSにおける座標に変換する
     * @param [in] value          変換対象の座標
     * @param [in] location       valueが観測された位置
     * @param [in] time           valueが観測された時刻
     * @param [in] coordinateType 座標の種類
     * 
     * @details 派生クラスはこれをオーバーライドして実際の座標変換処理を記述すること。
     */
    virtual Coordinate transformFromNonDerivedBaseCRS(const Eigen::Vector3d& value,const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType) const=0;

    /**
     * @brief このCRSにおいて観測された値valueを、親を辿った最上位の座標系における座標に変換する
     * @param [in] value          変換対象の座標
     * @param [in] time           valueが観測された時刻
     * @param [in] coordinateType 座標の種類
     * 
     * @details 派生クラスはこれをオーバーライドして実際の座標変換処理を記述すること。
     * 
     *          locationは原点(0,0,0)とみなす。
     */
    virtual Coordinate transformToNonDerivedBaseCRS(const Eigen::Vector3d& value, const Time& time, const CoordinateType& coordinateType) const;

    /**
     * @brief 親を辿った最上位の座標系において位置locationで観測された値valueを、このCRSにおける座標に変換する
     * @param [in] value          変換対象の座標
     * @param [in] time           valueが観測された時刻
     * @param [in] coordinateType 座標の種類
     * 
     * @details 派生クラスはこれをオーバーライドして実際の座標変換処理を記述すること。
     * 
     *          locationは原点(0,0,0)とみなす。
     */
    virtual Coordinate transformFromNonDerivedBaseCRS(const Eigen::Vector3d& value, const Time& time, const CoordinateType& coordinateType) const;
    ///@}

    /** @name 中間座標系との座標変換
     */
    ///@{
    /**
     * @brief 位置locationで観測された値valueを、中間座標系における座標に変換する。
     * 
     * @param [in] value    変換対象の座標
     * @param [in] location valueが観測された位置
     */
    Coordinate transformToIntermediateCRS(const Coordinate& value,const Coordinate& location) const;

    /**
     * @brief 中間座標系における位置locationで観測された値valueを、このCRSにおける座標に変換する。
     * 
     * @param [in] value    変換対象の座標
     * @param [in] location valueが観測された位置
     * 
     * @attention 引数がCoordinateの場合はその値自身がcrsを持っておりこのCRSへの直接変換が可能なのでこの関数はあまり意味がない。
     */
    Coordinate transformFromIntermediateCRS(const Coordinate& value,const Coordinate& location) const;

    /**
     * @brief 位置locationで観測された値valueを、中間座標系における座標に変換する。
     * 
     * @param [in] value    変換対象の座標
     * 
     * @details value自身が保持しているlocationの情報を使用する。
     */
    Coordinate transformToIntermediateCRS(const Coordinate& value) const;

    /**
     * @brief 中間座標系において観測された値valueを、このCRSにおける座標に変換する。
     * 
     * @param [in] value    変換対象の座標
     * 
     * @details value自身が保持しているlocationの情報を使用する。
     * 
     * @attention 引数がCoordinateの場合はその値自身がcrsを持っておりこのCRSへの直接変換が可能なのでこの関数はあまり意味がない。
     */
    Coordinate transformFromIntermediateCRS(const Coordinate& value) const;

    /**
     * @brief このCRSにおいて位置locationで観測された値valueを、中間座標系における座標に変換する
     * @param [in] value          変換対象の座標
     * @param [in] location       valueが観測された位置
     * @param [in] time           valueが観測された時刻
     * @param [in] coordinateType 座標の種類
     * 
     * @details 派生クラスはこれをオーバーライドして実際の座標変換処理を記述すること。
     */
    virtual Coordinate transformToIntermediateCRS(const Eigen::Vector3d& value,const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType) const=0;

    /**
     * @brief 中間座標系において位置locationで観測された値valueを、このCRSにおける座標に変換する
     * @param [in] value          変換対象の座標
     * @param [in] location       valueが観測された位置
     * @param [in] time           valueが観測された時刻
     * @param [in] coordinateType 座標の種類
     * 
     * @details 派生クラスはこれをオーバーライドして実際の座標変換処理を記述すること。
     */
    virtual Coordinate transformFromIntermediateCRS(const Eigen::Vector3d& value,const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType) const=0;

    /**
     * @brief このCRSにおいて観測された値valueを、中間座標系における座標に変換する
     * @param [in] value          変換対象の座標
     * @param [in] time           valueが観測された時刻
     * @param [in] coordinateType 座標の種類
     * 
     * @details 派生クラスはこれをオーバーライドして実際の座標変換処理を記述すること。
     * 
     *          locationは原点(0,0,0)とみなす。
     */
    virtual Coordinate transformToIntermediateCRS(const Eigen::Vector3d& value, const Time& time, const CoordinateType& coordinateType) const;

    /**
     * @brief 中間座標系において位置locationで観測された値valueを、このCRSにおける座標に変換する
     * @param [in] value          変換対象の座標
     * @param [in] time           valueが観測された時刻
     * @param [in] coordinateType 座標の種類
     * 
     * @details 派生クラスはこれをオーバーライドして実際の座標変換処理を記述すること。
     * 
     *          locationは原点(0,0,0)とみなす。
     */
    virtual Coordinate transformFromIntermediateCRS(const Eigen::Vector3d& value, const Time& time, const CoordinateType& coordinateType) const;
    ///@}
    protected:
    nl::json _base_j; //!< 親座標系を表すjson
    mutable std::shared_ptr<CoordinateReferenceSystem> _base; //!< 親座標系
    mutable std::shared_ptr<CoordinateReferenceSystem> _nonDerivedBase; //!< 親を辿った最上位の(=Derivedでない)座標系
    mutable std::shared_ptr<ProjectedCRS> _baseProjected; //!< 親座標系がProjectedCRSかどうか
};

ASRC_DECLARE_DERIVED_TRAMPOLINE(DerivedCRS)
    //「高度」の取得
    using Base::getHeight;
    using Base::getGeoidHeight;
    virtual double getHeight(const Eigen::Vector3d& location, const Time& time) const override{
        PYBIND11_OVERRIDE(double,Base,getHeight,location,time);
    }
    virtual double getGeoidHeight(const Eigen::Vector3d& location, const Time& time) const override{
        PYBIND11_OVERRIDE(double,Base,getGeoidHeight,location,time);
    }
    // CRSインスタンス間の変換可否
    virtual bool isTransformableTo(const std::shared_ptr<CoordinateReferenceSystem>& other) const override{
        PYBIND11_OVERRIDE(bool,Base,isTransformableTo,other);
    }
    virtual bool isTransformableFrom(const std::shared_ptr<CoordinateReferenceSystem>& other) const override{
        PYBIND11_OVERRIDE(bool,Base,isTransformableFrom,other);
    }
    // 座標変換
    using Base::transformTo;
    using Base::transformFrom;
    virtual Coordinate transformTo(const Eigen::Vector3d& value, const Eigen::Vector3d& location, const Time& time, const std::shared_ptr<CoordinateReferenceSystem>& dstCRS, const CoordinateType& coordinateType) const override{
        PYBIND11_OVERRIDE(Coordinate,Base,transformTo,value,location,time,dstCRS,coordinateType);
    }
    virtual Coordinate transformFrom(const Eigen::Vector3d& value, const Eigen::Vector3d& location, const Time& time, const std::shared_ptr<CoordinateReferenceSystem>& srcCRS, const CoordinateType& coordinateType) const override{
        PYBIND11_OVERRIDE(Coordinate,Base,transformFrom,value,location,time,srcCRS,coordinateType);
    }
    //
    // 以下は本クラスで追加定義するメンバ
    //
    virtual void setBaseCRS(const std::shared_ptr<CoordinateReferenceSystem>& newBaseCRS) const override{
        PYBIND11_OVERRIDE(void,Base,setBaseCRS,newBaseCRS);
    }
    //親座標系の取得
    virtual std::shared_ptr<CoordinateReferenceSystem> getBaseCRS() const override{
        PYBIND11_OVERRIDE(std::shared_ptr<CoordinateReferenceSystem>,Base,getBaseCRS);
    }
    virtual std::shared_ptr<CoordinateReferenceSystem> getNonDerivedBaseCRS() const override{
        PYBIND11_OVERRIDE(std::shared_ptr<CoordinateReferenceSystem>,Base,getNonDerivedBaseCRS);
    }
    //親座標系との座標変換
    using Base::transformToBaseCRS;
    using Base::transformFromBaseCRS;
    using Base::transformToNonDerivedBaseCRS;
    using Base::transformFromNonDerivedBaseCRS;
    using Base::transformToIntermediateCRS;
    using Base::transformFromIntermediateCRS;
    virtual Coordinate transformToBaseCRS(const Eigen::Vector3d& value,const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType) const override{
        PYBIND11_OVERRIDE_PURE(Coordinate,Base,transformToBaseCRS,value,location,time,coordinateType);
    }
    virtual Coordinate transformFromBaseCRS(const Eigen::Vector3d& value,const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType) const override{
        PYBIND11_OVERRIDE_PURE(Coordinate,Base,transformFromBaseCRS,value,location,time,coordinateType);
    }
    virtual Coordinate transformToBaseCRS(const Eigen::Vector3d& value, const Time& time, const CoordinateType& coordinateType) const override{
        PYBIND11_OVERRIDE(Coordinate,Base,transformToBaseCRS,value,time,coordinateType);
    }
    virtual Coordinate transformFromBaseCRS(const Eigen::Vector3d& value, const Time& time, const CoordinateType& coordinateType) const override{
        PYBIND11_OVERRIDE(Coordinate,Base,transformFromBaseCRS,value,time,coordinateType);
    }
    virtual Coordinate transformToNonDerivedBaseCRS(const Eigen::Vector3d& value,const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType) const override{
        PYBIND11_OVERRIDE_PURE(Coordinate,Base,transformToNonDerivedBaseCRS,value,location,time,coordinateType);
    }
    virtual Coordinate transformFromNonDerivedBaseCRS(const Eigen::Vector3d& value,const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType) const override{
        PYBIND11_OVERRIDE_PURE(Coordinate,Base,transformFromNonDerivedBaseCRS,value,location,time,coordinateType);
    }
    virtual Coordinate transformToNonDerivedBaseCRS(const Eigen::Vector3d& value, const Time& time, const CoordinateType& coordinateType) const override{
        PYBIND11_OVERRIDE(Coordinate,Base,transformToNonDerivedBaseCRS,value,time,coordinateType);
    }
    virtual Coordinate transformFromNonDerivedBaseCRS(const Eigen::Vector3d& value, const Time& time, const CoordinateType& coordinateType) const override{
        PYBIND11_OVERRIDE(Coordinate,Base,transformFromNonDerivedBaseCRS,value,time,coordinateType);
    }
    virtual Coordinate transformToIntermediateCRS(const Eigen::Vector3d& value,const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType) const override{
        PYBIND11_OVERRIDE_PURE(Coordinate,Base,transformToIntermediateCRS,value,location,time,coordinateType);
    }
    virtual Coordinate transformFromIntermediateCRS(const Eigen::Vector3d& value,const Eigen::Vector3d& location, const Time& time, const CoordinateType& coordinateType) const override{
        PYBIND11_OVERRIDE_PURE(Coordinate,Base,transformFromIntermediateCRS,value,location,time,coordinateType);
    }
    virtual Coordinate transformToIntermediateCRS(const Eigen::Vector3d& value, const Time& time, const CoordinateType& coordinateType) const override{
        PYBIND11_OVERRIDE(Coordinate,Base,transformToIntermediateCRS,value,time,coordinateType);
    }
    virtual Coordinate transformFromIntermediateCRS(const Eigen::Vector3d& value, const Time& time, const CoordinateType& coordinateType) const override{
        PYBIND11_OVERRIDE(Coordinate,Base,transformFromIntermediateCRS,value,time,coordinateType);
    }
};

void exportDerivedCRS(py::module &m, const std::shared_ptr<asrc::core::FactoryHelper>& factoryHelper);

ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
