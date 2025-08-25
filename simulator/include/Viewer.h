/**
 * Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
 * @file
 * 
 * @brief エピソードの可視化を行うためのクラス
 */
#pragma once
#include <iostream>
#include <vector>
#include <map>
#include <pybind11/pybind11.h>
#include "Callback.h"
#include "GUIDataFrameCreator.h"

namespace py=pybind11;
namespace nl=nlohmann;

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)

/**
 * @class Viewer
 * @brief エピソードの可視化を行うためのクラス
 * 
 * @details 一つのエピソード中に一つしか存在できない。
 */
ASRC_DECLARE_DERIVED_CLASS_WITH_TRAMPOLINE(Viewer,Callback)
    public:
    static const std::string baseName; // baseNameを上書き
    bool isValid;
    //constructors & destructor
    using BaseType::BaseType;
    virtual ~Viewer();
    //functions
    virtual void initialize() override;
    virtual void serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full) override;
    virtual void validate() override;
    virtual void display(); //!< 描画処理を記述する。 onInnerStepEnd() の中で呼ばれる。
    virtual void close(); //!< 終了処理を記述する。デストラクタの中で呼ばれる。
    virtual void onEpisodeBegin() override;
    virtual void onInnerStepBegin() override;
    virtual void onInnerStepEnd() override;
    virtual std::shared_ptr<GUIDataFrameCreator> getDataFrameCreator(); //!< 描画用データフレーム生成用オブジェクトを返す。
};
ASRC_DECLARE_DERIVED_TRAMPOLINE(Viewer)
    virtual void display() override{
        PYBIND11_OVERRIDE(void,Base,display);
    }
    virtual void close() override{
        PYBIND11_OVERRIDE(void,Base,close);
    }
    virtual std::shared_ptr<GUIDataFrameCreator> getDataFrameCreator() override{
        PYBIND11_OVERRIDE(std::shared_ptr<GUIDataFrameCreator>,Base,getDataFrameCreator);
    }
};
void exportViewer(py::module &m, const std::shared_ptr<asrc::core::FactoryHelper>& factoryHelper);

ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
