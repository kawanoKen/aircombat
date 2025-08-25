// Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#pragma once
#include <pybind11/pybind11.h>
#include "Utility.h"
#include "EntityIdentifier.h"
namespace py=pybind11;
namespace nl=nlohmann;

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)

class SimulationManagerAccessorForCallback;

ASRC_INLINE_NAMESPACE_BEGIN(util)

ASRC_DECLARE_BASE_REF_CLASS(GUIDataFrameCreator)
    /*GUI描画のために必要なデータフレームを生成するための基底クラス。
    SimulationManager内部で動くViewerクラスであれば直接各インスタンスにアクセスしても差し支えないが、
    ログデータや別プロセスから描画する際に描画処理を再利用することを念頭に置いて、
    nl::json型でデータフレームを生成することを標準仕様とする。
    //描画用なので各データの座標系はmanager->getRootCRS()に統一しておく。必要に応じてViewer側で変換する。
    */
    public:
    //constructors & destructor
    GUIDataFrameCreator();
    //functions
    virtual nl::json makeHeader(const std::shared_ptr<SimulationManagerAccessorForCallback>& manager);
    virtual nl::json makeFrame(const std::shared_ptr<SimulationManagerAccessorForCallback>& manager);
    virtual void serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full);
    std::map<EntityIdentifier,std::uint64_t> serializedAt;
    std::uint64_t tickAtLastSerialization;
};
ASRC_DECLARE_BASE_REF_TRAMPOLINE(GUIDataFrameCreator)
    virtual nl::json makeFrame(const std::shared_ptr<SimulationManagerAccessorForCallback>& manager) override{
        PYBIND11_OVERRIDE(nl::json,Base,makeFrame,manager);
    }
};
void exportGUIDataFrameCreator(py::module &m);

ASRC_NAMESPACE_END(util)
ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
