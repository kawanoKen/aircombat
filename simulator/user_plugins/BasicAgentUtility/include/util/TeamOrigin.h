// Copyright (c) 2021-2023 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#pragma once
#include "../Common.h"

ASRC_PLUGIN_NAMESPACE_BEGIN

namespace util{

//陣営座標系(進行方向が+x方向となるようにz軸まわりに回転させ、防衛ライン中央が原点となるように平行移動させた座標系)を表すクラス。
//MotionStateを使用しても良いがクォータニオンを経由することで浮動小数点演算に起因する余分な誤差が生じるため、もし可能な限り対称性を求めるのであればこの例のように符号反転で済ませたほうが良い。
//ただし、機体運動等も含めると全ての状態量に対して厳密に対称なシミュレーションとはならないため、ある程度の誤差は生じる。
class PYBIND11_EXPORT TeamOrigin{
    public:
    bool isEastSider;
    Eigen::Vector3d pos;//原点
    TeamOrigin();
    TeamOrigin(bool isEastSider_,double dLine);
    ~TeamOrigin();
    Eigen::Vector3d relBtoP(const Eigen::Vector3d& v) const;//陣営座標系⇛慣性座標系
    Eigen::Vector3d relPtoB(const Eigen::Vector3d& v) const;//慣性座標系⇛陣営座標系
    Eigen::Vector3d absBtoP(const Eigen::Vector3d& v) const;//陣営座標系⇛慣性座標系
    Eigen::Vector3d absPtoB(const Eigen::Vector3d& v) const;//慣性座標系⇛陣営座標系
};

void exportTeamOrigin(py::module &m);

}

ASRC_PLUGIN_NAMESPACE_END
