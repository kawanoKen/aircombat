// Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#pragma once
#include "Common.h"
#include <vector>
#include <map>
#include <pybind11/pybind11.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <ASRCAISim1/MathUtility.h>
#include <ASRCAISim1/Utility.h>
#include <ASRCAISim1/Reward.h>
#include "BasicAACRuler01.h"
namespace py=pybind11;
namespace nl=nlohmann;

ASRC_PLUGIN_NAMESPACE_BEGIN

ASRC_DECLARE_DERIVED_CLASS_WITH_TRAMPOLINE(BasicAACReward01,asrc::core::TeamReward)
    public:
    /*基本的な空対空戦闘のルールの得点要素に対応した基本報酬のクラス。
        実装上の仕様はR4年度コンテストで使用されたものと同一であり、パラメータ設定によってR5年度コンテストのルールに対応した報酬を表現できる。
        単に得点をそのまま報酬として扱いたい場合はScoreRewardクラスを使用してもよいが、必ずしも得点と報酬は連動すべきとは限らない。
        また、Rulerの一部の得点計算は戦闘終了時点まで保留されることがある一方で、報酬は遅延させずに与えたいということも想定される。
        そのため、陣営ごとの即時報酬として計算するクラスのひな形をサンプルとして実装するものである。
        なお、各得点増減要因の発生時の報酬を変化させたい場合は、modelConfigの以下のパラメータを設定することで実現可能である。（デフォルトは全て0またはtrueであり、得点の値と同等となる。）
        1. rElim：相手が全滅した際の追加報酬
        2. rElimE：自陣営が全滅した際の追加報酬
        3. rBreakRatio：相手の防衛ラインを突破した際の得点を報酬として与える際の倍率(1+rBreakRatio倍)
        4. rBreak：相手の防衛ラインを突破した際の追加報酬
        5. rBreakE：相手に防衛ラインを突破された際の追加報酬
        6. adjustBreakEnd (bool)：Rulerが進出度に応じた得点をしない条件で終了した際に進出度に応じた報酬を無効化するかどうか
        7. rTimeup：時間切れとなった際の追加報酬
        8. rDisq：自陣営がペナルティによる敗北条件を満たした際の追加報酬
        9. rDisqE：相手がペナルティによる敗北条件を満たした際の追加報酬
        10. rHitRatio：相手を撃墜した際の得点を報酬として与える際の倍率(1+rHitRatio倍)
        11. rHit：相手を撃墜した際の追加報酬
        12. rHitE：相手に自陣営の機体が撃墜された際の追加報酬
        13. rAdvRatio：進出度合いに応じた得点(の変化率)を報酬として与える際の倍率(1+rAdvRatio倍)
        14. acceptNegativeAdv：相手陣営の方が進出度合いが大きい時も負数として報酬化するかどうか(ゼロサム化を行う場合は意味なし)
        15. rCrashRatio：自陣営の機体が墜落した際の得点を報酬として与える際の倍率(1+rCrashRatio倍)
        16. rCrash：自陣営の機体が墜落した際の追加報酬
        17. rCrashE：相手が墜落した際の追加報酬
        18. rAliveRatio：自陣営の機体が得た生存点を報酬として与える際の倍率(1+rAliveRatio倍)
        19. rAlive: 自陣営の機体が生存点を得た際の追加報酬
        20. rAliveE：相手が生存点を得た際の追加報酬
        18. rOutRatio：場外ペナルティによる得点を報酬として与える際の倍率(1+rOutRatio倍)
        19. adjustZerosum：相手陣営の得点を減算してゼロサム化するかどうか
        20. rHitPerAircraft：BasicAACRuler01のscorePerAircraftと同様の発想で、rHitの値を「1機あたりの増減量」と「陣営全体での総増減量」のどちらとして解釈するかのフラグ。
        21. rHitEPerAircraft：BasicAACRuler01のscorePerAircraftと同様の発想で、rHitEの値を「1機あたりの増減量」と「陣営全体での総増減量」のどちらとして解釈するかのフラグ。
        22. rCrashPerAircraft：BasicAACRuler01のscorePerAircraftと同様の発想で、rHitCrashの値を「1機あたりの増減量」と「陣営全体での総増減量」のどちらとして解釈するかのフラグ。
        23. rCrashEPerAircraft：BasicAACRuler01のscorePerAircraftと同様の発想で、rHitCrashEの値を「1機あたりの増減量」と「陣営全体での総増減量」のどちらとして解釈するかのフラグ。
        20. rAlivePerAircraft：BasicAACRuler01のscorePerAircraftと同様の発想で、rAliveの値を「1機あたりの増減量」と「陣営全体での総増減量」のどちらとして解釈するかのフラグ。
        21. rAliveEPerAircraft：BasicAACRuler01のscorePerAircraftと同様の発想で、rAliveEの値を「1機あたりの増減量」と「陣営全体での総増減量」のどちらとして解釈するかのフラグ。
	*/
    bool debug;
    double rElim,rElimE,rBreakRatio,rBreak,rBreakE,rTimeup,rDisq,rDisqE,rHitRatio,rAdvRatio,rCrashRatio,rAliveRatio,rOutRatio;
    std::map<std::string,double> rHit,rHitE,rCrash,rCrashE,rAlive,rAliveE;
    std::map<std::string,std::map<std::string,double>> rHitScale,rHitEScale,rCrashScale,rCrashEScale,rAliveScale,rAliveEScale;
    bool rHitPerAircraft,rHitEPerAircraft,rCrashPerAircraft,rCrashEPerAircraft,rAlivePerAircraft,rAliveEPerAircraft;
    bool adjustBreakEnd,acceptNegativeAdv,adjustZerosum;
    std::shared_ptr<BasicAACRuler01> ruler;
    std::map<std::string,std::string> opponentName;
    std::map<std::string,double> advPrev;
    std::map<std::string,double> advOffset;
    std::map<std::string,double> breakTime;
    std::map<std::string,double> disqTime;
    std::map<std::string,double> eliminatedTime;
    //constructors & destructor
    using BaseType::BaseType;
    //functions
    virtual void initialize() override;
    virtual void serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full) override;
    void debugPrint(const std::string& reason,const std::string& team,double value);
    virtual void validate() override;
    virtual void onEpisodeBegin() override;
    virtual void onInnerStepEnd() override;
    virtual void onStepEnd() override;
    virtual double getRHit(const std::string& team,const std::string& modelName) const;
    virtual double getRHitE(const std::string& team,const std::string& modelName) const;
    virtual double getRCrash(const std::string& team,const std::string& modelName) const;
    virtual double getRCrashE(const std::string& team,const std::string& modelName) const;
    virtual double getRAlive(const std::string& team,const std::string& modelName) const;
    virtual double getRAliveE(const std::string& team,const std::string& modelName) const;
};
ASRC_DECLARE_DERIVED_TRAMPOLINE(BasicAACReward01)
    virtual double getRHit(const std::string& team,const std::string& modelName) const override{
        PYBIND11_OVERRIDE(double,Base,getRHit,team,modelName);
    }
    virtual double getRHitE(const std::string& team,const std::string& modelName) const override{
        PYBIND11_OVERRIDE(double,Base,getRHitE,team,modelName);
    }
    virtual double getRCrash(const std::string& team,const std::string& modelName) const override{
        PYBIND11_OVERRIDE(double,Base,getRCrash,team,modelName);
    }
    virtual double getRCrashE(const std::string& team,const std::string& modelName) const override{
        PYBIND11_OVERRIDE(double,Base,getRCrashE,team,modelName);
    }
};

void exportBasicAACReward01(py::module &m, const std::shared_ptr<asrc::core::FactoryHelper>& factoryHelper);

ASRC_PLUGIN_NAMESPACE_END
