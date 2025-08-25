/**
 * Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
 * @file
 * 
 * @brief 時刻系の取り扱いに関するもの。UTC時刻以外の時刻系も重要なためそれらの変換を含めて実装する。
 */
#pragma once
#include "Common.h"
#include "Utility.h"
#include <magic_enum/magic_enum.hpp>
#include <iomanip>
#include <ios>
#include <iostream>
#include <sstream>

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)

class SimulationManagerBase;

/**
 * @brief 時刻系を識別するための列挙体
 */
enum class PYBIND11_EXPORT TimeSystem{
    TT, //!< Terrestrial Time、地球時 : 天文学で用いられる基準の時刻系。
    TAI, //!< International Atomic Time、国際原子時 : 原子時計により維持されている時刻系。
    UT1, //!< Universal Time、世界時 : 地球の自転に対応した時刻系。ECI-ECEFの変換がUT1ベースで定義されるので重要。
    UTC, //!< Coordinated Universal Time、協定世界時 : 日常的に使用される時刻系。各国の標準時はUTCからの差で定められている。
    GPS, //!< GPS衛星が使用する時刻系。TAIより19秒遅い。
    JY, //!< Julian Year、ユリウス年
    JC, //!< Julian Century、ユリウス世紀
    INVALID, //!< 無効な時刻を表す。
};
DEFINE_SERIALIZE_ENUM_AS_STRING(TimeSystem)

inline std::ostream& operator<<(std::ostream& os, const TimeSystem& ts){
    os << magic_enum::enum_name(ts);
    return os;
}


struct Epoch;
/**
 * @class Time
 * @brief 時刻系の情報と合わせて時刻情報を表現するための構造体。
 * 
 * @details ECI(Earth Centered Inertial)座標系は2000-01-01T12:00:00 TTを基準とされていることが多いので、
 *          この構造体の内部では各時刻系における2000-01-01T12:00:00からの経過秒数として管理する。(ただし、JYは年、JCは世紀単位。)
 * 
 *          時刻と時間の混同をなるべく防止するため、double等の数値型との暗黙の変換は認めない。
 */
ASRC_DECLARE_NON_POLYMORPHIC_DATA_CLASS(Time)
   TimeSystem timeSystem; //!< 時刻系(省略時はTT)
   double value; //!< timeSysyem における2000-01-01T12:00:00からの経過秒数(JYは年、JCは世紀単位。)
    Time(); //!< デフォルトコンストラクタは無効な時刻を表す。
    Time(const double& t, const TimeSystem& ts); //!< 2000-01-01T12:00:00からの経過秒数で直接指定する。
    Time(const std::string& datetime, const TimeSystem& ts); //!< ISO8601相当のタイムゾーン無し版(YYYYMMDDThhmmss.SSS又はYYYY-MM-DDThh:mm:ss.SSS)の文字列で指定する。
    Time(const Epoch& ep); //!< Epoch と同じ時刻に設定する。
    Time(const nl::ordered_json& j); //!< jsonから値を設定する。
    explicit operator bool() const; //!< 有効な時刻を表している場合にtrueを返す。
    bool isValid() const; //!< 有効な時刻を表している場合にtrueを返す。
    void operator+=(const double& delta); //!< delta 秒加算する。
    void operator-=(const double& delta); //!< delta 秒減算する。
    Time to(const TimeSystem& ts) const; //!< 指定した時刻系での時刻に変換する。
    Time toTT() const; //!< TT時刻に変換する。
    Time toTAI() const; //!< TAI時刻に変換する。
    Time toUT1() const; //!< UT1時刻に変換する。
    Time toUTC() const; //!< UTC時刻に変換する。
    Time toGPS() const; //!< GPS時刻に変換する。
    Time toJY() const; //!< ユリウス年での表現に変換する。
    Time toJC() const; //!< ユリウス世紀での表現に変換する。
    static double parseDatetime(const std::string& datetime); //!< ISO8601相当のタイムゾーン無し版(YYYYMMDDThhmmss.SSS又はYYYY-MM-DDThh:mm:ss.SSS)の文字列を2000-01-01T12:00:00からの経過秒数に変換する。閏秒は無視する。
    static int days_from_civil(int y,int m, int d); //!< 日付(y,m,d)を与え、1970-01-01からの経過日数を返す。 https://howardhinnant.github.io/date_algorithm.html より。
    static std::tuple<int,int,int> civil_from_days(int z); //!< 1970-01-01からの経過日数を与え、日付(y,m,d)を返す。 https://howardhinnant.github.io/date_algorithm.html より。
    std::string toString() const; //!< ISO8601相当のタイムゾーン無し版(YYYY-MM-DDThh:mm:ss.SSS)の文字列表現を返す。
    /**
     * @brief シリアライズ処理
     */
    template<class Archive>
    void save(Archive & archive) const{
        archive(
            CEREAL_NVP(timeSystem),
            CEREAL_NVP(value)
        );
    }
    /**
     * @brief デシリアライズ処理
     */
    template<class Archive>
    void load(Archive & archive) {
        if constexpr (traits::same_archive_as<Archive,util::NLJSONInputArchive>){
            const nl::ordered_json& j=archive.getCurrentNode();
            if(j.is_number()){
                // for backward compatibility
                timeSystem=TimeSystem::TT;
                value=j;
                return;
            }else if(!j.is_object()){
                throw std::runtime_error("Time only accepts json object with a value and its timeSystem.");
            }
        }
        if(!util::try_load(archive,"timeSystem",timeSystem)){
            timeSystem=TimeSystem::TT;
        }
        if(!util::try_load(archive,"value",value)){
            value=0;
        }
    }
};

/**
 * @brief Time を ISO8601相当のタイムゾーン無し版(YYYY-MM-DDThh:mm:ss.SSS)の文字列表現に変換する
 */
inline std::ostream& operator<<(std::ostream& os, const Time& t){
    int days = (t.value + 12*3600)/86400;
    auto [year, month, day] = Time::civil_from_days(days + Time::days_from_civil(2000,1,1));
    double rem = t.value + 12*3600 - days*86400;
    int hour = rem/3600;
    rem -= hour*3600;
    int minute = rem/60;
    rem -= minute*60;
    int second = rem;
    rem -= second;
    int millisecond = rem*1000.0;
    auto prev = os.flags();
    os << std::right << std::setfill('0') << std::noshowpos << std::noshowpoint;
    os << std::setw(4) << year;
    os << '-' << std::setw(2) << month;
    os << '-' << std::setw(2) << day;
    os << 'T' << std::setw(2) << hour;
    os << ':' << std::setw(2) << minute;
    os << ':' << std::setw(2) << second;
    os << '.' << std::setw(3) << millisecond;
    os.setf(prev);
    os << " in " << t.timeSystem;
    return os;
}

/**
 * @brief 左辺の時刻に右辺の秒数を加算した時刻を返す。
 */
inline Time operator+(const Time& lhs,const double& rhs) {
    return std::move(Time(lhs.value + rhs,lhs.timeSystem));
}
/**
 * @brief 右辺の時刻に左辺の秒数を加算した時刻を返す。
 */
inline Time operator+(const double& lhs,const Time& rhs) {
    return std::move(Time(lhs + rhs.value,rhs.timeSystem));
}
/**
 * @brief 左辺の時刻から右辺の秒数を減算した時刻を返す。
 */
inline Time operator-(const Time& lhs,const double& rhs) {
    return std::move(Time(lhs.value - rhs,lhs.timeSystem));
}
/**
 * @brief 左辺の時刻から右辺の時刻を引いた秒数を返す。
 */
inline double operator-(const Time& lhs,const Time& rhs) {
    assert(lhs.timeSystem == rhs.timeSystem);
    return lhs.value - rhs.value;
}
/**
 * @brief 左辺の時刻と右辺の時刻が同一かどうかを返す。時刻系が異なる場合はTTに変換して揃えて比較する。
 */
inline bool operator==(const Time& lhs,const Time& rhs) {
    if(lhs.timeSystem == rhs.timeSystem){
        return lhs.value == rhs.value;
    }else{
        return lhs.toTT() == rhs.toTT();
    }
}
/**
 * @brief 左辺の時刻と右辺の時刻が異なるかどうかを返す。時刻系が異なる場合はTTに変換して揃えて比較する。
 */
inline bool operator!=(const Time& lhs,const Time& rhs) {
    if(lhs.timeSystem == rhs.timeSystem){
        return lhs.value != rhs.value;
    }else{
        return lhs.toTT() != rhs.toTT();
    }
}
/**
 * @brief 左辺の時刻が右辺の時刻がより過去かどうかを返す。時刻系が異なる場合はTTに変換して揃えて比較する。
 */
inline bool operator<(const Time& lhs,const Time& rhs) {
    if(lhs.timeSystem == rhs.timeSystem){
        return lhs.value < rhs.value;
    }else{
        return lhs.toTT() < rhs.toTT();
    }
}
/**
 * @brief 左辺の時刻が右辺の時刻がより未来かどうかを返す。時刻系が異なる場合はTTに変換して揃えて比較する。
 */
inline bool operator>(const Time& lhs,const Time& rhs) {
    if(lhs.timeSystem == rhs.timeSystem){
        return lhs.value > rhs.value;
    }else{
        return lhs.toTT() > rhs.toTT();
    }
}
/**
 * @brief 左辺の時刻が右辺の時刻以前かどうかを返す。時刻系が異なる場合はTTに変換して揃えて比較する。
 */
inline bool operator<=(const Time& lhs,const Time& rhs) {
    if(lhs.timeSystem == rhs.timeSystem){
        return !(lhs.value > rhs.value);
    }else{
        return lhs.toTT() <= rhs.toTT();
    }
}
/**
 * @brief 左辺の時刻が右辺の時刻以降かどうかを返す。時刻系が異なる場合はTTに変換して揃えて比較する。
 */
inline bool operator>=(const Time& lhs,const Time& rhs) {
    if(lhs.timeSystem == rhs.timeSystem){
        return !(lhs.value < rhs.value);
    }else{
        return lhs.toTT() >= rhs.toTT();
    }
}

/**
 * @class Epoch
 * @brief シミュレーション時刻の原点を表現するための構造体。時刻系間の変換も管理する。
 * 
 * @details  SimulationManageのconfigにおいて"/Manager/Epoch"として与える。
 *          シミュレーション中の時刻系指定のない実数型変数はその原点からの経過秒数を指すものとする。
 * 
 * @code {.json}
 *  {
 *      "Manager": {
 *          "Epoch": {
 *              "timeSystem": "TT",
 *              "datetime": "2000-01-01T12-00-00", // ISO8601相当のタイムゾーン無し版(YYYYMMDDThhmmss.SSS又はYYYY-MM-DDThh:mm:ss.SSS)
 *              "delta_tt_tai": 32.184, //optional
 *              "delta_tt_ut1": 67.6439, //optional
 *              "delta_ut1_utc": 0.0, //optional
 *              "delta_tai_gpu": 19.0 //optional
 *          }
 *      }
 *  }
 * @endcode
 */
ASRC_DECLARE_NON_POLYMORPHIC_DATA_CLASS(Epoch)
    friend class SimulationManagerBase; //!< deltaはシミュレーション実行管理クラスからのみ変更可能とする。
    TimeSystem givenTS; //!< インスタンス生成時に指定された時刻系
    std::string givenDatetime; //!< インスタンス生成時に指定されたISO8601相当のタイムゾーン無し版(YYYYMMDDThhmmss.SSS又はYYYY-MM-DDThh:mm:ss.SSS)の文字列表現
    Time origin; //!< TT換算での原点時刻
    Epoch();
    Epoch(const std::string& datetime, const TimeSystem& ts); //!< ISO8601相当のタイムゾーン無し版(YYYYMMDDThhmmss.SSS又はYYYY-MM-DDThh:mm:ss.SSS)の文字列で指定する。
    Epoch(const nl::ordered_json& j); //!< jsonから値を設定する。deltaの情報も含まれていた場合はそれも反映する。
    static Time to(const Time& t,const TimeSystem& ts); //!< Time の時刻系を変換する。
    static Time toTT(const Time& t); //!< Time の時刻系を TimeSystem::TT に変換する。
    static Time toTAI(const Time& t); //!< Time の時刻系を TimeSystem::TAI に変換する。
    static Time toUT1(const Time& t); //!< Time の時刻系を TimeSystem::UT1 に変換する。
    static Time toUTC(const Time& t); //!< Time の時刻系を TimeSystem::UTC に変換する。
    static Time toGPS(const Time& t); //!< Time の時刻系を TimeSystem::GPS に変換する。
    static Time toJY(const Time& t); //!< Time の時刻系を TimeSystem::JY に変換する。
    static Time toJC(const Time& t); //!< Time の時刻系を TimeSystem::JC に変換する。
    double delta(const Time& t) const; //!< 秒(TT)単位でのoriginからの経過時間
    double deltaYear(const Time& t) const; //!< ユリウス年単位でのoriginからの経過時間
    double deltaCentury(const Time& t) const; //!< ユリウス世紀数単位でのoriginからの経過時間
    /**
     * @brief deltaのシリアライズ処理
     */
    template<class Archive>
    static void saveDeltas(Archive & archive){
        archive(
            CEREAL_NVP(delta_tt_tai),
            CEREAL_NVP(delta_tt_ut1),
            CEREAL_NVP(delta_ut1_utc),
            CEREAL_NVP(delta_tai_gps)
        );
    }
    static nl::ordered_json getDeltas(); //!< 現在のdeltaの一覧をjson objectで返す。
    static double getDelta_tt_tai(); //!< TT-TAIを返す。
    static double getDelta_tt_ut1(); //!< ΔT=TT-UT1を返す。
    static double getDelta_ut1_utc(); //!< ΔUT1=UT1-UTCを返す。
    static double getDelta_tai_gps(); //!< TAI-GPUを返す。
    std::string toString() const;
    /**
     * @brief シリアライズ処理
     */
    template<class Archive>
    void save(Archive & archive) const{
        archive(
            cereal::make_nvp("timeSystem",givenTS),
            cereal::make_nvp("datetime",givenDatetime)
        );
    }
    /**
     * @brief デシリアライズ処理
     */
    template<class Archive>
    void load(Archive & archive) {
        if(!util::try_load(archive,"timeSystem",givenTS)){
            givenTS=TimeSystem::TT;
        }
        if(!util::try_load(archive,"datetime",givenDatetime)){
            givenDatetime="2000-01-01T12:00:00";
        }
        origin=toTT(Time(givenDatetime,givenTS));

        // nl::ordered_jsonの場合のみ、Deltaの読み込みも試す。
        if constexpr (traits::same_archive_as<Archive,util::NLJSONInputArchive>){
            loadDeltas(archive);
        }
    }
    private:
    static double delta_tt_tai; //!< TT-TAI≒32.184 (1977-01-01T00:00 TAI時点の差。一般的な用途では基本的にこの値で定数とみなして支障ない。)
    static double delta_tt_ut1; //!< ΔT=TT-UT1≒67.6439 (2015-01-01T00:00 UTC時点の値。)
    static double delta_ut1_utc; //!< ΔUT1=UT1-UTC ±0.9秒の範囲に収まるようにUTCが閏秒により調整されている。2024年時点はほぼ0。
    static double delta_tai_gps; //!< TAI-GPU=19
    /**
     * @brief deltaのデシリアライズ処理
     */
    template<class Archive>
    static void loadDeltas(Archive & archive){
        util::try_load(archive,"delta_tt_tai",delta_tt_tai);
        util::try_load(archive,"delta_tt_ut1",delta_tt_ut1);
        util::try_load(archive,"delta_ut1_utc",delta_ut1_utc);
        util::try_load(archive,"delta_tai_gps",delta_tai_gps);
    }
    static void setDeltas(const nl::ordered_json& j); //!< 各deltaをjson objectから読み込んで変更する。
    static void setDelta_tt_tai(const double& _delta_tt_tai=32.184); //!< TT-TAIを変更する。
    static void setDelta_tt_ut1(const double& _delta_tt_ut1=67.6439); //!< ΔT=TT-UT1を変更する。
    static void setDelta_ut1_utc(const double& _delta_ut1_utc=0.0); //!< ΔUT1=UT1-UTCを変更する。
    static void setDelta_tai_gps(const double& _delta_tai_gps=19.0); //!< TAI-GPUを変更する。
};

/**
 * @brief 左辺の Epoch に右辺の秒数を加算した Epoch を返す。
 */
inline Time operator+(const Epoch& lhs,const double& rhs) {
    return std::move(lhs.origin + rhs);
}
/**
 * @brief 右辺の Epoch に左辺の秒数を加算した Epoch を返す。
 */
inline Time operator+(const double& lhs,const Epoch& rhs) {
    return std::move(lhs + rhs.origin);
}
/**
 * @brief 左辺の Epoch から右辺の秒数を減算した Epoch を返す。
 */
inline Time operator-(const Epoch& lhs,const double& rhs) {
    return lhs.origin - rhs;
}
/**
 * @brief 左辺の Epoch から右辺の Epoch を引いた秒数を返す。
 */
inline double operator-(const Epoch& lhs,const Epoch& rhs) {
    return lhs.origin - rhs.origin;
}
/**
 * @brief 左辺の Epoch と右辺の Epoch が同一かどうかを返す。時刻系が異なる場合はTTに変換して揃えて比較する。
 */
inline bool operator==(const Epoch& lhs,const Epoch& rhs) noexcept{
    return lhs.origin == rhs.origin;
}
/**
 * @brief 左辺の Epoch と右辺の Epoch が異なるかどうかを返す。時刻系が異なる場合はTTに変換して揃えて比較する。
 */
inline bool operator!=(const Epoch& lhs,const Epoch& rhs) noexcept{
    return lhs.origin != rhs.origin;
}
/**
 * @brief 左辺の Epoch が右辺の Epoch がより未来かどうかを返す。時刻系が異なる場合はTTに変換して揃えて比較する。
 */
inline bool operator>(const Epoch& lhs,const Epoch& rhs) noexcept{
    return lhs.origin > rhs.origin;
}
/**
 * @brief 左辺の Epoch が右辺の Epoch がより過去かどうかを返す。時刻系が異なる場合はTTに変換して揃えて比較する。
 */
inline bool operator<(const Epoch& lhs,const Epoch& rhs) noexcept{
    return lhs.origin < rhs.origin;
}
/**
 * @brief 左辺の Epoch が右辺の Epoch 以降かどうかを返す。時刻系が異なる場合はTTに変換して揃えて比較する。
 */
inline bool operator>=(const Epoch& lhs,const Epoch& rhs) noexcept{
    return lhs.origin >= rhs.origin;
}
/**
 * @brief 左辺の Epoch が右辺の Epoch 以前かどうかを返す。時刻系が異なる場合はTTに変換して揃えて比較する。
 */
inline bool operator<=(const Epoch& lhs,const Epoch& rhs) noexcept{
    return lhs.origin <= rhs.origin;
}
inline std::ostream& operator<<(std::ostream& os, const Epoch& ep){
    os << "Epoch(" << ep.givenDatetime << " in " << ep.givenTS << ")";
    return os;
}

void exportTimeSystem(py::module &m);

ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
