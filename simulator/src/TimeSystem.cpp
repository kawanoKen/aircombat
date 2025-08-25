// Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#include "TimeSystem.h"
#include <iomanip>
#include <limits>
#include <cctype>

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)

using namespace util;

Time::Time():value(0.0),timeSystem(TimeSystem::INVALID){}
Time::Time(const double& t, const TimeSystem& ts):value(t),timeSystem(ts){}
Time::Time(const std::string& datetime, const TimeSystem& ts):Time(Time::parseDatetime(datetime),ts){}
Time::Time(const Epoch& ep):Time(ep.origin){
}
Time::Time(const nl::ordered_json& j){
    load_from_json(j);
}
Time::operator bool() const{
    return timeSystem!=TimeSystem::INVALID;
}
bool Time::isValid() const{
    return timeSystem!=TimeSystem::INVALID;
}
void Time::operator+=(const double& delta){
    value+=delta;
}
void Time::operator-=(const double& delta){
    value-=delta;
}
Time Time::to(const TimeSystem& ts) const{
    return Epoch::to(*this,ts);
}
Time Time::toTT() const{
    return Epoch::toTT(*this);
}
Time Time::toTAI() const{
    return Epoch::toTAI(*this);
}
Time Time::toUT1() const{
    return Epoch::toUT1(*this);
}
Time Time::toUTC() const{
    return Epoch::toUTC(*this);
}
Time Time::toGPS() const{
    return Epoch::toGPS(*this);
}
Time Time::toJY() const{
    return Epoch::toJY(*this);
}
Time Time::toJC() const{
    return Epoch::toJC(*this);
}
double Time::parseDatetime(const std::string& datetime){
    // ISO 8601相当のタイムゾーン無し文字列で与えること。
    // YYYYMMDDThhmmss.SSS 又は YYYY-MM-DDThh:mm:ss.SSS
    int year, month, day, hour, minute;
    double second;
    std::size_t pos=0;
    try{
        year=std::stoi(datetime.substr(pos,4));
        pos+=4;
        if(!std::isdigit(datetime.at(pos))){pos++;}
        month=std::stoi(datetime.substr(pos,2));
        pos+=2;
        if(!std::isdigit(datetime.at(pos))){pos++;}
        day=std::stoi(datetime.substr(pos,2));
        pos+=2;
        if(!std::isdigit(datetime.at(pos))){pos++;}
        hour=std::stoi(datetime.substr(pos,2));
        pos+=2;
        if(!std::isdigit(datetime.at(pos))){pos++;}
        minute=std::stoi(datetime.substr(pos,2));
        pos+=2;
        if(!std::isdigit(datetime.at(pos))){pos++;}
        std::istringstream iss(datetime.substr(pos)); // 最後は末尾に余計なものが付いていても良いようにstringstreamを使う。
        iss >> second;
    }catch(std::exception& ex){
        std::cout<<"Failed to parse datetime '"<<datetime<<"'"<<std::endl;
        std::cout<<ex.what()<<std::endl;
        throw ex;
    }
    return (
        (days_from_civil(year,month,day) - days_from_civil(2000,1,1)) * 86400 + 
        (hour - 12) * 3600 + 
        minute * 60 +
        second
    );
}
int Time::days_from_civil(int y,int m, int d){
    //日付(y,m,d)を与え、1970-01-01からの経過日数を返す。 https://howardhinnant.github.io/date_algorithm.html より。
    y -= m <= 2;
    const int era = (y >= 0 ? y : y-399) /400;
    const unsigned int yoe = y - era * 400;
    const unsigned int doy = (153*(m > 2 ? m-3 : m+9) + 2)/5 + d-1;
    const unsigned int doe = yoe * 365 + yoe /4 - yoe/100 + doy;
    return era * 146097 + doe - 719468;
}
std::tuple<int,int,int> Time::civil_from_days(int z){
    //1970-01-01からの経過日数を与え、日付(y,m,d)を返す。 https://howardhinnant.github.io/date_algorithm.html より。
    z += 719468;
    const int era = (z >= 0 ? z : z -146096) / 146097;
    const unsigned int doe = z - era * 146097;
    const unsigned int yoe = (doe - doe/1460 + doe/36524 - doe/146096) / 365;
    const int y = yoe + era * 400;
    const unsigned int doy = doe - (365*yoe + yoe/4 - yoe/100);
    const unsigned int mp = (5*doy + 2)/153;
    const unsigned int d = doy - (153*mp+2)/5 + 1;
    const unsigned int m = mp < 10 ? mp+3 : mp-9;
    return {y + (m <= 2), m, d};
}
std::string Time::toString() const{
    std::ostringstream ss;
    ss << std::setprecision(std::numeric_limits<double>::max_digits10);
    ss << *this;
    return ss.str();
}

double Epoch::delta_tt_tai=32.184; //TT-TAI≒32.184 (1977年1月1日00:00 TAI時点の差。一般的な用途では基本的にこの値で定数とみなして支障ない。)
double Epoch::delta_tt_ut1=67.6439; //ΔT=TT-UT1≒67.6439 (2015年1月1日00:00 UTC時点の値。)
double Epoch::delta_ut1_utc=0.0; //ΔUT1=UT1-UTC ±0.9秒の範囲に収まるようにUTCが閏秒により調整されている。2024年時点はほぼ0。
double Epoch::delta_tai_gps=19.0; //TAI-GPU=19
Epoch::Epoch():Epoch("2000-01-01T12:00:00",TimeSystem::TT){}
Epoch::Epoch(const std::string& datetime, const TimeSystem& ts):
givenTS(ts),
givenDatetime(datetime),
origin(Time(datetime,ts))
{
}
Epoch::Epoch(const nl::ordered_json& j){
    load_from_json(j);
}
Time Epoch::to(const Time& t,const TimeSystem& ts){
    if(t.timeSystem==ts){
        return t;
    }
    if(ts==TimeSystem::TT){
        return toTT(t);
    }else if(ts==TimeSystem::TAI){
        return toTAI(t);
    }else if(ts==TimeSystem::UT1){
        return toUT1(t);
    }else if(ts==TimeSystem::UTC){
        return toUTC(t);
    }else if(ts==TimeSystem::GPS){
        return toGPS(t);
    }else if(ts==TimeSystem::JY){
        return toJY(t);
    }else if(ts==TimeSystem::JC){
        return toJC(t);
    }else{
        throw std::runtime_error("Invalid Time.");
    }
}
Time Epoch::toTT(const Time& t){
    if(t.timeSystem==TimeSystem::TT){
        return t;
    }else if(t.timeSystem==TimeSystem::TAI){
        return Time(t.value+delta_tt_tai,TimeSystem::TT);
    }else if(t.timeSystem==TimeSystem::UT1){
        return Time(t.value+delta_tt_ut1,TimeSystem::TT);
    }else if(t.timeSystem==TimeSystem::UTC){
        return Time(t.value+delta_tt_ut1+delta_ut1_utc,TimeSystem::TT);
    }else if(t.timeSystem==TimeSystem::GPS){
        return Time(t.value+delta_tt_tai+delta_tai_gps,TimeSystem::TT);
    }else if(t.timeSystem==TimeSystem::JY){
        return Time(t.value*86400*365.25,TimeSystem::TT);
    }else if(t.timeSystem==TimeSystem::JC){
        return Time(t.value*86400*36525,TimeSystem::TT);
    }else{
        throw std::runtime_error("Invalid Time.");
    }
}
Time Epoch::toTAI(const Time& t){
    if(t.timeSystem==TimeSystem::TAI){
        return t;
    }else{
        return Time(toTT(t).value-delta_tt_tai,TimeSystem::TAI);
    }
}
Time Epoch::toUT1(const Time& t){
    if(t.timeSystem==TimeSystem::UT1){
        return t;
    }else{
        return Time(toTT(t).value-delta_tt_ut1,TimeSystem::UT1);
    }
}
Time Epoch::toUTC(const Time& t){
    if(t.timeSystem==TimeSystem::UTC){
        return t;
    }else{
        return Time(toTT(t).value-delta_tt_ut1-delta_ut1_utc,TimeSystem::UTC);
    }
}
Time Epoch::toGPS(const Time& t){
    if(t.timeSystem==TimeSystem::GPS){
        return t;
    }else{
        return Time(toTT(t).value-delta_tt_tai-delta_tai_gps,TimeSystem::GPS);
    }
}
Time Epoch::toJY(const Time& t){
    if(t.timeSystem==TimeSystem::JY){
        return t;
    }else{
        return Time(toTT(t).value/(86400.0*365.25),TimeSystem::JY);
    }
}
Time Epoch::toJC(const Time& t){
    if(t.timeSystem==TimeSystem::JC){
        return t;
    }else{
        return Time(toTT(t).value/(86400.0*36525.0),TimeSystem::JC);
    }
}
double Epoch::delta(const Time& t) const{
    return toTT(t)-origin;
}
double Epoch::deltaYear(const Time& t) const{
    return toJY(t)-toJY(origin);
}
double Epoch::deltaCentury(const Time& t) const{
    return toJC(t)-toJC(origin);
}
double Epoch::getDelta_tt_tai(){
    return delta_tt_tai;
}
double Epoch::getDelta_tt_ut1(){
    return delta_tt_ut1;
}
double Epoch::getDelta_ut1_utc(){
    return delta_ut1_utc;
}
double Epoch::getDelta_tai_gps(){
    return delta_tai_gps;
}
std::string Epoch::toString() const{
    std::ostringstream ss;
    ss << std::setprecision(std::numeric_limits<double>::max_digits10);
    ss << *this;
    return ss.str();
}
nl::ordered_json Epoch::getDeltas(){
    nl::ordered_json j;
    {
        ::asrc::core::util::NLJSONOutputArchive ar(j,false);
        saveDeltas(ar);
    }
    return std::move(j);
}
void Epoch::setDeltas(const nl::ordered_json& j){
    if(j.is_object()){
        ::asrc::core::util::NLJSONInputArchive ar(j,false);
        loadDeltas(ar);
    }else{
        throw std::runtime_error("Epoch only accepts json object for setting deltas.");
    }
}
void Epoch::setDelta_tt_tai(const double& _delta_tt_tai){
    delta_tt_tai=_delta_tt_tai;
}
void Epoch::setDelta_tt_ut1(const double& _delta_tt_ut1){
    delta_tt_ut1=_delta_tt_ut1;
}
void Epoch::setDelta_ut1_utc(const double& _delta_ut1_utc){
    delta_ut1_utc=_delta_ut1_utc;
}
void Epoch::setDelta_tai_gps(const double& _delta_tai_gps){
    delta_tai_gps=_delta_tai_gps;
}

void exportTimeSystem(py::module &m){
    using namespace pybind11::literals;

    expose_enum_value_helper(
        expose_enum_class<TimeSystem>(m,"TimeSystem")
        ,"TT"
        ,"TAI"
        ,"UT1"
        ,"UTC"
        ,"GPS"
        ,"JY"
        ,"JC"
        ,"INVALID"
    );

    expose_common_class<Time>(m,"Time")
    .def(py_init<>())
    .def(py_init<const double&,const TimeSystem&>(),"t"_a,"ts"_a)
    .def(py_init<const std::string&,const TimeSystem&>(),"datetime"_a,"ts"_a)
    .def(py_init<const Epoch&>())
    .def(py_init<const nl::ordered_json&>())
    DEF_READWRITE(Time,timeSystem)
    DEF_READWRITE(Time,value)
    DEF_FUNC(Time,isValid)
    DEF_FUNC(Time,to)
    DEF_FUNC(Time,toTT)
    DEF_FUNC(Time,toTAI)
    DEF_FUNC(Time,toUT1)
    DEF_FUNC(Time,toUTC)
    DEF_FUNC(Time,toGPS)
    DEF_FUNC(Time,toJY)
    DEF_FUNC(Time,toJC)
    DEF_STATIC_FUNC(Time,parseDatetime)
    DEF_STATIC_FUNC(Time,days_from_civil)
    DEF_STATIC_FUNC(Time,civil_from_days)
    .def("__bool__",[](const Time& t){return t.isValid();})
    .def(py::self + double())
    .def(double() + py::self)
    .def(py::self += double())
    .def(py::self - double())
    .def(py::self - py::self)
    .def(py::self -= double())
    .def(py::self == py::self)
    .def(py::self != py::self)
    .def(py::self < py::self)
    .def(py::self > py::self)
    .def(py::self <= py::self)
    .def(py::self >= py::self)
    .def("__str__",[](const Time& t){return t.toString();})
    ;

    expose_common_class<Epoch>(m,"Epoch")
    .def(py_init<const std::string&,const TimeSystem&>(),"datetime"_a,"ts"_a)
    .def(py_init<const nl::ordered_json&>())
    DEF_READWRITE(Epoch,givenTS)
    DEF_READWRITE(Epoch,givenDatetime)
    DEF_READWRITE(Epoch,origin)
    DEF_STATIC_FUNC(Epoch,to)
    DEF_STATIC_FUNC(Epoch,toTT)
    DEF_STATIC_FUNC(Epoch,toTAI)
    DEF_STATIC_FUNC(Epoch,toUT1)
    DEF_STATIC_FUNC(Epoch,toUTC)
    DEF_STATIC_FUNC(Epoch,toGPS)
    DEF_STATIC_FUNC(Epoch,toJY)
    DEF_STATIC_FUNC(Epoch,toJC)
    DEF_FUNC(Epoch,delta)
    DEF_FUNC(Epoch,deltaYear)
    DEF_FUNC(Epoch,deltaCentury)
    DEF_STATIC_FUNC(Epoch,getDeltas)
    DEF_STATIC_FUNC(Epoch,getDelta_tt_tai)
    DEF_STATIC_FUNC(Epoch,getDelta_tt_ut1)
    DEF_STATIC_FUNC(Epoch,getDelta_ut1_utc)
    DEF_STATIC_FUNC(Epoch,getDelta_tai_gps)
    .def(py::self + double())
    .def(double() + py::self)
    .def(py::self - py::self)
    .def(py::self - Time())
    .def(py::self - double())
    .def(Time() - py::self)
    .def(py::self == py::self)
    .def(py::self == Time())
    .def(py::self != py::self)
    .def(py::self != Time())
    .def(py::self < py::self)
    .def(py::self < Time())
    .def(py::self > py::self)
    .def(py::self > Time())
    .def(py::self <= py::self)
    .def(py::self <= Time())
    .def(py::self >= py::self)
    .def(py::self >= Time())
    .def("__str__",[](const Epoch& ep){return ep.toString();})
    ;
    py::implicitly_convertible<Epoch, Time>();
}

ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
