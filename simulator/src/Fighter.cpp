// Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#include "Fighter.h"
#include "Utility.h"
#include "Units.h"
#include "SimulationManager.h"
#include "Agent.h"
#include <boost/uuid/nil_generator.hpp>

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)
using namespace util;

void Fighter::initialize(){
    BaseType::initialize();
    //modelConfigで指定するもの
    m=getValueFromJsonKRD(modelConfig.at("dynamics"),"m",randomGen,10000.0);
    if(modelConfig.contains("stealth")){
        rcsScale=getValueFromJsonKRD(modelConfig.at("stealth"),"rcsScale",randomGen,1.0);
    }else{
        rcsScale=1.0;
    }
    if(modelConfig.contains("weapon")){
        numMsls=getValueFromJsonKRD(modelConfig.at("weapon"),"numMsls",randomGen,10);
    }else{
        numMsls=10;
    }
    enableThrustAfterFuelEmpty=getValueFromJsonKRD(modelConfig.at("propulsion"),"enableThrustAfterFuelEmpty",randomGen,false);
    fuelCapacity=getValueFromJsonKRD(modelConfig.at("propulsion"),"fuelCapacity",randomGen,5000.0);
    fuelRemaining=fuelCapacity;
    //instanceConfigで指定するもの
    datalinkName=getValueFromJsonKRD<std::string>(instanceConfig,"datalinkName",randomGen,"");
    //その他の内部変数
    isDatalinkEnabled=true;
    track.clear();
    trackSource.clear();
    target=Track3D();targetID=-1;
    observables["propulsion"]={
        {"fuelRemaining",fuelRemaining}
    };
    observables["spec"]={
        {"dynamics",{nl::json::object()}},
        {"weapon",{
            {"numMsls",numMsls}
        }},
        {"stealth",{
            {"rcsScale",rcsScale}
        }},
        {"sensor",nl::json::object()},
        {"propulsion",{
            {"fuelCapacity",fuelCapacity}
        }}
    };
    observables["shared"]={
        {"agent",nl::json::object()},
        {"fighter",nl::json::object()}
    };
}

void Fighter::serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full){
    BaseType::serializeInternalState(archive,full);

    if(full){
        ASRC_SERIALIZE_NVP(archive
            ,engine
            ,radar
            ,mws
            ,missiles
            ,dummyMissile
            ,numMsls
            ,datalinkName
            ,enableThrustAfterFuelEmpty
        )
    }
    ASRC_SERIALIZE_NVP(archive
        ,m
        ,rcsScale
        ,fuelCapacity
        ,missileTargets
        ,nextMsl
        ,remMsls
        ,isDatalinkEnabled
        ,track
        ,trackSource
        ,target
        ,targetID
        ,fuelRemaining
        ,optCruiseFuelFlowRatePerDistance
    )
}

void Fighter::makeChildren(){
    BaseType::makeChildren();
    //子要素のデフォルトのmodelConfigは、処理周期の指定を自身と同じとする
    nl::json defaultControllerModelConfig=nl::json::object();
    if(modelConfig.contains("firstTick")){
        defaultControllerModelConfig["firstTick"]=modelConfig.at("firstTick");
    }
    if(modelConfig.contains("interval")){
        defaultControllerModelConfig["interval"]=modelConfig.at("interval");
    }
    //子要素のinstanceConfig
    auto fullName=getFullName();
    //sensors
    nl::json sub={
        {"seed",randomGen()},//Entity
        {"entityFullName",fullName+"/Radar"},//PhysicalAsset
        {"parent",this->weak_from_this()},//PhysicalAsset
        {"isBound",true}//PhysicalAsset
    };
    if(modelConfig.at("sensor").contains("radar")){
        radar=createEntity<AircraftRadar>(
            isEpisodic(),
            "PhysicalAsset",
            modelConfig.at("/sensor/radar"_json_pointer),
            sub
        );
        observables["spec"]["sensor"]["radar"]=radar.lock()->observables["spec"];
    }
    if(modelConfig.at("sensor").contains("mws")){
        sub["seed"]=randomGen();
        sub["entityFullName"]=fullName+"/MWS";
        mws=createEntity<MWS>(
            isEpisodic(),
            "PhysicalAsset",
            modelConfig.at("/sensor/mws"_json_pointer),
            sub
        );
        observables["spec"]["sensor"]["mws"]=mws.lock()->observables["spec"];
    }

    //propulsion
    sub["entityFullName"]=fullName+"/Propulsion";
    if(modelConfig.contains("/propulsion/engine"_json_pointer)){
        engine=createEntity<Propulsion>(
            isEpisodic(),
            "PhysicalAsset",
            modelConfig.at("/propulsion/engine"_json_pointer),
            sub
        );
    }

    //weapons
    missiles.clear();
    missileTargets.clear();
    nextMsl=0;
    remMsls=numMsls;
    sub["isBound"]=true;
    for(int i=0;i<numMsls;++i){
        sub["seed"]=randomGen();
        sub["entityFullName"]=fullName+"/Missile"+std::to_string(i+1);
        missiles.push_back(
            createEntity<Missile>(
                isEpisodic(),
                "PhysicalAsset",
                modelConfig.at("/weapon/missile"_json_pointer),
                sub
            )
        );
        missileTargets.push_back(std::make_pair(Track3D(),false));
        generateCommunicationBuffer(
            "MissileComm:"+sub["entityFullName"].get<std::string>(),
            nl::json::array({"PhysicalAsset:"+getFullName(),"PhysicalAsset:"+sub["entityFullName"].get<std::string>()}),
            nl::json::array({"PhysicalAsset:"+getTeam()+"/.+"})
        );
    }
    if(numMsls<=0){
        //初期弾数0のとき、射程計算のみ可能とするためにダミーインスタンスを生成
        sub["seed"]=randomGen();
        sub["entityFullName"]=fullName+"/Missile(dummy)";
        dummyMissile=createEntity<Missile>(
                isEpisodic(),
                "PhysicalAsset",
                modelConfig.at("/weapon/missile"_json_pointer),
                sub
            );
        //CommunicationBufferは正規表現で対象Assetを特定するため、括弧はエスケープが必要
        std::string query=fullName+"/Missile\\(dummy\\)";
        generateCommunicationBuffer(
            "MissileComm:"+sub["entityFullName"].get<std::string>(),
            nl::json::array({"PhysicalAsset:"+getFullName(),"PhysicalAsset:"+query}),
            nl::json::array()
        );
    }else{
        dummyMissile=std::shared_ptr<Missile>(nullptr);
    }
    if(numMsls>0){
        observables["spec"]["weapon"]["missile"]=missiles[0].lock()->observables["spec"];
    }else{
        observables["spec"]["weapon"]["missile"]=dummyMissile.lock()->observables["spec"];
    }
    sub={
        {"seed",randomGen()},//Entity
        {"entityFullName",fullName+"/SensorDataSharer"},//Controller
        {"parent",this->weak_from_this()}//Controller
    };
    if(modelConfig.contains("/controller/sensorDataSharer"_json_pointer)){
        controllers["SensorDataSharer"]=createEntity<Controller>(isEpisodic(),"Controller",modelConfig.at("/controller/sensorDataSharer"_json_pointer),sub);
    }else{
        controllers["SensorDataSharer"]=createEntityByClassName<SensorDataSharer>(isEpisodic(),"Controller","SensorDataSharer",defaultControllerModelConfig,sub);
    }
    sub={
        {"seed",randomGen()},//Entity
        {"entityFullName",fullName+"/SensorDataSanitizer"},//Controller
        {"parent",this->weak_from_this()}//Controller
    };
    if(modelConfig.contains("/controller/sensorDataSanitizer"_json_pointer)){
        controllers["SensorDataSanitizer"]=createEntity<Controller>(isEpisodic(),"Controller",modelConfig.at("/controller/sensorDataSanitizer"_json_pointer),sub);
    }else{
        controllers["SensorDataSanitizer"]=createEntityByClassName<SensorDataSanitizer>(isEpisodic(),"Controller","SensorDataSanitizer",defaultControllerModelConfig,sub);
    }
    sub={
        {"seed",randomGen()},//Entity
        {"entityFullName",fullName+"/OtherDataSharer"},//Controller
        {"parent",this->weak_from_this()}//Controller
    };
    if(modelConfig.contains("/controller/otherDataSharer"_json_pointer)){
        controllers["OtherDataSharer"]=createEntity<Controller>(isEpisodic(),"Controller",modelConfig.at("/controller/otherDataSharer"_json_pointer),sub);
    }else{
        controllers["OtherDataSharer"]=createEntityByClassName<OtherDataSharer>(isEpisodic(),"Controller","OtherDataSharer",defaultControllerModelConfig,sub);
    }
    sub={
        {"seed",randomGen()},//Entity
        {"entityFullName",fullName+"/OtherDataSanitizer"},//Controller
        {"parent",this->weak_from_this()}//Controller
    };
    if(modelConfig.contains("/controller/otherDataSanitizer"_json_pointer)){
        controllers["OtherDataSanitizer"]=createEntity<Controller>(isEpisodic(),"Controller",modelConfig.at("/controller/otherDataSanitizer"_json_pointer),sub);
    }else{
        controllers["OtherDataSanitizer"]=createEntityByClassName<OtherDataSanitizer>(isEpisodic(),"Controller","OtherDataSanitizer",defaultControllerModelConfig,sub);
    }
    if(modelConfig.contains("/pilot/model"_json_pointer)){
        sub={
            {"seed",randomGen()},//Entity
            {"entityFullName",fullName+"/HumanIntervention"},//Controller
            {"parent",this->weak_from_this()}//Controller
        };
        controllers["HumanIntervention"]=createEntity<Controller>(isEpisodic(),"Controller",modelConfig.at("/pilot/model"_json_pointer),sub);
    }
    sub={
        {"seed",randomGen()},//Entity
        {"entityFullName",fullName+"/WeaponController"},//Controller
        {"parent",this->weak_from_this()}//Controller
    };
    if(modelConfig.contains("/weapon/controller"_json_pointer)){
        controllers["WeaponController"]=createEntity<Controller>(isEpisodic(),"Controller",modelConfig.at("/weapon/controller"_json_pointer),sub);
    }else{
        controllers["WeaponController"]=createEntityByClassName<WeaponController>(isEpisodic(),"Controller","WeaponController",defaultControllerModelConfig,sub);
    }
    sub={
        {"seed",randomGen()},//Entity
        {"entityFullName",fullName+"/FlightController"},//Controller
        {"parent",this->weak_from_this()}//Controller
    };
    controllers["FlightController"]=createEntity<Controller>(isEpisodic(),"Controller",modelConfig.at("/dynamics/controller"_json_pointer),sub);
}
void Fighter::validate(){
    BaseType::validate();
    isDatalinkEnabled = communicationBuffers.count(datalinkName)>0;
    //自機以外の味方機の誘導弾とのデータリンクに接続
    for(auto&& asset:manager->getAssets()){
        if(asset.lock()->getTeam()==team && isinstance<Fighter>(asset) && !this->isSame(asset.lock())){
            auto f=getShared<const Fighter>(asset);
            for(auto&& e:f->missiles){
                std::string bufferName="MissileComm:"+e.lock()->getFullName();
                manager->requestInvitationToCommunicationBuffer(bufferName,getShared<Asset>(this->shared_from_this()));
            }
        }
    }
    if(modelConfig.contains("/propulsion/optCruiseFuelFlowRatePerDistance"_json_pointer)){
        optCruiseFuelFlowRatePerDistance=modelConfig.at("/propulsion/optCruiseFuelFlowRatePerDistance"_json_pointer);
    }else{
        optCruiseFuelFlowRatePerDistance=calcOptimumCruiseFuelFlowRatePerDistance();
        std::cout<<"optCruiseFuelFlowRatePerDistance="<<optCruiseFuelFlowRatePerDistance<<" for \""<<getFactoryModelName()<<"\""<<std::endl;
        std::cout<<"maximumRange="<<getMaxReachableRange()<<std::endl;
    }
    observables["spec"]["propulsion"]["optCruiseFuelFlowRatePerDistance"]=optCruiseFuelFlowRatePerDistance;
}
void Fighter::setDependency(){
    //validate
    for(auto&& e:missiles){
        manager->addDependency(SimPhase::VALIDATE,shared_from_this(),e.lock());
    }
    if(!dummyMissile.expired()){
        manager->addDependency(SimPhase::VALIDATE,shared_from_this(),dummyMissile.lock());
    }
    //perceive
    if(!radar.expired()){
        manager->addDependency(SimPhase::PERCEIVE,radar.lock(),controllers["SensorDataSharer"].lock());
    }
    if(!mws.expired()){
        manager->addDependency(SimPhase::PERCEIVE,mws.lock(),controllers["SensorDataSharer"].lock());
    }
    for(auto&& e:missiles){
        manager->addDependency(SimPhase::PERCEIVE,controllers["SensorDataSanitizer"].lock(),e.lock());
        manager->addDependency(SimPhase::PERCEIVE,e.lock(),shared_from_this());
    }
    if(!dummyMissile.expired()){
        manager->addDependency(SimPhase::PERCEIVE,controllers["SensorDataSanitizer"].lock(),dummyMissile.lock());
        manager->addDependency(SimPhase::PERCEIVE,dummyMissile.lock(),shared_from_this());
    }
    manager->addDependency(SimPhase::PERCEIVE,shared_from_this(),controllers["OtherDataSharer"].lock());
    if(!agent.expired()){
        manager->addDependency(SimPhase::PERCEIVE,agent.lock(),controllers["OtherDataSharer"].lock());
    }
    //control
    if(!agent.expired()){
        manager->addDependency(SimPhase::CONTROL,agent.lock(),controllers["OtherDataSharer"].lock());
    }
    manager->addDependency(SimPhase::CONTROL,controllers["OtherDataSanitizer"].lock(),controllers["WeaponController"].lock());
    manager->addDependency(SimPhase::CONTROL,controllers["OtherDataSanitizer"].lock(),controllers["FlightController"].lock());
    if(controllers.count("HumanIntervention")>0){
        manager->addDependency(SimPhase::CONTROL,controllers["OtherDataSanitizer"].lock(),controllers["HumanIntervention"].lock());
        manager->addDependency(SimPhase::CONTROL,controllers["HumanIntervention"].lock(),controllers["WeaponController"].lock());
        manager->addDependency(SimPhase::CONTROL,controllers["HumanIntervention"].lock(),controllers["FlightController"].lock());
    }
    if(!engine.expired()){
        manager->addDependency(SimPhase::CONTROL,controllers["FlightController"].lock(),engine.lock());
    }
    for(auto&& e:missiles){
        manager->addDependency(SimPhase::CONTROL,controllers["WeaponController"].lock(),e.lock());
    }
    if(!dummyMissile.expired()){
        manager->addDependency(SimPhase::CONTROL,controllers["WeaponController"].lock(),dummyMissile.lock());
    }
    //behave
    if(!engine.expired()){
        manager->addDependency(SimPhase::BEHAVE,shared_from_this(),engine.lock());
    }

    //dynamic
    manager->addDependencyGenerator([manager=manager,wPtr=weak_from_this()](const std::shared_ptr<Asset>& asset){
        if(!wPtr.expired()){
            auto ptr=getShared<Type>(wPtr);
            if(
                asset->getTeam()==ptr->getTeam()
                && isinstance<Fighter>(asset)
            ){
                //perceive
                auto f=getShared<const Fighter>(asset);
                manager->addDependency(SimPhase::PERCEIVE,f->controllers.at("SensorDataSharer").lock(),ptr->controllers["SensorDataSanitizer"].lock());
                manager->addDependency(SimPhase::PERCEIVE,f->controllers.at("OtherDataSharer").lock(),ptr->controllers["OtherDataSanitizer"].lock());
                //control
                manager->addDependency(SimPhase::CONTROL,f->controllers.at("OtherDataSharer").lock(),ptr->controllers["OtherDataSanitizer"].lock());
            }
        }
    });
}
void Fighter::perceive(bool inReset){
    PhysicalAsset::perceive(inReset);
    observables["propulsion"]={
        {"fuelRemaining",fuelRemaining}
    };
    nl::json mslObs=nl::json::array();
    for(auto&& e:missiles){
        mslObs.push_back(e.lock()->observables);
    }
    observables["weapon"]={
        {"remMsls",remMsls},
        {"nextMsl",nextMsl},
        {"launchable",isLaunchable()},
        {"missiles",std::move(mslObs)}
    };
}
void Fighter::control(){
    //done by controllers
}
void Fighter::behave(){
    double dt=interval[SimPhase::BEHAVE]*manager->getBaseTimeStep();
    calcMotion(dt);
    if(!engine.expired()){
        fuelRemaining=std::max<double>(fuelRemaining-dt*engine.lock()->getFuelFlowRate(),0.0);
    }
    if(motion.getHeight()<0){//墜落
        manager->triggerEvent("Crash",this->weak_from_this());
        manager->requestToKillAsset(getShared<Asset>(this->shared_from_this()));
    }
    PhysicalAsset::behave();
}
void Fighter::kill(){
    if(!engine.expired()){
        engine.lock()->kill();
    }
    if(!radar.expired()){
        radar.lock()->kill();
    }
    if(!mws.expired()){
        mws.lock()->kill();
    }
    if(isDatalinkEnabled){
        //immidiate notification to friends. If you need more realistic behavior, other notification or estimation of aliveness scheme should be implemented.
        nl::json mslObs=nl::json::array();
        for(auto&& e:missiles){
            mslObs.push_back(e.lock()->observables);
        }
        communicationBuffers[datalinkName].lock()->send({
            {"fighterObservables",{{getFullName(),{
                {"isAlive",false},
                {"weapon",{{
                    {"remMsls",remMsls},
                    {"nextMsl",nextMsl},
                    {"launchable",false},
                    {"missiles",std::move(mslObs)}
                }}},
                {"time",manager->getTime()+manager->getBaseTimeStep()}
            }}}}
        },CommunicationBuffer::MERGE);
    }
    for(auto&& e:controllers){
        e.second.lock()->kill();
    }
    this->PhysicalAsset::kill();
}
double Fighter::getThrust(){
    if(!engine.expired()){
        if(!enableThrustAfterFuelEmpty && fuelRemaining<=0 && engine.lock()->getFuelFlowRate()>0){
            return 0;
        }else{
            return engine.lock()->getThrust();
        }
    }else{
        return 0;
    }
}
double Fighter::calcThrust(const nl::json& args){
    if(!engine.expired()){
        if(!enableThrustAfterFuelEmpty && fuelRemaining<=0 && engine.lock()->getFuelFlowRate()>0){
            return 0;
        }else{
            return engine.lock()->calcThrust(args);
        }
    }else{
        return 0;
    }
}
double Fighter::getMaxReachableRange(){
    if(optCruiseFuelFlowRatePerDistance<=0.0){
        //no fuel consumption
        return std::numeric_limits<double>::infinity();
    }else{
        return fuelRemaining/optCruiseFuelFlowRatePerDistance;
    }
}
std::pair<bool,Track3D> Fighter::isTracking(const std::weak_ptr<PhysicalAsset>& target_){
    if(target_.expired()){
        return std::make_pair(false,Track3D());
    }else{
        return isTracking(target_.lock()->getUUIDForTrack());
    }
}
std::pair<bool,Track3D> Fighter::isTracking(const Track3D& target_){
    if(target_.is_none()){
        return std::make_pair(false,Track3D());
    }else{
        return isTracking(target_.truth);
    }
}
std::pair<bool,Track3D> Fighter::isTracking(const boost::uuids::uuid& target_){
    if(target_==boost::uuids::nil_uuid()){
        return std::make_pair(false,Track3D());
    }else{
        if(isAlive()){
            for(auto& t:track){
                if(t.isSame(target_)){
                    return std::make_pair(true,t);
                }
            }
        }
        return std::make_pair(false,Track3D());
    }
}
bool Fighter::isLaunchable(){
    bool ret = remMsls>0;
    ret = ret && getShared<WeaponController>(controllers["WeaponController"])->isLaunchable();
    if(controllers.count("HumanIntervention")>0){
        ret = ret && getShared<HumanIntervention>(controllers["HumanIntervention"])->isLaunchable();
    }
    return ret;
}
bool Fighter::isLaunchableAt(const Track3D& target_){
    bool ret = isLaunchable() && !target_.is_none();
    ret = ret && getShared<WeaponController>(controllers["WeaponController"])->isLaunchableAt(target_);
    if(controllers.count("HumanIntervention")>0){
        ret = ret && getShared<HumanIntervention>(controllers["HumanIntervention"])->isLaunchableAt(target_);
    }
    return ret;
}
std::shared_ptr<Missile> Fighter::launchInVirtual(const std::shared_ptr<SimulationManager>& dstManager,const nl::json& launchCommand){
    if(manager->isSame(dstManager)){
        throw std::runtime_error("Fighter::launchInVirtual() failed. The given dstManager is same as the original manager of the fighter.");
    }
    auto newUUID=boost::uuids::random_generator()();
    nl::json j=modelConfig.at("/weapon/missile"_json_pointer);
    if(j.is_string()){
        j={
            {"modelName",j},
            {"modelConfig",nl::json::object()}
        };
    }else if(!j.is_object()){
        throw std::runtime_error("modelConfig['/weapon/missile'] should be string or object.");
    }
    j.merge_patch({
        {"isEpisodic",isEpisodic()},
        {"baseName","PhysicalAsset"},
        {"instanceConfig",{
            {"seed",randomGen()},
            {"uuid",newUUID},
            {"entityFullName",getFullName()+"/VirtualMissile"+boost::uuids::to_string(newUUID)},
            {"motion",motion},
            {"launch",true},
            {"launchCommand",launchCommand}
        }}
    });
    if(launchCommand.contains("motion")){
        j["/instanceConfig/motion"_json_pointer]=launchCommand.at("motion");
    }

    auto ret=dstManager->createEntity<Missile>(j);

    ret->generateCommunicationBuffer(
        "MissileComm:"+ret->getFullName(),
        nl::json::array({"PhysicalAsset:"+ret->getFullName()}),
        nl::json::array({"PhysicalAsset:"+getTeam()+"/.+"})
    );

    return ret;
}
void Fighter::setFlightControllerMode(const std::string& ctrlName){
    std::dynamic_pointer_cast<FlightController>(controllers["FlightController"].lock())->setMode(ctrlName);
}
Eigen::Vector3d Fighter::toEulerAngle(){
    Eigen::Vector3d ex=motion.relBtoH(Eigen::Vector3d(1,0,0),"NED");
    Eigen::Vector3d ey=motion.relBtoH(Eigen::Vector3d(0,1,0),"NED");
    Eigen::Vector3d horizontalY=Eigen::Vector3d(0,0,1).cross(ex).normalized();
    double sinRoll=horizontalY.cross(ey).dot(ex);
    double cosRoll=horizontalY.dot(ey);
    double rollAtt=atan2(sinRoll,cosRoll);
    return Eigen::Vector3d(rollAtt,motion.getEL(),motion.getAZ());
}
double Fighter::getRmax(const Eigen::Vector3d &rs,const Eigen::Vector3d &vs,const Eigen::Vector3d &rt,const Eigen::Vector3d &vt){
    if(numMsls>0){
        return missiles[0].lock()->getRmax(rs,vs,rt,vt,getParentCRS());
    }else{
        return dummyMissile.lock()->getRmax(rs,vs,rt,vt,getParentCRS());
    }
}
double Fighter::getRmax(const Eigen::Vector3d &rs,const Eigen::Vector3d &vs,const Eigen::Vector3d &rt,const Eigen::Vector3d &vt,const std::shared_ptr<CoordinateReferenceSystem>& crs){
    if(numMsls>0){
        return missiles[0].lock()->getRmax(rs,vs,rt,vt,crs);
    }else{
        return dummyMissile.lock()->getRmax(rs,vs,rt,vt,crs);
    }
}
double Fighter::getRmax(const Eigen::Vector3d &rs,const Eigen::Vector3d &vs,const Eigen::Vector3d &rt,const Eigen::Vector3d &vt,const double& aa){
    if(numMsls>0){
        return missiles[0].lock()->getRmax(rs,vs,rt,vt,aa,getParentCRS());
    }else{
        return dummyMissile.lock()->getRmax(rs,vs,rt,vt,aa,getParentCRS());
    }
}
double Fighter::getRmax(const Eigen::Vector3d &rs,const Eigen::Vector3d &vs,const Eigen::Vector3d &rt,const Eigen::Vector3d &vt,const double& aa,const std::shared_ptr<CoordinateReferenceSystem>& crs){
    if(numMsls>0){
        return missiles[0].lock()->getRmax(rs,vs,rt,vt,aa,crs);
    }else{
        return dummyMissile.lock()->getRmax(rs,vs,rt,vt,aa,crs);
    }
}
std::shared_ptr<EntityAccessor> Fighter::getAccessorImpl(){
    return std::make_shared<FighterAccessor>(getShared<Fighter>(this->shared_from_this()));
}

void SensorDataSharer::perceive(bool inReset){
    auto p=getShared<Fighter>(parent);
    if(p->isAlive_){
        if(p->isDatalinkEnabled){
            nl::json data={
                {"isAlive",p->isAlive()},
                {"sensor",nl::json::object()},
                {"time",manager->getTime()}
            };
            if(!p->radar.expired()){
                data["/sensor/radar"_json_pointer]=p->radar.lock()->observables;
            }
            if(!p->mws.expired()){
                data["/sensor/mws"_json_pointer]=p->mws.lock()->observables;
            }
            p->communicationBuffers[p->datalinkName].lock()->send({
                {"fighterObservables",{{p->getFullName(),data}}}
            },CommunicationBuffer::MERGE);
        }
    }
}

void SensorDataSanitizer::serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full){
    BaseType::serializeInternalState(archive,full);

    ASRC_SERIALIZE_NVP(archive
        ,lastSharedTime
    )
}

void SensorDataSanitizer::perceive(bool inReset){
    auto p=getShared<Fighter>(parent);
    if(p->isAlive_){
        p->track.clear();
        p->trackSource.clear();
        if(!p->radar.expired()){
            for(auto& e:p->radar.lock()->track){
                p->track.push_back(e.transformTo(p->getParentCRS()));
                p->trackSource.push_back({p->getFullName()});
            }
        }
        nl::json sharedFighterObservable=nl::json::object();
        nl::json& sf=p->observables["shared"]["fighter"];
        sf=nl::json::object();
        if(p->isDatalinkEnabled){
            std::pair<Time,nl::json> tmp=p->communicationBuffers[p->datalinkName].lock()->receive("fighterObservables");
            if(tmp.first){//valid data
                sharedFighterObservable=tmp.second;
            }
            Track3D same;
            int sameID=-1;
            int idx=0;
            for(auto&& e:sharedFighterObservable.items()){
                sf[e.key()]["isAlive"]=e.value().at("isAlive");
                if(e.key()!=p->getFullName()){
                    Time sent(e.value().at("time"));
                    if(lastSharedTime.count(e.key())==0 || lastSharedTime[e.key()]<sent){
                        lastSharedTime[e.key()]=sent;
                        if(e.value()["isAlive"].get<bool>()){
                            if(e.value().at("sensor").contains("radar")){
                                std::vector<Track3D> shared=e.value().at("/sensor/radar/track"_json_pointer);
                                sf[e.key()]["sensor"]=e.value()["sensor"];
                                for(auto&& rhs:shared){
                                    same=Track3D();
                                    sameID=-1;
                                    idx=0;
                                    for(auto& lhs:p->track){
                                        if(lhs.isSame(rhs)){
                                            same=lhs;
                                            sameID=idx;
                                            idx++;
                                        }
                                    }
                                    if(same.is_none()){
                                        p->track.push_back(rhs.transformTo(p->getParentCRS()));
                                        p->trackSource.push_back({e.key()});
                                    }else{
                                        same.addBuffer(rhs);
                                        p->trackSource[sameID].push_back(e.key());
                                    }
                                }
                            }
                        }
                    }
                }
            }
            for(auto& t:p->track){
                t.merge();
            }
        }
        p->observables["sensor"]=nl::json::object();
        if(!p->radar.expired()){
            p->observables["sensor"]["radar"]=p->radar.lock()->observables;
        }
        if(!p->mws.expired()){
            p->observables["sensor"]["mws"]=p->mws.lock()->observables;
        }
        p->observables["sensor"]["track"]=p->track;
        p->observables["sensor"]["trackSource"]=p->trackSource;
        if(!p->target.is_none()){
            Track3D old=p->target;
            p->target=Track3D();
            p->targetID=-1;
            int i=0;
            for(auto& t:p->track){
                if(old.isSame(t)){
                    p->target=t;
                    p->targetID=i;
                }
                ++i;
            }
        }else{
            p->targetID=-1;
        }
        for(int mslId=0;mslId<p->nextMsl;++mslId){
            Track3D old=p->missileTargets[mslId].first;
            p->missileTargets[mslId].second=false;
            for(auto& t:p->track){
                if(old.isSame(t)){
                    p->missileTargets[mslId].first=t;
                    p->missileTargets[mslId].second=true;
                    break;
                }
            }
        }
    }else{
        p->track.clear();
    }
}

void OtherDataSharer::perceive(bool inReset){
    auto p=getShared<Fighter>(parent);
    if(p->isAlive_){
        if(p->isDatalinkEnabled){
            nl::json dataPacket={
                {"fighterObservables",{
                    {p->getFullName(),{
                        {"isAlive",p->isAlive()},
                        {"spec",p->observables["spec"]},
                        {"motion",p->observables["motion"]},
                        {"propulsion",p->observables["propulsion"]},
                        {"weapon",p->observables["weapon"]},
                        {"time",manager->getTime()}
                    }}
                }}
            };
            if(!p->agent.expired()){
                dataPacket["agentObservables"]=nl::json::object();
                nl::json& agObs=dataPacket["agentObservables"];
                for(auto&&e:p->agent.lock()->observables.items()){
                    agObs[e.key()]={
                        {"obs",e.value()},
                        {"time",manager->getTime()}
                    };
                }
            }
            //撃墜された味方の射撃済誘導弾に関する諸元更新を引き継ぎ
            bool isFirst=false;//自身が生存機の中で先頭かどうか
            for(auto&& asset:manager->getAssets()){
                if(asset.lock()->getTeam()==team && isinstance<Fighter>(asset)){
                    auto f=getShared<const Fighter>(asset);
                    if(f->isAlive()){
                        isFirst = f->getFullName()==p->getFullName();
                        break;
                    }
                }
            }
            if(isFirst){
                for(auto&& asset:manager->getAssets()){
                    if(asset.lock()->getTeam()==team && isinstance<Fighter>(asset)){
                        auto f=getShared<const Fighter>(asset);
                        if(!f->isAlive()){
                            nl::json mslObs=nl::json::array();
                            for(auto&& e:f->missiles){
                                mslObs.push_back(e.lock()->observables);
                            }
                            dataPacket["fighterObservables"][f->getFullName()]["weapon"]={{"missiles",std::move(mslObs)}};
                            dataPacket["fighterObservables"][f->getFullName()]["time"]=manager->getTime();
                        }
                    }
                }
            }
            //送信
            p->communicationBuffers[p->datalinkName].lock()->send(dataPacket,CommunicationBuffer::MERGE);
        }
    }
}
void OtherDataSharer::control(){
    auto p=getShared<Fighter>(parent);
    if(p->isAlive_ && p->isDatalinkEnabled){
        if(!p->agent.expired()){
            nl::json agCom=nl::json::object();
            for(auto&&e:p->agent.lock()->commands.items()){
                agCom[e.key()]={
                    {"com",e.value()},
                    {"time",manager->getTime()}
                };
            }
            p->communicationBuffers[p->datalinkName].lock()->send({
                {"agentCommands",agCom}
            },CommunicationBuffer::MERGE);
        }
    }
}


void OtherDataSanitizer::serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full){
    BaseType::serializeInternalState(archive,full);

    ASRC_SERIALIZE_NVP(archive
        ,lastSharedTimeOfAgentObservable
        ,lastSharedTimeOfFighterObservable
        ,lastSharedTimeOfAgentCommand
    )
}

void OtherDataSanitizer::perceive(bool inReset){
    auto p=getShared<Fighter>(parent);
    if(p->isAlive_){
        std::pair<Time,nl::json> sharedAgentObservable={Time(),nl::json::object()};
        std::pair<Time,nl::json> sharedFighterObservable={Time(),nl::json::object()};
        if(p->isDatalinkEnabled){
            sharedAgentObservable=p->communicationBuffers[p->datalinkName].lock()->receive("agentObservables");
            if(!sharedAgentObservable.first){//invalid data
                sharedAgentObservable.second=nl::json::object();
            }
            sharedFighterObservable=p->communicationBuffers[p->datalinkName].lock()->receive("fighterObservables");
            if(!sharedFighterObservable.first){//invalid data
                sharedFighterObservable.second=nl::json::object();
            }
        }
        nl::json& sa=p->observables["shared"]["agent"];
        sa=nl::json::object();
        for(auto&& e:sharedAgentObservable.second.items()){
            Time sent(e.value().at("time"));
            if(lastSharedTimeOfAgentObservable.count(e.key())==0 || lastSharedTimeOfAgentObservable[e.key()]<sent){
                lastSharedTimeOfAgentObservable[e.key()]=sent;
                sa[e.key()]=e.value().at("obs");
            }
        }
        if(!p->agent.expired()){
            sa.merge_patch(p->agent.lock()->observables);
        }
        nl::json& sf=p->observables["shared"]["fighter"];
        for(auto&& e:sharedFighterObservable.second.items()){
            Time sent(e.value().at("time"));
            sf[e.key()]["isAlive"]=e.value().at("isAlive");
            if(lastSharedTimeOfFighterObservable.count(e.key())==0 || lastSharedTimeOfFighterObservable[e.key()]<sent){
                lastSharedTimeOfFighterObservable[e.key()]=sent;
                sf[e.key()]["spec"]=e.value().at("spec");
                if(e.value().at("isAlive").get<bool>()){
                    sf[e.key()]["motion"]=e.value().at("motion");
                }
                sf[e.key()]["motion"]=e.value().at("motion");
                sf[e.key()]["propulsion"]=e.value().at("propulsion");
                sf[e.key()]["weapon"]=e.value().at("weapon");
            }
        }
    }
}
void OtherDataSanitizer::control(){
    auto p=getShared<Fighter>(parent);
    if(p->isAlive_){
        nl::json agentCommand={
            {"motion",{
                {"roll",0.0},
                {"pitch",0.0},
                {"yaw",0.0},
                {"accel",0.0}
            }},
            {"weapon",{
                {"launch",false},
                {"target",Track3D()}
            }}
        };
        if(!p->agent.expired()){
            agentCommand=p->agent.lock()->commands.at(p->getFullName());
        }else{
            if(p->isDatalinkEnabled){
                auto tmp=p->communicationBuffers[p->datalinkName].lock()->receive("agentCommands");
                if(tmp.first){//valid data
                    auto sharedAgentCommand=tmp.second;
                    if(sharedAgentCommand.contains(p->getFullName())){
                        Time sent=sharedAgentCommand.at(p->getFullName()).at("time");
                        if(!lastSharedTimeOfAgentCommand || lastSharedTimeOfAgentCommand<sent){
                            lastSharedTimeOfAgentCommand=sent;
                            agentCommand=sharedAgentCommand.at(p->getFullName()).at("com");
                        }
                    }
                }
            }
        }
        p->commands["fromAgent"]=agentCommand;
        p->commands["motion"]=agentCommand["motion"];//もしHumanInterventionが無い場合はそのまま使われる
        p->commands["weapon"]=agentCommand["weapon"];//もしHumanInterventionが無い場合はそのまま使われる
    }
}
void HumanIntervention::initialize(){
    BaseType::initialize();
    capacity=getValueFromJsonKRD(modelConfig,"capacity",randomGen,1);
    delay=getValueFromJsonKRD(modelConfig,"delay",randomGen,3.0);
    cooldown=getValueFromJsonKRD(modelConfig,"cooldown",randomGen,0.999);
    lastShotApprovalTime=manager->getTime()-cooldown;
}
void HumanIntervention::serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full){
    BaseType::serializeInternalState(archive,full);

    if(full){
        ASRC_SERIALIZE_NVP(archive
            ,capacity
            ,delay
            ,cooldown
        )
    }
    ASRC_SERIALIZE_NVP(archive
        ,lastShotApprovalTime
        ,recognizedShotCommands
    )
}

bool HumanIntervention::isLaunchable(){
    return recognizedShotCommands.size()<capacity;
}
bool HumanIntervention::isLaunchableAt(const Track3D& target_){
    return isLaunchable() && !target_.is_none();
}
void HumanIntervention::control(){
    auto p=getShared<Fighter>(parent);
    if(p->isAlive_){
        p->commands["weapon"]={
            {"launch",false},
            {"target",Track3D()}
        };
        if(p->isLaunchableAt(p->commands["fromAgent"]["weapon"]["target"]) && p->commands["fromAgent"]["weapon"]["launch"]){
            if(recognizedShotCommands.size()<capacity
                && (recognizedShotCommands.size()==0 || manager->getTime()>=recognizedShotCommands.back().first+cooldown)
                && manager->getTime()>=lastShotApprovalTime+cooldown-delay
            ){
                recognizedShotCommands.push_back(std::make_pair(manager->getTime(),p->commands["fromAgent"]["weapon"]["target"]));
            }
        }
        if(recognizedShotCommands.size()>0){
            auto front=recognizedShotCommands.front();
            recognizedShotCommands.pop_front();
            if(manager->getTime()>=front.first+delay && p->isLaunchableAt(front.second)){
                //承認
                p->commands["weapon"]={
                    {"launch",true},
                    {"target",front.second}
                };
                lastShotApprovalTime=manager->getTime();
            }else{
                //判断中
                recognizedShotCommands.push_front(front);
            }
        }
        p->commands["motion"]=p->commands["fromAgent"]["motion"];
    }
}
void WeaponController::initialize(){
    BaseType::initialize();
    enable_loal=getValueFromJsonKRD(modelConfig,"enable_loal",randomGen,true);
    launchRangeLimit=getValueFromJsonKRD(modelConfig,"launchRangeLimit",randomGen,-1.0);
    offBoresightAngleLimit=deg2rad(getValueFromJsonKRD(modelConfig,"offBoresightAngleLimit",randomGen,-1.0));
    enable_utdc=getValueFromJsonKRD(modelConfig,"enable_utdc",randomGen,true);
    enable_takeover_guidance=getValueFromJsonKRD(modelConfig,"enable_takeover_guidance",randomGen,true);
}

void WeaponController::serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full){
    BaseType::serializeInternalState(archive,full);

    if(full){
        ASRC_SERIALIZE_NVP(archive
            ,enable_loal
            ,launchRangeLimit
            ,offBoresightAngleLimit
            ,enable_utdc
            ,enable_takeover_guidance
        )
    }
}

bool WeaponController::isLaunchable(){
    auto p=getShared<Fighter>(parent);
    return p->remMsls>0;
}
bool WeaponController::isLaunchableAt(const Track3D& target_){
    bool ret = isLaunchable() && !target_.is_none();
    auto p=getShared<Fighter>(parent);
    Eigen::Vector3d rpos=p->absPtoB(target_.pos(p->getParentCRS()));
    double L=rpos.norm();
    if(launchRangeLimit>0){
        ret = ret && L<=launchRangeLimit;
    }
    if(offBoresightAngleLimit>0){
        ret = ret && rpos(0)>L*cos(offBoresightAngleLimit);
    }
    if(ret){
        if(!enable_loal){
            double Lref=p->missiles[p->nextMsl].lock()->sensor.lock()->observables.at("/spec/Lref"_json_pointer);
            double thetaFOR=p->missiles[p->nextMsl].lock()->sensor.lock()->observables.at("/spec/thetaFOR"_json_pointer);
            return L<=Lref && rpos(0)>L*cos(thetaFOR);
        }
    }
    return ret;
}
void WeaponController::control(){
    auto p=getShared<Fighter>(parent);
    if(p->isAlive_){
        if(!p->commands.contains("weapon")){
            p->commands["weapon"]={
                {"launch",false},
                {"target",Track3D()}
            };
        }
        std::pair<bool,Track3D> trackingInfo=p->isTracking(p->commands["weapon"]["target"].get<Track3D>());
        p->target=trackingInfo.second;
        bool launchFlag=p->commands["weapon"]["launch"].get<bool>() && trackingInfo.first;
        if(!p->isLaunchableAt(p->target)){
            p->commands["weapon"]={
                {"launch",false},
                {"target",Track3D()}
            };
        }else if(launchFlag){
            p->missileTargets[p->nextMsl]=std::make_pair(p->target,true);
            p->communicationBuffers["MissileComm:"+p->missiles[p->nextMsl].lock()
                ->getFullName()].lock()->send({
                    {"launch",true},
                    {"target",p->missileTargets[p->nextMsl].first}
                },CommunicationBuffer::MERGE);
            p->nextMsl+=1;
            p->remMsls-=1;
            manager->triggerEvent("Shot",{this->weak_from_this()});
            p->commands["weapon"]={
                {"launch",false},
                {"target",Track3D()}
            };
        }
        if(enable_utdc){
            for(int mslId=0;mslId<p->nextMsl;++mslId){
                if(p->missileTargets[mslId].second){
                    p->communicationBuffers["MissileComm:"+p->missiles[mslId].lock()
                    ->getFullName()].lock()->send({
                        {"target",p->missileTargets[mslId].first}
                    },CommunicationBuffer::MERGE);
                }
            }
            if(enable_takeover_guidance){
                //撃墜された味方の射撃済誘導弾に対する制御を引き継ぎ
                nl::json& sf=p->observables["shared"]["fighter"];
                bool isFirst=false;//自身が生存機の中で先頭かどうか
                for(auto&& asset:manager->getAssets()){
                    if(asset.lock()->getTeam()==team && isinstance<Fighter>(asset)){
                        auto f=getShared<const Fighter>(asset);
                        if(f->isAlive()){
                            isFirst = f->getFullName()==p->getFullName();
                            break;
                        }
                    }
                }
                if(isFirst){
                    for(auto&& asset:manager->getAssets()){
                        if(asset.lock()->getTeam()==team && isinstance<Fighter>(asset)){
                            auto f=getShared<const Fighter>(asset);
                            if(!f->isAlive()){
                                for(auto&& e:f->missiles){
                                    auto msl=e.lock();
                                    if(msl->isAlive() && msl->hasLaunched){
                                        for(auto& t:p->track){
                                            if(msl->target.isSame(t)){
                                                p->communicationBuffers["MissileComm:"+msl->getFullName()].lock()->send({
                                                    {"target",t}
                                                },CommunicationBuffer::MERGE);
                                                break;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
void FlightController::initialize(){
    BaseType::initialize();
    mode="direct";
}

void FlightController::serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full){
    BaseType::serializeInternalState(archive,full);

    ASRC_SERIALIZE_NVP(archive
        ,mode
    )
}

void FlightController::control(){
    auto p=getShared<Fighter>(parent);
    if(p->isAlive_){
        if(!p->commands.contains("motion")){
            p->commands["motion"]=getDefaultCommand();
        }
        commands["motion"]=calc(p->commands["motion"]);
    }
}
nl::json FlightController::getDefaultCommand(){
    std::cout<<"Warning! FlightController::getDefaultCommand() should be overridden."<<std::endl;
    return nl::json::object();
}
nl::json FlightController::calc(const nl::json &cmd){
    std::cout<<"Warning! FlightController::calc(const nl::json&) should be overridden."<<std::endl;
    return cmd;
}
void FlightController::setMode(const std::string& ctrlName){
    mode=ctrlName;
}
FighterAccessor::FighterAccessor(const std::shared_ptr<Fighter>& f)
:PhysicalAssetAccessor(f)
,fighter(f)
{
}
void FighterAccessor::setFlightControllerMode(const std::string& ctrlName){
    fighter.lock()->setFlightControllerMode(ctrlName);
}
double FighterAccessor::getMaxReachableRange(){
    return fighter.lock()->getMaxReachableRange();
}
double FighterAccessor::getRmax(const Eigen::Vector3d &rs,const Eigen::Vector3d &vs,const Eigen::Vector3d &rt,const Eigen::Vector3d &vt){
    return fighter.lock()->getRmax(rs,vs,rt,vt);
}
double FighterAccessor::getRmax(const Eigen::Vector3d &rs,const Eigen::Vector3d &vs,const Eigen::Vector3d &rt,const Eigen::Vector3d &vt,const std::shared_ptr<CoordinateReferenceSystem>& crs){
    return fighter.lock()->getRmax(rs,vs,rt,vt,crs);
}
double FighterAccessor::getRmax(const Eigen::Vector3d &rs,const Eigen::Vector3d &vs,const Eigen::Vector3d &rt,const Eigen::Vector3d &vt,const double& aa){
    return fighter.lock()->getRmax(rs,vs,rt,vt,aa);
}
double FighterAccessor::getRmax(const Eigen::Vector3d &rs,const Eigen::Vector3d &vs,const Eigen::Vector3d &rt,const Eigen::Vector3d &vt,const double& aa,const std::shared_ptr<CoordinateReferenceSystem>& crs){
    return fighter.lock()->getRmax(rs,vs,rt,vt,aa,crs);
}
bool FighterAccessor::isLaunchable(){
    return fighter.lock()->isLaunchable();
}
bool FighterAccessor::isLaunchableAt(const Track3D& target_){
    return fighter.lock()->isLaunchableAt(target_);
}
std::shared_ptr<Missile> FighterAccessor::launchInVirtual(const std::shared_ptr<SimulationManager>& dstManager,const nl::json& launchCommand){
    return fighter.lock()->launchInVirtual(dstManager,launchCommand);
}

void exportFighter(py::module &m, const std::shared_ptr<asrc::core::FactoryHelper>& factoryHelper)
{
    using namespace pybind11::literals;
    expose_entity_subclass<Fighter>(m,"Fighter")
    DEF_FUNC(Fighter,getThrust)
    DEF_FUNC(Fighter,calcThrust)
    DEF_FUNC(Fighter,calcOptimumCruiseFuelFlowRatePerDistance)
    DEF_FUNC(Fighter,getMaxReachableRange)
    .def("isTracking",py::overload_cast<const std::weak_ptr<PhysicalAsset>&>(&Fighter::isTracking))
    .def("isTracking",py::overload_cast<const Track3D&>(&Fighter::isTracking))
    .def("isTracking",py::overload_cast<const boost::uuids::uuid&>(&Fighter::isTracking))
    DEF_FUNC(Fighter,isLaunchable)
    DEF_FUNC(Fighter,isLaunchableAt)
    DEF_FUNC(Fighter,launchInVirtual)
    DEF_FUNC(Fighter,setFlightControllerMode)
    DEF_FUNC(Fighter,calcMotion)
    DEF_FUNC(Fighter,toEulerAngle)
    .def("getRmax",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const Eigen::Vector3d&,const Eigen::Vector3d&>(&Fighter::getRmax))
    .def("getRmax",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const Eigen::Vector3d&,const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&>(&Fighter::getRmax))
    .def("getRmax",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const Eigen::Vector3d&,const Eigen::Vector3d&,const double&>(&Fighter::getRmax))
    .def("getRmax",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const Eigen::Vector3d&,const Eigen::Vector3d&,const double&,const std::shared_ptr<CoordinateReferenceSystem>&>(&Fighter::getRmax))
    DEF_READWRITE(Fighter,m)
    DEF_READWRITE(Fighter,rcsScale)
    DEF_READWRITE(Fighter,fuelCapacity)
    DEF_READWRITE(Fighter,engine)
    DEF_READWRITE(Fighter,radar)
    DEF_READWRITE(Fighter,mws)
    DEF_READWRITE(Fighter,missiles)
    DEF_READWRITE(Fighter,dummyMissile)
    DEF_READWRITE(Fighter,missileTargets)
    DEF_READWRITE(Fighter,nextMsl)
    DEF_READWRITE(Fighter,numMsls)
    DEF_READWRITE(Fighter,remMsls)
    DEF_READWRITE(Fighter,isDatalinkEnabled)
    DEF_READWRITE(Fighter,track)
    DEF_READWRITE(Fighter,trackSource)
    DEF_READWRITE(Fighter,datalinkName)
    DEF_READWRITE(Fighter,target)
    DEF_READWRITE(Fighter,targetID)
    DEF_READWRITE(Fighter,fuelRemaining)
    DEF_READWRITE(Fighter,optCruiseFuelFlowRatePerDistance)
    ;
    //FACTORY_ADD_CLASS(PhysicalAsset,Fighter) //Do not register to Factory because Fighter is abstract class.

    expose_entity_subclass<SensorDataSharer>(m,"SensorDataSharer")
    ;
    FACTORY_ADD_CLASS(Controller,SensorDataSharer)

    expose_entity_subclass<SensorDataSanitizer>(m,"SensorDataSanitizer")
    DEF_READWRITE(SensorDataSanitizer,lastSharedTime)
    ;
    FACTORY_ADD_CLASS(Controller,SensorDataSanitizer)

    expose_entity_subclass<OtherDataSharer>(m,"OtherDataSharer")
    ;
    FACTORY_ADD_CLASS(Controller,OtherDataSharer)

    expose_entity_subclass<OtherDataSanitizer>(m,"OtherDataSanitizer")
    DEF_READWRITE(OtherDataSanitizer,lastSharedTimeOfAgentObservable)
    DEF_READWRITE(OtherDataSanitizer,lastSharedTimeOfFighterObservable)
    DEF_READWRITE(OtherDataSanitizer,lastSharedTimeOfAgentCommand)
    ;
    FACTORY_ADD_CLASS(Controller,OtherDataSanitizer)

    expose_entity_subclass<HumanIntervention>(m,"HumanIntervention")
    DEF_FUNC(HumanIntervention,isLaunchable)
    DEF_FUNC(HumanIntervention,isLaunchableAt)
    DEF_READWRITE(HumanIntervention,capacity)
    DEF_READWRITE(HumanIntervention,recognizedShotCommands)
    ;
    FACTORY_ADD_CLASS(Controller,HumanIntervention)

    expose_entity_subclass<WeaponController>(m,"WeaponController")
    DEF_FUNC(WeaponController,isLaunchable)
    DEF_FUNC(WeaponController,isLaunchableAt)
    ;
    FACTORY_ADD_CLASS(Controller,WeaponController)

    expose_entity_subclass<FlightController>(m,"FlightController")
    DEF_FUNC(FlightController,getDefaultCommand)
    DEF_FUNC(FlightController,calc)
    DEF_FUNC(FlightController,setMode)
    ;
    //FACTORY_ADD_CLASS(Controller,FlightController)  //Do not register to Factory because FlightController is abstract class.

    expose_common_class<FighterAccessor>(m,"FighterAccessor")
    .def(py_init<const std::shared_ptr<Fighter>&>())
    DEF_FUNC(FighterAccessor,setFlightControllerMode)
    DEF_FUNC(FighterAccessor,getMaxReachableRange)
    .def("getRmax",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const Eigen::Vector3d&,const Eigen::Vector3d&>(&FighterAccessor::getRmax))
    .def("getRmax",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const Eigen::Vector3d&,const Eigen::Vector3d&,const std::shared_ptr<CoordinateReferenceSystem>&>(&FighterAccessor::getRmax))
    .def("getRmax",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const Eigen::Vector3d&,const Eigen::Vector3d&,const double&>(&FighterAccessor::getRmax))
    .def("getRmax",py::overload_cast<const Eigen::Vector3d&,const Eigen::Vector3d&,const Eigen::Vector3d&,const Eigen::Vector3d&,const double&,const std::shared_ptr<CoordinateReferenceSystem>&>(&FighterAccessor::getRmax))
    DEF_FUNC(FighterAccessor,isLaunchable)
    DEF_FUNC(FighterAccessor,isLaunchableAt)
    DEF_FUNC(FighterAccessor,launchInVirtual)
    ;
}

ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
