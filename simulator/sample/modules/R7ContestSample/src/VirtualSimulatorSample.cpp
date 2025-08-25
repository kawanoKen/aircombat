// Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#include "VirtualSimulatorSample.h"
#include <ASRCAISim1/Fighter.h>

ASRC_PLUGIN_NAMESPACE_BEGIN

using namespace asrc::core;
using namespace util;

using namespace BasicAgentUtility::util;

VirtualSimulatorSample::VirtualSimulatorSample(const nl::json& config){
	maxNumPerParent=getValueFromJsonKD(config,"maxNumPerParent",10);
	launchInterval=getValueFromJsonKD(config,"launchInterval",5);
	kShoot=getValueFromJsonKD(config,"kShoot",0.5);
}

void VirtualSimulatorSample::createVirtualSimulator(const std::shared_ptr<asrc::core::Agent>& agent){
	nl::json option={
        {"patch",{
            {"Manager",{
                {"TimeStep",{
                    {"defaultAgentStepInterval",agent->interval[SimPhase::AGENT_STEP]}
                }},
                {"skipNoAgentStep",false}
            }}
        }}
    };
    virtualSimulator=agent->manager->createVirtualSimulationManager(option);
    virtualSimulator->reset(std::nullopt,std::nullopt);
}

void VirtualSimulatorSample::step(const std::shared_ptr<asrc::core::Agent>& agent, const std::vector<Track3D>& tracks){
    auto now=agent->manager->getTickCount();
	for(auto&& [port, parent] : agent->parents){
		auto fullName=parent->getFullName();
		if(parent->isAlive()){
			MotionState motion(parent->observables.at("motion"));
			auto crs=agent->manager->getRootCRS();
			auto time=agent->manager->getTime();
			Track3D selfTrack(
				parent->getUUIDForTrack(),
				crs,
				time,
				motion.pos(crs),
				motion.vel(crs)
			);
			{
				// 仮想誘導弾の諸元を更新
				auto it=virtualMissiles[fullName].begin();
				minimumDistances[fullName]=std::numeric_limits<double>::infinity();
				bool hit=false;
				while(it!=virtualMissiles[fullName].end()){
					auto missile=*it;
					assert(missile);
					if(missile->isAlive()){
						missile->communicationBuffers["MissileComm:"+missile->getFullName()].lock()->send({
							{"target",selfTrack}
						},CommunicationBuffer::UpdateType::MERGE);

						double dt=agent->interval[SimPhase::AGENT_STEP]*agent->manager->getBaseTimeStep();
						if(missile->hitCheck(motion.pos(missile->getParentCRS()),motion.pos(missile->getParentCRS())-motion.vel(missile->getParentCRS())*dt)){
							// 命中
							missile->manager->requestToKillAsset(missile);
							hit=true;
						}
						minimumDistances[fullName]=std::min<double>(
							minimumDistances[fullName],
							(missile->pos()-motion.pos(missile->getParentCRS())).norm()
						);
						++it;
					}else{
						// 飛翔終了していたら消す
						it=virtualMissiles[fullName].erase(it);
					}
				}
				//std::cout<<fullName<<" tick="<<agent->manager->getTickCount()<<": virtualMissiles size="<<virtualMissiles[fullName].size()<<", minimumDistance="<<minimumDistances[fullName]<<", hit="<<hit<<std::endl;
			}
		}else{
			// parentが消えたら仮想誘導弾も消す
			if(virtualMissiles.contains(fullName)){
				for(auto&& missile : virtualMissiles[fullName]){
					missile->manager->requestToKillAsset(missile);
				}
				virtualMissiles.erase(virtualMissiles.find(fullName));
				minimumDistances.erase(minimumDistances.find(fullName));
			}
		}
	}

	// 仮想シミュレータの時間を進める
    while(virtualSimulator->getTickCount()<now){
        virtualSimulator->step(py::dict());
    }

	// 生存中のparentsに対して、新たな仮想誘導弾を生成する
	for(auto&& [port, parent] : agent->parents){
		auto fullName=parent->getFullName();
		if(parent->isAlive()){
			MotionState motion(parent->observables.at("motion"));
			auto crs=agent->manager->getRootCRS();
			auto time=agent->manager->getTime();
			Track3D selfTrack(
				parent->getUUIDForTrack(),
				crs,
				time,
				motion.pos(crs),
				motion.vel(crs)
			);
    		for(auto&& t:tracks){
				//std::cout<<"launch virtual missile at tick="<<manager->getTickCount()<<" (v "<<virtualSimulator->getTickCount()<<")"<<std::endl;
				if(virtualMissiles[fullName].size()<maxNumPerParent){
					MotionState shooterMotion(
						crs,
						time,
						t.pos(crs),
						t.vel(crs),
						Eigen::Vector3d(0,0,0),
						Quaternion(1,0,0,0),
						"FSD"
					);
					if(launchCondition(parent,shooterMotion,selfTrack)){
						nl::json launchCommand={
							{"target",selfTrack},
							{"motion",shooterMotion}
						};
						virtualMissiles[fullName].push_back(getShared<FighterAccessor>(parent)->launchInVirtual(virtualSimulator,launchCommand));
					}
				}
			}
		}
    }
}

bool VirtualSimulatorSample::launchCondition(const std::shared_ptr<asrc::core::PhysicalAssetAccessor>& parent,const MotionState& motion, const Track3D& track){
	return calcRNorm(parent,motion,track,false) <= kShoot;
}

void exportVirtualSimulatorSample(py::module &m)
{
	using namespace pybind11::literals;
	bind_stl_container<std::map<std::string,std::vector<std::shared_ptr<asrc::core::Missile>>>>(m,py::module_local(true));
    expose_common_class<VirtualSimulatorSample>(m,"VirtualSimulatorSample",py::module_local(true))
	ASRC_DEF_FUNC(VirtualSimulatorSample,createVirtualSimulator)
	ASRC_DEF_FUNC(VirtualSimulatorSample,step)
	ASRC_DEF_FUNC(VirtualSimulatorSample,launchCondition)
	ASRC_DEF_READWRITE(VirtualSimulatorSample,maxNumPerParent)
	ASRC_DEF_READWRITE(VirtualSimulatorSample,launchInterval)
	ASRC_DEF_READWRITE(VirtualSimulatorSample,virtualSimulator)
	ASRC_DEF_READWRITE(VirtualSimulatorSample,virtualMissiles)
	ASRC_DEF_READWRITE(VirtualSimulatorSample,minimumDistances)
	;
}

ASRC_PLUGIN_NAMESPACE_END
