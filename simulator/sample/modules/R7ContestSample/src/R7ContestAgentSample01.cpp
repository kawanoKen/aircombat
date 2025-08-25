// Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#include "R7ContestAgentSample01.h"
#include <utility>
#include <algorithm>
#include <iomanip>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <ASRCAISim1/Utility.h>
#include <ASRCAISim1/MathUtility.h>
#include <ASRCAISim1/Units.h>
#include <ASRCAISim1/FlightControllerUtility.h>
#include <ASRCAISim1/CoordinatedFighter.h>
#include <ASRCAISim1/SixDoFFighter.h>
#include <ASRCAISim1/Missile.h>
#include <ASRCAISim1/Track.h>
#include <ASRCAISim1/SimulationManager.h>
#include <ASRCAISim1/Ruler.h>
#include <ASRCAISim1/StaticCollisionAvoider2D.h>

ASRC_PLUGIN_NAMESPACE_BEGIN

using namespace asrc::core;
using namespace util;
using namespace R7ContestModels;

using namespace BasicAgentUtility::util;

R7ContestAgentSample01::ActionInfo::ActionInfo(){
	dstDir<<1,0,0;
	dstAlt=10000.0;
	velRecovery=false;
	asThrottle=false;
	keepVel=false;
	dstThrottle=1.0;
	dstV=300.0;
	launchFlag=false;
	target=Track3D();
}

void R7ContestAgentSample01::initialize(){
    BaseType::initialize();
	//modelConfigの読み込み
	//observation spaceの設定
	maxParentNum=getValueFromJsonK(modelConfig,"maxParentNum");
	maxFriendNum=getValueFromJsonK(modelConfig,"maxFriendNum");
	maxEnemyNum=getValueFromJsonK(modelConfig,"maxEnemyNum");
	maxFriendMissileNum=getValueFromJsonK(modelConfig,"maxFriendMissileNum");
	maxEnemyMissileNum=getValueFromJsonK(modelConfig,"maxEnemyMissileNum");
	assert(maxParentNum>=parents.size());

	horizontalNormalizer=getValueFromJsonKR(modelConfig,"horizontalNormalizer",randomGen);
	verticalNormalizer=getValueFromJsonKR(modelConfig,"verticalNormalizer",randomGen);
	fgtrVelNormalizer=getValueFromJsonKR(modelConfig,"fgtrVelNormalizer",randomGen);
	mslVelNormalizer=getValueFromJsonKR(modelConfig,"mslVelNormalizer",randomGen);

	use_observation_mask=getValueFromJsonK(modelConfig,"use_observation_mask");
	use_action_mask=getValueFromJsonK(modelConfig,"use_action_mask");
	raise_when_invalid_action=getValueFromJsonK(modelConfig,"raise_when_invalid_action");

	// 仮想シミュレータの使用
	// この例では各parentに最も近い仮想誘導弾との距離をparentのobservationに付加する
	use_virtual_simulator=getValueFromJsonK(modelConfig,"use_virtual_simulator");
	if(use_virtual_simulator){
		virtual_simulator_value_when_no_missile=getValueFromJsonK(modelConfig,"virtual_simulator_value_when_no_missile");
		nl::json sub={
			{"maxNumPerParent",getValueFromJsonK(modelConfig,"virtual_simulator_maxNumPerParent")},
			{"launchInterval",getValueFromJsonK(modelConfig,"virtual_simulator_launchInterval")},
			{"kShoot",getValueFromJsonK(modelConfig,"virtual_simulator_kShoot")}
		};
		virtualSimulator=std::make_shared<VirtualSimulatorSample>(sub);
	}

	//  2次元画像としてのobservation
	use_image_observation=getValueFromJsonK(modelConfig,"use_image_observation");
	if(use_image_observation){
		image_longitudinal_resolution=getValueFromJsonK(modelConfig,"image_longitudinal_resolution");
		image_lateral_resolution=getValueFromJsonK(modelConfig,"image_lateral_resolution");
		image_front_range=getValueFromJsonKR(modelConfig,"image_front_range",randomGen);
		image_back_range=getValueFromJsonKR(modelConfig,"image_back_range",randomGen);
		image_side_range=getValueFromJsonKR(modelConfig,"image_side_range",randomGen);
		image_horizon=getValueFromJsonKR(modelConfig,"image_horizon",randomGen);
		image_interval=getValueFromJsonKR(modelConfig,"image_interval",randomGen);
		image_rotate=getValueFromJsonKRD(modelConfig,"image_rotate",randomGen,false);
		image_relative_position=getValueFromJsonKRD(modelConfig,"image_relative_position",randomGen,true);
		draw_parent_trajectory=getValueFromJsonK(modelConfig,"draw_parent_trajectory");
		draw_friend_trajectory=getValueFromJsonK(modelConfig,"draw_friend_trajectory");
		draw_enemy_trajectory=getValueFromJsonK(modelConfig,"draw_enemy_trajectory");
		draw_friend_missile_trajectory=getValueFromJsonK(modelConfig,"draw_friend_missile_trajectory");
		draw_parent_radar_coverage=getValueFromJsonK(modelConfig,"draw_parent_radar_coverage");
		draw_friend_radar_coverage=getValueFromJsonK(modelConfig,"draw_friend_radar_coverage");
		draw_defense_line=getValueFromJsonK(modelConfig,"draw_defense_line");
		draw_side_line=getValueFromJsonK(modelConfig,"draw_side_line");
	}

	//  共通情報に関するobservation
	use_remaining_time=getValueFromJsonK(modelConfig,"use_remaining_time");
	if(use_remaining_time){
		remaining_time_clipping=getValueFromJsonKR(modelConfig,"remaining_time_clipping",randomGen);
	}
	common_dim=(
		(use_remaining_time ? 1 : 0)
	);

	//  parentsの状態量に関するobservation
	use_parent_last_action=getValueFromJsonK(modelConfig,"use_parent_last_action");
	// parent_dimはaction spaceの設定を読み込むまで確定しないので後回し

	//  味方機の状態量に関するobservation
	use_friend_position=getValueFromJsonK(modelConfig,"use_friend_position");
	use_friend_velocity=getValueFromJsonK(modelConfig,"use_friend_velocity");
	use_friend_initial_num_missile=getValueFromJsonK(modelConfig,"use_friend_initial_num_missile");
	use_friend_current_num_missile=getValueFromJsonK(modelConfig,"use_friend_current_num_missile");
	use_friend_attitude=getValueFromJsonK(modelConfig,"use_friend_attitude");
	use_friend_angular_velocity=getValueFromJsonK(modelConfig,"use_friend_angular_velocity");
	use_friend_current_fuel=getValueFromJsonK(modelConfig,"use_friend_current_fuel");
	use_friend_rcs_scale=getValueFromJsonK(modelConfig,"use_friend_rcs_scale");
	use_friend_radar_range=getValueFromJsonK(modelConfig,"use_friend_radar_range");
	use_friend_radar_coverage=getValueFromJsonK(modelConfig,"use_friend_radar_coverage");
	use_friend_maximum_speed_scale=getValueFromJsonK(modelConfig,"use_friend_maximum_speed_scale");
	use_friend_missile_thrust_scale=getValueFromJsonK(modelConfig,"use_friend_missile_thrust_scale");
	friend_dim=(
		(use_friend_position ? 3 : 0)
		+(use_friend_velocity ? 4 : 0)
		+(use_friend_initial_num_missile ? 1 : 0)
		+(use_friend_current_num_missile ? 1 : 0)
		+(use_friend_attitude ? 3 : 0)
		+(use_friend_angular_velocity ? 3 : 0)
		+(use_friend_current_fuel ? 1 : 0)
		+(use_friend_rcs_scale ? 1 : 0)
		+(use_friend_radar_range ? 1 : 0)
		+(use_friend_radar_coverage ? 1 : 0)
		+(use_friend_maximum_speed_scale ? 1 : 0)
		+(use_friend_missile_thrust_scale ? 1 : 0)
	);

	//  彼機の状態量に関するobservation
	use_enemy_position=getValueFromJsonK(modelConfig,"use_enemy_position");
	use_enemy_velocity=getValueFromJsonK(modelConfig,"use_enemy_velocity");
	enemy_dim=(
		(use_enemy_position ? 3 : 0)
		+(use_enemy_velocity ? 4 : 0)
	);

	//  味方誘導弾の状態量に関するobservation
	use_friend_missile_position=getValueFromJsonK(modelConfig,"use_friend_missile_position");
	use_friend_missile_velocity=getValueFromJsonK(modelConfig,"use_friend_missile_velocity");
	use_friend_missile_flight_time=getValueFromJsonK(modelConfig,"use_friend_missile_flight_time");
	use_friend_missile_target_distance=getValueFromJsonK(modelConfig,"use_friend_missile_target_distance");
	use_friend_missile_target_mode=getValueFromJsonK(modelConfig,"use_friend_missile_target_mode");
	use_friend_missile_target_position=getValueFromJsonK(modelConfig,"use_friend_missile_target_position");
	use_friend_missile_target_velocity=getValueFromJsonK(modelConfig,"use_friend_missile_target_velocity");
	friend_missile_dim=(
		(use_friend_missile_position ? 3 : 0)
		+(use_friend_missile_velocity ? 4 : 0)
		+(use_friend_missile_flight_time ? 1 : 0)
		+(use_friend_missile_target_distance ? 1 : 0)
		+(use_friend_missile_target_mode ? 3 : 0)
		+(use_friend_missile_target_position ? 3 : 0)
		+(use_friend_missile_target_velocity ? 4 : 0)
		+(use_friend_missile_thrust_scale ? 1 : 0)
	);

	//  彼側誘導弾の状態量に関するobservation
	use_enemy_missile_observer_position=getValueFromJsonK(modelConfig,"use_enemy_missile_observer_position");
	use_enemy_missile_direction=getValueFromJsonK(modelConfig,"use_enemy_missile_direction");
	use_enemy_missile_angular_velocity=getValueFromJsonK(modelConfig,"use_enemy_missile_angular_velocity");
	enemy_missile_dim=(
		(use_enemy_missile_observer_position ? 3 : 0)
		+(use_enemy_missile_direction ? 3 : 0)
		+(use_enemy_missile_angular_velocity ? 3 : 0)
	);

	// 我機と彼機の間の関係性に関するobservation
	use_our_missile_range=getValueFromJsonK(modelConfig,"use_our_missile_range");
	use_their_missile_range=getValueFromJsonK(modelConfig,"use_their_missile_range");
	friend_enemy_relative_dim=(
		(use_our_missile_range ? 3 : 0)
		+(use_their_missile_range ? 3 : 0)
	);

	//action spaceの設定
	//  左右旋回に関する設定
	dstAz_relative=getValueFromJsonK(modelConfig,"dstAz_relative");
	turnTable=getValueFromJsonK(modelConfig,"turnTable");
	turnTable*=deg2rad(1.0);
	std::sort(turnTable.begin(),turnTable.end());
	use_override_evasion=getValueFromJsonK(modelConfig,"use_override_evasion");
	if(use_override_evasion){
		evasion_turnTable=getValueFromJsonK(modelConfig,"evasion_turnTable");
		evasion_turnTable*=deg2rad(1.0);
		std::sort(evasion_turnTable.begin(),evasion_turnTable.end());
		assert(turnTable.size()==evasion_turnTable.size());
	}else{
		evasion_turnTable=turnTable;
	}
	//  上昇・下降に関する設定
	no_vertical_maneuver=getValueFromJsonK(modelConfig,"no_vertical_maneuver");
	if(!no_vertical_maneuver){
		use_altitude_command=getValueFromJsonK(modelConfig,"use_altitude_command");
		if(use_altitude_command){
			altTable=getValueFromJsonK(modelConfig,"altTable");
			std::sort(altTable.begin(),altTable.end());
			refAltInterval=getValueFromJsonK(modelConfig,"refAltInterval");
		}else{
			pitchTable=getValueFromJsonK(modelConfig,"pitchTable");
			std::sort(pitchTable.begin(),pitchTable.end());
			pitchTable*=deg2rad(1.0);
		}
	}else{
		use_altitude_command=false;
	}
	//  加減速に関する設定
	accelTable=getValueFromJsonK(modelConfig,"accelTable");
	std::sort(accelTable.begin(),accelTable.end());
	always_maxAB=getValueFromJsonK(modelConfig,"always_maxAB");
	//  射撃に関する設定
	use_Rmax_fire=getValueFromJsonK(modelConfig,"use_Rmax_fire");
	if(use_Rmax_fire){
		shotIntervalTable=getValueFromJsonK(modelConfig,"shotIntervalTable");
		std::sort(shotIntervalTable.begin(),shotIntervalTable.end());
		shotThresholdTable=getValueFromJsonK(modelConfig,"shotThresholdTable");
		std::sort(shotThresholdTable.begin(),shotThresholdTable.end());
	}
	//行動制限に関する設定
	//  高度制限に関する設定
	altMin=getValueFromJsonKRD(modelConfig,"altMin",randomGen,2000.0);
	altMax=getValueFromJsonKRD(modelConfig,"altMax",randomGen,15000.0);
	if(modelConfig.contains("altitudeKeeper")){
		nl::json sub=modelConfig.at("altitudeKeeper");
		altitudeKeeper=AltitudeKeeper(sub);
	}
	//  場外制限に関する設定
	dOutLimit=getValueFromJsonKRD(modelConfig,"dOutLimit",randomGen,5000.0);
    dOutLimitThreshold=getValueFromJsonKRD(modelConfig,"dOutLimitThreshold",randomGen,10000.0);
	dOutLimitStrength=getValueFromJsonKRD(modelConfig,"dOutLimitStrength",randomGen,2e-3);
	//  同時射撃数の制限に関する設定
	maxSimulShot=getValueFromJsonKRD(modelConfig,"maxSimulShot",randomGen,4);
	//  下限速度の制限に関する設定
	minimumV=getValueFromJsonKRD(modelConfig,"minimumV",randomGen,150.0);
	minimumRecoveryV=getValueFromJsonKRD(modelConfig,"minimumRecoveryV",randomGen,180.0);
	minimumRecoveryDstV=getValueFromJsonKRD(modelConfig,"minimumRecoveryDstV",randomGen,200.0);

	//前回の行動に関する情報
	last_action_dim=3+(1+maxEnemyNum);
	for(auto&& [port,parent] : parents){
		last_action_obs[parent->getFullName()]=Eigen::VectorXf::Zero(last_action_dim);
		actionInfos[parent->getFullName()]=ActionInfo();
	}

	//  parentsの状態量に関するobservation
	parent_dim=(
		(use_parent_last_action ? last_action_dim : 0)
		+(use_virtual_simulator ? 1 : 0)
	)+friend_dim;

	//spaceの確定
	py::module_ spaces=py::module_::import("gymnasium.spaces");
	float floatLow=std::numeric_limits<float>::lowest();
	float floatHigh=std::numeric_limits<float>::max();
	//  actionに関するもの
	py::dict single_action_space_dict;
	py::dict single_action_mask_space_dict;
	single_action_space_dict["turn"]=spaces.attr("Discrete")(turnTable.size());
	if(use_action_mask){
        Eigen::VectorXi shape=Eigen::VectorXi::Constant(1,turnTable.size());
        single_action_mask_space_dict["turn"]=spaces.attr("Box")(floatLow,floatHigh,shape,py::dtype::of<float>());
	}
	if(!no_vertical_maneuver){
		single_action_space_dict["pitch"]=spaces.attr("Discrete")(use_altitude_command ? altTable.size() : pitchTable.size());
		if(use_action_mask){
			Eigen::VectorXi shape=Eigen::VectorXi::Constant(1,use_altitude_command ? altTable.size() : pitchTable.size());
			single_action_mask_space_dict["pitch"]=spaces.attr("Box")(floatLow,floatHigh,shape,py::dtype::of<float>());
		}
	}
	if(!always_maxAB){
		single_action_space_dict["accel"]=spaces.attr("Discrete")(accelTable.size());
		if(use_action_mask){
			Eigen::VectorXi shape=Eigen::VectorXi::Constant(1,accelTable.size());
			single_action_mask_space_dict["accel"]=spaces.attr("Box")(floatLow,floatHigh,shape,py::dtype::of<float>());
		}
	}
	single_action_space_dict["target"]=spaces.attr("Discrete")(1+maxEnemyNum);
	if(use_action_mask){
		Eigen::VectorXi shape=Eigen::VectorXi::Constant(1,1+maxEnemyNum);
		single_action_mask_space_dict["target"]=spaces.attr("Box")(floatLow,floatHigh,shape,py::dtype::of<float>());
	}
	if(use_Rmax_fire){
		if(shotIntervalTable.size()>1){
			single_action_space_dict["shotInterval"]=spaces.attr("Discrete")(shotIntervalTable.size());
			if(use_action_mask){
				Eigen::VectorXi shape=Eigen::VectorXi::Constant(1,shotIntervalTable.size());
				single_action_mask_space_dict["shotInterval"]=spaces.attr("Box")(floatLow,floatHigh,shape,py::dtype::of<float>());
			}
		}
		if(shotThresholdTable.size()>1){
			single_action_space_dict["shotThreshold"]=spaces.attr("Discrete")(shotThresholdTable.size());
			if(use_action_mask){
				Eigen::VectorXi shape=Eigen::VectorXi::Constant(1,shotThresholdTable.size());
				single_action_mask_space_dict["shotThreshold"]=spaces.attr("Box")(floatLow,floatHigh,shape,py::dtype::of<float>());
			}
		}
	}
	py::object single_action_space=spaces.attr("Dict")(single_action_space_dict);
	py::object single_action_mask_space;
	if(use_action_mask){
		single_action_mask_space=spaces.attr("Dict")(single_action_mask_space_dict);
	}
	py::list action_space_list;
	py::list action_mask_space_list;
	for(auto&& [port, parent] : parents){
		action_space_list.append(single_action_space);
		if(use_action_mask){
			action_mask_space_list.append(single_action_mask_space);
		}
	}
	_action_space=spaces.attr("Tuple")(action_space_list);

	//  observationに関するもの
	py::dict observation_space_dict;
	py::dict observation_mask_space_dict;
	//共通情報(残り時間等)
	if(common_dim>0){
        Eigen::VectorXi shape=Eigen::VectorXi::Constant(1,common_dim);
        observation_space_dict["common"]=spaces.attr("Box")(floatLow,floatHigh,shape,py::dtype::of<float>());
	}

	//画像情報
	if(use_image_observation){
		numChannels=(
			(draw_parent_trajectory ? 1 : 0)
			+(draw_friend_trajectory ? 1 : 0)
			+(draw_enemy_trajectory ? 1 : 0)
			+(draw_friend_missile_trajectory ? 1 : 0)
			+(draw_parent_radar_coverage ? 1 : 0)
			+(draw_friend_radar_coverage ? 1 : 0)
			+(draw_defense_line ? 1 : 0)
			+(draw_side_line ? 1 : 0)
		);
		//画像データバッファの生成
		image_buffer=Eigen::Tensor<float,3>(numChannels,image_longitudinal_resolution,image_lateral_resolution);
		image_buffer.setZero();
		image_buffer_coords=Eigen::Tensor<double,3>(2,image_longitudinal_resolution,image_lateral_resolution);
		image_buffer_coords.setZero();
		for(int lon=0;lon<image_longitudinal_resolution;++lon){
			for(int lat=0;lat<image_lateral_resolution;++lat){
				image_buffer_coords(0,lon,lat)=(image_front_range+image_back_range)*(0.5+lat)/image_longitudinal_resolution-image_back_range;
				image_buffer_coords(1,lon,lat)=2.0*image_side_range*(0.5+lon)/image_lateral_resolution-image_side_range;
			}
		}
		//軌跡描画用の過去データバッファの生成
		for(int i=0;i<image_horizon;++i){
			image_past_data.push_back(InstantInfo());
		}
		//spaceの生成
		Eigen::VectorXi shape(3);
		shape<<numChannels,image_longitudinal_resolution,image_lateral_resolution;
		observation_space_dict["image"]=spaces.attr("Box")(floatLow,floatHigh,shape,py::dtype::of<float>());
	}

	//parentsの諸元
	if(maxParentNum>0 && parent_dim>0){
        Eigen::VectorXi shape=Eigen::Vector2i(maxParentNum,parent_dim);
        observation_space_dict["parent"]=spaces.attr("Box")(floatLow,floatHigh,shape,py::dtype::of<float>());
		if(use_observation_mask){
        	Eigen::VectorXi shape=Eigen::VectorXi::Constant(1,maxParentNum);
			observation_mask_space_dict["parent"]=spaces.attr("Box")(floatLow,floatHigh,shape,py::dtype::of<float>());
		}
	}

	//parents以外の味方機の諸元
	if(maxFriendNum>0 && friend_dim>0){
        Eigen::VectorXi shape=Eigen::Vector2i(maxFriendNum,friend_dim);
        observation_space_dict["friend"]=spaces.attr("Box")(floatLow,floatHigh,shape,py::dtype::of<float>());
		if(use_observation_mask){
        	Eigen::VectorXi shape=Eigen::VectorXi::Constant(1,maxFriendNum);
			observation_mask_space_dict["friend"]=spaces.attr("Box")(floatLow,floatHigh,shape,py::dtype::of<float>());
		}
	}

	//彼機
	if(maxEnemyNum>0 && enemy_dim>0){
        Eigen::VectorXi shape=Eigen::Vector2i(maxEnemyNum,enemy_dim);
        observation_space_dict["enemy"]=spaces.attr("Box")(floatLow,floatHigh,shape,py::dtype::of<float>());
		if(use_observation_mask){
        	Eigen::VectorXi shape=Eigen::VectorXi::Constant(1,maxEnemyNum);
			observation_mask_space_dict["enemy"]=spaces.attr("Box")(floatLow,floatHigh,shape,py::dtype::of<float>());
		}
	}

	//味方誘導弾
	if(maxFriendMissileNum>0 && friend_missile_dim>0){
        Eigen::VectorXi shape=Eigen::Vector2i(maxFriendMissileNum,friend_missile_dim);
        observation_space_dict["friend_missile"]=spaces.attr("Box")(floatLow,floatHigh,shape,py::dtype::of<float>());
		if(use_observation_mask){
        	Eigen::VectorXi shape=Eigen::VectorXi::Constant(1,maxFriendMissileNum);
			observation_mask_space_dict["friend_missile"]=spaces.attr("Box")(floatLow,floatHigh,shape,py::dtype::of<float>());
		}
	}

	//彼側誘導弾
	if(maxEnemyMissileNum>0 && enemy_missile_dim>0){
        Eigen::VectorXi shape=Eigen::Vector2i(maxEnemyMissileNum,enemy_missile_dim);
        observation_space_dict["enemy_missile"]=spaces.attr("Box")(floatLow,floatHigh,shape,py::dtype::of<float>());
		if(use_observation_mask){
        	Eigen::VectorXi shape=Eigen::VectorXi::Constant(1,maxEnemyMissileNum);
			observation_mask_space_dict["enemy_missile"]=spaces.attr("Box")(floatLow,floatHigh,shape,py::dtype::of<float>());
		}
	}

	//我機と彼機の間の関係性
	if(maxParentNum+maxFriendNum>0 && maxEnemyNum>0 && friend_enemy_relative_dim>0){
        Eigen::VectorXi shape=Eigen::Vector3i(maxParentNum+maxFriendNum,maxEnemyNum,friend_enemy_relative_dim);
        observation_space_dict["friend_enemy_relative"]=spaces.attr("Box")(floatLow,floatHigh,shape,py::dtype::of<float>());
		if(use_observation_mask){
        	Eigen::VectorXi shape=Eigen::Vector2i(maxParentNum+maxFriendNum,maxEnemyNum);
			observation_mask_space_dict["friend_enemy_relative"]=spaces.attr("Box")(floatLow,floatHigh,shape,py::dtype::of<float>());
		}
	}

	if(use_observation_mask){
		observation_space_dict["observation_mask"]=spaces.attr("Dict")(observation_mask_space_dict);
	}
	if(use_action_mask){
		observation_space_dict["action_mask"]=spaces.attr("Tuple")(action_mask_space_list);
	}
	_observation_space=spaces.attr("Dict")(observation_space_dict);
}
void R7ContestAgentSample01::serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full){
	if(use_virtual_simulator){
		throw std::runtime_error("VirtualSimulator is not serializable in the current version.");
	}

    BaseType::serializeInternalState(archive,full);

    if(full){
		//observation spaceの設定
        ASRC_SERIALIZE_NVP(archive
            ,maxParentNum
            ,maxFriendNum
			,maxEnemyNum
			,maxFriendMissileNum
			,maxEnemyMissileNum
            ,horizontalNormalizer
            ,verticalNormalizer
            ,fgtrVelNormalizer
            ,mslVelNormalizer
			,use_virtual_simulator
            ,use_image_observation
        )
		if(use_image_observation){
			ASRC_SERIALIZE_NVP(archive
				,image_longitudinal_resolution
				,image_lateral_resolution
				,image_front_range
				,image_back_range
				,image_side_range
				,image_horizon
				,image_interval
				,image_rotate
				,image_relative_position
			)
		}
        ASRC_SERIALIZE_NVP(archive
            ,use_remaining_time
        )
		if(use_remaining_time){
			ASRC_SERIALIZE_NVP(archive
				,remaining_time_clipping
			)
		}
        ASRC_SERIALIZE_NVP(archive
            ,use_parent_last_action
			,use_friend_position
			,use_friend_velocity
			,use_friend_initial_num_missile
			,use_friend_current_num_missile
			,use_friend_attitude
			,use_friend_angular_velocity
			,use_friend_current_fuel
			,use_friend_rcs_scale
			,use_friend_radar_range
			,use_friend_radar_coverage
			,use_friend_maximum_speed_scale
			,use_friend_missile_thrust_scale
			,use_enemy_position
			,use_enemy_velocity
			,use_friend_missile_position
			,use_friend_missile_velocity
			,use_friend_missile_flight_time
			,use_friend_missile_target_distance
			,use_friend_missile_target_mode
			,use_friend_missile_target_position
			,use_friend_missile_target_velocity
			,use_enemy_missile_observer_position
			,use_enemy_missile_direction
			,use_enemy_missile_angular_velocity
			,use_our_missile_range
			,use_their_missile_range
        )

		//action spaceの設定
		ASRC_SERIALIZE_NVP(archive
			,dstAz_relative
			,turnTable
			,use_override_evasion
			,evasion_turnTable
			,no_vertical_maneuver
			,use_altitude_command
		)
		if(!no_vertical_maneuver){
			if(use_altitude_command){
				ASRC_SERIALIZE_NVP(archive
					,altTable
					,refAltInterval
				)
			}else{
				ASRC_SERIALIZE_NVP(archive
					,pitchTable
				)
			}
		}
		ASRC_SERIALIZE_NVP(archive
			,accelTable
			,always_maxAB
			,use_Rmax_fire
		)
		if(use_Rmax_fire){
			ASRC_SERIALIZE_NVP(archive
				,shotIntervalTable
				,shotThresholdTable
			)
		}
		//行動制限に関する設定
		ASRC_SERIALIZE_NVP(archive
			,altMin
			,altMax
			,altitudeKeeper
			,dOutLimit
			,dOutLimitThreshold
			,dOutLimitStrength
			,maxSimulShot
			,minimumV
			,minimumRecoveryV
			,minimumRecoveryDstV
		)

		//内部変数
		if(use_image_observation){
			ASRC_SERIALIZE_NVP(archive
				,numChannels
				,image_buffer_coords
			)
		}
		ASRC_SERIALIZE_NVP(archive
			,common_dim
			,parent_dim
			,friend_dim
			,enemy_dim
			,friend_missile_dim
			,enemy_missile_dim
			,friend_enemy_relative_dim
			,last_action_dim
			,_observation_space
			,_action_space
            ,dOut
            ,dLine
		)
		if(isOutputArchive(archive)){
			auto rulerObs=manager->getRuler().lock()->observables;
			std::string eastSider=rulerObs.at("eastSider");
			call_operator(archive,cereal::make_nvp("isEastSider",getTeam()==eastSider));
		}else{
			bool isEastSider;
			call_operator(archive,CEREAL_NVP(isEastSider));
			teamOrigin=TeamOrigin(isEastSider,dLine);
		}
    }
	if(use_virtual_simulator){
		if(asrc::core::util::isInputArchive(archive) && !virtualSimulator->virtualSimulator){
		}
		call_operator(archive
			,cereal::make_nvp("virtualSimulator",makeInternalStateSerializer(virtualSimulator,full))
		);
	}
	if(use_image_observation){
		ASRC_SERIALIZE_NVP(archive
			,image_buffer
			,image_past_data
		)
	}
	ASRC_SERIALIZE_NVP(archive
		,last_action_obs
		,actionInfos
	)
}

void R7ContestAgentSample01::validate(){
	//Rulerに関する情報の取得
	auto rulerObs=manager->getRuler().lock()->observables;
	dOut=rulerObs.at("dOut");
	dLine=rulerObs.at("dLine");
	std::string eastSider=rulerObs.at("eastSider");
	teamOrigin=TeamOrigin(getTeam()==eastSider,dLine);

	//機体制御方法の設定、前回行動の初期化
    int pIdx=0;
    for(auto&& [port,parent] : parents){
		std::dynamic_pointer_cast<FighterAccessor>(parent)->setFlightControllerMode("fromDirAndVel");
		auto& actionInfo=actionInfos[parent->getFullName()];
		actionInfo.dstDir<<0,(getTeam()==eastSider ? -1 : +1),0;
		last_action_obs[parent->getFullName()](0)=atan2(actionInfo.dstDir(1),actionInfo.dstDir(0));
		actionInfo.lastShotTimes.clear();
		pIdx++;
	}

	//仮想シミュレータの使用
	if(use_virtual_simulator){
		virtualSimulator->createVirtualSimulator(getShared<Agent>(shared_from_this()));
	}
}
py::object R7ContestAgentSample01::observation_space(){
	return _observation_space;
}
py::object R7ContestAgentSample01::makeObs(){
	//observablesの収集
	extractFriendObservables();
	extractEnemyObservables();
	extractFriendMissileObservables();
	extractEnemyMissileObservables();

	//仮想シミュレータの使用
	if(use_virtual_simulator){
		virtualSimulator->step(getShared<Agent>(shared_from_this()),lastTrackInfo);
	}

	//observationの生成
	py::dict ret;
	py::dict observation_mask;

	//共通情報(残り時間等)
	if(common_dim>0){
		ret["common"]=makeCommonStateObservation();
	}

	//画像情報
	if(use_image_observation){
		makeImageObservation();
		ret["image"]=py::cast(image_buffer,py::return_value_policy::copy);
	}

	//parentsの諸元
	if(maxParentNum>0 && parent_dim>0){
		Eigen::MatrixXf parent_temporal=Eigen::MatrixXf::Zero(maxParentNum,parent_dim);
		Eigen::VectorXf parent_mask=Eigen::VectorXf::Zero(maxParentNum);
		int fIdx=0;
		for(auto&& [port,parent] : parents){
			if(fIdx>=maxParentNum){
				break;
			}
			auto& fObs=ourObservables[fIdx];
			auto& fMotion=ourMotion[fIdx];
			if(fObs.at("isAlive")){
				parent_temporal.block(fIdx,0,1,parent_dim)=makeParentStateObservation(parent,fObs,fMotion).transpose();
				parent_mask(fIdx)=1;
			}
			fIdx++;
		}
		ret["parent"]=py::cast(std::move(parent_temporal));
		if(use_observation_mask){
			observation_mask["parent"]=py::cast(std::move(parent_mask));
		}
	}

	//parents以外の味方機の諸元
	if(maxFriendNum>0 && friend_dim>0){
		Eigen::MatrixXf friend_temporal=Eigen::MatrixXf::Zero(maxFriendNum,friend_dim);
		Eigen::VectorXf friend_mask=Eigen::VectorXf::Zero(maxFriendNum);
		for(int fIdx=0;fIdx<ourMotion.size()-parents.size();++fIdx){
			if(fIdx>=maxFriendNum){
				break;
			}
			auto& fObs=ourObservables[fIdx+parents.size()];
			auto& fMotion=ourMotion[fIdx+parents.size()];
			if(fObs.at("isAlive")){
				friend_temporal.block(fIdx,0,1,friend_dim)=makeFriendStateObservation(fObs,fMotion).transpose();
				friend_mask(fIdx)=1;
			}
		}
		ret["friend"]=py::cast(std::move(friend_temporal));
		if(use_observation_mask){
			observation_mask["friend"]=py::cast(std::move(friend_mask));
		}
	}

	//彼機(味方の誰かが探知しているもののみ)
	if(maxEnemyNum>0 && enemy_dim>0){
		Eigen::MatrixXf enemy_temporal=Eigen::MatrixXf::Zero(maxEnemyNum,enemy_dim);
		Eigen::VectorXf enemy_mask=Eigen::VectorXf::Zero(maxEnemyNum);
		for(int tIdx=0;tIdx<lastTrackInfo.size();++tIdx){
			if(tIdx>=maxEnemyNum){
				break;
			}
			enemy_temporal.block(tIdx,0,1,enemy_dim)=makeEnemyStateObservation(lastTrackInfo[tIdx]).transpose();
			enemy_mask(tIdx)=1;
		}
		ret["enemy"]=py::cast(std::move(enemy_temporal));
		if(use_observation_mask){
			observation_mask["enemy"]=py::cast(std::move(enemy_mask));
		}
	}

	//味方誘導弾(射撃時刻が古いものから最大N発分)
	if(maxFriendMissileNum>0 && friend_missile_dim>0){
		Eigen::MatrixXf friend_missile_temporal=Eigen::MatrixXf::Zero(maxFriendMissileNum,friend_missile_dim);
		Eigen::VectorXf friend_missile_mask=Eigen::VectorXf::Zero(maxFriendMissileNum);
		for(int mIdx=0;mIdx<msls.size();++mIdx){
			const auto& mObs=msls[mIdx];
			if(mIdx>=maxFriendMissileNum || !(mObs.at("isAlive").get<bool>()&&mObs.at("hasLaunched").get<bool>())){
				break;
			}
			friend_missile_temporal.block(mIdx,0,1,friend_missile_dim)=makeFriendMissileStateObservation(mObs).transpose();
			friend_missile_mask(mIdx)=1;
		}
		ret["friend_missile"]=py::cast(std::move(friend_missile_temporal));
		if(use_observation_mask){
			observation_mask["friend_missile"]=py::cast(std::move(friend_missile_mask));
		}
	}

	//彼側誘導弾(観測者の正面に近い順にソート)
	if(maxEnemyMissileNum>0 && enemy_missile_dim>0){
		Eigen::MatrixXf enemy_missile_temporal=Eigen::MatrixXf::Zero(maxEnemyMissileNum,enemy_missile_dim);
		Eigen::VectorXf enemy_missile_mask=Eigen::VectorXf::Zero(maxEnemyMissileNum);
		std::vector<std::pair<Track2D,double>> allMWS;
		for(int fIdx=0;fIdx<ourMotion.size();++fIdx){
			if(ourObservables[fIdx].at("isAlive")){
				auto& fMotion=ourMotion[fIdx];
				for(auto&& m : mws[fIdx]){
					double angle=acos(std::clamp<double>(m.dir().dot(fMotion.dirBtoP(Eigen::Vector3d(1,0,0))),-1,1));
					allMWS.emplace_back(m,angle);
				}
			}
		}
		std::sort(allMWS.begin(),allMWS.end(),
			[](const std::pair<Track2D,double>& lhs,const std::pair<Track2D,double>& rhs)->bool {
				return lhs.second<rhs.second;
		});
		for(int mIdx=0;mIdx<allMWS.size();++mIdx){
			if(mIdx>=maxEnemyMissileNum){
				break;
			}
			enemy_missile_temporal.block(mIdx,0,1,enemy_missile_dim)=makeEnemyMissileStateObservation(allMWS[mIdx].first).transpose();
			enemy_missile_mask(mIdx)=1;
		}
		ret["enemy_missile"]=py::cast(std::move(enemy_missile_temporal));
		if(use_observation_mask){
			observation_mask["enemy_missile"]=py::cast(std::move(enemy_missile_mask));
		}
	}

	//我機と彼機の間の関係性
	if(maxParentNum+maxFriendNum>0 && maxEnemyNum>0 && friend_enemy_relative_dim>0){
		Eigen::Tensor<float,3> friend_enemy_relative_temporal(maxParentNum+maxFriendNum,maxEnemyNum,friend_enemy_relative_dim);
		friend_enemy_relative_temporal.setZero();
		Eigen::MatrixXf friend_enemy_relative_mask=Eigen::MatrixXf::Zero(maxParentNum+maxFriendNum,maxEnemyNum);
		std::shared_ptr<PhysicalAssetAccessor> firstAlive=nullptr;

		int fIdx=0;
		for(auto&& [port,parent] : parents){
			if(fIdx>=maxParentNum){
				break;
			}
			auto& fObs=ourObservables[fIdx];
			auto& fMotion=ourMotion[fIdx];
			if(fObs.at("isAlive")){
				if(!firstAlive){
					firstAlive=parent;
				}
				for(int tIdx=0;tIdx<lastTrackInfo.size();++tIdx){
					if(tIdx>=maxEnemyNum){
						break;
					}
					std::array<Eigen::Index, 3> startIndices{fIdx,tIdx,0};
					std::array<Eigen::Index, 3> sliceSizes{1,1,friend_enemy_relative_dim};
					friend_enemy_relative_temporal.slice(startIndices,sliceSizes)=makeFriendEnemyRelativeStateObservation(
						parent,fObs,fMotion,lastTrackInfo[tIdx]
					).reshape(sliceSizes);
					friend_enemy_relative_mask(fIdx,tIdx)=1;
				}
			}
			fIdx++;
		}
		for(int fIdx=0;fIdx<ourMotion.size()-parents.size();++fIdx){
			if(fIdx>=maxFriendNum){
				break;
			}
			auto& fObs=ourObservables[fIdx+parents.size()];
			auto& fMotion=ourMotion[fIdx+parents.size()];
			if(fObs.at("isAlive")){
				for(int tIdx=0;tIdx<lastTrackInfo.size();++tIdx){
					if(tIdx>=maxEnemyNum){
						break;
					}
					std::array<Eigen::Index, 3> startIndices{fIdx+(Eigen::Index)parents.size(),tIdx,0};
					std::array<Eigen::Index, 3> sliceSizes{1,1,friend_enemy_relative_dim};
					friend_enemy_relative_temporal.slice(startIndices,sliceSizes)=makeFriendEnemyRelativeStateObservation(
						firstAlive,fObs,fMotion,lastTrackInfo[tIdx]
					).reshape(sliceSizes);
					friend_enemy_relative_mask(fIdx+parents.size(),tIdx)=1;
				}
			}
		}
		ret["friend_enemy_relative"]=py::cast(std::move(friend_enemy_relative_temporal));
		if(use_observation_mask){
			observation_mask["friend_enemy_relative"]=friend_enemy_relative_mask;
		}
	}

	if(use_observation_mask && observation_mask.size()>0){
		ret["observation_mask"]=observation_mask;
	}

	if(use_action_mask){
		action_mask=makeActionMask();
		if(!action_mask.is_none()){
			ret["action_mask"]=action_mask;
		}
	}
	return ret;
}

void R7ContestAgentSample01::extractFriendObservables(){
	//味方機(parents→parents以外の順)
	ourMotion.clear();
	ourObservables.clear();
	std::shared_ptr<PhysicalAssetAccessor> firstAlive=nullptr;
	for(auto&& [port, parent] : parents){
		if(parent->isAlive()){
			firstAlive=parent;
			break;
		}
	}

	std::set<std::string> parentFullNames;
	// まずはparents
	for(auto&& [port, parent] : parents){
		parentFullNames.insert(parent->getFullName());
		if(parent->isAlive()){
			ourMotion.push_back(MotionState(parent->observables.at("motion")).transformTo(getLocalCRS()));
			//残存していればobservablesそのもの
			ourObservables.push_back(parent->observables);
		}else{
			ourMotion.push_back(MotionState());
			//被撃墜or墜落済なら本体の更新は止まっているので残存している親が代理更新したものを取得(誘導弾情報のため)
			ourObservables.push_back(
				firstAlive->observables.at("/shared/fighter"_json_pointer).at(parent->getFullName()));
		}
	}

	// その後にparents以外
	for(auto&& [fullName,fObs] : firstAlive->observables.at("/shared/fighter"_json_pointer).items()){
		if(!parentFullNames.contains(fullName)){
			if(fObs.at("isAlive")){
				ourMotion.push_back(MotionState(fObs.at("motion")).transformTo(getLocalCRS()));
			}else{
				ourMotion.push_back(MotionState());
			}
			ourObservables.push_back(fObs);
		}
	}
}

void R7ContestAgentSample01::extractEnemyObservables(){
	//彼機(味方の誰かが探知しているもののみ)
	//観測されている航跡を、自陣営の機体に近いものから順にソートしてlastTrackInfoに格納する。
	//lastTrackInfoは行動のdeployでも射撃対象の指定のために参照する。
	std::shared_ptr<PhysicalAssetAccessor> firstAlive=nullptr;
	for(auto&& [port, parent] : parents){
		if(parent->isAlive()){
			firstAlive=parent;
			break;
		}
	}

	lastTrackInfo.clear();
	for(auto&& t:firstAlive->observables.at("/sensor/track"_json_pointer)){
		lastTrackInfo.push_back(Track3D(t).transformTo(getLocalCRS()));
	}
	sortTrack3DByDistance(lastTrackInfo,ourMotion,true);
}

void R7ContestAgentSample01::extractFriendMissileObservables(){
	//味方誘導弾(射撃時刻の古い順にソート)
	msls.clear();
	for(int fIdx=0;fIdx<ourObservables.size();++fIdx){
		for(auto&& msl:ourObservables[fIdx].at("/weapon/missiles"_json_pointer)){
			msls.push_back(msl);
		}
	}
	std::sort(msls.begin(),msls.end(),
	[](const nl::json& lhs,const nl::json& rhs){
		Time lhsT,rhsT;
		if(lhs.at("isAlive").get<bool>() && lhs.at("hasLaunched").get<bool>()){
			lhsT=lhs.at("launchedT");
		}else{
			lhsT=Time(std::numeric_limits<double>::infinity(),TimeSystem::TT);
		}
		if(rhs.at("isAlive").get<bool>() && rhs.at("hasLaunched").get<bool>()){
			rhsT=rhs.at("launchedT");
		}else{
			rhsT=Time(std::numeric_limits<double>::infinity(),TimeSystem::TT);
		}
		return lhsT<rhsT;
	});
}
void R7ContestAgentSample01::extractEnemyMissileObservables(){
	//彼側誘導弾(各機の正面に近い順にソート)
	mws.clear();
	for(int fIdx=0;fIdx<ourMotion.size();++fIdx){
		auto& fObs=ourObservables[fIdx];
		auto& fMotion=ourMotion[fIdx];

		mws.push_back(std::vector<Track2D>());
		if(fObs.at("isAlive")){
			if(fObs.contains("/sensor/mws/track"_json_pointer)){
				for(auto&& mObs:fObs.at("/sensor/mws/track"_json_pointer)){
					mws[fIdx].push_back(Track2D(mObs).transformTo(getLocalCRS()));
				}
			}
			//自機正面に近い順にソート
			sortTrack2DByAngle(mws[fIdx],fMotion,Eigen::Vector3d(1,0,0),true);
		}
	}
}

Eigen::VectorXf R7ContestAgentSample01::makeCommonStateObservation(){
	Eigen::VectorXf ret=Eigen::VectorXf::Zero(common_dim);
	int ofs=0;

	//残り時間
	if(use_remaining_time){
    	auto rulerObs=manager->getRuler().lock()->observables;
		float maxTime=rulerObs.at("maxTime");
		ret(ofs)=std::min<float>((maxTime-manager->getElapsedTime())/60.0,remaining_time_clipping);
		ofs+=1;
	}

	//

	return std::move(ret);
}

void R7ContestAgentSample01::makeImageObservation(){
	Eigen::Vector3d pos,vel;
	double V,R;
	image_buffer.setZero();
	//現在の情報を収集
	InstantInfo current;
	//味方機(自機含む)の諸元
	for(int fIdx=0;fIdx<parents.size();++fIdx){
		if(ourObservables[fIdx].at("isAlive")){
			current.parent_pos.push_back(ourMotion[fIdx].pos());
		}
	}
	for(int fIdx=parents.size();fIdx<ourMotion.size();++fIdx){
		if(ourObservables[fIdx].at("isAlive")){
			current.friend_pos.push_back(ourMotion[fIdx].pos());
		}
	}
	//彼機(味方の誰かが探知しているもののみ)
	for(int tIdx=0;tIdx<lastTrackInfo.size();++tIdx){
		const auto& t=lastTrackInfo[tIdx];
		current.enemy_pos.push_back(t.pos());
	}
	//味方誘導弾(射撃時刻が古いものから最大N発分)
	for(int mIdx=0;mIdx<msls.size();++mIdx){
		const auto& m=msls[mIdx];
		if(!(m.at("isAlive").get<bool>()&&m.at("hasLaunched").get<bool>())){
			break;
		}
		MotionState mm=MotionState(m.at("motion")).transformTo(getLocalCRS());
		current.friend_msl_pos.push_back(mm.pos());
	}

	//画像の生成
	//ch0:parentsの軌跡
	//ch1:parents以外の味方軌跡
	//ch2:彼機軌跡
	//ch3:味方誘導弾軌跡
	//ch4:parentsの覆域
	//ch5:parents以外の味方覆域
	//ch6:彼我防衛ライン
	//ch7:場外ライン
	int stepCount=getStepCount();
	int bufferIdx=image_horizon-1-(stepCount%image_horizon);
	image_past_data[bufferIdx]=current;
	int tMax=std::min<int>(
        floor(stepCount/image_interval)*image_interval,
        floor((image_horizon-1)/image_interval)*image_interval
    );
	int ch=0;

	for(int t=tMax;t>=0;t-=image_interval){
		InstantInfo frame=image_past_data[(bufferIdx+t)%image_horizon];
		ch=0;
		if(draw_parent_trajectory){
			//ch0:parentsの軌跡
			for(int i=0;i<frame.parent_pos.size();++i){
				Eigen::Vector2i g=posToGrid(pos);
				int lon=g(0);
				int lat=g(1);
				plotPoint(
					frame.parent_pos[i],
					ch,
					1.0-1.0*t/image_horizon
				);
			}
			ch+=1;
		}

		if(draw_friend_trajectory){
			//ch1:parents以外の味方軌跡
			for(int i=0;i<frame.friend_pos.size();++i){
				Eigen::Vector2i g=posToGrid(pos);
				int lon=g(0);
				int lat=g(1);
				plotPoint(
					frame.friend_pos[i],
					ch,
					1.0-1.0*t/image_horizon
				);
			}
			ch+=1;
		}

		if(draw_enemy_trajectory){
			//ch2:彼機軌跡
			for(int i=0;i<frame.enemy_pos.size();++i){
				Eigen::Vector2i g=posToGrid(pos);
				int lon=g(0);
				int lat=g(1);
				plotPoint(
					frame.enemy_pos[i],
					ch,
					1.0-1.0*t/image_horizon
				);
			}
			ch+=1;
		}

		if(draw_friend_missile_trajectory){
			//ch3:味方誘導弾軌跡
			for(int i=0;i<frame.friend_msl_pos.size();++i){
				Eigen::Vector2i g=posToGrid(pos);
				int lon=g(0);
				int lat=g(1);
				plotPoint(
					frame.friend_msl_pos[i],
					ch,
					1.0-1.0*t/image_horizon
				);
			}
			ch+=1;
		}
	}
	
	//ch4:parentsのレーダ覆域, ch5:parents以外の味方レーダ覆域
	if(draw_parent_radar_coverage || draw_friend_radar_coverage){
		for(int i=0;i<ourMotion.size();++i){
			if(!draw_parent_radar_coverage && i<parents.size()){
				continue;
			}
			if(draw_parent_radar_coverage && i==parents.size()){
				ch+=1;
			}
			if(!draw_friend_radar_coverage && i>=parents.size()){
				continue;
			}
			if(ourObservables[i].at("isAlive")){
				if(ourObservables[i].contains("/spec/sensor/radar"_json_pointer)){
					Eigen::Vector3d ex=ourMotion[i].relBtoP(Eigen::Vector3d(1,0,0));
					double Lref=ourObservables[i].at("/spec/sensor/radar/Lref"_json_pointer);
					double thetaFOR=ourObservables[i].at("/spec/sensor/radar/thetaFOR"_json_pointer);
					for(int lon=0;lon<image_longitudinal_resolution;++lon){
						for(int lat=0;lat<image_lateral_resolution;++lat){
							Eigen::Vector3d pos=gridToPos(lon,lat,10000.0);
							Eigen::Vector3d rPos=pos-ourMotion[i].pos();
							double L=rPos.norm();
							if(L<=Lref && rPos.dot(ex)>=L*cos(thetaFOR)){
								plotPoint(
									pos,
									ch,
									1.0
								);
							}
						}
					}
				}
			}
		}
		if(draw_friend_radar_coverage){
			ch+=1;
		}
	}

	if(draw_defense_line){
		//ch6:彼我防衛ライン
		//彼側防衛ライン
		double eps=1e-10;//戦域と同一の描画範囲とした際に境界上になって描画されなくなることを避けるため、ごくわずかに内側にずらすとよい。
		plotLine(
			teamOrigin.relBtoP(Eigen::Vector3d(dLine-eps,-dOut+eps,0)),
			teamOrigin.relBtoP(Eigen::Vector3d(dLine-eps,dOut-eps,0)),
			ch,
			1.0
		);
		//我側防衛ライン
		plotLine(
			teamOrigin.relBtoP(Eigen::Vector3d(-dLine+eps,-dOut+eps,0)),
			teamOrigin.relBtoP(Eigen::Vector3d(-dLine+eps,dOut-eps,0)),
			ch,
			-1.0
		);
		ch+=1;
	}

	if(draw_side_line){
		//ch7:場外ライン
		//進行方向右側
		double eps=1e-10;//戦域と同一の描画範囲とした際に境界上になって描画されなくなることを避けるため、ごくわずかに内側にずらすとよい。
		plotLine(
			teamOrigin.relBtoP(Eigen::Vector3d(-dLine+eps,dOut-eps,0)),
			teamOrigin.relBtoP(Eigen::Vector3d(dLine-eps,dOut-eps,0)),
			ch,
			1.0
		);
		//進行方向左側
		plotLine(
			teamOrigin.relBtoP(Eigen::Vector3d(-dLine+eps,-dOut+eps,0)),
			teamOrigin.relBtoP(Eigen::Vector3d(dLine-eps,-dOut+eps,0)),
			ch,
			1.0
		);
		ch+=1;
	}
}

Eigen::VectorXf R7ContestAgentSample01::makeParentStateObservation(const std::shared_ptr<PhysicalAssetAccessor>& parent, const nl::json& fObs, const MotionState& fMotion){
	//parentの諸元
	Eigen::VectorXf ret=Eigen::VectorXf::Zero(parent_dim);
	int ofs=0;

	//前回の行動
	//前回のdeployで計算済だが、dstAzのみ現在のazからの差に置き換える
	if(use_parent_last_action){
		if(parent->isAlive()){
			auto parentFullName=parent->getFullName();
			MotionState myMotion=MotionState(parent->observables.at("motion")).transformTo(getLocalCRS());
			double deltaAz=last_action_obs[parentFullName](0)-myMotion.getAZ();
			last_action_obs[parentFullName](0)=atan2(sin(deltaAz),cos(deltaAz));
			ret.block(ofs,0,last_action_dim,1)=last_action_obs[parentFullName];
		}
		ofs+=last_action_dim;
	}

	// 仮想シミュレータの使用
	// この例では各parentに最も近い仮想誘導弾との距離をparentのobservationに付加する
	if(use_virtual_simulator){
		double d=virtual_simulator_value_when_no_missile;
		if(virtualSimulator->minimumDistances.contains(parent->getFullName())){
			d=std::min(d,virtualSimulator->minimumDistances.at(parent->getFullName()));
		ret(ofs)=d/horizontalNormalizer;
		ofs+=1;
		}
	}

	//味方機の共通諸元も使用
	ret.block(ofs,0,friend_dim,1)=makeFriendStateObservation(fObs,fMotion);

	return std::move(ret);
}

Eigen::VectorXf R7ContestAgentSample01::makeFriendStateObservation(const nl::json& fObs, const MotionState& fMotion){
	//味方機の諸元
	Eigen::VectorXf ret=Eigen::VectorXf::Zero(friend_dim);
	int ofs=0;

	//位置
	if(use_friend_position){
		Eigen::Vector3d pos=teamOrigin.relPtoB(fMotion.pos());//慣性座標系→陣営座標系に変換
		ret.block(ofs,0,3,1)=(pos.array()/Eigen::Array3d(horizontalNormalizer,horizontalNormalizer,verticalNormalizer)).cast<float>();
		ofs+=3;
	}

	//速度
	if(use_friend_velocity){
		Eigen::Vector3d vel=teamOrigin.relPtoB(fMotion.vel());//慣性座標系→陣営座標系に変換
		double V=vel.norm();
		ret(ofs)=V/fgtrVelNormalizer;
		ofs+=1;
		ret.block(ofs,0,3,1)=(vel/V).cast<float>();
		ofs+=3;
	}

	//初期弾数
	if(use_friend_initial_num_missile){
		ret(ofs)=fObs.at("/spec/weapon/numMsls"_json_pointer);
		ofs+=1;
	}

	//残弾数
	if(use_friend_current_num_missile){
		ret(ofs)=fObs.at("/weapon/remMsls"_json_pointer);
		ofs+=1;
	}

	//姿勢(バンク、α、β)
	if(use_friend_attitude){
		Eigen::Vector3d vb=fMotion.relPtoB(fMotion.vel());
		double V=vb.norm();
		Eigen::Vector3d ex=fMotion.dirBtoP(Eigen::Vector3d(1,0,0));
		Eigen::Vector3d ey=fMotion.dirBtoP(Eigen::Vector3d(0,1,0));
		Eigen::Vector3d down=fMotion.dirHtoP(Eigen::Vector3d(0,0,1),Eigen::Vector3d(0,0,0),"NED",false);
		Eigen::Vector3d horizontalY=down.cross(ex);
		double roll=0;
		if(horizontalY.norm()>0){
			horizontalY.normalize();
			double sinRoll=horizontalY.cross(ey).dot(ex);
			double cosRoll=horizontalY.dot(ey);
			roll=atan2(sinRoll,cosRoll);
		}
		double alpha=atan2(vb(2),vb(0));
		double beta=asin(vb(1)/V);
		ret(ofs)=roll;
		ret(ofs+1)=alpha;
		ret(ofs+2)=beta;
		ofs+3;
	}

	//角速度
	if(use_friend_angular_velocity){
        Eigen::Vector3d omegaB=fMotion.relPtoB(fMotion.omega());//回転だけを行いたいのでrelで変換
        ret.block(ofs,0,3,1)=omegaB.cast<float>();
        ofs+=3;
	}

	//余剰燃料(R7年度コンテストでは不要)
	if(use_friend_current_fuel){
		ret(ofs)=calcDistanceMargin(getTeam(),fMotion,fObs,manager->getRuler().lock()->observables);
		ofs+=1;
	}

	//R7年度コンテストオープン部門向け、機体性能の変動情報
	//RCSスケール
	if(use_friend_rcs_scale){
		if(fObs.contains("/spec/stealth/rcsScale"_json_pointer)){
			ret(ofs)=fObs.at("/spec/stealth/rcsScale"_json_pointer).get<double>();
		}else{
			ret(ofs)=1.0;
		}
		ofs+=1;
	}

	//レーダ探知距離
	if(use_friend_radar_range){
		if(fObs.contains("/spec/sensor/radar/Lref"_json_pointer)){
			ret(ofs)=fObs.at("/spec/sensor/radar/Lref"_json_pointer).get<double>()/horizontalNormalizer;
		}else{
			ret(ofs)=0.0;
		}
		ofs+=1;
	}

	//レーダ覆域(角度、ラジアン)
	if(use_friend_radar_coverage){
		if(fObs.contains("/spec/sensor/radar/thetaFOR"_json_pointer)){
			ret(ofs)=fObs.at("/spec/sensor/radar/thetaFOR"_json_pointer).get<double>();
		}else{
			ret(ofs)=0.0;
		}
		ofs+=1;
	}

	//最大速度倍率
	if(use_friend_maximum_speed_scale){
		if(fObs.contains("/spec/dynamics/scale/vMax"_json_pointer)){
			ret(ofs)=fObs.at("/spec/dynamics/scale/vMax"_json_pointer).get<double>();
		}else{
			ret(ofs)=1.0;
		}
		ofs+=1;
	}

	//誘導弾推力倍率
	if(use_friend_missile_thrust_scale){
		if(fObs.contains("/spec/weapon/missile/thrust"_json_pointer)){
			ret(ofs)=fObs.at("/spec/weapon/missile/thrust"_json_pointer).get<double>()/30017.9989;
		}else{
			ret(ofs)=1.0;
		}
		ofs+=1;
	}

	//射撃遅延時間(秒)は直接観測不可(observablesからもparentからも見えない)

	return std::move(ret);
}

Eigen::VectorXf R7ContestAgentSample01::makeEnemyStateObservation(const Track3D& track){
	//彼機の諸元
	Eigen::VectorXf ret=Eigen::VectorXf::Zero(enemy_dim);
	int ofs=0;

	//位置
	if(use_enemy_position){
		Eigen::Vector3d pos=teamOrigin.relPtoB(track.pos());//慣性座標系→陣営座標系に変換
		ret.block(ofs,0,3,1)=(pos.array()/Eigen::Array3d(horizontalNormalizer,horizontalNormalizer,verticalNormalizer)).cast<float>();
		ofs+=3;
	}

	//速度
	if(use_enemy_velocity){
		Eigen::Vector3d vel=teamOrigin.relPtoB(track.vel());//慣性座標系→陣営座標系に変換
		double V=vel.norm();
		ret(ofs)=V/fgtrVelNormalizer;
		ofs+=1;
		ret.block(ofs,0,3,1)=(vel/V).cast<float>();
		ofs+=3;
	}

	return std::move(ret);
}

Eigen::VectorXf R7ContestAgentSample01::makeFriendMissileStateObservation(const nl::json& mObs){
	//味方誘導弾の諸元
	Eigen::VectorXf ret=Eigen::VectorXf::Zero(friend_missile_dim);
	int ofs=0;

	MotionState mm=MotionState(mObs.at("motion")).transformTo(getLocalCRS());

	//位置
	if(use_friend_missile_position){
		Eigen::Vector3d pos=teamOrigin.relPtoB(mm.pos());//慣性座標系→陣営座標系に変換
		ret.block(ofs,0,3,1)=(pos.array()/Eigen::Array3d(horizontalNormalizer,horizontalNormalizer,verticalNormalizer)).cast<float>();
		ofs+=3;
	}

	//速度
	if(use_friend_missile_velocity){
		Eigen::Vector3d vel=teamOrigin.relPtoB(mm.vel());//慣性座標系→陣営座標系に変換
		double V=vel.norm();
		ret(ofs)=V/mslVelNormalizer;
		ofs+=1;
		ret.block(ofs,0,3,1)=(vel/V).cast<float>();
		ofs+=3;
	}

	//飛翔時間(分)
	if(use_friend_missile_flight_time){
		Time launchedT=mObs.at("launchedT");
		ret(ofs)=(manager->getTime()-launchedT)/60.0;
		ofs+=1;
	}

	//目標情報(距離、誘導状態、推定目標位置、推定目標速度)
	Track3D mTgt=Track3D(mObs.at("target")).transformTo(getLocalCRS()).extrapolateTo(manager->getTime());

	//目標との距離
	if(use_friend_missile_target_distance){
		ret(ofs)=1-std::min<double>(1.0,(mTgt.pos()-mm.pos()).norm()/horizontalNormalizer);
		ofs+=1;
	}

	//目標への誘導状態
	if(use_friend_missile_target_mode){
		Missile::Mode mode=jsonToEnum<Missile::Mode>(mObs.at("mode"));
		if(mode==Missile::Mode::GUIDED){
			ret.block(ofs,0,3,1)<<1,0,0;
		}else if(mode==Missile::Mode::SELF){
			ret.block(ofs,0,3,1)<<0,1,0;
		}else{
			ret.block(ofs,0,3,1)<<0,0,1;
		}
		ofs+=3;
	}

	//目標の位置
	if(use_friend_missile_target_position){
		Eigen::Vector3d pos=teamOrigin.relPtoB(mTgt.pos());//慣性座標系→陣営座標系に変換
		ret.block(ofs,0,3,1)=(pos.array()/Eigen::Array3d(horizontalNormalizer,horizontalNormalizer,verticalNormalizer)).cast<float>();
		ofs+=3;
	}

	//目標の速度
	if(use_friend_missile_target_velocity){
		Eigen::Vector3d vel=teamOrigin.relPtoB(mTgt.vel());//慣性座標系→陣営座標系に変換
		double V=vel.norm();
		ret(ofs)=V/mslVelNormalizer;
		ofs+=1;
		ret.block(ofs,0,3,1)=(vel/V).cast<float>();
		ofs+=3;
	}

	//R7年度コンテストオープン部門向け、機体性能の変動情報
	//誘導弾推力倍率
	if(use_friend_missile_thrust_scale){
		if(mObs.contains("/spec/thrust"_json_pointer)){
			ret(ofs)=mObs.at("/spec/thrust"_json_pointer).get<double>()/30017.9989;
		}else{
			ret(ofs)=1.0;
		}
		ofs+=1;
	}

	return std::move(ret);
}

Eigen::VectorXf R7ContestAgentSample01::makeEnemyMissileStateObservation(const Track2D& track){
	Eigen::VectorXf ret=Eigen::VectorXf::Zero(enemy_missile_dim);
	int ofs=0;

	//観測点
	if(use_enemy_missile_observer_position){
		Eigen::Vector3d origin=teamOrigin.relPtoB(track.origin());//慣性座標系→陣営座標系に変換
		ret.block(ofs,0,3,1)=(origin.array()/Eigen::Array3d(horizontalNormalizer,horizontalNormalizer,verticalNormalizer)).cast<float>();
		ofs+=3;
	}

	//方向
	if(use_enemy_missile_direction){
		Eigen::Vector3d dir=teamOrigin.relPtoB(track.dir());//慣性座標系→陣営座標系に変換 //teamOriginはrelとdirを区別しない
		ret.block(ofs,0,3,1)=dir.cast<float>();
		ofs+=3;
	}

	//方向変化率
	if(use_enemy_missile_angular_velocity){
		Eigen::Vector3d omega=teamOrigin.relPtoB(track.omega());//慣性座標系→陣営座標系に変換。//回転だけを行いたいのでrelで変換
		ret.block(ofs,0,3,1)=omega.cast<float>();
		ofs+=3;
	}

	return std::move(ret);
}

Eigen::Tensor<float,1> R7ContestAgentSample01::makeFriendEnemyRelativeStateObservation(const std::shared_ptr<PhysicalAssetAccessor>& parent,const nl::json& fObs, const MotionState& fMotion, const Track3D& track){
	//我機と彼機の間の関係性に関する諸元
	Eigen::Tensor<float,1> ret(friend_enemy_relative_dim);
	ret.setZero();
	int ofs=0;
	if(use_our_missile_range){
		ret(ofs)=calcRHead(parent,fMotion,track,false)/horizontalNormalizer;//RHead
		ret(ofs+1)=calcRTail(parent,fMotion,track,false)/horizontalNormalizer;//RTail
		ret(ofs+2)=calcRNorm(parent,fMotion,track,false);//現在の距離をRTail〜RHeadで正規化したもの
		ofs+=3;
	}
	if(use_their_missile_range){
		ret(ofs)=calcRHeadE(parent,fMotion,track,false)/horizontalNormalizer;//RHead
		ret(ofs+1)=calcRTailE(parent,fMotion,track,false)/horizontalNormalizer;//RTail
		ret(ofs+2)=calcRNormE(parent,fMotion,track,false);//現在の距離をRTail〜RHeadで正規化したもの
		ofs+=3;
	}

	return std::move(ret);
}

py::object R7ContestAgentSample01::makeActionMask(){
	//無効な行動を示すマスクを返す
	//有効な場合は1、無効な場合は0とする。
	if(use_action_mask){
		//このサンプルでは射撃目標のみマスクする。
		Eigen::VectorXf target_mask=Eigen::VectorXf::Zero(1+maxEnemyNum);
		target_mask(0)=1;//「射撃なし」はつねに有効
		for(int tIdx=0;tIdx<lastTrackInfo.size();++tIdx){
			if(tIdx>=maxEnemyNum){
				break;
			}
			target_mask(1+tIdx)=1;
		}

		py::list ret;
		for(auto&& [port,parent] : parents){
			py::dict mask;
			mask["turn"]=py::cast(Eigen::VectorXf::Constant(turnTable.size(),1));
			if(!no_vertical_maneuver){
				mask["pitch"]=py::cast(Eigen::VectorXf::Constant(use_altitude_command ? altTable.size() : pitchTable.size(),1));
			}
			if(!always_maxAB){
				mask["accel"]=py::cast(Eigen::VectorXf::Constant(accelTable.size(),1));
			}
			mask["target"]=py::cast(target_mask);
			if(use_Rmax_fire){
				if(shotIntervalTable.size()>1){
					mask["shotInterval"]=py::cast(Eigen::VectorXf::Constant(shotIntervalTable.size(),1));
				}
				if(shotThresholdTable.size()>1){
					mask["shotThreshold"]=py::cast(Eigen::VectorXf::Constant(shotThresholdTable.size(),1));
				}
			}
			ret.append(mask);
		}
		return ret;
	}else{
		return py::none();
	}
}

py::object R7ContestAgentSample01::action_space(){
	return _action_space;
}

void R7ContestAgentSample01::deploy(py::object action_){
	//observablesの収集
	extractFriendObservables();
	//extractEnemyObservables(); // 彼機情報だけは射撃対象の選択と連動するので更新してはいけない。
	extractFriendMissileObservables();
	extractEnemyMissileObservables();

	//actionのパース
	py::sequence action=py::reinterpret_borrow<py::sequence>(action_);
	if(use_action_mask && raise_when_invalid_action){
		//有効な行動かどうかを確かめる(形は合っていることを前提とする)
    	int pIdx=0;
		py::sequence maskAsList=py::reinterpret_borrow<py::sequence>(action_mask);
    	for(auto&& [port,parent] : parents){
			py::dict myAction=py::reinterpret_borrow<py::dict>(action[pIdx]);
			py::dict myActionMask=py::reinterpret_borrow<py::dict>(maskAsList[pIdx]);
			for(auto&& [key,mask] : myActionMask){
				if(mask[myAction[key]].cast<int>()!=1){
					std::ostringstream oss;
					oss<<getFullName()<<"'s action '"<<key.cast<std::string>()<<"' at "<<myAction[key].cast<int>()
					<<" for "<<parent->getFullName()<<" is masked but sampled somehow. Check your sampler method.";
					throw std::runtime_error(oss.str());
				}
			}
		}
	}
	last_action_obs.clear();
    int pIdx=0;
    for(auto&& [port,parent] : parents){
		auto parentFullName=parent->getFullName();
		last_action_obs[parentFullName]=Eigen::VectorXf::Zero(last_action_dim);
        if(!parent->isAlive()){
			pIdx++;
			continue;
		}
		auto& actionInfo=actionInfos[parentFullName];
		auto& myMotion=ourMotion[pIdx];
		auto& myObs=ourObservables[pIdx];
		auto& myMWS=mws[pIdx];
		py::dict myAction=py::reinterpret_borrow<py::dict>(action[pIdx]);

		//左右旋回
		double deltaAz=turnTable(myAction["turn"].cast<int>());
		double dstAz;
		if(myMWS.size()>0 && use_override_evasion){
			deltaAz=evasion_turnTable(myAction["turn"].cast<int>());
			Eigen::Vector3d dr=Eigen::Vector3d::Zero();
			for(auto& m:myMWS){
				dr+=m.dir();
			}
			dr.normalize();
			dstAz=atan2(-dr(1),-dr(0))+deltaAz;
			actionInfo.dstDir<<cos(dstAz),sin(dstAz),0;
		}else if(dstAz_relative){
			actionInfo.dstDir=myMotion.dirHtoP(Eigen::Vector3d(cos(deltaAz),sin(deltaAz),0),Eigen::Vector3d(0,0,0),"FSD",false);
		}else{
			actionInfo.dstDir=teamOrigin.relBtoP(Eigen::Vector3d(cos(deltaAz),sin(deltaAz),0));
		}
		dstAz=atan2(actionInfo.dstDir(1),actionInfo.dstDir(0));
		last_action_obs[parentFullName](0)=dstAz;

		//上昇・下降
		double dstPitch;
		if(!no_vertical_maneuver){
			if(use_altitude_command){
				double refAlt=std::round<int>(myMotion.getHeight()/refAltInterval)*refAltInterval;
				actionInfo.dstAlt=std::clamp<double>(refAlt+altTable(myAction["pitch"].cast<int>()),altMin,altMax);
				dstPitch=0;//dstAltをcommandsに与えればSixDoFFlightControllerのaltitudeKeeperで別途計算されるので0でよい。
			}else{
				dstPitch=pitchTable(myAction["pitch"].cast<int>());
			}
		}else{
			dstPitch=0;
		}
		actionInfo.dstDir<<actionInfo.dstDir(0)*cos(dstPitch),actionInfo.dstDir(1)*cos(dstPitch),-sin(dstPitch);
		last_action_obs[parentFullName](1)=use_altitude_command ? actionInfo.dstAlt : dstPitch;
	
		//加減速
		double V=myMotion.vel.norm();
		if(always_maxAB){
			actionInfo.asThrottle=true;
			actionInfo.keepVel=false;
			actionInfo.dstThrottle=1.0;
			last_action_obs[parentFullName](2)=1.0;
		}else{
			actionInfo.asThrottle=false;
			double accel=accelTable(myAction["accel"].cast<int>());
			actionInfo.dstV=V+accel;
			actionInfo.keepVel = accel==0.0;
			last_action_obs[parentFullName](2)=accel/std::max(accelTable(accelTable.size()-1),accelTable(0));
		}
		//下限速度の制限
		if(V<minimumV){
			actionInfo.velRecovery=true;
		}
		if(V>=minimumRecoveryV){
			actionInfo.velRecovery=false;
		}
		if(actionInfo.velRecovery){
			actionInfo.dstV=minimumRecoveryDstV;
			actionInfo.asThrottle=false;
		}

		//射撃
		//actionのパース
		int shotTarget=myAction["target"].cast<int>()-1;
		double shotInterval,shotThreshold;
		if(use_Rmax_fire){
			if(shotIntervalTable.size()>1){
				shotInterval=shotIntervalTable(myAction["shotInterval"].cast<int>());
			}else{
				shotInterval=shotIntervalTable(0);
			}
			if(shotThresholdTable.size()>1){
				shotThreshold=shotThresholdTable(myAction["shotThreshold"].cast<int>());
			}else{
				shotThreshold=shotThresholdTable(0);
			}
		}
		//射撃可否の判断、射撃コマンドの生成
		int flyingMsls=0;
		if(myObs.contains("/weapon/missiles"_json_pointer)){
			for(auto&& msl:myObs.at("/weapon/missiles"_json_pointer)){
				if(msl.at("isAlive").get<bool>() && msl.at("hasLaunched").get<bool>()){
					flyingMsls++;
				}
			}
		}
		if(
			shotTarget>=0 && 
			shotTarget<lastTrackInfo.size() && 
			getShared<FighterAccessor>(parent)->isLaunchableAt(lastTrackInfo[shotTarget]) &&
			flyingMsls<maxSimulShot
		){
			if(use_Rmax_fire){
				double rMin=std::numeric_limits<double>::infinity();
				Track3D t=lastTrackInfo[shotTarget];
				double r=calcRNorm(parent,myMotion,t,false);
				if(r<=shotThreshold){
					//射程の条件を満たしている
					if(actionInfo.lastShotTimes.count(t.truth)==0){
						actionInfo.lastShotTimes[t.truth]=manager->getEpoch();
					}
					if(manager->getTime()-actionInfo.lastShotTimes[t.truth]>=shotInterval){
						//射撃間隔の条件を満たしている
						actionInfo.lastShotTimes[t.truth]=manager->getTime();
					}else{
						//射撃間隔の条件を満たさない
						shotTarget=-1;
					}
				}else{
					//射程の条件を満たさない
					shotTarget=-1;
				}
			}
		}else{
			shotTarget=-1;
		}
		last_action_obs[parentFullName](3+(shotTarget+1))=1;
		if(shotTarget>=0){
			actionInfo.launchFlag=true;
			actionInfo.target=lastTrackInfo[shotTarget];
		}else{
			actionInfo.launchFlag=false;
			actionInfo.target=Track3D();
		}
	
		observables[parentFullName]["decision"]={
			{"Roll",nl::json::array({"Don't care"})},
			{"Fire",nl::json::array({actionInfo.launchFlag,actionInfo.target})}
		};
		if(myMWS.size()>0 && use_override_evasion){
			observables[parentFullName]["decision"]["Horizontal"]=nl::json::array({"Az_NED",dstAz});
		}else{
			if(dstAz_relative){
				observables[parentFullName]["decision"]["Horizontal"]=nl::json::array({"Az_BODY",deltaAz});
			}else{
				observables[parentFullName]["decision"]["Horizontal"]=nl::json::array({"Az_NED",dstAz});
			}
		}
		if(use_altitude_command){
			observables[parentFullName]["decision"]["Vertical"]=nl::json::array({"Pos",-actionInfo.dstAlt});
		}else{
			observables[parentFullName]["decision"]["Vertical"]=nl::json::array({"El",dstPitch});
		}
		if(actionInfo.asThrottle){
			observables[parentFullName]["decision"]["Throttle"]=nl::json::array({"Throttle",actionInfo.dstThrottle});
		}else{
			observables[parentFullName]["decision"]["Throttle"]=nl::json::array({"Vel",actionInfo.dstV});
		}
		pIdx++;
	}
}
void R7ContestAgentSample01::control(){
	//observablesの収集
	extractFriendObservables();
	//extractEnemyObservables(); // 彼機情報だけは射撃対象の選択と連動するので更新してはいけない。
	extractFriendMissileObservables();
	extractEnemyMissileObservables();

	//Setup collision avoider
	StaticCollisionAvoider2D avoider;
	//北側
	nl::json c={
		{"p1",Eigen::Vector3d(+dOut,-5*dLine,0)},
		{"p2",Eigen::Vector3d(+dOut,+5*dLine,0)},
		{"infinite_p1",true},
		{"infinite_p2",true},
		{"isOneSide",true},
		{"inner",Eigen::Vector2d(0.0,0.0)},
        {"limit",dOutLimit},
        {"threshold",dOutLimitThreshold},
		{"adjustStrength",dOutLimitStrength}
	};
	avoider.borders.push_back(std::make_shared<LinearSegment>(c));
	//南側
	c={
		{"p1",Eigen::Vector3d(-dOut,-5*dLine,0)},
		{"p2",Eigen::Vector3d(-dOut,+5*dLine,0)},
		{"infinite_p1",true},
		{"infinite_p2",true},
		{"isOneSide",true},
		{"inner",Eigen::Vector2d(0.0,0.0)},
        {"limit",dOutLimit},
        {"threshold",dOutLimitThreshold},
		{"adjustStrength",dOutLimitStrength}
	};
	avoider.borders.push_back(std::make_shared<LinearSegment>(c));
	//東側
	c={
		{"p1",Eigen::Vector3d(-5*dOut,+dLine,0)},
		{"p2",Eigen::Vector3d(+5*dOut,+dLine,0)},
		{"infinite_p1",true},
		{"infinite_p2",true},
		{"isOneSide",true},
		{"inner",Eigen::Vector2d(0.0,0.0)},
        {"limit",dOutLimit},
        {"threshold",dOutLimitThreshold},
		{"adjustStrength",dOutLimitStrength}
	};
	avoider.borders.push_back(std::make_shared<LinearSegment>(c));
	//西側
	c={
		{"p1",Eigen::Vector3d(-5*dOut,-dLine,0)},
		{"p2",Eigen::Vector3d(+5*dOut,-dLine,0)},
		{"infinite_p1",true},
		{"infinite_p2",true},
		{"isOneSide",true},
		{"inner",Eigen::Vector2d(0.0,0.0)},
        {"limit",dOutLimit},
        {"threshold",dOutLimitThreshold},
		{"adjustStrength",dOutLimitStrength}
	};
	avoider.borders.push_back(std::make_shared<LinearSegment>(c));

    int pIdx=0;
    for(auto&& [port,parent] : parents){
		auto parentFullName=parent->getFullName();
        if(!parent->isAlive()){
			pIdx++;
			continue;
		}
		auto& actionInfo=actionInfos[parentFullName];
		auto& myMotion=ourMotion[pIdx];
		auto& myObs=ourObservables[pIdx];
		MotionState originalMyMotion(myObs.at("motion")); //機体側にコマンドを送る際には元のparent座標系での値が必要

		//戦域逸脱を避けるための方位補正
		actionInfo.dstDir=avoider(myMotion,actionInfo.dstDir);

		//高度方向の補正(actionがピッチ指定の場合)
		if(!use_altitude_command){
			double n=sqrt(actionInfo.dstDir(0)*actionInfo.dstDir(0)+actionInfo.dstDir(1)*actionInfo.dstDir(1));
			double dstPitch=atan2(-actionInfo.dstDir(2),n);
			//高度下限側
			Eigen::Vector3d bottom=altitudeKeeper(myMotion,actionInfo.dstDir,altMin);
			double minPitch=atan2(-bottom(2),sqrt(bottom(0)*bottom(0)+bottom(1)*bottom(1)));
			//高度上限側
			Eigen::Vector3d top=altitudeKeeper(myMotion,actionInfo.dstDir,altMax);
			double maxPitch=atan2(-top(2),sqrt(top(0)*top(0)+top(1)*top(1)));
			dstPitch=std::clamp<double>(dstPitch,minPitch,maxPitch);
			double cs=cos(dstPitch);
			double sn=sin(dstPitch);
			actionInfo.dstDir=Eigen::Vector3d(actionInfo.dstDir(0)/n*cs,actionInfo.dstDir(1)/n*cs,-sn);
		}
		commands[parentFullName]={
			{"motion",{
				{"dstDir",originalMyMotion.dirAtoP(actionInfo.dstDir,myMotion.pos(),getLocalCRS())}//元のparent座標系に戻す
			}},
			{"weapon",{
				{"launch",actionInfo.launchFlag},
				{"target",actionInfo.target}
			}}
		};
		if(use_altitude_command){
			commands[parentFullName]["motion"]["dstAlt"]=actionInfo.dstAlt;
		}
		if(actionInfo.asThrottle){
			commands[parentFullName]["motion"]["dstThrottle"]=actionInfo.dstThrottle;
		}else if(actionInfo.keepVel){
			commands[parentFullName]["motion"]["dstAccel"]=0.0;
		}else{
			commands[parentFullName]["motion"]["dstV"]=actionInfo.dstV;
		}
		actionInfo.launchFlag=false;
		pIdx++;
	}
}
py::object R7ContestAgentSample01::convertActionFromAnother(const nl::json& decision,const nl::json& command){
	//observablesの収集
	extractFriendObservables();
	//extractEnemyObservables(); // 彼機情報だけは射撃対象の選択と連動するので更新してはいけない。
	extractFriendMissileObservables();
	extractEnemyMissileObservables();
	//double interval=getStepInterval()*manager->getBaseTimeStep();
	Eigen::VectorXd tmp_d;
	int tmp_i;
	py::list ret;
    int pIdx=0;
    for(auto&& [port,parent] : parents){
		py::dict myAction;
		auto parentFullName=parent->getFullName();
        if(!parent->isAlive()){
			//左右旋回
			tmp_d=turnTable-Eigen::VectorXd::Constant(turnTable.rows(),0.0);
			tmp_d.cwiseAbs().minCoeff(&tmp_i);
			myAction["turn"]=tmp_i;
			//上昇・下降
			if(!no_vertical_maneuver){
				if(use_altitude_command){
					tmp_d=altTable-Eigen::VectorXd::Constant(altTable.rows(),0.0);
				}else{
					tmp_d=pitchTable-Eigen::VectorXd::Constant(pitchTable.rows(),0.0);
				}
				tmp_d.cwiseAbs().minCoeff(&tmp_i);
				myAction["pitch"]=tmp_i;
			}
			//加減速
			if(!always_maxAB){
				tmp_d=accelTable-Eigen::VectorXd::Constant(accelTable.rows(),0.0);
				tmp_d.cwiseAbs().minCoeff(&tmp_i);
				myAction["accel"]=tmp_i;
			}
			//射撃
			myAction["target"]=0;
			if(use_Rmax_fire){
				if(shotIntervalTable.size()>1){
					myAction["shotInterval"]=0;
				}
				if(shotThresholdTable.size()>1){
					myAction["shotThreshold"]=shotThresholdTable.size()-1;
				}
			}
			pIdx++;
			ret.append(myAction);
			continue;
		}
		auto& actionInfo=actionInfos[parentFullName];
		MotionState& myMotion=ourMotion[pIdx];
		//左右旋回
		std::string decisionType=decision[parentFullName]["Horizontal"][0];
		double value=decision[parentFullName]["Horizontal"][1];
		nl::json motionCmd=command[parentFullName].at("motion");
		double dstAz_ex=0.0;
		if(decisionType=="Az_NED"){
			dstAz_ex=value;
		}else if(decisionType=="Az_BODY"){
			dstAz_ex=value+myMotion.getAZ();
		}else if(motionCmd.contains("dstDir")){
			Eigen::Vector3d dd=motionCmd.at("dstDir");
			dstAz_ex=atan2(dd(1),dd(0));
		}else{
			throw std::runtime_error("Given unsupported expert's decision & command for this Agent class.");
		}
		double deltaAz_ex=0.0;
		auto& myMWS=mws[pIdx];
		if(use_override_evasion && myMWS.size()>0){
			Eigen::Vector3d dr=Eigen::Vector3d::Zero();
			for(auto& m:myMWS){
				dr+=m.dir();
			}
			dr.normalize();
			deltaAz_ex=dstAz_ex-atan2(-dr(1),-dr(0));
			deltaAz_ex=atan2(sin(deltaAz_ex),cos(deltaAz_ex));
			tmp_d=evasion_turnTable-Eigen::VectorXd::Constant(evasion_turnTable.rows(),deltaAz_ex);
		}else{
			Eigen::Vector3d dstDir_ex;
			if(dstAz_relative){
				dstDir_ex = myMotion.dirPtoH(Eigen::Vector3d(cos(dstAz_ex),sin(dstAz_ex),0),myMotion.pos(),"FSD",false);
			}else{
				dstDir_ex = teamOrigin.relPtoB(Eigen::Vector3d(cos(dstAz_ex),sin(dstAz_ex),0));
			}
			deltaAz_ex = atan2(dstDir_ex(1),dstDir_ex(0));
			tmp_d=turnTable-Eigen::VectorXd::Constant(turnTable.rows(),deltaAz_ex);
		}
		tmp_d.cwiseAbs().minCoeff(&tmp_i);
		myAction["turn"]=tmp_i;

		//上昇・下降
		if(!no_vertical_maneuver){
			decisionType=decision[parentFullName]["Vertical"][0];
			value=decision[parentFullName]["Vertical"][1];
			double dstPitch_ex=0.0;
			double deltaAlt_ex=0.0;
			double refAlt=std::round<int>(myMotion.getHeight()/refAltInterval)*refAltInterval;
			if(decisionType=="El"){
				dstPitch_ex=value;
				deltaAlt_ex=altitudeKeeper.inverse(myMotion,dstPitch_ex)-refAlt;
			}else if(decisionType=="Pos"){
				dstPitch_ex=altitudeKeeper.getDstPitch(myMotion,-value);
				deltaAlt_ex=-value-refAlt;
			}else if(motionCmd.contains("dstDir")){
				Eigen::Vector3d dd=motionCmd.at("dstDir");
				dstPitch_ex=-atan2(dd(2),sqrt(dd(0)*dd(0)+dd(1)*dd(1)));
				deltaAlt_ex=altitudeKeeper.inverse(myMotion,dstPitch_ex)-refAlt;
			}else{
				throw std::runtime_error("Given unsupported expert's decision & command for this Agent class.");
			}
			dstPitch_ex=atan2(sin(dstPitch_ex),cos(dstPitch_ex));
			if(use_altitude_command){
				tmp_d=altTable-Eigen::VectorXd::Constant(altTable.rows(),deltaAlt_ex);
			}else{
				tmp_d=pitchTable-Eigen::VectorXd::Constant(pitchTable.rows(),dstPitch_ex);
			}
			tmp_d.cwiseAbs().minCoeff(&tmp_i);
			myAction["pitch"]=tmp_i;
		}

		//加減速
		if(!always_maxAB){
			//本サンプルでは「現在の速度と目標速度の差」をactionとしてdstVを機体へのコマンドとしているため、
			//教師役の行動が目標速度以外(スロットルや加速度)の場合、正確な変換は困難である。
			//そのため、最も簡易的な変換の例として、最大減速、等速、最大加速の3値への置換を実装している。
			decisionType=decision[parentFullName]["Throttle"][0];
			value=decision[parentFullName]["Throttle"][1];
			double V=myMotion.vel.norm();
			int accelIdx=-1;
			double dstV_ex;
			if(motionCmd.contains("dstV")){
				dstV_ex=motionCmd.at("dstV");
			}else if(decisionType=="Vel"){
				dstV_ex=value;
			}else if(decisionType=="Throttle"){
				//0〜1のスロットルで指定していた場合
				double th=0.3;
				if(value>1-th){
					accelIdx=accelTable.size()-1;
				}else if(value<th){
					accelIdx=0;
				}else{
					dstV_ex=V;
				}
			}else if(decisionType=="Accel"){
				//加速度ベースの指定だった場合
				double eps=0.5;
				if(abs(value)<eps){
					dstV_ex=V;
				}else if(value>0){
					accelIdx=accelTable.size()-1;
				}else{
					accelIdx=0;
				}
			}else{
				throw std::runtime_error("Given unsupported expert's decision & command for this Agent class.");
			}
			if(accelIdx<0){
				double deltaV_ex=dstV_ex-V;
				tmp_d=accelTable-Eigen::VectorXd::Constant(accelTable.rows(),deltaV_ex);
				tmp_d.cwiseAbs().minCoeff(&tmp_i);
				myAction["accel"]=tmp_i;
			}else{
				myAction["accel"]=accelIdx;
			}
		}

		//射撃
		int shotTarget_ex=-1;
		Track3D expertTarget=decision[parentFullName]["Fire"][1];
		if(decision[parentFullName]["Fire"][0].get<bool>()){
			for(int tIdx=0;tIdx<lastTrackInfo.size();++tIdx){
				const auto& t=lastTrackInfo[tIdx];
				if(t.isSame(expertTarget)){
					shotTarget_ex=tIdx;
				}
			}
			if(shotTarget_ex>=maxEnemyNum){
				shotTarget_ex=-1;
			}
		}
		myAction["target"]=shotTarget_ex+1;
		if(use_Rmax_fire){
			if(shotTarget_ex<0){
				//射撃なしの場合、間隔と射程の条件は最も緩いものとする
				if(shotIntervalTable.size()>1){
					myAction["shotInterval"]=0;
				}
				if(shotThresholdTable.size()>1){
					myAction["shotThreshold"]=shotThresholdTable.size()-1;
				}
			}else{
				//射撃ありの場合、間隔と射程の条件は最も厳しいものとする
				if(actionInfo.lastShotTimes.count(expertTarget.truth)==0){
					actionInfo.lastShotTimes[expertTarget.truth]=manager->getEpoch();
				}
				if(shotIntervalTable.size()>1){
					double shotInterval_ex=manager->getTime()-actionInfo.lastShotTimes[expertTarget.truth];
					myAction["shotInterval"]=0;
					for(int i=shotIntervalTable.size()-1;i>=0;i--){
						if(shotIntervalTable(i)<shotInterval_ex){
							myAction["shotInterval"]=i;
							break;
						}
					}
				}
				if(shotThresholdTable.size()>1){
					double r=calcRNorm(parent,myMotion,expertTarget,false);
					myAction["shotThreshold"]=shotThresholdTable.size()-1;
					for(int i=0;i<shotThresholdTable.size();i++){
						if(r<shotThresholdTable(i)){
							myAction["shotThreshold"]=i;
							break;
						}
					}
				}
			}
		}
		ret.append(myAction);
		pIdx++;
	}
	return ret;
}
void R7ContestAgentSample01::controlWithAnotherAgent(const nl::json& decision,const nl::json& command){
	//基本的にはオーバーライド不要だが、模倣時にActionと異なる行動を取らせたい場合に使用する。
	control();
	//例えば、以下のようにcommandを置換すると射撃のみexpertと同タイミングで行うように変更可能。
	//commands[parent->getFullName()]["weapon"]=command.at(parent->getFullName()).at("weapon");
}

bool R7ContestAgentSample01::isInside(const int& lon, const int& lat){
    //ピクセル座標(lon,lat)が画像の範囲内かどうかを返す。
	return 0<=lon && lon<image_longitudinal_resolution && 0<=lat && lat<image_lateral_resolution;
}
Eigen::Vector2i R7ContestAgentSample01::rposToGrid(const Eigen::Vector3d& dr){
	//基準点からの相対位置drに対応するピクセル座標(lon,lat)を返す。
	int lon=floor((dr(1)+image_side_range)*image_lateral_resolution/(2*image_side_range));
	int lat=floor((dr(0)+image_back_range)*image_longitudinal_resolution/(image_front_range+image_back_range));
	return Eigen::Vector2i(lon,lat);
}
Eigen::Vector3d R7ContestAgentSample01::gridToRpos(const int& lon, const int& lat, const double& alt){
	//指定した高度においてピクセル座標(lon,lat)に対応する基準点からの相対位置drを返す。
	return Eigen::Vector3d(
		image_buffer_coords(0,lon,lat),
		image_buffer_coords(1,lon,lat),
		-alt
	);
}
Eigen::Vector2i R7ContestAgentSample01::posToGrid(const Eigen::Vector3d& pos){
	//慣性座標系での絶対位置posに対応するピクセル座標(lon,lat)を返す。
	Eigen::Vector3d origin;
	if(image_relative_position){
		//生存中のparentsの中で先頭のparentの位置が基準点
		for(int fIdx=0;fIdx<parents.size();++fIdx){
			if(ourObservables[fIdx].at("isAlive")){
				origin=ourMotion[fIdx].pos();
				break;
			}
		}
	}else{
		//自陣営防衛ライン中央が基準点
		origin=teamOrigin.pos;
	}
	Eigen::Vector3d rpos=pos-origin;
	if(image_rotate){
		//生存中のparentsの中で先頭のparentのH座標系に回転
		for(int fIdx=0;fIdx<parents.size();++fIdx){
			if(ourObservables[fIdx].at("isAlive")){
				rpos=ourMotion[fIdx].relPtoH(rpos,origin,"FSD",false);
				break;
			}
		}
	}else{
		//慣性座標系から陣営座標系に回転
		rpos=teamOrigin.relPtoB(rpos);
	}
	return rposToGrid(rpos);
}
Eigen::Vector3d R7ContestAgentSample01::gridToPos(const int& lon, const int& lat, const double& alt){
	//指定した高度においてピクセル座標(lon,lat)に対応する慣性座標系での絶対位置posを返す。
	Eigen::Vector3d rpos=gridToRpos(lon,lat,alt);
	Eigen::Vector3d origin;
	if(image_relative_position){
		//生存中のparentsの中で先頭のparentの位置が基準点
		for(int fIdx=0;fIdx<parents.size();++fIdx){
			if(ourObservables[fIdx].at("isAlive")){
				origin=ourMotion[fIdx].pos();
				break;
			}
		}
	}else{
		//自陣営防衛ライン中央が基準点
		origin=teamOrigin.pos;
	}
	if(image_rotate){
		//生存中のparentsの中で先頭のparentのH座標系から慣性座標系に回転
		for(int fIdx=0;fIdx<parents.size();++fIdx){
			if(ourObservables[fIdx].at("isAlive")){
				rpos=ourMotion[fIdx].relHtoP(rpos,ourMotion[fIdx].absPtoH(origin,"FSD",false),"FSD",false);
				break;
			}
		}
	}else{
		//陣営座標系から慣性座標系に回転
		rpos=teamOrigin.relBtoP(rpos);
	}
	return rpos+origin;
}
void R7ContestAgentSample01::plotPoint(const Eigen::Vector3d& pos,const int& ch, const float& value){
	//画像バッファのチャネルchで絶対位置posに対応する点の値をvalueにする。
	Eigen::Vector2i g=posToGrid(pos);
	int lon=g(0);
	int lat=g(1);
	if(isInside(lon,lat)){
		image_buffer(ch,lon,lat)=value;
	}
}
void R7ContestAgentSample01::plotLine(const Eigen::Vector3d& pBegin,const Eigen::Vector3d& pEnd,const int& ch, const float& value){
    //画像バッファのチャネルchで慣性座標系での絶対位置pBeginからpEndまでの線分に対応する各点の値をvalueにする。
	//線分の描画にはブレゼンハムのアルゴリズムを使用している。
	Eigen::Vector2i gBegin=posToGrid(pBegin);
	Eigen::Vector2i gEnd=posToGrid(pEnd);
	int x0=gBegin(0);
	int y0=gBegin(1);
	int x1=gEnd(0);
	int y1=gEnd(1);
	bool steep=abs(y1-y0)>abs(x1-x0);
	if(steep){
		std::swap(x0,y0);
		std::swap(x1,y1);
	}
	if(x0>x1){
		std::swap(x0,x1);
		std::swap(y0,y1);
	}
	int deltax=x1-x0;
	int deltay=abs(y1-y0);
	int error=deltax/2;
	int ystep = y0<y1 ? 1 : -1;
	int y=y0;
	for(int x=x0;x<=x1;++x){
		if(steep){
			if(isInside(y,x)){
				image_buffer(ch,y,x)=value;
			}
		}else{
			if(isInside(x,y)){
				image_buffer(ch,x,y)=value;
			}
		}
		error-=deltay;
		if(error<0){
			y+=ystep;
			error+=deltax;
		}
	}
}

void exportR7ContestAgentSample01(py::module &m, const std::shared_ptr<asrc::core::FactoryHelper>& factoryHelper)
{
	using namespace pybind11::literals;
	bind_stl_container<std::vector<R7ContestAgentSample01::InstantInfo>>(m,py::module_local(true)); // python側からimage_past_dataを参照したい場合はこれが必要
	bind_stl_container<std::vector<R7ContestAgentSample01::ActionInfo>>(m,py::module_local(true)); // python側からactionInfosを参照したい場合はこれが必要
    expose_entity_subclass<R7ContestAgentSample01>(m,"R7ContestAgentSample01",py::module_local(true)) // 投稿時はpy::module_local(true)を引数に加えてビルドしたものを用いること！
	;
    FACTORY_ADD_CLASS(Agent,R7ContestAgentSample01)
}

ASRC_PLUGIN_NAMESPACE_END
