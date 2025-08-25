// Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#pragma once
#include <boost/uuid/uuid.hpp>
#include <ASRCAISim1/Agent.h>
#include <ASRCAISim1/MotionState.h>
#include <ASRCAISim1/Sensor.h>
#include <Eigen/CXX11/Tensor>
#include <ASRCAISim1/FlightControllerUtility.h>
#include <BasicAgentUtility/util/FuelManagementUtility.h>
#include <BasicAgentUtility/util/MissileRangeUtility.h>
#include <BasicAgentUtility/util/TeamOrigin.h>
#include <BasicAgentUtility/util/sortTrack3DByDistance.h>
#include <BasicAgentUtility/util/sortTrack2DByAngle.h>
#include "VirtualSimulatorSample.h"
#include "Common.h"

ASRC_PLUGIN_NAMESPACE_BEGIN

// 名前空間全体をusingすることも可能ではあるが推奨はしない。
// using directive is possible, but not recommended unless this module will not be inherited by another module.
//using namespace asrc::core; 

ASRC_DECLARE_DERIVED_CLASS_WITHOUT_TRAMPOLINE(R7ContestAgentSample01,asrc::core::Agent)
	/*第4回空戦AIチャレンジ向けの観測・行動空間の構築例。
		編隊全体で1つのAgentを割り当てる中央集権方式と、1機につき1つのAgentを割り当てる分散方式の両方に対応している。

		- observablesの抽出
			parentsのobservablesの取り扱いを容易にするため、以下の4つのメンバ関数によって種類ごとに自身のメンバ変数に格納する機能をもっている。
			- extractFriendObservables()…味方機のobservablesとmotionをparents→それ以外の順にメンバ変数ourObservablesとourMotionに格納する。
			- extractFriendMissileObservables()…味方誘導弾のobservablesを射撃時刻が古い順にソートしてメンバ変数mslsに格納する。
			- extractFriendEnemyObservables()…味方全体で共有している彼機のTrack3Dを最寄りの味方との距離が小さい順にソートしてメンバ変数lastTrackInfoに格納する。
			- extractFriendEnemyMissileObservables()…味方各機が探知している彼側誘導弾のTrack2Dを、自機正面に近い順にソートしてメンバ変数mwsに格納する。

		- observationの形式
			- 観測空間は複雑な構造を表現できるように、Dictとしている。
			- Blue側でもRed側でも同じになるように、自機座標系でないベクトル量は陣営座標系(自陣営の進行方向が+x方向となるようにz軸まわりに回転させ、防衛ライン中央が原点となるように平行移動させた座標系)で表現する。
			- Dict spaceの各要素の仕様は以下のとおり。
				それぞれ、modelConfigに与えるuse_xxxフラグの値によって内訳をカスタマイズ可能である。
				そのフラグによって空となった要素は出力されない。
				- common (Box)
					以下の要素を1次元のBox(shape=[common_dim])で格納したもの
					- 残り時間(1)…分単位の値。remaining_time_clippingにより上側がクリッピングされる。
				- image (Box)
					- use_image_observationをtrueとした場合のみ生成する。
					- 画像の解像度はimage_longitudinal_resolution×image_lateral_resolutionとする。
					- ピクセルのインデックスは画像の左下を(0,0)とする。
					- 戦域を描画する座標系(描画座標系)の+x方向は画像の上方向、+y方向は画像の右方向とする。
					- 描画座標系の原点は、image_relative_positionをTrueとした場合は自機位置、Falseとした場合は自陣営防衛ライン中央とする。
					- 描画座標系の+x方向は、image_rotateをTrueとした場合は自機正面、Falseとした場合は自陣営の進行すべき方向とする。
					- 描画座標系の+y方向は、戦域を真上から見た図として描画されるような向きに取る。
					- x軸方向(縦方向)の描画範囲はimage_front_range+image_back_rangeとし、原点は画像下端からimage_back_range/(image_front_range+image_back_range)の位置とする。
					- y軸方向(横方向)の描画範囲は2×image_side_rangeとし、原点は画像の中央とする。
					- 各チャネルの内容は以下の通り。それぞれ、draw_xxxフラグで有効、無効を切り替えられる。
						1. parentsの軌跡…現時刻が1.0、image_horizon秒前が0となるような線形減衰で、image_interval秒間隔で描画する。
						2. parents以外の味方の軌跡…現時刻が1.0、image_horizon秒前が0となるような線形減衰で、image_interval秒間隔で描画する。
						3. 彼機の軌跡…現時刻が1.0、image_horizon秒前が0となるような線形減衰で、image_interval秒間隔で描画する。
						4. 味方誘導弾の軌跡…現時刻が1.0、image_horizon秒前が0となるような線形減衰で、image_interval秒間隔で描画する。
						5. parentsのレーダ覆域…現時刻の覆域内を1とする。
						6. parents以外のレーダ覆域…現時刻の覆域内を1とする。
						7. 彼我防衛ライン…彼側の防衛ラインを1、我側の防衛ラインを-1とする。
						8. 場外ライン…ライン上を1とする。
				- parent (Box)
					parentsの諸元を2次元のBox(shape=[maxParentNum,parent_dim])で格納したもの
					- 直前の行動(4+maxEnemyNum)
						1. 左右旋回…前回の目標方位を現在の方位からの差分で格納。
						2. 上昇・下降…前回の目標高度またはピッチ角を格納。no_vertical_maneuverがtrueの場合は常に0とする。
						3. 加減速…前回の目標加速度を、accelTableの絶対値が最大の要素で正規化したものを格納。always_maxABがtrueの場合は常に1とする。
						4. 射撃…前回の射撃コマンドを、射撃なし+射撃対象IDのone hotで格納。
					- 自身を目標とした彼側の仮想誘導弾との距離の最小値(1)…use_virtual_simulator=Trueとした場合のみ。0発の場合は別途コンフィグで指定された定数。horizontalNormalizerで正規化する。
					- friendに使用される全ての諸元(friend_dim)
				- friend (Box)
					parents以外の味方機の諸元を2次元のBox(shape=[maxFriendNum,friend_dim])で格納したもの
					- 位置(3)…x,y成分をhorizontalNormalizerで、z成分をverticalNormalizerで正規化したもの。
					- 速度(4)…速度のノルムをfgtrVelNormalizerで正規化したものと、速度方向の単位ベクトルの4次元に分解したもの。
					- 初期弾数(1)
					- 残弾数(1)
					- 姿勢(3)…バンク角、α(迎角)、β(横滑り角)の3次元をradで表現したもの。
					- 角速度(3)…機体座標系での角速度をrad/sで表現したもの。
					- 余剰燃料(1)…現在の余剰燃料を距離に換算したものを、2*dLineで正規化して-1〜+1にクリッピングしたもの。
					- RCSスケール(1)
					- レーダ探知距離(1)…horizontalNormalizerで正規化。
					- レーダ覆域(1)…角度をラジアンで
					- 最大速度倍率(1)
					- 誘導弾推力倍率(1)…基準推力(この例ではハードコーディング)の値で正規化。
				- enemy (Box)
					- 位置(3)…x,y成分をhorizontalNormalizerで、z成分をverticalNormalizerで正規化したもの。
					- 速度(4)…速度のノルムをfgtrVelNormalizerで正規化したものと、速度方向の単位ベクトルの4次元に分解したもの。
				- friend_missile (Box)
					- 位置(3)…x,y成分をhorizontalNormalizerで、z成分をverticalNormalizerで正規化したもの。
					- 速度(4)…速度のノルムをfgtrVelNormalizerで正規化したものと、速度方向の単位ベクトルの4次元に分解したもの。
					- 飛翔時間(1)…射撃からの経過時間を分単位で。
					- 目標との距離(1)…horizontalNormalizerで正規化
					- 目標への誘導状態(3)…Missile::Mode(GUIDED,SELF,MEMORY)のone-hotベクトル
					- 目標の位置(3)…x,y成分をhorizontalNormalizerで、z成分をverticalNormalizerで正規化したもの。
					- 目標の速度(4)…速度のノルムをfgtrVelNormalizerで正規化したものと、速度方向の単位ベクトルの4次元に分解したもの。
					- 誘導弾推力倍率(1)…基準推力(この例ではハードコーディング)の値で正規化。
				- enemy_missile (Box)
					- 観測点の位置(3)…x,y成分をhorizontalNormalizerで、z成分をverticalNormalizerで正規化したもの。
					- 方向(3)…検出された方向(陣営座標系)
					- 方向変化率(3)…検出された方向の変化率(陣営座標系)
				- friend_enemy_relative (Box)
					ここはparentsとparents以外の両方を対象とする。
					- 味方から敵へのRHead(1)…horizontalNormalizerで正規化
					- 味方から敵へのRTail(1)…horizontalNormalizerで正規化
					- 味方から敵へのRNorm(1)…現在の彼我間距離をRTail〜RHeadが0〜1となるように正規化したもの
					- 敵から味方へのRHead(1)…horizontalNormalizerで正規化
					- 敵から味方へのRTail(1)…horizontalNormalizerで正規化
					- 敵から味方へのRNorm(1)…現在の彼我間距離をRTail〜RHeadが0〜1となるように正規化したもの
				- observation_mask (Dict)
					use_observation_maskをtrueにしたときのみ生成される、各要素に有効な値が入っているかどうかを表すマスク。
					意味上はboolだが、floatのBoxで格納し、1が有効、0が無効を表すものとする。
					- parent [maxParentNum]
					- friend [maxFriendNum]
					- enemy [maxEnemyNum]
					- friend_missile [maxFriendMissileNum]
					- enemy_missile [maxEnemyMissileNum]
					- friend_enemy_relative [maxParentNum+max]
				- action_mask (Tuple[Dict])
					use_action_maskをtrueにしたときのみ生成される、各parentの各行動空間の有効な選択肢を表すマスク。
					意味上はboolだが、floatのBoxで格納し、1が有効、0が無効を表すものとする。
					このサンプルではtargetのみマスクを計算し、それ以外の行動については全て有効(1)を出力する。
					なお、強化学習ライブラリの中には探索時に無効な行動を除外する機能がないものも多いため、
					そのようなライブラリを使用する場合にこのマスクを使ってlogitsに-infを入れたりすると発散する可能性がある。

		- actionの形式
			各parentのactionを表すDictをparentの数だけ並べたTupleとする。
			Dictの内訳は以下の通り。
			- turn (Discrete) 左右旋回を表す。
				ある基準方位を0として目標方位(右を正)で指定する。
				基準方位は、dstAz_relativeフラグをTrueとした場合、自機正面となり、Falseとした場合、自陣営の進行すべき方向となる。
				目標方位の選択肢はturnTableで与える。
				また、use_override_evasionフラグをTrueとした場合、MWS検出時の基準方位と目標方位テーブルを上書きすることが可能。
				基準方位は検出された誘導弾の到来方位の平均値と逆向きとし、目標方位テーブルはevasion_turnTableで与える。
			- pitch (Discrete) 上昇・下降を表す。
				no_vertical_maneuverフラグがFalseの場合のみ追加する。ユース部門はFalseを推奨する。
				use_altitude_commandフラグをTrueとした場合、基準高度からの目標高度差で指定する。
				基準高度はrefAltInterval間隔で最寄りの高度とし、目標高度差の選択肢はaltTableで与える。
				use_altitude_commandフラグをFalseとした場合、水平を0とした目標ピッチ角(上昇を正))で指定する。
				目標ピッチ角の選択肢はpitchTableで与える。
			- accel (Discrete) 加減速を表す。
				always_maxABをTrueとした場合は常時最大推力を出力するものとし、spaceからは削除する。
				Falseとした場合は基準速度(=現在速度)に対する速度差として目標速度を指定する。
				速度差の選択肢はaccelTableで与える。
			- target (Discrete) 射撃対象を表す。
				0を射撃なし、1〜maxEnemyNumを対応するlastTrackInfoのTrack3DとしたDiscrete形式で指定する。
			- shotInterval (Discrete) 射撃間隔(秒)を表す。
				use_Rmax_fireフラグをTrueとした場合のみspaceに加える。
				選択したTrack3Dに対して前回の射撃からこの秒数以上経過している場合のみ射撃を行う。
				選択肢はshotIntervalTableで与える。
			- shotThreshold (Discrete) 射程条件を表す。
				use_Rmax_fireフラグをTrueとした場合のみspaceに加える。
				RTailを0、RHeadを1として距離を正規化した値で、閾値を下回った場合に射撃を行う。
				選択肢はshotThresholdTableで与える。

		- 行動制限について
			このサンプルでは、いくつかの観点でAIの行動判断を上書きして、一般に望ましくないと思われる挙動を抑制する例を実装している。
			1. 高度の制限
			altMinを下限、altMaxを上限とした高度範囲内で行動するように、それを逸脱しそうな場合に制限をかける。
			spaceが目標高度の場合は単純なクリッピングで範囲を限定する。
			spaceが目標ピッチ角の場合は、簡易な高度制御則を用いて制限ピッチ角を計算し、それを用いてクリッピングを行う。
			2. 場外の制限
			戦域中心からdOutLimitの位置に引いた基準ラインの超過具合に応じて目標方位に制限をかける。
			無限遠で基準ラインに直交、基準ライン上で基準ラインと平行になるようなatanスケールでの角度補正を行う。
			補正を行う範囲は、戦域端からdOutLimitThresholdの位置に引いたラインからとする。
			3. 同時射撃数の制限
			自機の飛翔中誘導弾がmaxSimulShot発以上のときは新たな射撃を行えないようにしている。
			4. 下限速度の制限
			速度がminimumVを下回った場合、minimumRecoveryVを上回るまでの間、
			目標速度をminimumRecovertDstVに固定することで、低速域での飛行を抑制している。

		- Attributes:
			- modelConfigで設定するもの
				- observation spaceの設定
					- maxParentNum (int): observation,actionに用いるparentsの最大数
					- maxFriendNum (int): observationに用いるparents以外の味方機の最大数
					- maxEnemyNum (int): observationに用いる彼機航跡の最大数
					- maxFriendMissileNum (int): observationに用いる我側誘導弾の最大数
					- maxEnemyMissileNum (int): observationに用いる彼側誘導弾の最大数
					- horizontalNormalizer (float): 水平方向の位置・距離の正規化のための除数
					- verticalNormalizer (float): 高度方向の正規化のための除数
					- fgtrVelNormalizer (float): 機体速度の正規化のための除数
					- mslVelNormalizer (float): 誘導弾速度の正規化のための除数
					- 仮想シミュレータの使用
						- use_virtual_simulator (bool): 仮想シミュレータを使用するかどうか
						- virtual_simulator_value_when_no_missile (float): 仮想誘導弾が存在しないときに用いる距離定数。大きい値を推奨。
						- virtual_simulator_maxNumPerParent (int): 各parentあたりの最大仮想誘導弾数
						- virtual_simulator_launchInterval (int): 仮想誘導弾の生成間隔 (agent step数単位)
						- virtual_simulator_kShoot (float): 仮想誘導弾の射撃条件(彼から我へのRNormがこの値以下のとき射撃)
					- 2次元画像としてのobservation
						- use_image_observation (bool): 戦域を2次元画像で表現したobservationを使用するかどうか
						- image_longitudinal_resolution (int): 前後方向の解像度
						- image_lateral_resolution (int): 左右方向の解像度
						- image_front_range (float): 前方の描画範囲(m)
						- image_back_range (float): 後方の描画範囲(m)
						- image_side_range (float): 側方の描画範囲(m)
						- image_horizon (int): 軌跡を描画する秒数
						- image_interval (int): 軌跡を描画する間隔
						- image_rotate (bool): 画像化の基準座標系。trueの場合は生存中のparentsのうち先頭の機体の正面を-x軸とするように回転した座標系。falseの場合は陣営座標系とする。
						- image_relative_position (bool): 画像化の基準位置。trueの場合は生存中のparentsのうち先頭の機体の位置、falseの場合は自陣営防衛ライン中央とする。
						- draw_parent_trajectory (bool): parentsの軌跡を描画するかどうか
						- draw_friend_trajectory (bool): parents以外の味方の軌跡を描画するかどうか
						- draw_enemy_trajectory (bool): 彼機の軌跡を描画するかどうか
						- draw_friend_missile_trajectory (bool): 味方誘導弾の軌跡を描画するかどうか
						- draw_parent_radar_coverage (bool): parentsのレーダ覆域を描画するかどうか
						- draw_friend_radar_coverage (bool): parents以外のレーダ覆域を描画するかどうか
						- draw_defense_line (bool): 防衛ラインを描画するかどうか
						- draw_side_line (bool): 南北の場外ラインを描画するかどうか
					- 共通情報に関するobservation
						- use_remaining_time (bool): 残り時間の情報を入れるかどうか
						- remaining_time_clipping (float): 残り時間の情報を入れる際の上側クリッピング値
					- parentsの状態量に関するobservation
						- use_parent_last_action (bool): 前回の行動情報を入れるかどうか
					- 味方機の状態量に関するobservation
						- use_friend_position (bool): 位置情報を入れるかどうか
						- use_friend_velocity (bool): 速度情報を入れるかどうか
						- use_friend_initial_num_missile (bool): 初期弾数情報を入れるかどうか
						- use_friend_current_num_missile (bool): 残弾数情報を入れるかどうか
						- use_friend_attitude (bool): 姿勢情報を入れるかどうか
						- use_friend_angular_velocity (bool): 角速度情報を入れるかどうか
						- use_friend_current_fuel (bool): 残燃料情報を入れるかどうか
						- use_friend_rcs_scale (bool): RCS情報を入れるかどうか
						- use_friend_radar_range (bool): レーダ探知距離情報を入れるかどうか
						- use_friend_radar_coverage (bool): レーダ覆域情報を入れるかどうか
						- use_friend_maximum_speed_scale (bool): 最大速度倍率情報を入れるかどうか
						- use_friend_missile_thrust_scale (bool): 誘導弾推力倍率情報を入れるかどうか(誘導弾側のオンオフとも連動)
					- 彼機の状態量に関するobservation
						- use_enemy_position (bool): 位置情報を入れるかどうか
						- use_enemy_velocity (bool): 速度情報を入れるかどうか
					- 味方誘導弾の状態量に関するobservation
						- use_friend_missile_position (bool): 位置情報を入れるかどうか
						- use_friend_missile_velocity (bool): 速度情報を入れるかどうか
						- use_friend_missile_flight_time (bool): 飛翔時間情報を入れるかどうか
						- use_friend_missile_target_distance (bool): 目標との距離情報を入れるかどうか
						- use_friend_missile_target_mode (bool): 目標への誘導状態を入れるかどうか
						- use_friend_missile_target_position (bool): 目標の位置情報を入れるかどうか
						- use_friend_missile_target_velocity (bool): 目標の速度情報を入れるかどうか
						- use_friend_missile_thrust_scale (bool): 誘導弾推力倍率情報を入れるかどうか(戦闘機側のオンオフとも連動)
					- 彼側誘導弾の状態量に関するobservation
						- use_enemy_missile_observer_position (bool): 観測者の位置情報を入れるかどうか
						- use_enemy_missile_direction (bool): 検出方向情報を入れるかどうか
						- use_enemy_missile_angular_velocity (bool): 検出方向変化率情報を入れるかどうか
					- 我機と彼機の間の関係性に関するobservation
						- use_our_missile_range (bool): 我機から彼機への射程情報を入れるかどうか
						- use_their_missile_range (bool): 彼機から我機への射程情報を入れるかどうか
				- action spaceの設定
					- 左右旋回に関する設定
						- dstAz_relative (bool): 旋回の原点に関する設定。trueの場合は自機正面、falseの場合は自陣営の進行方向が原点となる。
						- turnTable (list[float]): 通常時の目標方位(deg)テーブル(右が正)
						- evasion_turnTable (list[float]): MWS作動時の目標方位(deg)テーブル(検出した誘導弾を背にした方位を0とし、右が正)
						- use_override_evasion (bool): MWS作動時の目標方位テーブルを専用のものに置き換えるかどうか
					- 上昇・下降に関する設定
						- no_vertical_maneuver (bool): 上昇・下降のアクションを用いるかどうか。ユース部門はfalseを推奨する。
						- pitchTable (list[float]): 目標ピッチ角(deg)テーブル(上昇が正)
						- altTable (list[float]): 目標高度(m)テーブル(最寄りの基準高度からの高度差で指定)
						- refAltInterval (float): 基準高度グリッドの間隔(m)
						- use_altitude_command (bool): ピッチ角と高度のどちらを使用するか。trueでピッチ、falseで高度を使用する。
					- 加減速に関する設定
						- accelTable (list[float]): 目標加速度テーブル
						- always_maxAB (bool): 常時maxABするかどうか。
					- 射撃に関する設定
						- shotIntervalTable (list[float]) 同一目標に対する射撃間隔(秒)テーブル
						- shotThresholdTable (list[float]): 射撃閾値テーブル(RTailを0、RHeadを1とした線形空間での閾値)
						- use_Rmax_fire (bool): 既存の射程テーブルを用いて射撃判定を行うかどうか。
				- 行動制限に関する設定
					- 高度制限に関する設定(ピッチ角をactionにしている場合のみ有効)
						- altMin (float): 下限高度
						- altMax (float): 上限高度
						- altitudeKeeper (dict): ピッチ角制限を計算するための高度制御則。サンプルではFlightControllerUtility.hのAltitudeKeeperクラスを使用しており、configではそのパラメータをdictで指定する。
					- 場外制限に関する設定
						- dOutLimit (float): 場外防止の基準ラインの距離
						- dOutLimitThreshold (float): 場外防止を開始する距離
						- dOutLimitStrength (float): 場外防止の復元力に関する係数
					- 同時射撃数の制限に関する設定
						- maxSimulShot (int): 自身が同時に射撃可能な誘導弾数の上限
					- 下限速度の制限に関する設定
						- minimumV (float): 低速域からの回復を開始する速度
						- minimumRecoveryV (float): 低速域からの回復を終了する速度
						- minimumRecoveryDstV (float): 低速域からの回復時に設定する目標速度
			- 内部変数
				- observationに関するもの
					- _observation_space (gym.spaces.Space): 観測空間を表すgym.spaces.Space
					- ourMotion (list[MotionState]): 味方各機のMotionState
					- ourObservables (list[nl::json]): 味方各機のobservables
					- lastTrackInfo (list[Track3D]): 自陣営が捉えている彼機航跡
					- msls (list[nl::json]): 自陣営の誘導弾(のobservables)
					- mws (list[list[Track2D]]): 各味方が検出している彼側誘導弾航跡
					- virtualSimulator (VirtualSimulator): 仮想シミュレータ
					- 2次元画像としてのobservation
						- numChannels (int): チャネル数
						- image_buffer (3-dim ndarray): 画像データバッファ。shape=[numChannels,image_longitudinal_resolution,image_lateral_resolution]
						- image_buffer_coords (3-dim ndarray): ピクセル座標(lon,lat)と戦域座標(x,y)のマッピング。shape=[2,image_longitudinal_resolution,image_lateral_resolution]
						- image_past_data (list[InstantInfo]): 軌跡描画用の過去データのバッファ
						- struct InstantInfo: 機軌跡描画用のフレームデータ
							parent_pos (list[3-elem 1-dim ndarray]): parentsの位置
							friend_pos (list[3-elem 1-dim ndarray]): parents以外の味方の位置
							friend_msl_pos (list[3-elem 1-dim ndarray]): 味方誘導弾の位置
							enemy_pos (list[3-elem 1-dim ndarray]): 彼機の位置
					- common_dim (int): 共通情報(残り時間等)の次元
					- parent_dim (int): parent1機あたりの次元
					- friend_dim (int): parent以外の味方1機あたりの次元
					- enemy_dim (int): 彼機1機あたりの次元
					- friend_missile_dim (int): 味方誘導弾1発あたりの次元
					- enemy_missile_dim (int): 彼側誘導弾1発あたりの次元
					- friend_enemy_relative_dim (int): 我機と彼機の関係性1組あたりの次元
					- last_action_dim (int): parent1機あたりの前回の行動の次元
					- last_action_obs (dict[str,1-dim ndarray]): 各parentsの前回の行動を格納するバッファ。shape=[last_action_dim]
				- actionに関するもの
					- _action_space (gym.spaces.Space): 行動空間を表すgym.spaces.Space
					- struct ActionInfo: 機体に対するコマンドを生成するための構造体
						- dstDir (3-elem 1-dim ndarray): 目標進行方向
						- dstAlt (float): 目標高度
						- velRecovery (bool): 下限速度制限からの回復中かどうか
						- asThrottle (bool): 加減速についてスロットルでコマンドを生成するかどうか
						- keepVel (bool): 加減速について等速(dstAccel=0)としてコマンドを生成するかどうか
						- dstThrottle (float): 目標スロットル
						- dstV (float): 目標速度
						- launchFlag (bool): /射撃するかどうか
						- target (Track3D): 射撃対象
						- lastShotTimes (dict[boost::uuids::uuid,Time]): 各Trackに対する直前の射撃時刻
					actionInfos (dict[str,ActionInfo]): 各parentsの行動を表す変数
				- Ruler、陣営に関するもの
					- dOut (float): 戦域中心から場外ラインまでの距離
					- dLine (float): 戦域中心から防衛ラインまでの距離
					- teamOrigin (BasicAgentUtility::util::TeamOrigin): 陣営座標系(進行方向が+x方向となるようにz軸まわりに回転させ、防衛ライン中央が原点となるように平行移動させた座標系)を表すクラス
	*/

	// asrc::core::をなるべく省略したい場合は、クラス内で一度だけalias宣言する方が安全である。
    // Alias declration inside the class is safer way to prevent writing "asrc::core::" everywhere.
	using AltitudeKeeper=asrc::core::util::AltitudeKeeper;
	using MotionState=asrc::core::MotionState;
	using PhysicalAssetAccessor=asrc::core::PhysicalAssetAccessor;
	using Track2D=asrc::core::Track2D;
	using Track3D=asrc::core::Track3D;
	using Time=asrc::core::Time;
	public:
	//modelConfigで設定するもの
	//  observation spaceの設定
	int maxParentNum; //observation,actionに用いるparentsの最大数
	int maxFriendNum; //observationに用いるparents以外の味方機の最大数
	int maxEnemyNum; //observationに用いる彼機航跡の最大数
	int maxFriendMissileNum; //observationに用いる我側誘導弾の最大数
	int maxEnemyMissileNum; //observationに用いる彼側誘導弾の最大数
	double horizontalNormalizer; //水平方向の位置・距離の正規化のための除数
	double verticalNormalizer; //高度方向の正規化のための除数
	double fgtrVelNormalizer; //機体速度の正規化のための除数
	double mslVelNormalizer; //誘導弾速度の正規化のための除数
	bool use_observation_mask; //無効なobservationに関するmask情報を含めるかどうか
	bool use_action_mask; //無効なactionに関するmask情報を含めるかどうか
	bool raise_when_invalid_action; //無効なactionが渡された場合に例外を投げるかどうか
	bool use_virtual_simulator; //仮想シミュレータを使用するかどうか
	double virtual_simulator_value_when_no_missile; //仮想誘導弾が存在しないときに用いる距離定数。大きい値を推奨。
	//    2次元画像としてのobservation
	bool use_image_observation; //戦域を2次元画像で表現したobservationを使用するかどうか
	int image_longitudinal_resolution; //前後方向の解像度
	int image_lateral_resolution; //左右方向の解像度
	double image_front_range; //前方の描画範囲(m)
	double image_back_range; //後方の描画範囲(m)
	double image_side_range; //側方の描画範囲(m)
	int image_horizon; //軌跡を描画する秒数
	int image_interval; //軌跡を描画する間隔
	bool image_rotate; //画像化の基準座標系。trueの場合自機正面を-x軸とするように回転。falseの場合、慣性座標系そのまま(赤青の陣営による反転は行う)。
	bool image_relative_position; //画像化の基準位置。trueの場合自機位置、falseの場合自陣営防衛ライン中央とする。
	bool draw_parent_trajectory; //parentsの軌跡を描画するかどうか
	bool draw_friend_trajectory; //parents以外の味方の軌跡を描画するかどうか
	bool draw_enemy_trajectory; //彼機の軌跡を描画するかどうか
	bool draw_friend_missile_trajectory; //味方誘導弾の軌跡を描画するかどうか
	bool draw_parent_radar_coverage; //parentsのレーダ覆域を描画するかどうか
	bool draw_friend_radar_coverage; //parents以外のレーダ覆域を描画するかどうか
	bool draw_defense_line; //防衛ラインを描画するかどうか
	bool draw_side_line; //南北の場外ラインを描画するかどうか
	//    共通情報に関するobservation
	bool use_remaining_time; //残り時間の情報を入れるかどうか
	double remaining_time_clipping; //残り時間の情報を入れる際の上側クリッピング値
	//    parentsの状態量に関するobservation
	bool use_parent_last_action; // 前回の行動情報を入れるかどうか
	//    味方機の状態量に関するobservation
	bool use_friend_position; // 位置情報を入れるかどうか
	bool use_friend_velocity; // 速度情報を入れるかどうか
	bool use_friend_initial_num_missile; //初期弾数情報を入れるかどうか
	bool use_friend_current_num_missile; //残弾数情報を入れるかどうか
	bool use_friend_attitude; //姿勢情報を入れるかどうか
	bool use_friend_angular_velocity; //角速度情報を入れるかどうか
	bool use_friend_current_fuel; //残燃料情報を入れるかどうか
	bool use_friend_rcs_scale; //RCS情報を入れるかどうか
	bool use_friend_radar_range; //レーダ探知距離情報を入れるかどうか
	bool use_friend_radar_coverage; //レーダ覆域情報を入れるかどうか
	bool use_friend_maximum_speed_scale; //最大速度倍率情報を入れるかどうか
	bool use_friend_missile_thrust_scale; //誘導弾推力倍率情報を入れるかどうか(誘導弾側のオンオフとも連動)
	//    彼機の状態量に関するobservation
	bool use_enemy_position; // 位置情報を入れるかどうか
	bool use_enemy_velocity; // 速度情報を入れるかどうか
	//    味方誘導弾の状態量に関するobservation
	bool use_friend_missile_position; // 位置情報を入れるかどうか
	bool use_friend_missile_velocity; // 速度情報を入れるかどうか
	bool use_friend_missile_flight_time; // 飛翔時間情報を入れるかどうか
	bool use_friend_missile_target_distance; // 目標との距離情報を入れるかどうか
	bool use_friend_missile_target_mode; //目標への誘導状態を入れるかどうか
	bool use_friend_missile_target_position; // 目標の位置情報を入れるかどうか
	bool use_friend_missile_target_velocity; // 目標の速度情報を入れるかどうか
	//bool use_friend_missile_thrust_scale; //誘導弾推力倍率情報を入れるかどうか(戦闘機側のオンオフとも連動)
	//    彼側誘導弾の状態量に関するobservation
	bool use_enemy_missile_observer_position; // 観測者の位置情報を入れるかどうか
	bool use_enemy_missile_direction; // 検出方向情報を入れるかどうか
	bool use_enemy_missile_angular_velocity; //検出方向変化率情報を入れるかどうか
	//   我機と彼機の間の関係性に関するobservation
	bool use_our_missile_range; // 我機から彼機への射程情報を入れるかどうか
	bool use_their_missile_range; // 彼機から我機への射程情報を入れるかどうか

	//  action spaceの設定
	//    左右旋回に関する設定
	bool dstAz_relative; //旋回の原点に関する設定。trueの場合は自機正面、falseの場合は自陣営の進行方向が原点となる。
	Eigen::VectorXd turnTable; //通常時の目標方位(deg)テーブル(右が正)
	Eigen::VectorXd evasion_turnTable; //MWS作動時の目標方位(deg)テーブル(検出した誘導弾を背にした方位を0とし、右が正)
	bool use_override_evasion; //MWS作動時の目標方位テーブルを専用のものに置き換えるかどうか
	//    上昇・下降に関する設定
	bool no_vertical_maneuver; //上昇・下降のアクションを用いるかどうか。ユース部門はfalseを推奨する。
	Eigen::VectorXd pitchTable; //目標ピッチ角(deg)テーブル(上昇が正)
	Eigen::VectorXd altTable; //目標高度(m)テーブル(最寄りの基準高度からの高度差で指定)
	double refAltInterval; //基準高度グリッドの間隔(m)
	bool use_altitude_command; //ピッチ角と高度のどちらを使用するか。trueでピッチ、falseで高度を使用する。
	//    加減速に関する設定
	Eigen::VectorXd accelTable; //目標加速度テーブル
	bool always_maxAB; //常時maxABするかどうか。
	//    射撃に関する設定
	Eigen::VectorXd shotIntervalTable; //同一目標に対する射撃間隔(秒)テーブル
	Eigen::VectorXd shotThresholdTable; //射撃閾値テーブル(RTailを0、RHeadを1とした線形空間での閾値)
	bool use_Rmax_fire; //既存の射程テーブルを用いて射撃判定を行うかどうか。

	//  行動制限に関する設定
	//    高度制限に関する設定(ピッチ角をactionにしている場合のみ有効)
	double altMin; //下限高度
	double altMax; //上限高度
	AltitudeKeeper altitudeKeeper; //ピッチ角制限を計算するための高度制御則。サンプルではFighterControllerのモデルと同じものを同じ設定で使用している。
	//    場外制限に関する設定
	double dOutLimit; //場外防止の基準ラインの距離。
	double dOutLimitThreshold; //場外防止を開始する距離。
	double dOutLimitStrength; //場外防止の復元力に関する係数。
	//    同時射撃数の制限に関する設定
	int maxSimulShot; //自身が同時に射撃可能な誘導弾数の上限
	//  下限速度の制限に関する設定
	double minimumV; //低速域からの回復を開始する速度
	double minimumRecoveryV; //低速域からの回復を終了する速度
	double minimumRecoveryDstV; //低速域からの回復時に設定する目標速度

	//内部変数
	//  observationに関するもの
	py::object _observation_space; //観測空間を表すgym.spaces.Space
	std::vector<MotionState> ourMotion; //味方各機のMotionState
	std::vector<nl::json> ourObservables; //味方各機のobservables
	std::vector<Track3D> lastTrackInfo; //自陣営が捉えている彼機航跡
	std::vector<nl::json> msls; //自陣営の誘導弾(のobservables)
	std::vector<std::vector<Track2D>> mws; //味方各機のMWS航跡
	std::shared_ptr<VirtualSimulatorSample> virtualSimulator; //仮想シミュレータ
	//    2次元画像としてのobservation
	int numChannels;//チャネル数
	Eigen::Tensor<float,3> image_buffer; //画像データバッファ[ch×lon×lat]
	Eigen::Tensor<double,3> image_buffer_coords; //ピクセル座標(lon,lat)と戦域座標(x,y)のマッピング[2×lon×lat]
	ASRC_DECLARE_NON_POLYMORPHIC_DATA_CLASS(InstantInfo)
		//軌跡描画用のフレームデータ
		public:
		std::vector<Eigen::Vector3d> parent_pos; //parentsの位置
		std::vector<Eigen::Vector3d> friend_pos; //parents以外の味方の位置
		std::vector<Eigen::Vector3d> friend_msl_pos; //味方誘導弾の位置
		std::vector<Eigen::Vector3d> enemy_pos; //彼機の位置
    	template<class Archive>
    	void serialize(Archive & archive){
			ASRC_SERIALIZE_NVP(archive
				,parent_pos
				,friend_pos
				,friend_msl_pos
				,enemy_pos
			)
		}
	};
	std::vector<InstantInfo> image_past_data; //軌跡描画用の過去データのバッファ

	int common_dim; //共通情報(残り時間等)の次元
	int parent_dim; //parent1機あたりの次元
	int friend_dim; //parent以外の味方機あたりの次元
	int enemy_dim; //彼機1機あたりの次元
	int friend_missile_dim; //味方誘導弾1発あたりの次元
	int enemy_missile_dim; //彼側誘導弾1発あたりの次元
	int friend_enemy_relative_dim; //我機と彼機の関係性1組あたりの次元

	int last_action_dim; //parent1機あたりの前回の行動の次元
	std::map<std::string,Eigen::VectorXf> last_action_obs; //前回の行動を格納するバッファ

	//  actionに関するもの
	py::object _action_space; //行動空間を表すgym.spaces.Space

	ASRC_DECLARE_NON_POLYMORPHIC_DATA_CLASS(ActionInfo)
		//    機体に対するコマンドを生成するための変数をまとめた構造体
		public:
		ActionInfo();
		Eigen::Vector3d dstDir; //目標進行方向
		double dstAlt; //目標高度
		bool velRecovery; //下限速度制限からの回復中かどうか
		bool asThrottle; //加減速についてスロットルでコマンドを生成するかどうか
		bool keepVel; //加減速について等速(dstAccel=0)としてコマンドを生成するかどうか
		double dstThrottle; //目標スロットル
		double dstV; //目標速度
		bool launchFlag; //射撃するかどうか
		Track3D target; //射撃対象
		std::map<boost::uuids::uuid,Time> lastShotTimes; //各Trackに対する直前の射撃時刻
    	template<class Archive>
    	void serialize(Archive & archive){
			ASRC_SERIALIZE_NVP(archive
				,dstDir
				,dstAlt
				,velRecovery
				,asThrottle
				,keepVel
				,dstThrottle
				,dstV
				,launchFlag
				,target
				,lastShotTimes
			)
		}
	};
	std::map<std::string,ActionInfo> actionInfos;
	py::object action_mask;
	
	//Ruler、陣営に関するもの
	double dOut; //戦域中心から場外ラインまでの距離
	double dLine; //戦域中心から防衛ラインまでの距離

	//陣営座標系(進行方向が+x方向となるようにz軸まわりに回転させ、防衛ライン中央が原点となるように平行移動させた座標系)を表すクラス。
	//MotionStateを使用しても良いがクォータニオンを経由することで浮動小数点演算に起因する余分な誤差が生じるため、もし可能な限り対称性を求めるのであればこの例のように符号反転で済ませたほうが良い。
	//ただし、機体運動等も含めると全ての状態量に対して厳密に対称なシミュレーションとはならないため、ある程度の誤差は生じる。
	BasicAgentUtility::util::TeamOrigin teamOrigin;

	public:
    //constructors & destructor
    using BaseType::BaseType;
    //functions
    virtual void initialize() override;
    virtual void serializeInternalState(asrc::core::util::AvailableArchiveTypes & archive, bool full) override;
	virtual void validate() override;
	virtual py::object observation_space() override;
	virtual py::object makeObs() override;
	virtual void extractFriendObservables();
	virtual void extractFriendMissileObservables();
	virtual void extractEnemyObservables();
	virtual void extractEnemyMissileObservables();
	virtual void makeImageObservation();
	virtual Eigen::VectorXf makeCommonStateObservation();
	virtual Eigen::VectorXf makeParentStateObservation(const std::shared_ptr<PhysicalAssetAccessor>& parent, const nl::json& fObs, const MotionState& fMotion);
	virtual Eigen::VectorXf makeFriendStateObservation(const nl::json& fObs, const MotionState& fMotion);
	virtual Eigen::VectorXf makeEnemyStateObservation(const Track3D& track);
	virtual Eigen::VectorXf makeFriendMissileStateObservation(const nl::json& mObs);
	virtual Eigen::VectorXf makeEnemyMissileStateObservation(const Track2D& track);
	virtual Eigen::Tensor<float,1> makeFriendEnemyRelativeStateObservation(const std::shared_ptr<PhysicalAssetAccessor>& parent,const nl::json& fObs, const MotionState& fMotion, const Track3D& track);
	virtual py::object makeActionMask();
	virtual py::object action_space() override;
	virtual void deploy(py::object action) override;
	virtual void control() override;
	virtual py::object convertActionFromAnother(const nl::json& decision,const nl::json& command) override;
	virtual void controlWithAnotherAgent(const nl::json& decision,const nl::json& command);
	//画像化するための関数
	bool isInside(const int& lon, const int& lat); //ピクセル座標が画像内かどうか
	Eigen::Vector2i rposToGrid(const Eigen::Vector3d& dr); //相対位置→ピクセル
	Eigen::Vector3d gridToRpos(const int& lon, const int& lat, const double& alt); //ピクセル+高度→相対位置
	Eigen::Vector2i posToGrid(const Eigen::Vector3d& pos); //絶対位置→ピクセル
	Eigen::Vector3d gridToPos(const int& lon, const int& lat, const double& alt); //ピクセル+高度→絶対位置
	void plotPoint(const Eigen::Vector3d& pos,const int& ch, const float& value); //点を描画
	void plotLine(const Eigen::Vector3d& pBegin,const Eigen::Vector3d& pEnd,const int& ch, const float& value); //線分を描画
};

void exportR7ContestAgentSample01(py::module &m, const std::shared_ptr<asrc::core::FactoryHelper>& factoryHelper);

ASRC_PLUGIN_NAMESPACE_END

ASRC_PYBIND11_MAKE_OPAQUE(std::vector<ASRC_PLUGIN_NAMESPACE::R7ContestAgentSample01::InstantInfo>); // python側からimage_past_dataを参照したい場合はこれが必要
ASRC_PYBIND11_MAKE_OPAQUE(std::vector<ASRC_PLUGIN_NAMESPACE::R7ContestAgentSample01::ActionInfo>); // python側からactionInfosを参照したい場合はこれが必要
