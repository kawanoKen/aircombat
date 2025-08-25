# Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
from math import *
from gymnasium import spaces
import numpy as np
import sys
from ASRCAISim1.core import (
	AltitudeKeeper,
	Agent,
	deg2rad,
	getValueFromJsonK,
	getValueFromJsonKR,
	getValueFromJsonKRD,
	LinearSegment,
	nljson,
	Missile,
	MotionState,
	PhysicalAssetAccessor,
	StaticCollisionAvoider2D,
	Time,
	TimeSystem,
	Track2D,
	Track3D,
	serialize_attr_with_type_info,
	serialize_by_func,
	serialize_internal_state_by_func,
	save_with_type_info,
	load_with_type_info,
	isOutputArchive,
	isInputArchive,
)
from BasicAgentUtility.util import (
	calcDistanceMargin,
	calcRHead,
	calcRTail,
	calcRNorm,
	calcRHeadE,
	calcRTailE,
	calcRNormE,
	TeamOrigin,
	sortTrack3DByDistance,
	sortTrack2DByAngle
)

from .PyVirtualSimulatorSample import PyVirtualSimulatorSample

class R7ContestPyAgentSample01(Agent):
	"""第4回空戦AIチャレンジ向けの観測・行動空間の構築例。
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
	"""

	class InstantInfo:
		#軌跡描画用のフレームデータ
		def __init__(self):
			self.parent_pos=[] #parentsの位置
			self.friend_pos=[] #parents以外の味方の位置
			self.friend_msl_pos=[] #味方誘導弾の位置
			self.enemy_pos=[] #彼機の位置
		def serialize(self, archive):
			serialize_attr_with_type_info(archive, self
				,"parent_pos"
				,"friend_pos"
				,"friend_msl_pos"
				,"enemy_pos"
			)
		_allow_cereal_serialization_in_cpp = True
		def save(self, archive):
			self.serialize(archive)
		@classmethod
		def static_load(cls, archive):
			ret=cls()
			ret.serialize(archive)
			return ret

	class ActionInfo:
		#機体に対するコマンドを生成するための変数をまとめた構造体
		def __init__(self):
			self.dstDir=np.array([1.0,0.0,0.0]) #目標進行方向
			self.dstAlt=10000.0 #目標高度
			self.velRecovery=False #下限速度制限からの回復中かどうか
			self.asThrottle=False #加減速についてスロットルでコマンドを生成するかどうか
			self.keepVel=False #加減速について等速(dstAccel=0)としてコマンドを生成するかどうか
			self.dstThrottle=1.0 #目標スロットル
			self.dstV=300 #目標速度
			self.launchFlag=False #射撃するかどうか
			self.target=Track3D() #射撃対象
			self.lastShotTimes={} #各Trackに対する直前の射撃時刻
		def serialize(self, archive):
			serialize_attr_with_type_info(archive, self
				,"dstDir"
				,"dstAlt"
				,"velRecovery"
				,"asThrottle"
				,"keepVel"
				,"dstThrottle"
				,"dstV"
				,"launchFlag"
				,"target"
				,"lastShotTimes"
			)
		_allow_cereal_serialization_in_cpp = True
		def save(self, archive):
			self.serialize(archive)
		@classmethod
		def static_load(cls, archive):
			ret=cls()
			ret.serialize(archive)
			return ret

	def initialize(self):
		super().initialize()
		#modelConfigの読み込み
		#observation spaceの設定
		self.maxParentNum=getValueFromJsonK(self.modelConfig,"maxParentNum")
		self.maxFriendNum=getValueFromJsonK(self.modelConfig,"maxFriendNum")
		self.maxEnemyNum=getValueFromJsonK(self.modelConfig,"maxEnemyNum")
		self.maxFriendMissileNum=getValueFromJsonK(self.modelConfig,"maxFriendMissileNum")
		self.maxEnemyMissileNum=getValueFromJsonK(self.modelConfig,"maxEnemyMissileNum")
		assert self.maxParentNum>=len(self.parents)

		self.horizontalNormalizer=getValueFromJsonKR(self.modelConfig,"horizontalNormalizer",self.randomGen)
		self.verticalNormalizer=getValueFromJsonKR(self.modelConfig,"verticalNormalizer",self.randomGen)
		self.fgtrVelNormalizer=getValueFromJsonKR(self.modelConfig,"fgtrVelNormalizer",self.randomGen)
		self.mslVelNormalizer=getValueFromJsonKR(self.modelConfig,"mslVelNormalizer",self.randomGen)

		self.use_observation_mask=getValueFromJsonK(self.modelConfig,"use_observation_mask")
		self.use_action_mask=getValueFromJsonK(self.modelConfig,"use_action_mask")
		self.raise_when_invalid_action=getValueFromJsonK(self.modelConfig,"raise_when_invalid_action")

		# 仮想シミュレータの使用
		# この例では各parentに最も近い仮想誘導弾との距離をparentのobservationに付加する
		self.use_virtual_simulator=getValueFromJsonK(self.modelConfig,"use_virtual_simulator")
		if self.use_virtual_simulator:
			self.virtual_simulator_value_when_no_missile=getValueFromJsonK(self.modelConfig,"virtual_simulator_value_when_no_missile")
			self.virtualSimulator=PyVirtualSimulatorSample({
				"maxNumPerParent": getValueFromJsonK(self.modelConfig,"virtual_simulator_maxNumPerParent"),
				"launchInterval": getValueFromJsonK(self.modelConfig,"virtual_simulator_launchInterval"),
				"kShoot": getValueFromJsonK(self.modelConfig,"virtual_simulator_kShoot")
			})
	
		#  2次元画像としてのobservation
		self.use_image_observation=getValueFromJsonK(self.modelConfig,"use_image_observation")
		if self.use_image_observation:
			self.image_longitudinal_resolution=getValueFromJsonK(self.modelConfig,"image_longitudinal_resolution")
			self.image_lateral_resolution=getValueFromJsonK(self.modelConfig,"image_lateral_resolution")
			self.image_front_range=getValueFromJsonKR(self.modelConfig,"image_front_range",self.randomGen)
			self.image_back_range=getValueFromJsonKR(self.modelConfig,"image_back_range",self.randomGen)
			self.image_side_range=getValueFromJsonKR(self.modelConfig,"image_side_range",self.randomGen)
			self.image_horizon=getValueFromJsonKR(self.modelConfig,"image_horizon",self.randomGen)
			self.image_interval=getValueFromJsonKR(self.modelConfig,"image_interval",self.randomGen)
			self.image_rotate=getValueFromJsonKRD(self.modelConfig,"image_rotate",self.randomGen,False)
			self.image_relative_position=getValueFromJsonKRD(self.modelConfig,"image_relative_position",self.randomGen,True)
			self.draw_parent_trajectory=getValueFromJsonK(self.modelConfig,"draw_parent_trajectory")
			self.draw_friend_trajectory=getValueFromJsonK(self.modelConfig,"draw_friend_trajectory")
			self.draw_enemy_trajectory=getValueFromJsonK(self.modelConfig,"draw_enemy_trajectory")
			self.draw_friend_missile_trajectory=getValueFromJsonK(self.modelConfig,"draw_friend_missile_trajectory")
			self.draw_parent_radar_coverage=getValueFromJsonK(self.modelConfig,"draw_parent_radar_coverage")
			self.draw_friend_radar_coverage=getValueFromJsonK(self.modelConfig,"draw_friend_radar_coverage")
			self.draw_defense_line=getValueFromJsonK(self.modelConfig,"draw_defense_line")
			self.draw_side_line=getValueFromJsonK(self.modelConfig,"draw_side_line")

		#  共通情報に関するobservation
		self.use_remaining_time=getValueFromJsonK(self.modelConfig,"use_remaining_time")
		if self.use_remaining_time:
			self.remaining_time_clipping=getValueFromJsonKR(self.modelConfig,"remaining_time_clipping",self.randomGen)
		self.common_dim=(
			(1 if self.use_remaining_time else 0)
		)

		#  parentsの状態量に関するobservation
		#  parent_dimはaction spaceの設定を読み込むまで確定しないので後回し
		self.use_parent_last_action=getValueFromJsonK(self.modelConfig,"use_parent_last_action")

		#  味方機の状態量に関するobservation
		self.use_friend_position=getValueFromJsonK(self.modelConfig,"use_friend_position")
		self.use_friend_velocity=getValueFromJsonK(self.modelConfig,"use_friend_velocity")
		self.use_friend_initial_num_missile=getValueFromJsonK(self.modelConfig,"use_friend_initial_num_missile")
		self.use_friend_current_num_missile=getValueFromJsonK(self.modelConfig,"use_friend_current_num_missile")
		self.use_friend_attitude=getValueFromJsonK(self.modelConfig,"use_friend_attitude")
		self.use_friend_angular_velocity=getValueFromJsonK(self.modelConfig,"use_friend_angular_velocity")
		self.use_friend_current_fuel=getValueFromJsonK(self.modelConfig,"use_friend_current_fuel")
		self.use_friend_rcs_scale=getValueFromJsonK(self.modelConfig,"use_friend_rcs_scale")
		self.use_friend_radar_range=getValueFromJsonK(self.modelConfig,"use_friend_radar_range")
		self.use_friend_radar_coverage=getValueFromJsonK(self.modelConfig,"use_friend_radar_coverage")
		self.use_friend_maximum_speed_scale=getValueFromJsonK(self.modelConfig,"use_friend_maximum_speed_scale")
		self.use_friend_missile_thrust_scale=getValueFromJsonK(self.modelConfig,"use_friend_missile_thrust_scale")
		self.friend_dim=(
			(3 if self.use_friend_position else 0)
			+(4 if self.use_friend_velocity else 0)
			+(1 if self.use_friend_initial_num_missile else 0)
			+(1 if self.use_friend_current_num_missile else 0)
			+(3 if self.use_friend_attitude else 0)
			+(3 if self.use_friend_angular_velocity else 0)
			+(1 if self.use_friend_current_fuel else 0)
			+(1 if self.use_friend_rcs_scale else 0)
			+(1 if self.use_friend_radar_range else 0)
			+(1 if self.use_friend_radar_coverage else 0)
			+(1 if self.use_friend_maximum_speed_scale else 0)
			+(1 if self.use_friend_missile_thrust_scale else 0)
		)

		#  彼機の状態量に関するobservation
		self.use_enemy_position=getValueFromJsonK(self.modelConfig,"use_enemy_position")
		self.use_enemy_velocity=getValueFromJsonK(self.modelConfig,"use_enemy_velocity")
		self.enemy_dim=(
			(3 if self.use_enemy_position else 0)
			+(4 if self.use_enemy_velocity else 0)
		)

		#  味方誘導弾の状態量に関するobservation
		self.use_friend_missile_position=getValueFromJsonK(self.modelConfig,"use_friend_missile_position")
		self.use_friend_missile_velocity=getValueFromJsonK(self.modelConfig,"use_friend_missile_velocity")
		self.use_friend_missile_flight_time=getValueFromJsonK(self.modelConfig,"use_friend_missile_flight_time")
		self.use_friend_missile_target_distance=getValueFromJsonK(self.modelConfig,"use_friend_missile_target_distance")
		self.use_friend_missile_target_mode=getValueFromJsonK(self.modelConfig,"use_friend_missile_target_mode")
		self.use_friend_missile_target_position=getValueFromJsonK(self.modelConfig,"use_friend_missile_target_position")
		self.use_friend_missile_target_velocity=getValueFromJsonK(self.modelConfig,"use_friend_missile_target_velocity")
		self.friend_missile_dim=(
			(3 if self.use_friend_missile_position else 0)
			+(4 if self.use_friend_missile_velocity else 0)
			+(1 if self.use_friend_missile_flight_time else 0)
			+(1 if self.use_friend_missile_target_distance else 0)
			+(3 if self.use_friend_missile_target_mode else 0)
			+(3 if self.use_friend_missile_target_position else 0)
			+(4 if self.use_friend_missile_target_velocity else 0)
			+(1 if self.use_friend_missile_thrust_scale else 0)
		)

		#  彼側誘導弾の状態量に関するobservation
		self.use_enemy_missile_observer_position=getValueFromJsonK(self.modelConfig,"use_enemy_missile_observer_position")
		self.use_enemy_missile_direction=getValueFromJsonK(self.modelConfig,"use_enemy_missile_direction")
		self.use_enemy_missile_angular_velocity=getValueFromJsonK(self.modelConfig,"use_enemy_missile_angular_velocity")
		self.enemy_missile_dim=(
			(3 if self.use_enemy_missile_observer_position else 0)
			+(3 if self.use_enemy_missile_direction else 0)
			+(3 if self.use_enemy_missile_angular_velocity else 0)
		)

		# 我機と彼機の間の関係性に関するobservation
		self.use_our_missile_range=getValueFromJsonK(self.modelConfig,"use_our_missile_range")
		self.use_their_missile_range=getValueFromJsonK(self.modelConfig,"use_their_missile_range")
		self.friend_enemy_relative_dim=(
			(3 if self.use_our_missile_range else 0)
			+(3 if self.use_their_missile_range else 0)
		)

		# action spaceの設定
		# 左右旋回に関する設定
		self.dstAz_relative=getValueFromJsonK(self.modelConfig,"dstAz_relative")
		self.turnTable=np.array(sorted(getValueFromJsonK(self.modelConfig,"turnTable")),dtype=np.float64)
		self.turnTable*=deg2rad(1.0)
		self.use_override_evasion=getValueFromJsonK(self.modelConfig,"use_override_evasion")
		if self.use_override_evasion:
			self.evasion_turnTable=np.array(sorted(getValueFromJsonK(self.modelConfig,"evasion_turnTable")),dtype=np.float64)
			self.evasion_turnTable*=deg2rad(1.0)
			assert len(self.turnTable)==len(self.evasion_turnTable)
		else:
			self.evasion_turnTable=self.turnTable
		# 上昇・下降に関する設定
		self.no_vertical_maneuver=getValueFromJsonK(self.modelConfig,"no_vertical_maneuver")
		if not self.no_vertical_maneuver:
			self.use_altitude_command=getValueFromJsonK(self.modelConfig,"use_altitude_command")
			if self.use_altitude_command:
				self.altTable=np.array(sorted(getValueFromJsonK(self.modelConfig,"altTable")),dtype=np.float64)
				self.refAltInterval=getValueFromJsonK(self.modelConfig,"refAltInterval")
			else:
				self.pitchTable=np.array(sorted(getValueFromJsonK(self.modelConfig,"pitchTable")),dtype=np.float64)
				self.pitchTable*=deg2rad(1.0)
				self.refAltInterval=1.0
		else:
			self.use_altitude_command=False
		# 加減速に関する設定
		self.accelTable=np.array(sorted(getValueFromJsonK(self.modelConfig,"accelTable")),dtype=np.float64)
		self.always_maxAB=getValueFromJsonK(self.modelConfig,"always_maxAB")
		# 射撃に関する設定
		self.use_Rmax_fire=getValueFromJsonK(self.modelConfig,"use_Rmax_fire")
		if self.use_Rmax_fire:
			self.shotIntervalTable=np.array(sorted(getValueFromJsonK(self.modelConfig,"shotIntervalTable")),dtype=np.float64)
			self.shotThresholdTable=np.array(sorted(getValueFromJsonK(self.modelConfig,"shotThresholdTable")),dtype=np.float64)
		#行動制限に関する設定
		#  高度制限に関する設定
		self.altMin=getValueFromJsonKRD(self.modelConfig,"altMin",self.randomGen,2000.0)
		self.altMax=getValueFromJsonKRD(self.modelConfig,"altMax",self.randomGen,15000.0)
		self.altitudeKeeper=AltitudeKeeper(self.modelConfig().get("altitudeKeeper",{}))
		# 場外制限に関する設定
		self.dOutLimit=getValueFromJsonKRD(self.modelConfig,"dOutLimit",self.randomGen,5000.0)
		self.dOutLimitThreshold=getValueFromJsonKRD(self.modelConfig,"dOutLimitThreshold",self.randomGen,10000.0)
		self.dOutLimitStrength=getValueFromJsonKRD(self.modelConfig,"dOutLimitStrength",self.randomGen,2e-3)
		# 同時射撃数の制限に関する設定
		self.maxSimulShot=getValueFromJsonKRD(self.modelConfig,"maxSimulShot",self.randomGen,4)
		# 下限速度の制限に関する設定
		self.minimumV=getValueFromJsonKRD(self.modelConfig,"minimumV",self.randomGen,150.0)
		self.minimumRecoveryV=getValueFromJsonKRD(self.modelConfig,"minimumRecoveryV",self.randomGen,180.0)
		self.minimumRecoveryDstV=getValueFromJsonKRD(self.modelConfig,"minimumRecoveryDstV",self.randomGen,200.0)

		#前回の行動に関する情報
		self.last_action_dim=3+(1+self.maxEnemyNum)
		self.last_action_obs={}
		self.actionInfos={}
		for port,parent in self.parents.items():
			self.last_action_obs[parent.getFullName()]=np.zeros([self.last_action_dim],dtype=np.float32)
			self.actionInfos[parent.getFullName()]=self.ActionInfo()

		#  parentsの状態量に関するobservation
		self.parent_dim=(
			(self.last_action_dim if self.use_parent_last_action else 0)
			+ (1 if self.use_virtual_simulator else 0)
		)+self.friend_dim

		#spaceの確定
		floatLow=np.finfo(np.float32).min
		floatHigh=np.finfo(np.float32).max
		#  actionに関するもの
		single_action_space_dict={}
		single_action_mask_space_dict={}

		single_action_space_dict["turn"]=spaces.Discrete(len(self.turnTable))
		if self.use_action_mask:
			single_action_mask_space_dict["turn"]=spaces.Box(floatLow,floatHigh,
				shape=[len(self.turnTable)],
				dtype=np.float32)

		if not self.no_vertical_maneuver:
			single_action_space_dict["pitch"]=spaces.Discrete(len(self.altTable) if self.use_altitude_command else len(self.pitchTable))
			if self.use_action_mask:
				single_action_mask_space_dict["pitch"]=spaces.Box(floatLow,floatHigh,
					shape=[len(self.altTable) if self.use_altitude_command else len(self.pitchTable)],
					dtype=np.float32)
		
		if not self.always_maxAB:
			single_action_space_dict["accel"]=spaces.Discrete(len(self.accelTable))
			if self.use_action_mask:
				single_action_mask_space_dict["accel"]=spaces.Box(floatLow,floatHigh,
					shape=[len(self.accelTable)],
					dtype=np.float32)

		single_action_space_dict["target"]=spaces.Discrete(1+self.maxEnemyNum)
		if self.use_action_mask:
			single_action_mask_space_dict["target"]=spaces.Box(floatLow,floatHigh,
				shape=[1+self.maxEnemyNum],
				dtype=np.float32)

		if self.use_Rmax_fire:
			if len(self.shotIntervalTable)>1:
				single_action_space_dict["shotInterval"]=spaces.Discrete(len(self.shotIntervalTable))
				if self.use_action_mask:
					single_action_mask_space_dict["shotInterval"]=spaces.Box(floatLow,floatHigh,
						shape=[len(self.shotIntervalTable)],
						dtype=np.float32)

			if len(self.shotThresholdTable)>1:
				single_action_space_dict["shotThreshold"]=spaces.Discrete(len(self.shotThresholdTable))
				if self.use_action_mask:
					single_action_mask_space_dict["shotThreshold"]=spaces.Box(floatLow,floatHigh,
						shape=[len(self.shotThresholdTable)],
						dtype=np.float32)

		single_action_space=spaces.Dict(single_action_space_dict)
		if self.use_action_mask:
			single_action_mask_space=spaces.Dict(single_action_mask_space_dict)
		action_space_list=[]
		action_mask_space_list=[]
		for port, parent in self.parents.items():
			action_space_list.append(single_action_space)
			if self.use_action_mask:
				action_mask_space_list.append(single_action_mask_space)
		self._action_space=spaces.Tuple(action_space_list)

		#  observationに関するもの
		observation_space_dict={}
		observation_mask_space_dict={}
		#共通情報(残り時間等)
		if self.common_dim>0:
			observation_space_dict["common"]=spaces.Box(floatLow,floatHigh,
				shape=[self.common_dim],
				dtype=np.float32)

		#画像情報
		if self.use_image_observation:
			self.numChannels=(
				(1 if self.draw_parent_trajectory else 0)
				+(1 if self.draw_friend_trajectory else 0)
				+(1 if self.draw_enemy_trajectory else 0)
				+(1 if self.draw_friend_missile_trajectory else 0)
				+(1 if self.draw_parent_radar_coverage else 0)
				+(1 if self.draw_friend_radar_coverage else 0)
				+(1 if self.draw_defense_line else 0)
				+(1 if self.draw_side_line else 0)
			)
			#画像データバッファの生成
			self.image_buffer=np.zeros([self.numChannels,self.image_longitudinal_resolution,self.image_lateral_resolution],dtype=np.float32)
			self.image_buffer_coords=np.zeros([2,self.image_longitudinal_resolution,self.image_lateral_resolution])
			for lon in range(self.image_longitudinal_resolution):
				for lat in range(self.image_lateral_resolution):
					self.image_buffer_coords[0,lon,lat]=(self.image_front_range+self.image_back_range)*(0.5+lat)/self.image_longitudinal_resolution-self.image_back_range
					self.image_buffer_coords[1,lon,lat]=2.0*self.image_side_range*(0.5+lon)/self.image_lateral_resolution-self.image_side_range
			#軌跡描画用の過去データバッファの生成
			self.image_past_data=[self.InstantInfo() for i in range(self.image_horizon)]
			#spaceの生成
			observation_space_dict["image"]=spaces.Box(floatLow,floatHigh,
				shape=[self.numChannels,self.image_longitudinal_resolution,self.image_lateral_resolution],
				dtype=np.float32)

		#parentsの諸元
		if self.maxParentNum>0 and self.parent_dim>0:
			observation_space_dict["parent"]=spaces.Box(floatLow,floatHigh,
				shape=[self.maxParentNum,self.parent_dim],
				dtype=np.float32)
			if self.use_observation_mask:
				observation_mask_space_dict["parent"]=spaces.Box(floatLow,floatHigh,
					shape=[self.maxParentNum],
					dtype=np.float32)

		#parents以外の味方機の諸元
		if self.maxFriendNum>0 and self.friend_dim>0:
			observation_space_dict["friend"]=spaces.Box(floatLow,floatHigh,
				shape=[self.maxFriendNum,self.friend_dim],
				dtype=np.float32)
			if self.use_observation_mask:
				observation_mask_space_dict["friend"]=spaces.Box(floatLow,floatHigh,
					shape=[self.maxFriendNum],
					dtype=np.float32)

		#彼機
		if self.maxEnemyNum>0 and self.enemy_dim>0:
			observation_space_dict["enemy"]=spaces.Box(floatLow,floatHigh,
				shape=[self.maxEnemyNum,self.enemy_dim],
				dtype=np.float32)
			if self.use_observation_mask:
				observation_mask_space_dict["enemy"]=spaces.Box(floatLow,floatHigh,
					shape=[self.maxEnemyNum],
					dtype=np.float32)

		#味方誘導弾
		if self.maxFriendMissileNum>0 and self.friend_missile_dim>0:
			observation_space_dict["friend_missile"]=spaces.Box(floatLow,floatHigh,
				shape=[self.maxFriendMissileNum,self.friend_missile_dim],
				dtype=np.float32)
			if self.use_observation_mask:
				observation_mask_space_dict["friend_missile"]=spaces.Box(floatLow,floatHigh,
					shape=[self.maxFriendMissileNum],
					dtype=np.float32)

		#彼側誘導弾
		if self.maxEnemyMissileNum>0 and self.enemy_missile_dim>0:
			observation_space_dict["enemy_missile"]=spaces.Box(floatLow,floatHigh,
				shape=[self.maxEnemyMissileNum,self.enemy_missile_dim],
				dtype=np.float32)
			if self.use_observation_mask:
				observation_mask_space_dict["enemy_missile"]=spaces.Box(floatLow,floatHigh,
					shape=[self.maxEnemyMissileNum],
					dtype=np.float32)

		#我機と彼機の間の関係性
		if self.maxParentNum+self.maxFriendNum>0 and self.maxEnemyNum>0 and self.friend_enemy_relative_dim>0:
			observation_space_dict["friend_enemy_relative"]=spaces.Box(floatLow,floatHigh,
				shape=[self.maxParentNum+self.maxFriendNum,self.maxEnemyNum,self.friend_enemy_relative_dim],
				dtype=np.float32)
			if self.use_observation_mask:
				observation_mask_space_dict["friend_enemy_relative"]=spaces.Box(floatLow,floatHigh,
					shape=[self.maxParentNum+self.maxFriendNum,self.maxEnemyNum],
					dtype=np.float32)

		if self.use_observation_mask:
			observation_space_dict["observation_mask"]=spaces.Dict(observation_mask_space_dict)

		if self.use_action_mask:
			observation_space_dict["action_mask"]=spaces.Tuple(action_mask_space_list)

		self._observation_space=spaces.Dict(observation_space_dict)

	def serializeInternalState(self, archive, full: bool):
		if self.use_virtual_simulator:
			raise ValueError("VirtualSimulator is not serializable in the current version.")

		super().serializeInternalState(archive, full)

		if full:
			#observation spaceの設定
			serialize_attr_with_type_info(archive, self
				,"maxParentNum"
				,"maxFriendNum"
				,"maxEnemyNum"
				,"maxFriendMissileNum"
				,"maxEnemyMissileNum"
				,"horizontalNormalizer"
				,"verticalNormalizer"
				,"fgtrVelNormalizer"
				,"mslVelNormalizer"
				,"use_virtual_simulator"
				,"use_image_observation"
			)
			if self.use_virtual_simulator:
				serialize_attr_with_type_info(archive, self
					,"virtual_simulator_value_when_no_missile"
				)
				serialize_internal_state_by_func(archive, full, "virtualSimulator", self.virtualSimulator.serializeInternalState)

			if self.use_image_observation:
				serialize_attr_with_type_info(archive, self
					,"image_longitudinal_resolution"
					,"image_lateral_resolution"
					,"image_front_range"
					,"image_back_range"
					,"image_side_range"
					,"image_horizon"
					,"image_interval"
					,"image_rotate"
					,"image_relative_position"
				)
			serialize_attr_with_type_info(archive, self
				,"use_remaining_time"
			)
			if self.use_remaining_time:
				serialize_attr_with_type_info(archive, self
					,"remaining_time_clipping"
				)

			serialize_attr_with_type_info(archive, self
				,"use_parent_last_action"
				,"use_friend_position"
				,"use_friend_velocity"
				,"use_friend_initial_num_missile"
				,"use_friend_current_num_missile"
				,"use_friend_attitude"
				,"use_friend_angular_velocity"
				,"use_friend_current_fuel"
				,"use_friend_rcs_scale"
				,"use_friend_radar_range"
				,"use_friend_radar_coverage"
				,"use_friend_maximum_speed_scale"
				,"use_friend_missile_thrust_scale"
				,"use_enemy_position"
				,"use_enemy_velocity"
				,"use_friend_missile_position"
				,"use_friend_missile_velocity"
				,"use_friend_missile_flight_time"
				,"use_friend_missile_target_distance"
				,"use_friend_missile_target_mode"
				,"use_friend_missile_target_position"
				,"use_friend_missile_target_velocity"
				,"use_enemy_missile_observer_position"
				,"use_enemy_missile_direction"
				,"use_enemy_missile_angular_velocity"
				,"use_our_missile_range"
				,"use_their_missile_range"
			)

			# action spaceの設定
			serialize_attr_with_type_info(archive, self
				,"dstAz_relative"
				,"turnTable"
				,"use_override_evasion"
				,"evasion_turnTable"
				,"no_vertical_maneuver"
				,"use_altitude_command"
			)
			if not self.no_vertical_maneuver:
				if self.use_altitude_command:
					serialize_attr_with_type_info(archive, self
						,"altTable"
						,"refAltInterval"
					)
				else:
					serialize_attr_with_type_info(archive, self
						,"pitchTable"
					)
			serialize_attr_with_type_info(archive, self
				,"accelTable"
				,"always_maxAB"
				,"use_Rmax_fire"
			)
			if self.use_Rmax_fire:
				serialize_attr_with_type_info(archive, self
					,"shotIntervalTable"
					,"shotThresholdTable"
				)
			# 行動制限に関する設定
			serialize_attr_with_type_info(archive, self
				,"altMin"
				,"altMax"
				,"altitudeKeeper"
				,"dOutLimit"
				,"dOutLimitThreshold"
				,"dOutLimitStrength"
				,"maxSimulShot"
				,"minimumV"
				,"minimumRecoveryV"
				,"minimumRecoveryDstV"
			)
			# 内部変数
			if self.use_image_observation:
				serialize_attr_with_type_info(archive, self
					,"numChannels"
					,"image_buffer_coords"
				)
			serialize_attr_with_type_info(archive, self
				,"common_dim"
				,"parent_dim"
				,"friend_dim"
				,"enemy_dim"
				,"friend_missile_dim"
				,"enemy_missile_dim"
				,"friend_enemy_relative_dim"
				,"last_action_dim"
				,"_observation_space"
				,"_action_space"
				,"dOut"
				,"dLine"
			)
			if isOutputArchive(archive):
				rulerObs=self.manager.getRuler()().observables()
				eastSider=rulerObs["eastSider"]
				save_with_type_info(archive, "isEastSider", self.getTeam()==eastSider)
			else:
				isEastSider=load_with_type_info(archive, "isEastSider")
				self.teamOrigin=TeamOrigin(isEastSider,self.dLine)
		if self.use_image_observation:
			serialize_attr_with_type_info(archive, self
				,"image_buffer"
				,"image_past_data"
			)
		serialize_attr_with_type_info(archive, self
			,"last_action_obs"
			,"actionInfos"
		)

	def validate(self):
		#Rulerに関する情報の取得
		rulerObs=self.manager.getRuler()().observables()
		self.dOut=rulerObs["dOut"]
		self.dLine=rulerObs["dLine"]
		eastSider=rulerObs["eastSider"]
		self.teamOrigin=TeamOrigin(self.getTeam()==eastSider,self.dLine)
		#機体制御方法の設定、前回行動の初期化
		for port,parent in self.parents.items():
			parent.setFlightControllerMode("fromDirAndVel")
			actionInfo=self.actionInfos[parent.getFullName()]
			actionInfo.dstDir=np.array([0., -1. if self.getTeam()==eastSider else +1., 0.])
			self.last_action_obs[parent.getFullName()][0]=atan2(actionInfo.dstDir[1],actionInfo.dstDir[0])
			actionInfo.lastShotTimes={}

		# 仮想シミュレータの使用
		if self.use_virtual_simulator:
			self.virtualSimulator.createVirtualSimulator(self)
	def observation_space(self):
		return self._observation_space
	def makeObs(self):
		#observablesの収集
		self.extractFriendObservables()
		self.extractEnemyObservables()
		self.extractFriendMissileObservables()
		self.extractEnemyMissileObservables()

		# 仮想シミュレータの使用
		if self.use_virtual_simulator:
			self.virtualSimulator.step(self,self.lastTrackInfo)

		#observationの生成
		ret={}
		observation_mask={}

		#共通情報(残り時間等)
		if self.common_dim>0:
			ret["common"]=self.makeCommonStateObservation()
		
		#画像情報
		if self.use_image_observation:
			self.makeImageObservation()
			ret["image"]=self.image_buffer

		#parentsの諸元
		if self.maxParentNum>0 and self.parent_dim>0:
			parent_temporal=np.zeros([self.maxParentNum,self.parent_dim],dtype=np.float32)
			parent_mask=np.zeros([self.maxParentNum],dtype=np.float32)
			fIdx=0
			for port,parent in self.parents.items():
				if fIdx>=self.maxParentNum:
					break

				fObs=self.ourObservables[fIdx]
				fMotion=self.ourMotion[fIdx]
				if fObs.at("isAlive"):
					parent_temporal[fIdx,:]=self.makeParentStateObservation(parent,fObs,fMotion)
					parent_mask[fIdx]=1
				fIdx+=1

			ret["parent"]=parent_temporal
			if self.use_observation_mask:
				observation_mask["parent"]=parent_mask
		
		#parents以外の味方機の諸元
		if self.maxFriendNum>0 and self.friend_dim>0:
			friend_temporal=np.zeros([self.maxFriendNum,self.friend_dim],dtype=np.float32)
			friend_mask=np.zeros([self.maxFriendNum],dtype=np.float32)
			for fIdx in range(len(self.ourMotion)-len(self.parents)):
				if fIdx>=self.maxFriendNum:
					break

				fObs=self.ourObservables[fIdx+len(self.parents)]
				fMotion=self.ourMotion[fIdx+len(self.parents)]
				if fObs.at("isAlive"):
					friend_temporal[fIdx,:]=self.makeFriendStateObservation(fObs,fMotion)
					friend_mask[fIdx]=1

			ret["friend"]=friend_temporal
			if self.use_observation_mask:
				observation_mask["friend"]=friend_mask

		#彼機(味方の誰かが探知しているもののみ)
		if self.maxEnemyNum>0 and self.enemy_dim>0:
			enemy_temporal=np.zeros([self.maxEnemyNum,self.enemy_dim],dtype=np.float32)
			enemy_mask=np.zeros([self.maxEnemyNum],dtype=np.float32)
			for tIdx,track in enumerate(self.lastTrackInfo):
				if tIdx>=self.maxEnemyNum:
					break

				enemy_temporal[tIdx,:]=self.makeEnemyStateObservation(track)
				enemy_mask[tIdx]=1

			ret["enemy"]=enemy_temporal
			if self.use_observation_mask:
				observation_mask["enemy"]=enemy_mask

		#味方誘導弾(射撃時刻が古いものから最大N発分)
		if self.maxFriendMissileNum>0 and self.friend_missile_dim>0:
			friend_missile_temporal=np.zeros([self.maxFriendMissileNum,self.friend_missile_dim],dtype=np.float32)
			friend_missile_mask=np.zeros([self.maxFriendMissileNum],dtype=np.float32)
			for mIdx,mObs in enumerate(self.msls):
				if mIdx>=self.maxFriendMissileNum or not (mObs.at("isAlive") and mObs.at("hasLaunched")):
					break

				friend_missile_temporal[mIdx,:]=self.makeFriendMissileStateObservation(mObs)
				friend_missile_mask[mIdx]=1

			ret["friend_missile"]=friend_missile_temporal
			if self.use_observation_mask:
				observation_mask["friend_missile"]=friend_missile_mask

		#彼側誘導弾(観測者の正面に近い順にソート)
		if self.maxEnemyMissileNum>0 and self.enemy_missile_dim>0:
			enemy_missile_temporal=np.zeros([self.maxEnemyMissileNum,self.enemy_missile_dim],dtype=np.float32)
			enemy_missile_mask=np.zeros([self.maxEnemyMissileNum],dtype=np.float32)
			allMWS=[]
			for fIdx,fMotion in enumerate(self.ourMotion):
				if self.ourObservables[fIdx].at("isAlive"):
					for m in self.mws[fIdx]:
						angle=np.arccos(np.clip(m.dir().dot(fMotion.dirBtoP(np.array([1,0,0]))),-1,1))
						allMWS.append([m,angle])

			allMWS.sort(key=lambda x: x[1])

			for mIdx,m in enumerate(allMWS):
				if mIdx>=self.maxEnemyMissileNum:
					break

				enemy_missile_temporal[mIdx,:]=self.makeEnemyMissileStateObservation(m[0])
				enemy_missile_mask[mIdx]=1

			ret["enemy_missile"]=enemy_missile_temporal
			if self.use_observation_mask:
				observation_mask["enemy_missile"]=enemy_missile_mask
	
		#我機と彼機の間の関係性
		if self.maxParentNum+self.maxFriendNum>0 and self.maxEnemyNum>0 and self.friend_enemy_relative_dim>0:
			friend_enemy_relative_temporal=np.zeros([self.maxParentNum+self.maxFriendNum,self.maxEnemyNum,self.friend_enemy_relative_dim],dtype=np.float32)
			friend_enemy_relative_mask=np.zeros([self.maxParentNum+self.maxFriendNum,self.maxEnemyNum],dtype=np.float32)
			firstAlive=None

			fIdx=0
			for port,parent in self.parents.items():
				if fIdx>=self.maxParentNum:
					break

				fObs=self.ourObservables[fIdx]
				fMotion=self.ourMotion[fIdx]
				if fObs.at("isAlive"):
					if firstAlive is None:
						firstAlive=parent

					for tIdx,track in enumerate(self.lastTrackInfo):
						if tIdx>=self.maxEnemyNum:
							break

						friend_enemy_relative_temporal[fIdx,tIdx,:]=self.makeFriendEnemyRelativeStateObservation(parent,fObs,fMotion,track)
						friend_enemy_relative_mask[fIdx,tIdx]=1
				fIdx+=1

			for fIdx in range(len(self.ourMotion)-len(self.parents)):
				if fIdx>=self.maxFriendNum:
					break

				fObs=self.ourObservables[fIdx+len(self.parents)]
				fMotion=self.ourMotion[fIdx+len(self.parents)]
				if fObs.at("isAlive"):
					for tIdx,track in enumerate(self.lastTrackInfo):
						if tIdx>=self.maxEnemyNum:
							break

						friend_enemy_relative_temporal[fIdx+len(self.parents),tIdx,:]=self.makeFriendEnemyRelativeStateObservation(firstAlive,fObs,fMotion,track)
						friend_enemy_relative_mask[fIdx+len(self.parents),tIdx]=1

			ret["friend_enemy_relative"]=friend_enemy_relative_temporal
			if self.use_observation_mask:
				observation_mask["friend_enemy_relative"]=friend_enemy_relative_mask

		if self.use_observation_mask and len(observation_mask)>0:
			ret["observation_mask"]=observation_mask

		if self.use_action_mask:
			self.action_mask=self.makeActionMask()
			if not self.action_mask is None:
				ret["action_mask"]=self.action_mask

		return ret

	def extractFriendObservables(self):
		#味方機(parents→parents以外の順)
		self.ourMotion=[]
		self.ourObservables=[]
		firstAlive=None
		for port,parent in self.parents.items():
			if parent.isAlive():
				firstAlive=parent
				break

		parentFullNames=set()
		# まずはparents
		for port, parent in self.parents.items():
			parentFullNames.add(parent.getFullName())
			if parent.isAlive():
				self.ourMotion.append(MotionState(parent.observables["motion"]).transformTo(self.getLocalCRS()))
				#残存していればobservablesそのもの
				self.ourObservables.append(parent.observables)
			else:
				self.ourMotion.append(MotionState())
				#被撃墜or墜落済なら本体の更新は止まっているので残存している親が代理更新したものを取得(誘導弾情報のため)
				self.ourObservables.append(
					firstAlive.observables.at_p("/shared/fighter").at(parent.getFullName()))

		# その後にparents以外
		for fullName,fObs in firstAlive.observables.at_p("/shared/fighter").items():
			if not fullName in parentFullNames:
				if fObs.at("isAlive"):
					self.ourMotion.append(MotionState(fObs["motion"]).transformTo(self.getLocalCRS()))
				else:
					self.ourMotion.append(MotionState())

				self.ourObservables.append(fObs)

	def extractEnemyObservables(self):
		#彼機(味方の誰かが探知しているもののみ)
		#観測されている航跡を、自陣営の機体に近いものから順にソートしてlastTrackInfoに格納する。
		#lastTrackInfoは行動のdeployでも射撃対象の指定のために参照する。
		firstAlive=None
		for port,parent in self.parents.items():
			if parent.isAlive():
				firstAlive=parent
				break

		self.lastTrackInfo=[Track3D(t).transformTo(self.getLocalCRS()) for t in firstAlive.observables.at_p("/sensor/track")]
		sortTrack3DByDistance(self.lastTrackInfo,self.ourMotion,True)

	def extractFriendMissileObservables(self):
		#味方誘導弾(射撃時刻の古い順にソート)
		def launchedT(m):
			return Time(m["launchedT"]) if m["isAlive"] and m["hasLaunched"] else Time(np.inf,TimeSystem.TT)
		self.msls=sorted(sum([[m for m in f.at_p("/weapon/missiles")] for f in self.ourObservables],[]),key=launchedT)

	def extractEnemyMissileObservables(self):
		#彼側誘導弾(各機の正面に近い順にソート)
		self.mws=[]
		for fIdx,fMotion in enumerate(self.ourMotion):
			fObs=self.ourObservables[fIdx]
			self.mws.append([])
			if fObs["isAlive"]:
				if fObs.contains_p("/sensor/mws/track"):
					for mObs in fObs.at_p("/sensor/mws/track"):
						self.mws[fIdx].append(Track2D(mObs).transformTo(self.getLocalCRS()))
				sortTrack2DByAngle(self.mws[fIdx],fMotion,np.array([1,0,0]),True)

	def makeCommonStateObservation(self):
		ret=np.zeros([self.common_dim],dtype=np.float32)
		ofs=0

		#残り時間
		if self.use_remaining_time:
			rulerObs=self.manager.getRuler()().observables
			maxTime=rulerObs["maxTime"]()
			ret[ofs]=min((maxTime-self.manager.getElapsedTime())/60.0,self.remaining_time_clipping)
			ofs+=1

		#

		return ret

	def makeImageObservation(self):
		self.image_buffer=np.zeros([self.numChannels,self.image_longitudinal_resolution,self.image_lateral_resolution],dtype=np.float32)
		#現在の情報を収集
		current=self.InstantInfo()
		#味方機(自機含む)の諸元
		for fIdx in range(len(self.parents)):
			if self.ourObservables[fIdx]["isAlive"]():
				current.parent_pos.append(self.ourMotion[fIdx].pos())
		for fIdx in range(len(self.parents),len(self.ourMotion)):
			if self.ourObservables[fIdx]["isAlive"]():
				current.friend_pos.append(self.ourMotion[fIdx].pos())
		#彼機(味方の誰かが探知しているもののみ)
		for tIdx,t in enumerate(self.lastTrackInfo):
			current.enemy_pos.append(t.pos())
		#味方誘導弾(射撃時刻が古いものから最大N発分)
		for mIdx,m in enumerate(self.msls):
			if not (m["isAlive"]() and m["hasLaunched"]()):
				break
			mm=MotionState(m["motion"]).transformTo(self.getLocalCRS())
			current.friend_msl_pos.append(mm.pos())

		#画像の生成
		#ch0:parentsの軌跡
		#ch1:parents以外の味方軌跡
		#ch2:彼機軌跡
		#ch3:味方誘導弾軌跡
		#ch4:parentsの覆域
		#ch5:parents以外の味方覆域
		#ch6:彼我防衛ライン
		#ch7:場外ライン
		stepCount=self.getStepCount()
		bufferIdx=self.image_horizon-1-(stepCount%self.image_horizon)
		self.image_past_data[bufferIdx]=current
		tMax=min(
			floor(stepCount/self.image_interval)*self.image_interval,
			floor((self.image_horizon-1)/self.image_interval)*self.image_interval
		)
		ch=0

		for t in range(tMax,-1,-self.image_interval):
			frame=self.image_past_data[(bufferIdx+t)%self.image_horizon]
			ch=0

			if self.draw_parent_trajectory:
				#ch0:parentsの軌跡
				for i in range(len(frame.parent_pos)):
					self.plotPoint(
						frame.parent_pos[i],
						ch,
						1.0-1.0*t/self.image_horizon
					)
				ch+=1

			if self.draw_friend_trajectory:
				#ch1:parents以外の味方軌跡
				for i in range(len(frame.friend_pos)):
					self.plotPoint(
						frame.friend_pos[i],
						ch,
						1.0-1.0*t/self.image_horizon
					)
				ch+=1

			if self.draw_enemy_trajectory:
				#ch2:彼機軌跡
				for i in range(len(frame.enemy_pos)):
					self.plotPoint(
						frame.enemy_pos[i],
						ch,
						1.0-1.0*t/self.image_horizon
					)
				ch+=1

			if self.draw_friend_missile_trajectory:
				#ch3:味方誘導弾軌跡
				for i in range(1,len(frame.friend_msl_pos)):
					self.plotPoint(
						frame.friend_msl_pos[i],
						ch,
						1.0-1.0*t/self.image_horizon
					)
				ch+=1

		#ch4:parentsのレーダ覆域, ch5:parents以外の味方レーダ覆域
		if self.draw_parent_radar_coverage or self.draw_friend_radar_coverage:
			for i in range(len(self.ourMotion)):
				if not self.draw_parent_radar_coverage and i<len(self.parents):
					continue
				if self.draw_parent_radar_coverage and i==len(self.parents):
					ch+=1
				if not self.draw_friend_radar_coverage and i>=len(self.parents):
					continue
				if self.ourObservables[i]["isAlive"]():
					if self.ourObservables[i].contains_p("/spec/sensor/radar"):
						ex=self.ourMotion[i].relBtoP(np.array([1,0,0]))
						Lref=self.ourObservables[i].at_p("/spec/sensor/radar/Lref")()
						thetaFOR=self.ourObservables[i].at_p("/spec/sensor/radar/thetaFOR")()
						for lon in range(self.image_longitudinal_resolution):
							for lat in range(self.image_lateral_resolution):
								pos=self.gridToPos(lon,lat,10000.0)
								rPos=pos-self.ourMotion[i].pos()
								L=np.linalg.norm(rPos)
								if L<=Lref and np.dot(rPos,ex)>=L*cos(thetaFOR):
									self.plotPoint(
										pos,
										ch,
										1.0
									)
			if self.draw_friend_radar_coverage:
				ch+=1

		if self.draw_defense_line:
			#ch6:彼我防衛ライン
			#彼側防衛ライン
			eps=1e-10 #戦域と同一の描画範囲とした際に境界上になって描画されなくなることを避けるため、ごくわずかに内側にずらすとよい。
			self.plotLine(
				self.teamOrigin.relBtoP(np.array([self.dLine-eps,-self.dOut+eps,0])),
				self.teamOrigin.relBtoP(np.array([self.dLine-eps,self.dOut-eps,0])),
				ch,
				1.0
			)
			#我側防衛ライン
			self.plotLine(
				self.teamOrigin.relBtoP(np.array([-self.dLine+eps,-self.dOut+eps,0])),
				self.teamOrigin.relBtoP(np.array([-self.dLine+eps,self.dOut-eps,0])),
				ch,
				-1.0
			)
			ch+=1

		if self.draw_side_line:
			#ch7:場外ライン
			#進行方向右側
			eps=1e-10 #戦域と同一の描画範囲とした際に境界上になって描画されなくなることを避けるため、ごくわずかに内側にずらすとよい。
			self.plotLine(
				self.teamOrigin.relBtoP(np.array([-self.dLine+eps,self.dOut-eps,0])),
				self.teamOrigin.relBtoP(np.array([self.dLine-eps,self.dOut-eps,0])),
				ch,
				1.0
			)
			#進行方向左側
			self.plotLine(
				self.teamOrigin.relBtoP(np.array([-self.dLine+eps,-self.dOut+eps,0])),
				self.teamOrigin.relBtoP(np.array([self.dLine-eps,-self.dOut+eps,0])),
				ch,
				1.0
			)
			ch+=1

	def makeParentStateObservation(self, parent: PhysicalAssetAccessor, fObs: nljson, fMotion: MotionState):
		ret=np.zeros([self.parent_dim],dtype=np.float32)
		ofs=0

		#前回の行動
		#前回のdeployで計算済だが、dstAzのみ現在のazからの差に置き換える
		if self.use_parent_last_action:
			if parent.isAlive():
				parentFullName=parent.getFullName()
				myMotion=MotionState(parent.observables["motion"]).transformTo(self.getLocalCRS())
				deltaAz=self.last_action_obs[parentFullName][0]-myMotion.getAZ()
				self.last_action_obs[parentFullName][0]=atan2(sin(deltaAz),cos(deltaAz))
				ret[ofs:ofs+self.last_action_dim]=self.last_action_obs[parentFullName]
				ofs+=self.last_action_dim

		# 仮想シミュレータの使用
		# この例では各parentに最も近い仮想誘導弾との距離をparentのobservationに付加する
		if self.use_virtual_simulator:
			d=self.virtual_simulator_value_when_no_missile
			if parent.getFullName() in self.virtualSimulator.minimumDistances:
				d=min(d,self.virtualSimulator.minimumDistances[parent.getFullName()])
			ret[ofs]=d/self.horizontalNormalizer
			ofs+=1

		#味方機の共通諸元も使用
		ret[ofs:]=self.makeFriendStateObservation(fObs,fMotion)
		return ret

	def makeFriendStateObservation(self, fObs: nljson, fMotion: MotionState):
		#味方機の諸元
		ret=np.zeros([self.friend_dim],dtype=np.float32)
		ofs=0

		#位置
		if self.use_friend_position:
			pos=self.teamOrigin.relPtoB(fMotion.pos()) #慣性座標系→陣営座標系に変換
			ret[ofs:ofs+3]=pos/np.array([self.horizontalNormalizer,self.horizontalNormalizer,self.verticalNormalizer])
			ofs+=3

		#速度
		if self.use_friend_velocity:
			vel=self.teamOrigin.relPtoB(fMotion.vel()) #慣性座標系→陣営座標系に変換
			V=np.linalg.norm(vel)
			ret[ofs]=V/self.fgtrVelNormalizer
			ofs+=1
			ret[ofs:ofs+3]=vel/V
			ofs+=3

		#初期弾数
		if self.use_friend_initial_num_missile:
			ret[ofs]=fObs.at_p("/spec/weapon/numMsls")()
			ofs+=1

		#残弾数
		if self.use_friend_current_num_missile:
			ret[ofs]=fObs.at_p("/weapon/remMsls")()
			ofs+=1

		#姿勢(バンク、α、β)
		if self.use_friend_attitude:
			vb=fMotion.relPtoB(fMotion.vel())
			V=np.linalg.norm(vb)
			ex=fMotion.dirBtoP(np.array([1.,0.,0.]))
			ey=fMotion.dirBtoP(np.array([0.,1.,0.]))
			down=fMotion.dirHtoP(np.array([0,0,1]),np.array([0.,0.,0.]),"NED",False)
			horizontalY=np.cross(down,ex)
			roll=0
			N=np.linalg.norm(horizontalY)
			if N>0:
				horizontalY/=N
				sinRoll=np.dot(np.cross(horizontalY,ey),ex)
				cosRoll=np.dot(horizontalY,ey)
				roll=atan2(sinRoll,cosRoll)
			alpha=atan2(vb[2],vb[0])
			beta=asin(vb[1]/V)
			ret[ofs]=roll
			ret[ofs+1]=alpha
			ret[ofs+2]=beta
			ofs+=3

		#角速度
		if self.use_friend_angular_velocity:
			omegaB=fMotion.relPtoB(fMotion.omega())#回転だけを行いたいのでrelで変換
			ret[ofs:ofs+3]=omegaB
			ofs+=3

		#余剰燃料(R7年度コンテストでは不要)
		if self.use_friend_current_fuel:
			ret[ofs]=calcDistanceMargin(self.getTeam(),fMotion,fObs,self.manager.getRuler()().observables)
			ofs+=1

		#R7年度コンテストオープン部門向け、機体性能の変動情報
		#RCSスケール
		if self.use_friend_rcs_scale:
			if fObs.contains_p("/spec/stealth/rcsScale"):
				ret[ofs]=fObs.at_p("/spec/stealth/rcsScale")()
			else:
				ret[ofs]=1.0
			ofs+=1

		#レーダ探知距離
		if self.use_friend_radar_range:
			if fObs.contains_p("/spec/sensor/radar/Lref"):
				ret[ofs]=fObs.at_p("/spec/sensor/radar/Lref")()/self.horizontalNormalizer
			else:
				ret[ofs]=0.0
			ofs+=1

		#レーダ覆域(角度、ラジアン)
		if self.use_friend_radar_coverage:
			if fObs.contains_p("/spec/sensor/radar/thetaFOR"):
				ret[ofs]=fObs.at_p("/spec/sensor/radar/thetaFOR")()
			else:
				ret[ofs]=0.0
			ofs+=1

		#最大速度倍率
		if self.use_friend_maximum_speed_scale:
			if fObs.contains_p("/spec/dynamics/scale/vMax"):
				ret[ofs]=fObs.at_p("/spec/dynamics/scale/vMax")()
			else:
				ret[ofs]=1.0
			ofs+=1

		#誘導弾推力倍率
		if self.use_friend_missile_thrust_scale:
			if fObs.contains_p("/spec/weapon/missile/thrust"):
				ret[ofs]=fObs.at_p("/spec/weapon/missile/thrust")()/30017.9989
			else:
				ret[ofs]=1.0
			ofs+=1

		#射撃遅延時間(秒)は直接観測不可(observablesからもparentからも見えない)

		return ret
	def makeEnemyStateObservation(self, track: Track3D):
		#彼機の諸元
		ret=np.zeros([self.enemy_dim],dtype=np.float32)
		ofs=0

		#位置
		if self.use_enemy_position:
			pos=self.teamOrigin.relPtoB(track.pos()) #慣性座標系→陣営座標系に変換
			ret[ofs:ofs+3]=pos/np.array([self.horizontalNormalizer,self.horizontalNormalizer,self.verticalNormalizer])
			ofs+=3

		#速度
		if self.use_enemy_velocity:
			vel=self.teamOrigin.relPtoB(track.vel()) #慣性座標系→陣営座標系に変換
			V=np.linalg.norm(vel)
			ret[ofs]=V/self.fgtrVelNormalizer
			ofs+=1
			ret[ofs:ofs+3]=vel/V
			ofs+=3

		return ret

	def makeFriendMissileStateObservation(self, mObs: nljson):
		#味方誘導弾の諸元
		ret=np.zeros([self.friend_missile_dim],dtype=np.float32)
		ofs=0

		mm=MotionState(mObs["motion"]).transformTo(self.getLocalCRS())
		#位置
		if self.use_friend_missile_position:
			pos=self.teamOrigin.relPtoB(mm.pos()) #慣性座標系→陣営座標系に変換
			ret[ofs:ofs+3]=pos/np.array([self.horizontalNormalizer,self.horizontalNormalizer,self.verticalNormalizer])
			ofs+=3

		#速度
		if self.use_friend_missile_velocity:
			vel=self.teamOrigin.relPtoB(mm.vel()) #慣性座標系→陣営座標系に変換
			V=np.linalg.norm(vel)
			ret[ofs]=V/self.mslVelNormalizer
			ofs+=1
			ret[ofs:ofs+3]=vel/V
			ofs+=3

		#飛翔時間(分)
		if self.use_friend_missile_flight_time:
			launchedT=Time(mObs["launchedT"])
			ret[ofs]=(self.manager.getTime()-launchedT)/60.0
			ofs+=1

		#目標情報(距離、誘導状態、推定目標位置、推定目標速度)
		mTgt=Track3D(mObs["target"]).transformTo(self.getLocalCRS()).extrapolateTo(self.manager.getTime())

		#目標との距離
		if self.use_friend_missile_target_distance:
			ret[ofs]=1.0-min(1.0,np.linalg.norm(mTgt.pos()-mm.pos())/self.horizontalNormalizer)
			ofs+=1

		#目標への誘導状態
		if self.use_friend_missile_target_mode:
			mode=mObs["mode"]()
			if mode==Missile.Mode.GUIDED.name:
				ret[ofs:ofs+3]=np.array([1,0,0])
			elif mode==Missile.Mode.SELF.name:
				ret[ofs:ofs+3]=np.array([0,1,0])
			else:#if mode==Missile.Mode.MEMORY.name:
				ret[ofs:ofs+3]=np.array([0,0,1])
			ofs+=3

		#目標の位置
		if self.use_friend_missile_target_position:
			pos=self.teamOrigin.relPtoB(mTgt.pos()) #慣性座標系→陣営座標系に変換
			ret[ofs:ofs+3]=pos/np.array([self.horizontalNormalizer,self.horizontalNormalizer,self.verticalNormalizer])
			ofs+=3

		#目標の速度
		if self.use_friend_missile_target_velocity:
			vel=self.teamOrigin.relPtoB(mTgt.vel()) #慣性座標系→陣営座標系に変換
			V=np.linalg.norm(vel)
			ret[ofs]=V/self.mslVelNormalizer
			ofs+=1
			ret[ofs:ofs+3]=vel/V
			ofs+=3

		#R7年度コンテストオープン部門向け、機体性能の変動情報
		#誘導弾推力倍率
		if self.use_friend_missile_thrust_scale:
			if mObs.contains_p("/spec/thrust"):
				ret[ofs]=mObs.at_p("/spec/thrust")()/30017.9989
			else:
				ret[ofs]=1.0
			ofs+=1

		return ret

	def makeEnemyMissileStateObservation(self, track: Track2D):
		#彼側誘導弾の諸元
		ret=np.zeros([self.enemy_missile_dim],dtype=np.float32)
		ofs=0
	
		#観測点
		if self.use_enemy_missile_observer_position:
			origin=self.teamOrigin.relPtoB(track.origin()) #慣性座標系→陣営座標系に変換
			ret[ofs:ofs+3]=origin/np.array([self.horizontalNormalizer,self.horizontalNormalizer,self.verticalNormalizer])
			ofs+=3

		#方向
		if self.use_enemy_missile_direction:
			dir=self.teamOrigin.relPtoB(track.dir()) #慣性座標系→陣営座標系に変換 #teamOriginはrelとdirを区別しない
			ret[ofs:ofs+3]=dir
			ofs+=3

		#方向変化率
		if self.use_enemy_missile_angular_velocity:
			omega=self.teamOrigin.relPtoB(track.omega()) #慣性座標系→陣営座標系に変換。/#回転だけを行いたいのでrelで変換
			ret[ofs:ofs+3]=omega
			ofs+=3

		return ret
	def makeFriendEnemyRelativeStateObservation(self, parent: PhysicalAssetAccessor, fObs: nljson, fMotion: MotionState, track: Track3D):
		#我機と彼機の間の関係性に関する諸元
		ret=np.zeros([self.friend_enemy_relative_dim],dtype=np.float32)
		ofs=0
		if self.use_our_missile_range:
			ret[ofs]=calcRHead(parent,fMotion,track,False)/self.horizontalNormalizer#RHead
			ret[ofs+1]=calcRTail(parent,fMotion,track,False)/self.horizontalNormalizer#RTail
			ret[ofs+2]=calcRNorm(parent,fMotion,track,False)#現在の距離をRTail〜RHeadで正規化したもの
			ofs+=3

		if self.use_their_missile_range:
			ret[ofs]=calcRHeadE(parent,fMotion,track,False)/self.horizontalNormalizer#RHead
			ret[ofs+1]=calcRTailE(parent,fMotion,track,False)/self.horizontalNormalizer#RTail
			ret[ofs+2]=calcRNormE(parent,fMotion,track,False)#現在の距離をRTail〜RHeadで正規化したもの
			ofs+=3

		return ret

	def makeActionMask(self):
		#無効な行動を示すマスクを返す
		#有効な場合は1、無効な場合は0とする。
		if self.use_action_mask:
			#このサンプルでは射撃目標のみマスクする。
			target_mask=np.zeros([1+self.maxEnemyNum],dtype=np.float32)
			target_mask[0]=1#「射撃なし」はつねに有効
			for tIdx,track in enumerate(self.lastTrackInfo):
				if tIdx>=self.maxEnemyNum:
					break
				target_mask[1+tIdx]=1

			ret=[]
			for port,parent in self.parents.items():
				mask={}
				mask["turn"]=np.full([len(self.turnTable)],1,dtype=np.float32)
				if not self.no_vertical_maneuver:
					mask["pitch"]=np.full([len(self.altTable) if self.use_altitude_command else len(self.pitchTable)],1,dtype=np.float32)
				if not self.always_maxAB:
					mask["accel"]=np.full([len(self.accelTable)],1,dtype=np.float32)
				mask["target"]=target_mask
				if self.use_Rmax_fire:
					if len(self.shotIntervalTable)>1:
						mask["shotInterval"]=np.full([len(self.shotIntervalTable)],1,dtype=np.float32)
					if len(self.shotThresholdTable)>1:
						mask["shotThreshold"]=np.full([len(self.shotThresholdTable)],1,dtype=np.float32)
				ret.append(mask)
			return ret
		else:
			return None

	def action_space(self):
		return self._action_space

	def deploy(self,action):
		#observablesの収集
		self.extractFriendObservables()
		#self.extractEnemyObservables() # 彼機情報だけは射撃対象の選択と連動するので更新してはいけない。
		self.extractFriendMissileObservables()
		self.extractEnemyMissileObservables()

		#actionのパース
		if self.use_action_mask and self.raise_when_invalid_action:
			#有効な行動かどうかを確かめる(形は合っていることを前提とする)
			for pIdx,parent in enumerate(self.parents.values()):
				for key,mask in self.action_mask[pIdx].items():
					if mask[action[pIdx][key]]!=1:
						raise ValueError(
							self.getFullName()+"'s action '"+key+"' at "+str(action[pIdx][key])
							+" for "+parent.getFullName()+" is masked but sampled somehow. Check your sampler method."
						)

		self.last_action_obs={}
		for pIdx,parent in enumerate(self.parents.values()):
			parentFullName=parent.getFullName()
			self.last_action_obs[parentFullName]=np.zeros([self.last_action_dim],dtype=np.float32)
			if not parent.isAlive():
				continue
			actionInfo=self.actionInfos[parentFullName]
			myMotion=self.ourMotion[pIdx]
			myObs=self.ourObservables[pIdx]
			myMWS=self.mws[pIdx]
			myAction=action[pIdx]

			#左右旋回
			deltaAz=self.turnTable[myAction["turn"]]
			if len(myMWS)>0 and self.use_override_evasion:
				deltaAz=self.evasion_turnTable[myAction["turn"]]
				dr=np.zeros([3])
				for m in myMWS:
					dr+=m.dir()
				dr/=np.linalg.norm(dr)
				dstAz=atan2(-dr[1],-dr[0])+deltaAz
				actionInfo.dstDir=np.array([cos(dstAz),sin(dstAz),0])
			elif self.dstAz_relative:
				actionInfo.dstDir=myMotion.dirHtoP(np.array([cos(deltaAz),sin(deltaAz),0]),np.array([0.,0.,0.]),"FSD",False)
			else:
				actionInfo.dstDir=self.teamOrigin.relBtoP(np.array([cos(deltaAz),sin(deltaAz),0]))
			dstAz=atan2(actionInfo.dstDir[1],actionInfo.dstDir[0])
			self.last_action_obs[parentFullName][0]=dstAz

			#上昇・下降
			if not self.no_vertical_maneuver:
				if self.use_altitude_command:
					refAlt=round(myMotion.getHeight()/self.refAltInterval)*self.refAltInterval
					actionInfo.dstAlt=max(self.altMin,min(self.altMax,refAlt+self.altTable[myAction["pitch"]]))
					dstPitch=0#dstAltをcommandsに与えればSixDoFFlightControllerのaltitudeKeeperで別途計算されるので0でよい。
				else:
					dstPitch=self.pitchTable[myAction["pitch"]]
			else:
				dstPitch=0
			actionInfo.dstDir=np.array([actionInfo.dstDir[0]*cos(dstPitch),actionInfo.dstDir[1]*cos(dstPitch),-sin(dstPitch)])
			self.last_action_obs[parentFullName][1]=actionInfo.dstAlt if self.use_altitude_command else dstPitch

			#加減速
			V=np.linalg.norm(myMotion.vel())
			if self.always_maxAB:
				actionInfo.asThrottle=True
				actionInfo.keepVel=False
				actionInfo.dstThrottle=1.0
				self.last_action_obs[parentFullName][2]=1.0
			else:
				actionInfo.asThrottle=False
				accel=self.accelTable[myAction["accel"]]
				actionInfo.dstV=V+accel
				actionInfo.keepVel = accel==0.0
				self.last_action_obs[parentFullName][2]=accel/max(self.accelTable[-1],self.accelTable[0])
			#下限速度の制限
			if V<self.minimumV:
				actionInfo.velRecovery=True
			if V>=self.minimumRecoveryV:
				actionInfo.velRecovery=False
			if actionInfo.velRecovery:
				actionInfo.dstV=self.minimumRecoveryDstV
				actionInfo.asThrottle=False

			#射撃
			#actionのパース
			shotTarget=myAction["target"]-1
			if self.use_Rmax_fire:
				if len(self.shotIntervalTable)>1:
					shotInterval=self.shotIntervalTable[myAction["shotInterval"]]
				else:
					shotInterval=self.shotIntervalTable[0]
				if len(self.shotThresholdTable)>1:
					shotThreshold=self.shotThresholdTable[myAction["shotThreshold"]]
				else:
					shotThreshold=self.shotThresholdTable[0]
			#射撃可否の判断、射撃コマンドの生成
			flyingMsls=0
			if myObs.contains_p("/weapon/missiles"):
				for msl in myObs.at_p("/weapon/missiles"):
					if msl.at("isAlive")() and msl.at("hasLaunched")():
						flyingMsls+=1
			if (
				shotTarget>=0 and
				shotTarget<len(self.lastTrackInfo) and
				parent.isLaunchableAt(self.lastTrackInfo[shotTarget]) and
				flyingMsls<self.maxSimulShot
			):
				if self.use_Rmax_fire:
					rMin=np.inf
					t=self.lastTrackInfo[shotTarget]
					r=calcRNorm(parent,myMotion,t,False)
					if r<=shotThreshold:
						#射程の条件を満たしている
						if not t.truth in actionInfo.lastShotTimes:
							actionInfo.lastShotTimes[t.truth]=self.manager.getEpoch()
						if self.manager.getTime()-actionInfo.lastShotTimes[t.truth]>=shotInterval:
							#射撃間隔の条件を満たしている
							actionInfo.lastShotTimes[t.truth]=self.manager.getTime()
						else:
							#射撃間隔の条件を満たさない
							shotTarget=-1
					else:
						#射程の条件を満たさない
						shotTarget=-1
			else:
				shotTarget=-1
			self.last_action_obs[parentFullName][3+(shotTarget+1)]=1
			if shotTarget>=0:
				actionInfo.launchFlag=True
				actionInfo.target=self.lastTrackInfo[shotTarget]
			else:
				actionInfo.launchFlag=False
				actionInfo.target=Track3D()

			self.observables[parentFullName]["decision"]={
				"Roll":("Don't care"),
				"Fire":(actionInfo.launchFlag,actionInfo.target.to_json())
			}
			if len(myMWS)>0 and self.use_override_evasion:
				self.observables[parentFullName]["decision"]["Horizontal"]=("Az_NED",dstAz)
			else:
				if self.dstAz_relative:
					self.observables[parentFullName]["decision"]["Horizontal"]=("Az_BODY",deltaAz)
				else:
					self.observables[parentFullName]["decision"]["Horizontal"]=("Az_NED",dstAz)
			if self.use_altitude_command:
				self.observables[parentFullName]["decision"]["Vertical"]=("Pos",-actionInfo.dstAlt)
			else:
				self.observables[parentFullName]["decision"]["Vertical"]=("El",dstPitch)
			if actionInfo.asThrottle:
				self.observables[parentFullName]["decision"]["Throttle"]=("Throttle",actionInfo.dstThrottle)
			else:
				self.observables[parentFullName]["decision"]["Throttle"]=("Vel",actionInfo.dstV)

	def control(self):
		#observablesの収集
		self.extractFriendObservables()
		#self.extractEnemyObservables() # 彼機情報だけは射撃対象の選択と連動するので更新してはいけない。
		self.extractFriendMissileObservables()
		self.extractEnemyMissileObservables()

		#Setup collision avoider
		avoider=StaticCollisionAvoider2D()
		#北側
		c={
			"p1":np.array([+self.dOut,-5*self.dLine,0]),
			"p2":np.array([+self.dOut,+5*self.dLine,0]),
			"infinite_p1":True,
			"infinite_p2":True,
			"isOneSide":True,
			"inner":np.array([0.0,0.0]),
			"limit":self.dOutLimit,
			"threshold":self.dOutLimitThreshold,
			"adjustStrength":self.dOutLimitStrength,
		}
		avoider.borders.append(LinearSegment(c))
		#南側
		c={
			"p1":np.array([-self.dOut,-5*self.dLine,0]),
			"p2":np.array([-self.dOut,+5*self.dLine,0]),
			"infinite_p1":True,
			"infinite_p2":True,
			"isOneSide":True,
			"inner":np.array([0.0,0.0]),
			"limit":self.dOutLimit,
			"threshold":self.dOutLimitThreshold,
			"adjustStrength":self.dOutLimitStrength,
		}
		avoider.borders.append(LinearSegment(c))
		#東側
		c={
			"p1":np.array([-5*self.dOut,+self.dLine,0]),
			"p2":np.array([+5*self.dOut,+self.dLine,0]),
			"infinite_p1":True,
			"infinite_p2":True,
			"isOneSide":True,
			"inner":np.array([0.0,0.0]),
			"limit":self.dOutLimit,
			"threshold":self.dOutLimitThreshold,
			"adjustStrength":self.dOutLimitStrength,
		}
		avoider.borders.append(LinearSegment(c))
		#西側
		c={
			"p1":np.array([-5*self.dOut,-self.dLine,0]),
			"p2":np.array([+5*self.dOut,-self.dLine,0]),
			"infinite_p1":True,
			"infinite_p2":True,
			"isOneSide":True,
			"inner":np.array([0.0,0.0]),
			"limit":self.dOutLimit,
			"threshold":self.dOutLimitThreshold,
			"adjustStrength":self.dOutLimitStrength,
		}
		avoider.borders.append(LinearSegment(c))
		for pIdx,parent in enumerate(self.parents.values()):
			parentFullName=parent.getFullName()
			if not parent.isAlive():
				continue
			actionInfo=self.actionInfos[parentFullName]
			myMotion=self.ourMotion[pIdx]
			myObs=self.ourObservables[pIdx]
			originalMyMotion=MotionState(myObs["motion"]) #機体側にコマンドを送る際には元のparent座標系での値が必要

			#戦域逸脱を避けるための方位補正
			actionInfo.dstDir=avoider(myMotion,actionInfo.dstDir)

			#高度方向の補正(actionがピッチ指定の場合)
			if not self.use_altitude_command:
				n=sqrt(actionInfo.dstDir[0]*actionInfo.dstDir[0]+actionInfo.dstDir[1]*actionInfo.dstDir[1])
				dstPitch=atan2(-actionInfo.dstDir[2],n)
				#高度下限側
				bottom=self.altitudeKeeper(myMotion,actionInfo.dstDir,self.altMin)
				minPitch=atan2(-bottom[2],sqrt(bottom[0]*bottom[0]+bottom[1]*bottom[1]))
				#高度上限側
				top=self.altitudeKeeper(myMotion,actionInfo.dstDir,self.altMax)
				maxPitch=atan2(-top[2],sqrt(top[0]*top[0]+top[1]*top[1]))
				dstPitch=max(minPitch,min(maxPitch,dstPitch))
				cs=cos(dstPitch)
				sn=sin(dstPitch)
				actionInfo.dstDir=np.array([actionInfo.dstDir[0]/n*cs,actionInfo.dstDir[1]/n*cs,-sn])
			self.commands[parentFullName]={
				"motion":{
					"dstDir":originalMyMotion.dirAtoP(actionInfo.dstDir,myMotion.pos(),self.getLocalCRS()) #元のparent座標系に戻す
				},
				"weapon":{
					"launch":actionInfo.launchFlag,
					"target":actionInfo.target.to_json()
				}
			}
			if self.use_altitude_command:
				self.commands[parentFullName]["motion"]["dstAlt"]=actionInfo.dstAlt
			if actionInfo.asThrottle:
				self.commands[parentFullName]["motion"]["dstThrottle"]=actionInfo.dstThrottle
			elif actionInfo.keepVel:
				self.commands[parentFullName]["motion"]["dstAccel"]=0.0
			else:
				self.commands[parentFullName]["motion"]["dstV"]=actionInfo.dstV
			actionInfo.launchFlag=False

	def convertActionFromAnother(self,decision,command):#模倣対象の行動または制御出力と近い行動を計算する
		#observablesの収集
		self.extractFriendObservables()
		#self.extractEnemyObservables() # 彼機情報だけは射撃対象の選択と連動するので更新してはいけない。
		self.extractFriendMissileObservables()
		self.extractEnemyMissileObservables()
		#interval=self.getStepInterval()*self.manager.getBaseTimeStep()
		ret=[]
		for pIdx,parent in enumerate(self.parents.values()):
			myAction={}
			parentFullName=parent.getFullName()
			if not parent.isAlive():
				#左右旋回
				tmp_d=self.turnTable-0.0
				tmp_i=np.argmin(abs(tmp_d))
				myAction["turn"]=tmp_i
				#上昇・下降
				if not self.no_vertical_maneuver:
					if self.use_altitude_command:
						tmp_d=self.altTable-0.0
					else:
						tmp_d=self.pitchTable-0.0
					tmp_i=np.argmin(abs(tmp_d))
					myAction["pitch"]=tmp_i
				#加減速
				if not self.always_maxAB:
					tmp_d=self.accelTable-0.0
					tmp_i=np.argmin(abs(tmp_d))
					myAction["accel"]=tmp_i
				#射撃
				myAction["target"]=0
				if self.use_Rmax_fire:
					if len(self.shotIntervalTable)>1:
						myAction["shotInterval"]=0
					if len(self.shotThresholdTable)>1:
						myAction["shotThreshold"]=len(self.shotThresholdTable)-1
				ret.append(myAction)
				continue
			actionInfo=self.actionInfos[parentFullName]
			myMotion=self.ourMotion[pIdx]
			#左右旋回
			decisionType=decision[parentFullName]["Horizontal"][0]()
			value=decision[parentFullName]["Horizontal"][1]()
			motionCmd=command[parentFullName]["motion"]()
			dstAz_ex=0.0
			if decisionType=="Az_NED":
				dstAz_ex=value
			elif decisionType=="Az_BODY":
				dstAz_ex=value+myMotion.getAZ()
			elif "dstDir" in motionCmd:
				dd=motionCmd["dstDir"]
				dstAz_ex=atan2(dd[1],dd[0])
			else:
				raise ValueError("Given unsupported expert's decision & command for this Agent class.")
			deltaAz_ex=0.0
			myMWS=self.mws[pIdx]
			if len(myMWS)>0 and self.use_override_evasion:
				dr=np.zeros([3])
				for m in myMWS:
					dr+=m.dir()
				dr/=np.linalg.norm(dr)
				deltaAz_ex=dstAz_ex-atan2(-dr[1],-dr[0])
				deltaAz_ex=atan2(sin(deltaAz_ex),cos(deltaAz_ex))
				tmp_d=self.evasion_turnTable-deltaAz_ex
			else:
				if self.dstAz_relative:
					dstDir_ex = myMotion.dirPtoH(np.array([cos(dstAz_ex),sin(dstAz_ex),0]),myMotion.pos(),"FSD",False)
				else:
					dstDir_ex = self.teamOrigin.relPtoB(np.array([cos(dstAz_ex),sin(dstAz_ex),0]))
				deltaAz_ex = atan2(dstDir_ex[1],dstDir_ex[0])
				tmp_d=self.turnTable-deltaAz_ex
			tmp_i=np.argmin(abs(tmp_d))
			myAction["turn"]=tmp_i

			#上昇・下降
			if not self.no_vertical_maneuver:
				decisionType=decision[parentFullName]["Vertical"][0]()
				value=decision[parentFullName]["Vertical"][1]()
				dstPitch_ex=0.0
				deltaAlt_ex=0.0
				refAlt=round(myMotion.getHeight()/self.refAltInterval)*self.refAltInterval
				if decisionType=="El":
					dstPitch_ex=-value
					deltaAlt_ex=self.altitudeKeeper.inverse(myMotion,dstPitch_ex)-refAlt
				elif decisionType=="Pos":
					dstPitch_ex=self.altitudeKeeper.getDstPitch(myMotion,-value)
					deltaAlt_ex=-value-refAlt
				elif "dstDir" in motionCmd:
					dd=motionCmd["dstDir"]
					dstPitch_ex=-atan2(dd[2],sqrt(dd[0]*dd[0]+dd[1]*dd[1]))
					deltaAlt_ex=self.altitudeKeeper.inverse(myMotion,dstPitch_ex)-refAlt
				else:
					raise ValueError("Given unsupported expert's decision & command for this Agent class.")
				dstPitch_ex=atan2(sin(dstPitch_ex),cos(dstPitch_ex))
				if self.use_altitude_command:
					tmp_d=self.altTable-deltaAlt_ex
				else:
					tmp_d=self.pitchTable-dstPitch_ex
				tmp_i=np.argmin(abs(tmp_d))
				myAction["pitch"]=tmp_i

			#加減速
			if not self.always_maxAB:
				#本サンプルでは「現在の速度と目標速度の差」をactionとしてdstVを機体へのコマンドとしているため、
				#教師役の行動が目標速度以外(スロットルや加速度)の場合、正確な変換は困難である。
				#そのため、最も簡易的な変換の例として、最大減速、等速、最大加速の3値への置換を実装している。
				decisionType=decision[parentFullName]["Throttle"][0]()
				value=decision[parentFullName]["Throttle"][1]()
				V=np.linalg.norm(myMotion.vel())
				accelIdx=-1
				if "dstV" in motionCmd:
					dstV_ex=motionCmd["dstV"]
				elif decisionType=="Vel":
					dstV_ex=value
				elif decisionType=="Throttle":
					#0〜1のスロットルで指定していた場合
					th=0.3
					if value>1-th:
						accelIdx=len(self.accelTable)-1
					elif value<th:
						accelIdx=0
					else:
						dstV_ex=V
				elif decisionType=="Accel":
					#加速度ベースの指定だった場合
					eps=0.5
					if abs(value)<eps:
						dstV_ex=V
					elif value>0:
						accelIdx=len(self.accelTable)-1
					else:
						accelIdx=0
				else:
					raise ValueError("Given unsupported expert's decision & command for this Agent class.")
				if accelIdx<0:
					deltaV_ex=dstV_ex-V
					tmp_d=self.accelTable-deltaV_ex
					tmp_i=np.argmin(abs(tmp_d))
					myAction["accel"]=tmp_i
				else:
					myAction["accel"]=accelIdx

			#射撃
			shotTarget_ex=-1
			expertTarget=Track3D(decision[parentFullName]["Fire"][1])
			if decision[parentFullName]["Fire"][0]():
				for tIdx,t in enumerate(self.lastTrackInfo):
					if t.isSame(expertTarget):
						shotTarget_ex=tIdx
				if shotTarget_ex>=self.maxEnemyNum:
					shotTarget_ex=-1
			myAction["target"]=shotTarget_ex+1
			if self.use_Rmax_fire:
				if shotTarget_ex<0:
					#射撃なしの場合、間隔と射程の条件は最も緩いものとする
					if len(self.shotIntervalTable)>1:
						myAction["shotInterval"]=0
					if len(self.shotThresholdTable)>1:
						myAction["shotThreshold"]=len(self.shotThresholdTable)-1
				else:
					#射撃ありの場合、間隔と射程の条件は最も厳しいものとする
					if not expertTarget.truth in actionInfo.lastShotTimes:
						actionInfo.lastShotTimes[expertTarget.truth]=self.manager.getEpoch()
					if len(self.shotIntervalTable)>1:
						shotInterval_ex=self.manager.getTime()-actionInfo.lastShotTimes[expertTarget.truth]
						myAction["shotInterval"]=0
						for i in range(len(self.shotIntervalTable)-1,-1,-1):
							if self.shotIntervalTable[i]<shotInterval_ex:
								myAction["shotInterval"]=i
								break
					if len(self.shotThresholdTable)>1:
						r=calcRNorm(parent,myMotion,expertTarget,False)
						myAction["shotThreshold"]=len(self.shotThresholdTable)-1
						for i in range(len(self.shotThresholdTable)):
							if r<self.shotThresholdTable[i]:
								myAction["shotThreshold"]=i
								break
			ret.append(myAction)
		return ret

	def controlWithAnotherAgent(self,decision,command):
		#基本的にはオーバーライド不要だが、模倣時にActionと異なる行動を取らせたい場合に使用する。
		self.control()
		#例えば、以下のようにcommandを置換すると射撃のみexpertと同タイミングで行うように変更可能。
		#self.commands[parent.getFullName()]["weapon"]=command[parent.getFullName()]["weapon"]

	def isInside(self,lon,lat):
		"""ピクセル座標(lon,lat)が画像の範囲内かどうかを返す。
		"""
		return 0<=lon and lon<self.image_longitudinal_resolution and 0<=lat and lat<self.image_lateral_resolution

	def rposToGrid(self,dr):
		"""基準点からの相対位置drに対応するピクセル座標(lon,lat)を返す。
		"""
		lon=floor((dr[1]+self.image_side_range)*self.image_lateral_resolution/(2*self.image_side_range))
		lat=floor((dr[0]+self.image_back_range)*self.image_longitudinal_resolution/(self.image_front_range+self.image_back_range))
		return np.array([lon,lat])

	def gridToRpos(self,lon,lat,alt):
		"""指定した高度においてピクセル座標(lon,lat)に対応する基準点からの相対位置drを返す。
		"""
		return np.array([
			self.image_buffer_coords[0,lon,lat],
			self.image_buffer_coords[1,lon,lat],
			-alt
		])

	def posToGrid(self,pos):
		"""絶対位置posに対応するピクセル座標(lon,lat)を返す。
		"""
		if self.image_relative_position:
			#生存中のparentsの中で先頭のparentの位置が基準点
			for fIdx in range(len(self.parents)):
				if self.ourObservables[fIdx].at("isAlive"):
					origin=self.ourMotion[fIdx].pos()
					break
		else:
			#自陣営防衛ライン中央が基準点
			origin=self.teamOrigin.pos
		rpos=pos-origin
		if self.image_rotate:
			#生存中のparentsの中で先頭のparentのH座標系に回転
			for fIdx in range(len(self.parents)):
				if self.ourObservables[fIdx].at("isAlive"):
					rpos=self.ourMotion[fIdx].relPtoH(rpos,origin,"FSD",False)
					break
		else:
			#慣性座標系から陣営座標系に回転
			rpos=self.teamOrigin.relPtoB(rpos)
		return self.rposToGrid(rpos)

	def gridToPos(self,lon,lat,alt):
		"""指定した高度においてピクセル座標(lon,lat)に対応する絶対位置posを返す。
		"""
		rpos=self.gridToRpos(lon,lat,alt)
		if self.image_relative_position:
			#生存中のparentsの中で先頭のparentの位置が基準点
			for fIdx in range(len(self.parents)):
				if self.ourObservables[fIdx].at("isAlive"):
					origin=self.ourMotion[fIdx].pos()
					break
		else:
			#自陣営防衛ライン中央が基準点
			origin=self.teamOrigin.pos
		if self.image_rotate:
			#生存中のparentsの中で先頭のparentのH座標系から慣性座標系に回転
			for fIdx in range(len(self.parents)):
				if self.ourObservables[fIdx].at("isAlive"):
					rpos=self.ourMotion[fIdx].relHtoP(rpos,self.ourMotion[fIdx].absPtoH(origin,"FSD",False),"FSD",False)
					break
		else:
			#陣営座標系から慣性座標系に回転
			rpos=self.teamOrigin.relBtoP(rpos)
		return rpos+origin

	def plotPoint(self,pos,ch,value):
		"""画像バッファのチャネルchで慣性座標系での絶対位置posに対応する点の値をvalueにする。
		"""
		g=self.posToGrid(pos)
		lon=g[0]
		lat=g[1]
		if self.isInside(lon,lat):
			self.image_buffer[ch,lon,lat]=value

	def plotLine(self,pBegin,pEnd,ch,value):
		"""画像バッファのチャネルchで慣性座標系での絶対位置pBeginからpEndまでの線分に対応する各点の値をvalueにする。
		線分の描画にはブレゼンハムのアルゴリズムを使用している。
		"""
		gBegin=self.posToGrid(pBegin)
		gEnd=self.posToGrid(pEnd)
		x0=gBegin[0]
		y0=gBegin[1]
		x1=gEnd[0]
		y1=gEnd[1]
		steep=abs(y1-y0)>abs(x1-x0)
		if steep:
			x0,y0=y0,x0
			x1,y1=y1,x1
		if x0>x1:
			x0,x1=x1,x0
			y0,y1=y1,y0
		deltax=x1-x0
		deltay=abs(y1-y0)
		error=deltax/2
		ystep = 1 if y0<y1 else -1
		y=y0
		for x in range(x0,x1+1):
			if steep:
				if self.isInside(y,x):
					self.image_buffer[ch,y,x]=value
			else:
				if self.isInside(x,y):
					self.image_buffer[ch,x,y]=value
			error-=deltay
			if error<0:
				y+=ystep
				error+=deltax
