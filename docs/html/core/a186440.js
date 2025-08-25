var a186440 =
[
    [ "シミュレーション登場物について", "a186446.html", "a186446" ],
    [ "シミュレーションの処理の流れについて", "a186452.html", [
      [ "シミュレーションの処理の流れ", "a186452.html#section_simulation_flow", null ],
      [ "非周期的なイベントの発生及びイベントハンドラ", "a186452.html#section_simulation_event_handler", null ]
    ] ],
    [ "シミュレータの生成及び外部インターフェース", "a186451.html", [
      [ "SimulationManager クラス", "a186451.html#section_simulation_execution_SimulationManager", [
        [ "コンストラクタ", "a186451.html#section_simulation_execution_SimulationManager_constructor", null ],
        [ "コンフィグの書式", "a186451.html#section_simulation_execution_SimulationManager_config_format", null ],
        [ "PhysiacalAsset の生成について", "a186451.html#section_simulation_execution_asset_config_dispatch", null ],
        [ "Agent の生成について", "a186451.html#section_simulation_execution_agent_config_dispatch", null ]
      ] ],
      [ "gymnasium インターフェース", "a186451.html#section_simulation_execution_as_gymnasium_env", [
        [ "GymManager クラス", "a186451.html#section_simulation_execution_GymManager", [
          [ "コンストラクタ(<b>init</b>)", "a186451.html#section_simulation_execution_GymManager_init", null ],
          [ "コンフィグの書き換え", "a186451.html#section_simulation_execution_GymManager_reconfigure", [
            [ "インスタンス生成時の書き換え", "a186451.html#section_simulation_execution_GymManager_reconfigure_at_instantiation", null ],
            [ "エピソード開始時の書き換え", "a186451.html#section_simulation_execution_GymManager_reconfigure_on_episode_begin", null ]
          ] ]
        ] ],
        [ "SinglizedEnv クラス", "a186451.html#section_simulation_execution_SinglizedEnv", null ]
      ] ],
      [ "SimpleEvaluator", "a186451.html#section_simulation_execution_SimpleEvaluator", null ],
      [ "StandalonePolicy", "a186451.html#section_simulation_execution_StandalonePolicy", [
        [ "Agentのfull nameの書式", "a186451.html#section_simulation_execution_Agent_naming_convention", null ]
      ] ]
    ] ],
    [ "時刻系の表現について", "a186456.html", [
      [ "Time", "a186456.html#section_simulation_Time", [
        [ "Time クラスのjson表現", "a186456.html#section_simulation_TimeSystem_Time_json", null ]
      ] ],
      [ "Epoch", "a186456.html#section_simulation_Epoch", [
        [ "Epoch クラスのjson表現", "a186456.html#section_simulation_TimeSystem_Epoch_json", null ]
      ] ]
    ] ],
    [ "座標系の取り扱いについて", "a186445.html", [
      [ "座標系の分類とクラス階層", "a186445.html#section_simulation_Coordinate_hierarchy", [
        [ "GeodeticCRS", "a186445.html#section_simulation_Coordinate_GeodeticCRS", [
          [ "制約", "a186445.html#section_simulation_Coordinate_GeodeticCRS_limitation", null ]
        ] ],
        [ "PureFlatCRS", "a186445.html#section_simulation_Coordinate_PureFlatCRS", null ],
        [ "DerivedCRS", "a186445.html#section_simulation_Coordinate_DerivedCRS", [
          [ "AffineCRS", "a186445.html#section_simulation_Coordinate_AffineCRS", null ],
          [ "TopocentricCRS", "a186445.html#section_simulation_Coordinate_TopocentricCRS", null ],
          [ "ProjectedCRS", "a186445.html#section_simulation_Coordinate_ProjectedCRS", null ]
        ] ]
      ] ],
      [ "座標軸の順序について", "a186445.html#section_simulation_Coordinate_axis_order", [
        [ "GeodeticCRS の場合", "a186445.html#section_simulation_Coordinate_axis_geodetic", null ],
        [ "物体固定座標系の直交座標軸の場合", "a186445.html#section_simulation_Coordinate_axis_body_cartesian", null ],
        [ "局所水平座標系の直交座標軸の場合", "a186445.html#section_simulation_Coordinate_axis_topocentric_cartesian", null ],
        [ "球座標系の場合", "a186445.html#section_simulation_Coordinate_axis_spherical", null ]
      ] ],
      [ "座標値の種類", "a186445.html#section_simulation_Coordinate_CoordinateType", null ],
      [ "CoordinateReferenceSystem オブジェクトを用いた座標変換", "a186445.html#section_simulation_Coordinate_CRS_transformation", null ],
      [ "座標値を表すデータ型「 Coordinate 」", "a186445.html#section_simulation_Coordinate_Coordinate", [
        [ "Coordinate クラスのjson表現", "a186445.html#section_simulation_Coordinate_Coordinate_json", null ],
        [ "Coordinate オブジェクトの座標変換", "a186445.html#section_simulation_Coordinate_Coordinate_transformation", null ]
      ] ],
      [ "CRSモデルのプリセット", "a186445.html#section_simulation_Coordinate_preset_models", [
        [ "SimulationManager固有のCRSインスタンスのプリセット", "a186445.html#section_simulation_Coordinate_preset_instances", null ]
      ] ],
      [ "シミュレーション中に必ず定義される座標系", "a186445.html#section_simulation_Coordinate_in_simulation", [
        [ "SimulationManager::rootCRS", "a186445.html#section_simulation_Coordinate_in_simulation_rootCRS", null ],
        [ "Ruler::localCRS", "a186445.html#section_simulation_Coordinate_in_simulation_Ruler_localCRS", null ],
        [ "Agent::localCRS", "a186445.html#section_simulation_Coordinate_in_simulation_Agent_localCRS", null ]
      ] ]
    ] ],
    [ "運動状態の表現方法(MotionState)", "a186454.html", [
      [ "MotionState クラスのjson表現", "a186454.html#section_simulation_MotionState_MotionState_json", null ],
      [ "時刻の外挿", "a186454.html#section_simulation_MotionState_extrapolation", null ],
      [ "座標変換", "a186454.html#section_simulation_MotionState_transformation", [
        [ "個別の状態量の座標変換", "a186454.html#section_simulation_MotionState_state_transformation", null ],
        [ "MotionStateオブジェクト全体の座標変換", "a186454.html#section_simulation_MotionState_entire_transformation", null ],
        [ "<a class=\"el\" href=\"a186450.html#section_simulation_entity_PhysicalAsset\">PhysicalAsset</a>に固定された座標系(<a class=\"el\" href=\"a186445.html#section_simulation_Coordinate_AffineCRS\">AffineCRS</a>)としての座標変換", "a186454.html#section_simulation_MotionState_transformation_as_AffineCRS", [
          [ "座標種別 xxx の選択肢", "a186454.html#autotoc_md10", null ],
          [ "座標系 Y,Z の選択肢", "a186454.html#autotoc_md11", null ]
        ] ]
      ] ]
    ] ],
    [ "航跡情報の表現方法(TrackBase, Track3D, Track2D)", "a186457.html", [
      [ "3次元航跡 (Track3D)", "a186457.html#section_simulation_Track_Track3D", [
        [ "Track3D クラスのjson表現", "a186457.html#section_simulation_Track_Track3D_json", null ]
      ] ],
      [ "2次元航跡 (Track2D)", "a186457.html#section_simulation_Track_Track2D", [
        [ "Track2D クラスのjson表現", "a186457.html#section_simulation_Track_Track2D_json", null ]
      ] ],
      [ "時刻の外挿", "a186457.html#section_simulation_Track_extrapolation", null ],
      [ "航跡のマージ", "a186457.html#section_simulation_Track_merge", null ],
      [ "同一性の判定", "a186457.html#section_simulation_Track_identification", null ],
      [ "座標変換", "a186457.html#section_simulation_Track_transformation", [
        [ "個別の状態量の座標変換", "a186457.html#section_simulation_Track_state_transformation", null ],
        [ "Trackオブジェクト全体の座標変換", "a186457.html#section_simulation_Track_entire_transformation", null ]
      ] ]
    ] ],
    [ "CommunicationBuffer による Asset 間通信の表現", "a186443.html", null ],
    [ "Accessorクラスによるアクセス制限", "a186441.html", [
      [ "SimulationManagerへのアクセス制限", "a186441.html#section_simulation_accessor_SimulationManager", null ],
      [ "Entity へのアクセス制限", "a186441.html#section_simulation_accessor_Entity", [
        [ "Asset へのアクセス制限", "a186441.html#section_simulation_accessor_Asset", [
          [ "PhysicalAsset へのアクセス制限", "a186441.html#section_simulation_accessor_PhysicalAsset", null ],
          [ "Fighter へのアクセス制限", "a186441.html#section_simulation_accessor_Fighter", null ]
        ] ],
        [ "Ruler へのアクセス制限", "a186441.html#section_simulation_accessor_Ruler", null ]
      ] ]
    ] ],
    [ "オブジェクトのシリアライゼーション", "a186455.html", [
      [ "既存オブジェクトへの参照としてのシリアライゼーション", "a186455.html#section_simulation_serialization_json_as_reference", [
        [ "既存の Entity インスタンスへの参照としてのjson表現", "a186455.html#section_simulation_Entity_as_reference_json", null ],
        [ "既存の EntityManager インスタンスへの参照としてのjson表現", "a186455.html#section_simulation_EntityManager_as_reference_json", null ]
      ] ],
      [ "値渡しでシリアライゼーションを行うクラス", "a186455.html#section_simulation_serialization_json_as_value", null ],
      [ "内部状態のシリアライゼーション (Experimental)", "a186455.html#autotoc_md7", [
        [ "Entity の内部状態の保存と復元", "a186455.html#autotoc_md8", null ],
        [ "SimulationManagerの内部状態の保存と復元", "a186455.html#autotoc_md9", null ]
      ] ]
    ] ],
    [ "jsonによる確率的パラメータ設定", "a186453.html", [
      [ "関数名に付けるK,R,Dの意味", "a186453.html#autotoc_md3", null ],
      [ "getValueFromJsonR(j,gen) のサンプリング仕様", "a186453.html#autotoc_md4", null ]
    ] ],
    [ "ConfigDispatcher によるjson object の再帰的な変換", "a186444.html", [
      [ "ConfigDispatcherの使用方法", "a186444.html#section_simulation_config_dispatcher_usage", null ],
      [ "再帰的な展開の仕様", "a186444.html#section_simulation_config_dispatcher_dispatch", null ]
    ] ]
];