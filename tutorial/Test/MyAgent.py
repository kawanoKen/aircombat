import numpy as np
from ASRCAISim1.core import Agent, getValueFromJsonK, getValueFromJsonKR, getValueFromJsonKRD, LinearSegment, MotionState, Track3D, Track2D, Time, TimeSystem, deg2rad, serialize_attr_with_type_info, StaticCollisionAvoider2D,AltitudeKeeper# type: ignore
from BasicAgentUtility.util import TeamOrigin, sortTrack3DByDistance, sortTrack2DByAngle, calcRNorm # type: ignore
from math import atan2, cos, sin, sqrt
from gymnasium import spaces


class SampleAgent(Agent):
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
        self.own = self.getTeam()
        self.common_dim = 1
        self.maxParentNum=getValueFromJsonK(self.modelConfig,"maxParentNum")
        self.maxFriendNum=getValueFromJsonK(self.modelConfig,"maxFriendNum")
        self.maxEnemyNum=getValueFromJsonK(self.modelConfig,"maxEnemyNum")
        self.maxFriendMissileNum=getValueFromJsonK(self.modelConfig,"maxFriendMissileNum")
        self.maxEnemyMissileNum=getValueFromJsonK(self.modelConfig,"maxEnemyMissileNum")
        self.use_observation_mask=getValueFromJsonK(self.modelConfig,"use_observation_mask")
        self.use_action_mask=getValueFromJsonK(self.modelConfig,"use_action_mask")
        self.remaining_time_clipping=getValueFromJsonKR(self.modelConfig,"remaining_time_clipping",self.randomGen)
        self.friend_dim=7
        self.horizontalNormalizer=getValueFromJsonKR(self.modelConfig,"horizontalNormalizer",self.randomGen)
        self.verticalNormalizer=getValueFromJsonKR(self.modelConfig,"verticalNormalizer",self.randomGen)
        self.fgtrVelNormalizer=getValueFromJsonKR(self.modelConfig,"fgtrVelNormalizer",self.randomGen)
        self.enemy_dim=7
        self.friend_missile_dim=7
        self.mslVelNormalizer=getValueFromJsonKR(self.modelConfig,"mslVelNormalizer",self.randomGen)
        self.enemy_missile_dim = 3


        #actionに関するもの
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

        self.actionInfos={}
        for port,parent in self.parents.items():
            self.actionInfos[parent.getFullName()]=self.ActionInfo()

        # 加減速に関する設定
        self.accelTable=np.array(sorted(getValueFromJsonK(self.modelConfig,"accelTable")),dtype=np.float64)

        #行動制限に関する設定
        # 場外制限に関する設定
        self.dOutLimit=getValueFromJsonKRD(self.modelConfig,"dOutLimit",self.randomGen,5000.0)
        self.dOutLimitThreshold=getValueFromJsonKRD(self.modelConfig,"dOutLimitThreshold",self.randomGen,10000.0)
        self.dOutLimitStrength=getValueFromJsonKRD(self.modelConfig,"dOutLimitStrength",self.randomGen,2e-3)

        #  高度制限に関する設定
        self.altMin=getValueFromJsonKRD(self.modelConfig,"altMin",self.randomGen,2000.0)
        self.altMax=getValueFromJsonKRD(self.modelConfig,"altMax",self.randomGen,15000.0)
        self.altitudeKeeper=AltitudeKeeper(self.modelConfig().get("altitudeKeeper",{}))

        # 同時射撃数の制限に関する設定
        self.maxSimulShot=getValueFromJsonKRD(self.modelConfig,"maxSimulShot",self.randomGen,4)

        # 下限速度の制限に関する設定
        self.minimumV=getValueFromJsonKRD(self.modelConfig,"minimumV",self.randomGen,150.0)
        self.minimumRecoveryV=getValueFromJsonKRD(self.modelConfig,"minimumRecoveryV",self.randomGen,180.0)
        self.minimumRecoveryDstV=getValueFromJsonKRD(self.modelConfig,"minimumRecoveryDstV",self.randomGen,200.0)


    def validate(self):
        #Rulerに関する情報の取得
        rulerObs=self.manager.getRuler()().observables()
        self.dOut=rulerObs["dOut"] # 戦域中心から場外ラインまでの距離
        self.dLine=rulerObs["dLine"] # 戦域中心から防衛ラインまでの距離
        self.teamOrigin=TeamOrigin(self.own==rulerObs["eastSider"],self.dLine) # 陣営座標系変換クラス定義


    def makeObs(self):
        obs = {}
        observation_mask={}
        
        # common(残り時間)
        ret=np.zeros([self.common_dim],dtype=np.float32)
        rulerObs=self.manager.getRuler()().observables
        maxTime=rulerObs['maxTime']()
        ret[0]=min((maxTime-self.manager.getElapsedTime())/60.0, self.remaining_time_clipping)

        obs['common'] = ret

        #味方機(parents→parents以外の順)
        ret = np.zeros([self.maxParentNum, self.friend_dim],dtype=np.float32)
        parent_mask=np.zeros([self.maxParentNum],dtype=np.float32)
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
        fIdx = 0
        for port,parent in self.parents.items():
            if fIdx>=self.maxParentNum:
                break
            fObs=self.ourObservables[fIdx]
            fMotion=self.ourMotion[fIdx]
            if fObs.at("isAlive"):
                parent_mask[fIdx]=1
                pos=self.teamOrigin.relPtoB(fMotion.pos()) #慣性座標系→陣営座標系に変換
                vel=self.teamOrigin.relPtoB(fMotion.vel()) #慣性座標系→陣営座標系に変換
                a=np.zeros([self.friend_dim],dtype=np.float32)
                ofs = 0
                a[ofs:ofs+3]=pos/np.array([self.horizontalNormalizer,self.horizontalNormalizer,self.verticalNormalizer])
                ofs += 3
                V=np.linalg.norm(vel)
                a[ofs]=V/self.fgtrVelNormalizer
                ofs+=1
                a[ofs:ofs+3]=vel/max(V, 1e-5)
                ret[fIdx, :] = a
            fIdx+=1

        obs['parent'] = ret
        if self.use_observation_mask:
            observation_mask['parent']=parent_mask

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

        ret=np.zeros([self.maxEnemyNum,self.enemy_dim],dtype=np.float32)
        enemy_mask=np.zeros([self.maxEnemyNum],dtype=np.float32)
        for tIdx,track in enumerate(self.lastTrackInfo):
            if tIdx>=self.maxEnemyNum:
                break
            t=np.zeros([self.enemy_dim],dtype=np.float32)
            ofs=0
            pos=self.teamOrigin.relPtoB(track.pos()) #慣性座標系→陣営座標系に変換
            t[ofs:ofs+3]=pos/np.array([self.horizontalNormalizer,self.horizontalNormalizer,self.verticalNormalizer])
            ofs+=3
            vel=self.teamOrigin.relPtoB(track.vel()) #慣性座標系→陣営座標系に変換
            V=np.linalg.norm(vel)
            t[ofs]=V/self.fgtrVelNormalizer
            ofs+=1
            t[ofs:ofs+3]=vel/max(V, 1e-5)
            ofs+=3
            ret[tIdx,:]=t
            enemy_mask[tIdx]=1

        obs['enemy']=ret
        if self.use_observation_mask:
            observation_mask['enemy']=enemy_mask


        #味方誘導弾(射撃時刻の古い順にソート
        def launchedT(m):
            return Time(m["launchedT"]) if m["isAlive"] and m["hasLaunched"] else Time(np.inf,TimeSystem.TT)
        self.msls=sorted(sum([[m for m in f.at_p("/weapon/missiles")] for f in self.ourObservables],[]),key=launchedT)
        ret=np.zeros([self.maxFriendMissileNum,self.friend_missile_dim],dtype=np.float32)
        friend_missile_mask=np.zeros([self.maxFriendMissileNum],dtype=np.float32)
        for mIdx,mObs in enumerate(self.msls):
            if mIdx>=self.maxFriendMissileNum or not (mObs.at("isAlive") and mObs.at("hasLaunched")):
                break
            a=np.zeros([self.friend_missile_dim],dtype=np.float32)
            ofs=0
            mm=MotionState(mObs["motion"]).transformTo(self.getLocalCRS())
            pos=self.teamOrigin.relPtoB(mm.pos()) #慣性座標系→陣営座標系に変換
            a[ofs:ofs+3]=pos/np.array([self.horizontalNormalizer,self.horizontalNormalizer,self.verticalNormalizer])
            ofs+=3
            vel=self.teamOrigin.relPtoB(mm.vel()) #慣性座標系→陣営座標系に変換
            V=np.linalg.norm(vel)
            a[ofs]=V/self.mslVelNormalizer
            ofs+=1
            a[ofs:ofs+3]=vel/max(V, 1e-5)
            ret[mIdx,:]=a
            friend_missile_mask[mIdx]=1
        
        obs['friend_missile'] = ret
        if self.use_observation_mask:
            observation_mask['friend_missile'] = friend_missile_mask

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
        ret=np.zeros([self.maxEnemyMissileNum,self.enemy_missile_dim],dtype=np.float32)
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
            a=np.zeros([self.enemy_missile_dim],dtype=np.float32)
            ofs=0
            origin=self.teamOrigin.relPtoB(m[0].origin()) #慣性座標系→陣営座標系に変換
            a[ofs:ofs+3]=origin/np.array([self.horizontalNormalizer,self.horizontalNormalizer,self.verticalNormalizer])
            ret[mIdx,:]=a
            enemy_missile_mask[mIdx]=1
        
        obs['enemy_missile'] = ret
        if self.use_observation_mask:
            observation_mask['enemy_missile']=enemy_missile_mask

        if self.use_observation_mask and len(observation_mask)>0:
            obs['observation_mask']=observation_mask

        if self.use_action_mask:
            self.action_mask=self.makeActionMask()
            if not self.action_mask is None:
                obs['action_mask']=self.action_mask


        return obs


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
                mask["accel"]=np.full([len(self.accelTable)],1,dtype=np.float32)
                mask["target"]=target_mask
                ret.append(mask)
            return ret
        else:
            return None


    def observation_space(self):
        floatLow=np.finfo(np.float32).min
        floatHigh=np.finfo(np.float32).max
        obs_space = {
            'common': spaces.Box(floatLow,floatHigh,
                                 shape=[self.common_dim],
                                 dtype=np.float32),
            'parent': spaces.Box(floatLow,floatHigh,
                                 shape=[self.maxParentNum,self.friend_dim],
                                 dtype=np.float32),
            'enemy': spaces.Box(floatLow,floatHigh,
                                shape=[self.maxEnemyNum,self.enemy_dim],
                                dtype=np.float32),
            'friend_missile': spaces.Box(floatLow,floatHigh,
                                         shape=[self.maxFriendMissileNum,self.friend_missile_dim],
                                         dtype=np.float32),
            'enemy_missile': spaces.Box(floatLow,floatHigh,
                                        shape=[self.maxEnemyMissileNum,self.enemy_missile_dim],
                                        dtype=np.float32)
        }
        if self.use_observation_mask:
            observation_mask = {
                'parent': spaces.Box(floatLow,floatHigh,
                                     shape=[self.maxParentNum],
                                     dtype=np.float32),
                'enemy': spaces.Box(floatLow,floatHigh,
                                    shape=[self.maxEnemyNum],
                                    dtype=np.float32),
                'friend_missile': spaces.Box(floatLow,floatHigh,
                                             shape=[self.maxFriendMissileNum],
                                             dtype=np.float32),
                'enemy_missile': spaces.Box(floatLow,floatHigh,
                                            shape=[self.maxEnemyMissileNum],
                                            dtype=np.float32)
            }
            obs_space['observation_mask'] = spaces.Dict(observation_mask) # type: ignore

        if self.use_action_mask:
            single_action_mask_space_dict = {
                'turn': spaces.Box(floatLow,floatHigh,
                                   shape=[len(self.turnTable)],
                                   dtype=np.float32),
                'accel': spaces.Box(floatLow,floatHigh,
                                    shape=[len(self.accelTable)],
                                    dtype=np.float32),
                'target': spaces.Box(floatLow,floatHigh,
                                     shape=[1+self.maxEnemyNum],
                                     dtype=np.float32)
            }
            single_action_mask_space=spaces.Dict(single_action_mask_space_dict) # type: ignore
            action_mask_space_list=[]
            for port, parent in self.parents.items():
                action_mask_space_list.append(single_action_mask_space)
            obs_space['action_mask'] = spaces.Tuple(action_mask_space_list) # type: ignore

        return spaces.Dict(obs_space) # type: ignore


    def action_space(self):
        single_action_space_dict={
            'turn': spaces.Discrete(len(self.turnTable)),
            'target': spaces.Discrete(1+self.maxEnemyNum),
            'accel': spaces.Discrete(len(self.accelTable))
        }
        single_action_space=spaces.Dict(single_action_space_dict) # type: ignore
        action_space_list=[]
        for port, parent in self.parents.items():
            action_space_list.append(single_action_space)
        return spaces.Tuple(action_space_list) # 行動空間の定義


    def deploy(self, action):
        #observablesの収集
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


        # 彼機情報だけは射撃対象の選択と連動するので更新してはいけない。
        #味方誘導弾(射撃時刻の古い順にソート)
        def launchedT(m):
            return Time(m["launchedT"]) if m["isAlive"] and m["hasLaunched"] else Time(np.inf,TimeSystem.TT)
        self.msls=sorted(sum([[m for m in f.at_p("/weapon/missiles")] for f in self.ourObservables],[]),key=launchedT)

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

        for pIdx,parent in enumerate(self.parents.values()):
            parentFullName=parent.getFullName()
            if not parent.isAlive():
                continue
            actionInfo=self.actionInfos[parentFullName]
            myMotion=self.ourMotion[pIdx]
            myObs=self.ourObservables[pIdx]
            myMWS=self.mws[pIdx]
            myAction=action[pIdx]

            #左右旋回
            deltaAz=self.turnTable[myAction["turn"]]
            actionInfo.dstDir=self.teamOrigin.relBtoP(np.array([cos(deltaAz),sin(deltaAz),0]))
            dstAz=atan2(actionInfo.dstDir[1],actionInfo.dstDir[0])

            #上昇・下降
            dstPitch=0
            actionInfo.dstDir=np.array([actionInfo.dstDir[0]*cos(dstPitch),actionInfo.dstDir[1]*cos(dstPitch),-sin(dstPitch)])

            #加減速
            V=np.linalg.norm(myMotion.vel())
            actionInfo.asThrottle=False
            accel=self.accelTable[myAction["accel"]]
            actionInfo.dstV=V+accel
            actionInfo.keepVel = accel==0.0

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

            #射撃可否の判断、射撃コマンドの生成
            flyingMsls=0
            if myObs.contains_p("/weapon/missiles"):
                for msl in myObs.at_p("/weapon/missiles"):
                    if msl.at("isAlive")() and msl.at("hasLaunched")():
                        flyingMsls+=1
            if not (
                shotTarget>=0 and
                shotTarget<len(self.lastTrackInfo) and
                parent.isLaunchableAt(self.lastTrackInfo[shotTarget]) and
                flyingMsls<self.maxSimulShot
            ):
                shotTarget=-1
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
            print(dstAz, deltaAz, dstPitch, actionInfo.dstThrottle, actionInfo.dstV)
            if len(myMWS)>0 and self.use_override_evasion:
                self.observables[parentFullName]["decision"]["Horizontal"]=("Az_NED",dstAz)
            else:
                if self.dstAz_relative:
                    self.observables[parentFullName]["decision"]["Horizontal"]=("Az_BODY",deltaAz)
                else:
                    self.observables[parentFullName]["decision"]["Horizontal"]=("Az_NED",dstAz)
            self.observables[parentFullName]["decision"]["Vertical"]=("El",dstPitch)
            if actionInfo.asThrottle:
                self.observables[parentFullName]["decision"]["Throttle"]=("Throttle",actionInfo.dstThrottle)
            else:
                self.observables[parentFullName]["decision"]["Throttle"]=("Vel",actionInfo.dstV)


    def control(self):
        #observablesの収集
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


        # 彼機情報だけは射撃対象の選択と連動するので更新してはいけない。
        #味方誘導弾(射撃時刻の古い順にソート)
        def launchedT(m):
            return Time(m["launchedT"]) if m["isAlive"] and m["hasLaunched"] else Time(np.inf,TimeSystem.TT)
        self.msls=sorted(sum([[m for m in f.at_p("/weapon/missiles")] for f in self.ourObservables],[]),key=launchedT)

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

            #高度方向の補正
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
            if actionInfo.asThrottle:
                self.commands[parentFullName]["motion"]["dstThrottle"]=actionInfo.dstThrottle
            elif actionInfo.keepVel:
                self.commands[parentFullName]["motion"]["dstAccel"]=0.0
            else:
                self.commands[parentFullName]["motion"]["dstV"]=actionInfo.dstV
            actionInfo.launchFlag=False
