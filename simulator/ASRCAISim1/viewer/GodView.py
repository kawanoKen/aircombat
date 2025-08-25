# Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)

from math import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import json
import os
import pygame
import numpy as np
from ASRCAISim1.utility.GraphicUtility import *
from ASRCAISim1.core import (
    Coordinate,
    CoordinateType,
    Missile,
    MotionState,
    Time,
    serialize_attr_with_type_info,
)
from ASRCAISim1.viewer.PygameViewer import PygameViewer, PygamePanel

class GodView(PygameViewer):
    """pygameを用いて、画面の上側に戦域を上から見た図を、下側に戦域を南から見た図を描画する例。
    機体、誘導弾、センサの覆域、防衛ラインの描画に加え、現在の時刻、得点、報酬、Agentの情報等を表示する。
    """
    defaultPanelConfigPath = os.path.join(os.path.dirname(__file__),"../config","GodViewUIConfig.json")
    def initialize(self):
        super().initialize()
        # 描画用のCRSの指定に関する情報を取得
        self.crsConfig=None
        if "crs" in self.instanceConfig:
            self.crsConfig=self.instanceConfig["crs"]
        elif "crs" in self.modelConfig:
            self.crsConfig=self.modelConfig["crs"]
        self.panelConfig.update({
            "width": self.width,
            "height": self.height,
            "crsConfig": self.crsConfig,
        })

    def serializeInternalState(self, archive, full: bool):
        super().serializeInternalState(archive, full)
    def makePanel(self):
        """画面描画を行うPanelオブジェクトを返す。
        """
        return GodViewPanel(self.panelConfig)

class GodViewPanel(PygamePanel):
    """pygameを用いて、画面の上側に戦域を上から見た図を、下側に戦域を南から見た図を描画する例。
    機体、誘導弾、センサの覆域、防衛ラインの描画に加え、現在の時刻、得点、報酬、Agentの情報等を表示する。
    """
    configKeysToBePopped = [
        "width",
        "height",
        "horizontalUnit",
        "vUnit",
        "altUnit",
        "uiScalerType",
        "crsConfig",
    ]
    def __init__(self,config):
        super().__init__(config)
        self.dynamicConfigPaths = ["", "/*/fighter", "/*/missile"]
        self.width=config["width"]
        self.height=config["height"]
        self.fieldMargin=np.array(self.getUIConfig("fieldMargin","/Default/region")) #各軸の表示範囲の余裕。
        self.xyScaleType=self.getUIConfig("xyScaleType","/Default/region") #x軸とy軸の縮尺。sameとしたら同じになる。fitとしたら個別に拡大・縮小される。
        self.regionType=self.getUIConfig("regionType","/Default/region") #各軸の表示範囲。fixとしたら固定。fitとしたら全機が入るように拡大・縮小される。
        self.slideRegion=self.getUIConfig("slideRegion","/Default/region")
        self.w_margin=self.getUIConfig("w_margin","/Default/region")
        self.h_margin=self.getUIConfig("h_margin","/Default/region")
        self.h_xy=self.getUIConfig("h_xy","/Default/region")
        self.h_yz=self.getUIConfig("h_yz","/Default/region")
        self.draw_xy=self.getUIConfig("draw_xy","/Default/region")
        self.draw_yz=self.getUIConfig("draw_yz","/Default/region")
        self.horizontalScaleFactors={
            "m":1.0,
            "km":1000.0,
            "NM":1852.0,
        }
        self.horizontalUnit=config["horizontalUnit"]
        assert(self.horizontalUnit in self.horizontalScaleFactors)
        self.vScaleFactors={
            "m/s":1.0,
            "kt":1852.0/3600.0,
            "km/h":1000.0/3600.0,
        }
        self.vUnit=config["vUnit"]
        assert(self.vUnit in self.vScaleFactors)
        self.altScaleFactors={
            "m":1.0,
            "ft":0.3048,
        }
        self.altUnit=config["altUnit"]
        assert(self.altUnit in self.altScaleFactors)
        uiScalerType=config["uiScalerType"] # "Min", "Max", "None" or float
        if uiScalerType == "Min":
            self.uiScaler=min(self.width/1280.0,self.height/720.0)
        elif uiScalerType == "Max":
            self.uiScaler=max(self.width/1280.0,self.height/720.0)
        elif isinstance(uiScalerType,float):
            self.uiScaler=uiScalerType
        else:
            self.uiScaler=1.0
        self.font=pygame.font.Font('freesansbold.ttf',round(12*self.uiScaler))

        self.drawFighter = {
            "1": self.drawFighter1,
            "2": self.drawFighter2,
        }
        self.drawMissile = {
            "1": self.drawMissile1,
            "2": self.drawMissile2,
        }
        self.crsConfig=config.get("crsConfig",None)

    def serializeInternalState(self, archive, full: bool):
        super().serializeInternalState(archive, full)
        if full:
            serialize_attr_with_type_info(archive, self
                ,"viewer"
                ,"isReplayMode"
                ,"rootCRS"
                ,"crs"
            )
            if self.isReplayMode:
                serialize_attr_with_type_info(archive, self, "lastAliveIndex")
        if full or not self.isReplayMode:
            serialize_attr_with_type_info(archive, self
                ,"time"
                ,"wake"
                ,"times"
                ,"wake_offset"
                ,"from_indices"
                ,"wake_from"
            )
    def identifyFighterModelToDraw(self,f):
        """ 描画用の機体種別を識別する。
        """
        return "Default"

    def identifyMissileModelToDraw(self,m):
        """ 描画用の誘導弾種別を識別する。
        """
        return "Default"

    def drawFighter1(self, f, frame):
        """ 第1回〜第3回空戦AIチャレンジで配布された標準の機体描画関数。
        """
        fighterModel = self.identifyFighterModelToDraw(f)
        glColor4f(*self.getUIConfig("color","/"+f["team"]+"/fighter/"+fighterModel+"/symbol"))

        if fighterModel == "Large":
            n=4
        else: #Small
            n=3
        mo=MotionState(f["motion"]).transformTo(self.crs)
        pf=self.regToSur(self.simToReg(mo.pos(),False))
        V=np.linalg.norm(np.array(f["vel"]))
        vn=self.regToSur(self.simToReg(mo.pos()+mo.vel()/V*1000.0,False))-pf
        vn[0]/=np.linalg.norm(vn[0])
        vn[1]/=np.linalg.norm(vn[1])

        fw1=pf+vn*8
        fw2=pf+vn*18
        if self.draw_xy:
            drawCircle2D(pf[0],4,n,vn[0])
            drawCircle2D(pf[0],5,n,vn[0])
            drawCircle2D(pf[0],6,n,vn[0])
            drawCircle2D(pf[0],7,n,vn[0])
            drawCircle2D(pf[0],8,n,vn[0])
        if self.draw_yz:
            drawCircle2D(pf[1],4,n,vn[1])
            drawCircle2D(pf[1],5,n,vn[1])
            drawCircle2D(pf[1],6,n,vn[1])
            drawCircle2D(pf[1],7,n,vn[1])
            drawCircle2D(pf[1],8,n,vn[1])
        glColor4f(0,0,0,1)
        if self.draw_xy:
            drawCircle2D(pf[0],3,n,vn[0])
            drawCircle2D(pf[0],9,n,vn[0])
            drawLine2D(fw1[0][0],fw1[0][1],fw2[0][0],fw2[0][1])
        if self.draw_yz:
            drawCircle2D(pf[1],3,n,vn[1])
            drawCircle2D(pf[1],9,n,vn[1])
            drawLine2D(fw1[1][0],fw1[1][1],fw2[1][0],fw2[1][1])

    def drawFighter2(self, f, frame):
        """ 第3回空戦AIチャレンジ参加者のkimper氏により追加された機体描画関数。
            使用許可が得られたため標準のGodViewに選択肢として追加するもの。
        """
        mo=MotionState(f["motion"]).transformTo(self.crs)
        pos=mo.pos()
        body_b=np.array([[1.5,0,0],[-0.5,0.8,0],[-0.5,-0.8,0]])
        fin_b=np.array([[1.5,0,0],[-0.5,0,0],[-0.7,0,-0.9]])

        fighterModel = self.identifyFighterModelToDraw(f)

        size=self.getUIConfig("size","/"+f["team"]+"/fighter/"+fighterModel+"/symbol")
        top_color=self.getUIConfig("color","/"+f["team"]+"/fighter/"+fighterModel+"/symbol")
        bottom_color=self.getUIConfig("color","/"+f["team"]+"/fighter/"+fighterModel+"/symbol-back", no_exception=True)
        if bottom_color is None:
            if f["team"]==frame["ruler"]["eastSider"]: #Blue
                bottom_color=(0.0,0.0,0.3,1)
            else: #Red
                bottom_color=(0.3,0.0,0.0,1)

        body=[mo.relBtoA(p,self.crs) for p in body_b*size]
        fin=[mo.relBtoA(p,self.crs) for p in fin_b*size]
        bottom_vec=mo.relBtoA([0,0,1],self.crs)

        #bodyの下面の色を変えるので、裏返ったときに隠れるfinから描画
        grid_scale = np.array(self.getUIConfig("scale","/Default/grid"))
        nodes=[self.regToSur(self.simToReg(pos+pt*grid_scale/grid_scale[0],False)) for pt in fin]
        nodes0=[n[0] for n in nodes]
        nodes1=[n[1] for n in nodes]

        glColor4f(*top_color)
        if self.draw_xy:
            fillTriangle2D(nodes0)
        if self.draw_yz:
            fillTriangle2D(nodes1)

        nodes=[self.regToSur(self.simToReg(pos+pt*grid_scale/grid_scale[0],False)) for pt in body]
        nodes0=[n[0] for n in nodes]
        nodes1=[n[1] for n in nodes]

        #xy図
        if self.draw_xy:
            if bottom_vec[2]>0:
                glColor4f(*top_color)
            else:
                glColor4f(*bottom_color)
            fillTriangle2D(nodes0)

        #yz図
        if self.draw_yz:
            if bottom_vec[0]>0:
                glColor4f(*top_color)
            else:
                glColor4f(*bottom_color)
            fillTriangle2D(nodes1)

    def drawMissile1(self, m, frame):
        """ 第1回〜第3回空戦AIチャレンジで配布された標準の誘導弾機体描画関数。
        """
        missileModel=self.identifyMissileModelToDraw(m)
        if(m["mode"]==Missile.Mode.SELF.name):
            missileMode = "self"
        elif(m["mode"]==Missile.Mode.GUIDED.name):
            if(m["sensor"]["isActive"]):
                missileMode = "guided-active"
            else:
                missileMode = "guided-inactive"
        else:
            if(m["sensor"]["isActive"]):
                missileMode = "memory-active"
            else:
                missileMode = "memory-inactive"

        glColor4f(*self.getUIConfig("color","/"+m["team"]+"/missile/"+missileModel+"/"+missileMode+"/symbol"))
        pm=self.regToSur(self.simToReg(m["pos"]))
        if self.draw_xy:
            fillCircle2D(pm[0],4,8)
        if self.draw_yz:
            fillCircle2D(pm[1],4,8)

    def drawMissile2(self, m, frame):
        """ 第3回空戦AIチャレンジ参加者のkimper氏により追加された誘導弾描画関数。
            使用許可が得られたため標準のGodViewに選択肢として追加するもの。
        """
        missileModel=self.identifyMissileModelToDraw(m)

        mo=MotionState(m["motion"]).transformTo(self.crs)
        pos=mo.pos()
        vel=mo.vel()
        vel/=np.linalg.norm(vel)
        size=self.getUIConfig("size","/"+m["team"]+"/missile/"+missileModel+"/guided-inactive/symbol")

        glColor4f(*self.getUIConfig("color","/"+m["team"]+"/missile/"+missileModel+"/guided-inactive/symbol"))
        
        grid_scale = np.array(self.getUIConfig("scale","/Default/grid"))
        pm0=self.regToSur(self.simToReg(pos,False))
        pm1=self.regToSur(self.simToReg(pos-vel*size*grid_scale/grid_scale[0],False))
        glLineWidth(3.0)
        if self.draw_xy:
            drawLine2D(pm0[0][0],pm0[0][1],pm1[0][0],pm1[0][1])
        if self.draw_yz:
            drawLine2D(pm0[1][0],pm0[1][1],pm1[1][0],pm1[1][1])
        glLineWidth(1.0)

    def drawCone(self, f, L, angle, N):
        """ fの正面を起点に距離L、頂角angleの球面錐を描画する。
            色設定はこの関数の外で行う。
            Lはm、angleはradで指定する。
        """
        ex=np.array(f["ex"])
        ey=np.array(f["ey"])
        ez=np.array(f["ez"])
        xy=sqrt(ex[0]*ex[0]+ex[1]*ex[1])
        pitchAngle=atan2(-ex[2],xy)
        eyHorizontal=np.cross(np.array([0.,0.,1.]),ex)
        sn=np.linalg.norm(eyHorizontal)
        if(abs(sn)<1e-6):
            eyHorizontal=np.array([0.,1.,0.])
        else:
            eyHorizontal/=sn
        exHorizontal=np.cross(eyHorizontal,np.array([0.,0.,1.]))
        exHorizontal/=np.linalg.norm(exHorizontal)
        cs=np.dot(ex,exHorizontal)
        if self.draw_xy:
            #XY平面プラス側
            psxy=[self.simToReg(f["pos"])]+[self.simToReg(
                f["pos"]+L*(cos(-angle+angle*2*i/N)*exHorizontal+sin(-angle+angle*2*i/N)*eyHorizontal))
                for i in range(N+1)]
            psxy=polygonRegionCut(psxy,[0.0,1.0],0)
            glBegin(GL_TRIANGLE_FAN)
            for p in psxy:
                tmp = Draw2D.convertPos(self.regToSur(p)[0])
                glVertex2d(tmp[0], tmp[1])
            glEnd()
            #XY平面マイナス側
            psxy=[self.simToReg(f["pos"])]+[self.simToReg(
                f["pos"]-L*(cos(-angle+angle*2*i/N)*exHorizontal*abs(sin(pitchAngle))+sin(-angle+angle*2*i/N)*eyHorizontal))
                for i in range(N+1)]
            psxy=polygonRegionCut(psxy,[0.0,1.0],0)
            glBegin(GL_TRIANGLE_FAN)
            for p in psxy:
                tmp = Draw2D.convertPos(self.regToSur(p)[0])
                glVertex2d(tmp[0], tmp[1])
            glEnd()
            exInYZ=(ex*np.array([0,1,1]))
            sn=np.linalg.norm(exInYZ)
            if(sn<1e-6):
                exInYZ=np.array([0.,1.,0.])
                absSinYaw=1.0
                eyz=np.array([0.,0.,1.])
            else:
                exInYZ=exInYZ/sn
                eyz=np.cross(exInYZ,np.array([1.,0.,0.]))
                absSinYaw=abs(asin(ex[0]))
        if self.draw_yz:
            #YZ平面プラス側
            psyz=[self.simToReg(f["pos"])]+[self.simToReg(
                f["pos"]+L*(cos(-angle+angle*2*i/N)*exInYZ+sin(-angle+angle*2*i/N)*eyz))
                for i in range(N+1)]
            psyz=polygonRegionCut(psyz,[0.0,1.0],2)
            glBegin(GL_TRIANGLE_FAN)
            for p in psyz:
                tmp = Draw2D.convertPos(self.regToSur(p)[1])
                glVertex2d(tmp[0], tmp[1])
            glEnd()
            #YZ平面マイナス側
            psyz=[self.simToReg(f["pos"])]+[self.simToReg(
                f["pos"]-L*(cos(-angle+angle*2*i/N)*exInYZ*absSinYaw+sin(-angle+angle*2*i/N)*eyz))
                for i in range(N+1)]
            psyz=polygonRegionCut(psyz,[0.0,1.0],2)
            glBegin(GL_TRIANGLE_FAN)
            for p in psyz:
                tmp = Draw2D.convertPos(self.regToSur(p)[1])
                glVertex2d(tmp[0], tmp[1])
            glEnd()

    def getPolicyNameToDisplay(self, frame: dict) -> str:
        """各陣営のAgentを表す名称を取得するもの。
            デフォルトではframe内で最初に見つかった所属Agentのpolicy名となる。
            派生クラスで適宜カスタマイズすること。

        Args:
            frame (dict): 現在のフレームデータ

        Returns:
            str: Agent名表示用の文字列
        """
        ruler=frame["ruler"]
        policyName = {}
        for team in [ruler["westSider"], ruler["eastSider"]]:
            policyName[team]=""
            for agentFullName,agent in frame["agents"].items():
                if agent["team"]==team:
                    _,_,policy=agentFullName.split(":")
                    if policy.startswith("Policy_"):
                        policyName[team]=policy[7:]
                    else:
                        policyName[team]=policy
                    break
        return policyName

    def getRewardStringToDisplay(self, team: str, frame: dict) -> str:
        """報酬を表示する際の文字列を取得するもの。

        Args:
            team str: 計算対象の陣営
            frame dict: 現在のフレームデータ

        Returns:
            str: 報酬表示用の文字列
        """
        def stringifyTotalRewards(tr):
            if isinstance(tr,float):
                return "{:.1f}".format(tr)
            else:
                if len(tr)==1:
                    return "{:.1f}".format(next(iter(tr.values())))
                else:
                    return "{"+",".join([k+":{:.1f}".format(v) for k,v in tr.items()])+"}"
        reward=[stringifyTotalRewards(frame["totalRewards"][agentFullName]) for agentFullName,agent in frame["agents"].items() if agent["team"]==team]
        return str(reward)


    def setTime(self, time):
        # 描画用の時刻を設定する。基本的にはdisplayの先頭でframe["time"]を与えればよい。
        # 座標変換等で他の時刻として計算したい場合に設定し直すものとする。
        self.time=time
    def calcRegion(self,frame):
        """x,y,z軸の描画対象範囲を計算する。
        """
        ruler=frame["ruler"]
        self.fgtrRegion=[
            np.min(np.r_[[self.transformToLocalCRS(f["pos"]) for f in frame["fighters"].values()]],0),
            np.max(np.r_[[self.transformToLocalCRS(f["pos"]) for f in frame["fighters"].values()]],0)
        ]
        ruler=frame["ruler"]
        self.fieldRegion=[
            np.array([-ruler["dOut"],-ruler["dLine"],-ruler["hLim"]]),
            np.array([ruler["dOut"],ruler["dLine"],0])
        ]
        self.bothRegion=[
            np.min(np.r_[[self.fgtrRegion[i],self.fieldRegion[i]]],0) for i in range(2)
        ]
        xyAspect=self.h_xy*Draw2D.height/(Draw2D.width*(1-self.w_margin*2))
        minR=np.array([(self.bothRegion[0][i] if self.regionType[i]=='fit' else self.fieldRegion[0][i]) for i in range(3)])
        maxR=np.array([(self.bothRegion[1][i] if self.regionType[i]=='fit' else self.fieldRegion[1][i]) for i in range(3)])
        mid=(minR+maxR)/2.0
        delta=(maxR-minR)/2.0
        minR=mid-delta*(self.fieldMargin+np.array([1,1,1]))
        maxR=mid+delta*(self.fieldMargin+np.array([1,1,1]))
        if(self.xyScaleType=='same'):
            if(xyAspect*(maxR[1]-minR[1])>(maxR[0]-minR[0])):
                mid=(maxR[0]+minR[0])/2.0
                delta=xyAspect*(maxR[1]-minR[1])/2.0
                minR[0]=mid-delta
                maxR[0]=mid+delta
            elif(xyAspect*(maxR[1]-minR[1])<(maxR[0]-minR[0])):
                mid=(maxR[1]+minR[1])/2.0
                delta=(maxR[0]-minR[0])/xyAspect/2.0
                minR[1]=mid-delta
                maxR[1]=mid+delta
        #slide
        slide=np.array([0.0,0.0,0.0])
        if self.slideRegion:
            for i in range(3):
                if self.fgtrRegion[0][i] < minR[i]:
                    slide[i] -= minR[i] - self.fgtrRegion[0][i]
                if self.fgtrRegion[1][i] > maxR[i]:
                    slide[i] += self.fgtrRegion[1][i] - maxR[i]    
        self.region=[minR+slide,maxR+slide]
    def transformToLocalCRS(self, value, coordinateType: CoordinateType=CoordinateType.POSITION_ABS):
        if isinstance(value, Coordinate):
            return value(self.crs)
        else:
            return self.crs.transformFrom(value, self.time, self.rootCRS, coordinateType)()
    def simToReg(self, sim, needTransformation: bool=True):#returns (xy,yz)
        """シミュレーション中の基準座標系で表された位置simを、
        x,y,z各軸を描画範囲で正規化し0〜1にして返す。
        """
        if needTransformation:
            sim = self.transformToLocalCRS(sim)
        return (sim-self.region[0])/(self.region[1]-self.region[0])
    def regToSur(self,reg):#'xy' or 'yz'
        """各軸正規化された位置regを、
        xy図、yz図上で対応する位置に表示されるような、画面全体のxy座標に変換して返す。
        """
        xy=np.array([
            Draw2D.width*(self.w_margin+reg[1]*(1.0-2*self.w_margin)),
            Draw2D.height*((self.h_margin*2+self.h_yz)+reg[0]*(self.h_xy))
        ])
        yz=np.array([
            Draw2D.width*(self.w_margin+reg[1]*(1.0-2*self.w_margin)),
            Draw2D.height*(self.h_margin+(1.0-reg[2])*self.h_yz)
        ])
        return np.array([xy,yz])
    def makeGrid(self,interval):
        """戦域の区切り線を描く。
        """
        lower=np.ceil(self.region[0]/interval)
        upper=np.floor(self.region[1]/interval)
        cnt=upper-lower
        for i in range(int(lower[0]),int(upper[0])+1):
            d=(i*interval[0]-self.region[0][0])/(self.region[1][0]-self.region[0][0])
            p=[self.regToSur(x)[0] for x in [[d,0,0],[d,1,1]]]
            drawLine2D(p[0][0],p[0][1],p[1][0],p[1][1])
        for i in range(int(lower[1]),int(upper[1])+1):
            d=(i*interval[1]-self.region[0][1])/(self.region[1][1]-self.region[0][1])
            p=[self.regToSur(x) for x in [[0,d,0],[1,d,1]]]
            drawLine2D(p[0][0][0],p[0][0][1],p[1][0][0],p[1][0][1])
            drawLine2D(p[0][1][0],p[0][1][1],p[1][1][0],p[1][1][1])
        for i in range(int(lower[2]),int(upper[2])+1):
            d=(i*interval[2]-self.region[0][2])/(self.region[1][2]-self.region[0][2])
            p=[self.regToSur(x)[1] for x in [[0,0,d],[1,1,d]]]
            drawLine2D(p[0][0],p[0][1],p[1][0],p[1][1])    
    def onEpisodeBegin(self, viewer_, header, frame):
        """エピソードの開始時(reset関数の最後)に呼ばれる。
        """
        self.viewer=viewer_
        self.isReplayMode = False
        self.wake = {}
        self.times = []
        self.wake_offset = 0
        self.from_indices = {}
        self.wake_from = {} # {name: [idx for time in frames]}
        self.rootCRS = self.viewer().createOrGetEntity(header["coordinateReferenceSystem"]["root"])
        if self.crsConfig is None:
            self.crs = self.viewer().createOrGetEntity(frame["ruler"]["localCRS"])
        else:
            self.crs = self.viewer().createOrGetEntity(self.crsConfig)
        assert self.crs is not None
    def onReplayBegin(self, viewer_, header, frames):
        """リプレイデータの読み込み完了時に呼ばれる。
        """
        self.viewer=viewer_
        self.isReplayMode = True
        self.wake = {}
        self.times = [f.get("elapsedTime",f["time"]) for f in frames]
        self.wake_offset = 0
        self.from_indices = {}
        self.wake_from = {} # {name: [idx for time in frames]}
        self.rootCRS = self.viewer().createOrGetEntity(header["coordinateReferenceSystem"]["root"])
        if self.crsConfig is None:
            self.crs = self.viewer().createOrGetEntity(frames[0]["ruler"]["localCRS"])
        else:
            self.crs = self.viewer().createOrGetEntity(self.crsConfig)
        assert self.crs is not None

        count = 0
        for count, frame in enumerate(frames):
            names = list(frame["fighters"].keys())
            for name, f in frame["fighters"].items():
                if name in self.wake:
                    self.wake[name].append(np.array(f["pos"]))
                    fighterModel = self.identifyFighterModelToDraw(f)
                    wake_length=self.getUIConfig("length","/"+f["team"]+"/fighter/"+fighterModel+"/trajectory")
                    while self.times[count]-self.times[self.from_indices[name]] > wake_length:
                        self.from_indices[name] = self.from_indices[name] + 1
                    self.wake_from[name].append(self.from_indices[name])
                else:
                    self.wake[name] = [None] * count + [np.array(f["pos"])]
                    self.wake_from[name] = [count]
                    self.from_indices[name] = count
            for name in self.wake:
                if not name in names:
                    self.wake[name].append(None)
                    self.from_indices[name] = count + 1
        lastFrame = {}
        self.lastAliveIndex = {}
        for frame in frames:
            for name, fgtr in frame["fighters"].items():
                if fgtr["isAlive"]:
                    lastFrame[name] = frame
                else:
                    if not name in self.lastAliveIndex:
                        self.lastAliveIndex[name] = lastFrame[name]["frameIndex"]
    def display(self,frame):
        """画面描画を行う。実際の描画処理はdisplayImplに記述する。
        """
        glClearColor(0.8, 0.8, 0.8, 1.0)
        glClear(GL_COLOR_BUFFER_BIT |GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_MODELVIEW)
        Draw2D.setShape(self.width,self.height)
        Draw2D.begin()
        self.displayImpl(frame)
        Draw2D.end()
        glFlush()
        pygame.display.flip()
        #pygame.display.update()
        #self.clock.tick(self.fps)
    def displayImpl(self,frame):
        """実際の描画を処理を記述する。
        """
        self.setTime(Time(frame["time"]))
        self.calcRegion(frame)
        horizontalScaleFactor=self.horizontalScaleFactors[self.horizontalUnit]
        vScaleFactor=self.vScaleFactors[self.vUnit]
        altScaleFactor=self.altScaleFactors[self.altUnit]
        ruler=frame["ruler"]
        bgColorF=self.getUIConfig("color","/Default/background")
        bgColorI=self.intColor(bgColorF)
        glColor4f(*bgColorF)
        p1,p2=self.regToSur([0,0,0]),self.regToSur([1,1,1])
        if self.draw_xy:
            fillRect2D(p1[0][0],p1[0][1],p2[0][0]-p1[0][0],p2[0][1]-p1[0][1])
        if self.draw_yz:
            fillRect2D(p1[1][0],p1[1][1],p2[1][0]-p1[1][0],p2[1][1]-p1[1][1])
        glColor4f(*self.getUIConfig("color","/Default/grid"))
        self.makeGrid(np.array([horizontalScaleFactor,horizontalScaleFactor,altScaleFactor])*np.array(self.getUIConfig("scale","/Default/grid")))
        glLineWidth(3.0)
        #防衛ライン、南北境界線
        nw=self.regToSur(self.simToReg(np.array([+ruler["dOut"],-ruler["dLine"],0]),False))
        ne=self.regToSur(self.simToReg(np.array([+ruler["dOut"],+ruler["dLine"],0]),False))
        sw=self.regToSur(self.simToReg(np.array([-ruler["dOut"],-ruler["dLine"],0]),False))
        se=self.regToSur(self.simToReg(np.array([-ruler["dOut"],+ruler["dLine"],0]),False))
        uw=self.regToSur(self.simToReg(np.array([+ruler["dOut"],-ruler["dLine"],-ruler["hLim"]]),False))
        ue=self.regToSur(self.simToReg(np.array([+ruler["dOut"],+ruler["dLine"],-ruler["hLim"]]),False))
        d=(ruler["dLine"]-self.region[0][1])/(self.region[1][1]-self.region[0][1])
        p=[self.regToSur(x) for x in [[0,d,0],[1,d,1]]]
        glColor4f(*self.getUIConfig("color","/Default/edge"))
        if self.draw_xy:
            drawLine2D(nw[0][0],nw[0][1],ne[0][0],ne[0][1])
            drawLine2D(sw[0][0],sw[0][1],se[0][0],se[0][1])
        if self.draw_yz:
            drawLine2D(nw[1][0],nw[1][1],ne[1][0],ne[1][1])
            drawLine2D(uw[1][0],uw[1][1],ue[1][0],ue[1][1])
        glColor4f(*self.getUIConfig("color","/"+ruler["eastSider"]+"/edge"))
        if self.draw_xy:
            drawLine2D(ne[0][0],ne[0][1],se[0][0],se[0][1])
        if self.draw_yz:
            drawLine2D(ne[1][0],ne[1][1],ue[1][0],ue[1][1])
        glColor4f(*self.getUIConfig("color","/"+ruler["westSider"]+"/edge"))
        if self.draw_xy:
            drawLine2D(nw[0][0],nw[0][1],sw[0][0],sw[0][1])
        if self.draw_yz:
            drawLine2D(nw[1][0],nw[1][1],uw[1][0],uw[1][1])
        glLineWidth(1.0)
        if self.getUIConfig("visibility","/Default/time"):
            colorF=self.getUIConfig("color","/Default/time")
            colorI=self.intColor(colorF)
            mode = self.getUIConfig("mode","/Default/time")
            assert mode in ["elapsed", "calendar", "both"], mode
            txt = "Time: "
            if mode in ["elapsed", "both"]:
                txt += "{:6.2f}s".format(frame.get("elapsedTime",frame["time"]))
            if mode == "both":
                txt += " ("
            if mode in ["calendar", "both"]:
                txt += str(Time(frame["time"]))
            if mode == "both":
                txt += ")"
            pos=(30*self.uiScaler,self.height-(24+10)*self.uiScaler)
            drawText2D(txt,pygame.font.Font('freesansbold.ttf',round(16*self.uiScaler)),pos,colorI,bgColorI)
        if self.getUIConfig("visibility","/Default/score"):
            colorF=self.getUIConfig("color","/Default/score")
            colorI=self.intColor(colorF)
            txt="Score: "+", ".join([team+"={:.2f}".format(score) for team,score in frame["scores"].items()])
            pos=(30*self.uiScaler,self.height-(24+40)*self.uiScaler)
            drawText2D(txt,pygame.font.Font('freesansbold.ttf',round(16*self.uiScaler)),pos,colorI,bgColorI)

        # policy名
        if self.getUIConfig("visibility","/Default/policy"):
            policyName = self.getPolicyNameToDisplay(frame)
            txt=policyName[ruler["westSider"]]
            colorF=self.getUIConfig("color","/"+ruler["westSider"]+"/policy")
            colorI=self.intColor(colorF)
            img=pygame.font.Font('freesansbold.ttf',round(16*self.uiScaler)).render(txt,True,bgColorI,None)
            pos=(self.width/2-20*self.uiScaler-img.get_width(),self.height-(24+10)*self.uiScaler)
            drawText2D(txt,pygame.font.Font('freesansbold.ttf',round(16*self.uiScaler)),pos,colorI,bgColorI)
            txt="vs"
            colorF=self.getUIConfig("color","/Default/policy")
            colorI=self.intColor(colorF)
            img=pygame.font.Font('freesansbold.ttf',round(16*self.uiScaler)).render(txt,True,bgColorI,None)
            pos=(self.width/2-img.get_width()/2,self.height-(24+10)*self.uiScaler)
            drawText2D(txt,pygame.font.Font('freesansbold.ttf',round(16*self.uiScaler)),pos,colorI,bgColorI)
            txt=policyName[ruler["eastSider"]]
            colorF=self.getUIConfig("color","/"+ruler["eastSider"]+"/policy")
            colorI=self.intColor(colorF)
            img=pygame.font.Font('freesansbold.ttf',round(16*self.uiScaler)).render(txt,True,bgColorI,None)
            pos=(self.width/2+20*self.uiScaler,self.height-(24+10)*self.uiScaler)
            drawText2D(txt,pygame.font.Font('freesansbold.ttf',round(16*self.uiScaler)),pos,colorI,bgColorI)

        #報酬
        positions = [
            (30*self.uiScaler,self.height-(24+130)*self.uiScaler),
            (30*self.uiScaler,self.height-(24+150)*self.uiScaler),
            (30*self.uiScaler,self.height-(24+170)*self.uiScaler),
            (30*self.uiScaler,self.height-(24+190)*self.uiScaler)
        ]
        for team in [ruler["westSider"], ruler["eastSider"]]:
            if self.getUIConfig("visibility","/"+team+"/reward"):
                colorF=self.getUIConfig("color","/"+team+"/policy")
                colorI=self.intColor(colorF)
                txt="Total reward of "+team
                pos=positions.pop(0)
                drawText2D(txt,pygame.font.Font('freesansbold.ttf',round(14*self.uiScaler)),pos,(team),bgColorI)
                txt=self.getRewardStringToDisplay(team,frame)
                pos=positions.pop(0)
                drawText2D(txt,pygame.font.Font('freesansbold.ttf',round(14*self.uiScaler)),pos,(team),bgColorI)

        #機体
        if not self.isReplayMode:
            self.times.append(frame.get("elapsedTime",frame["time"]))
            names = list(frame["fighters"].keys())
            for name in self.wake:
                if not name in names:
                    if frame["fighters"][name]["isAlive"]: # GodViewでは常に可観測なのでこの分岐には入らないはず
                        self.wake[name].append(None)
                        self.from_indices[name] = len(self.times) - 1
        leading={ruler["westSider"]:-ruler["dLine"],ruler["eastSider"]:-ruler["dLine"]}

        for fullName,f in frame["fighters"].items():
            if f["isAlive"]:
                leading[f["team"]]=max(np.dot(ruler["forwardAx"][f["team"]],self.transformToLocalCRS(f["pos"])[0:2]),leading[f["team"]])
                fighterModel = self.identifyFighterModelToDraw(f)
                # 軌跡
                if self.getUIConfig("visibility","/"+f["team"]+"/fighter/"+fighterModel+"/trajectory"):
                    if self.isReplayMode:
                        if f["isAlive"]:
                            index = frame["frameIndex"]
                        else:
                            index = self.lastAliveIndex[fullName]
                        wake_now = (
                            self.wake[fullName][self.wake_from[fullName][index]:index+1]
                            if self.wake[fullName][index] is not None
                            else []
                        )
                    else:
                        if f["isAlive"]:
                            count = len(self.times) - 1
                            if fullName in self.wake:
                                self.wake[fullName].append(np.array(f["pos"]))
                                wake_length=self.getUIConfig("length","/"+f["team"]+"/fighter/"+fighterModel+"/trajectory")
                                while self.times[count]-self.times[self.from_indices[fullName]] > wake_length:
                                    self.from_indices[fullName] = self.from_indices[fullName] + 1
                                self.wake_from[fullName].append(self.from_indices[fullName])
                            else:
                                self.wake[fullName] = [None] * count + [np.array(f["pos"])]
                                self.wake_from[fullName] = [count]
                                self.from_indices[fullName] = count
                        wake_now = (
                            self.wake[fullName][self.wake_from[fullName][-1]:]
                            if self.wake[fullName][-1] is not None
                            else []
                        )
                    if len(wake_now) > 1:
                        glColor4f(*self.getUIConfig("color","/"+f["team"]+"/fighter/"+fighterModel+"/trajectory"))
                        if self.draw_xy:
                            drawing = False
                            for pos in reversed(wake_now):
                                if pos is None:
                                    if drawing:
                                        glEnd()
                                        drawing = False
                                else:
                                    if not drawing:
                                        glBegin(GL_LINE_STRIP)
                                        drawing = True
                                    glVertex2d(*Draw2D.convertPos(self.regToSur(self.simToReg(pos))[0]))
                            if drawing:
                                glEnd()
                                drawing = False
                        if self.draw_yz:
                            drawing = False
                            for pos in reversed(wake_now):
                                if pos is None:
                                    if drawing:
                                        glEnd()
                                        drawing = False
                                else:
                                    if not drawing:
                                        glBegin(GL_LINE_STRIP)
                                        drawing = True
                                    glVertex2d(*Draw2D.convertPos(self.regToSur(self.simToReg(pos))[1]))
                            if drawing:
                                glEnd()
                                drawing = False

                # 機体シンボル
                if self.getUIConfig("visibility","/"+f["team"]+"/fighter/"+fighterModel+"/symbol"):
                    symbolType = self.getUIConfig("type","/"+f["team"]+"/fighter/"+fighterModel+"/symbol")
                    self.drawFighter[symbolType](f, frame)

                #誘導弾ロックオン範囲
                if self.getUIConfig("visibility","/"+f["team"]+"/fighter/"+fighterModel+"/lock-on range"):
                    glColor4f(*self.getUIConfig("color","/"+f["team"]+"/fighter/"+fighterModel+"/lock-on range"))
                    m=f["observables"]["weapon"]["missiles"][0]
                    L=m["spec"]["sensor"]["Lref"]
                    angle=m["spec"]["sensor"]["thetaFOR"]
                    N=16
                    self.drawCone(f, L, angle, N)

                #レーダ覆域
                if self.getUIConfig("visibility","/"+f["team"]+"/fighter/"+fighterModel+"/radar coverage"):
                    glColor4f(*self.getUIConfig("color","/"+f["team"]+"/fighter/"+fighterModel+"/radar coverage"))
                    radarSpec=f["observables"]["spec"]["sensor"].get("radar", None)
                    if radarSpec is not None:
                        L=radarSpec["Lref"]
                        angle=radarSpec.get("thetaFOR", np.pi/2)
                        N=16
                        self.drawCone(f, L, angle, N)

                # 機体諸元
                if self.getUIConfig("visibility","/"+f["team"]+"/fighter/"+fighterModel+"/status"):
                    pf=self.regToSur(self.simToReg(np.array(f["pos"])))
                    V=int(round(np.linalg.norm(np.array(f["vel"]))))
                    alt=int(self.rootCRS.getHeight(np.array(f["pos"]),self.time))
                    colorF=self.getUIConfig("color","/"+f["team"]+"/fighter/"+fighterModel+"/status")
                    colorI=self.intColor(colorF)
                    fTxt=f["name"]+": v="+str(round(V/vScaleFactor))+self.vUnit+", m="+str(f["remMsls"])+",h="+str(round(alt/altScaleFactor))+self.altUnit
                    if self.draw_xy:
                        drawText2D(fTxt,self.font,(pf[0][0]+10,pf[0][1]+0),colorI)
                    if self.draw_yz:
                        drawText2D(fTxt,self.font,(pf[1][0]+10,pf[1][1]+0),colorI)

        #誘導弾
        for m in frame["missiles"].values():
            if(m["isAlive"] and m["hasLaunched"]):
                missileModel=self.identifyMissileModelToDraw(m)
                if(m["mode"]==Missile.Mode.SELF.name):
                    missileMode = "self"
                elif(m["mode"]==Missile.Mode.GUIDED.name):
                    if(m["sensor"]["isActive"]):
                        missileMode = "guided-active"
                    else:
                        missileMode = "guided-inactive"
                else:
                    if(m["sensor"]["isActive"]):
                        missileMode = "memory-active"
                    else:
                        missileMode = "memory-inactive"

                #シンボル
                if self.getUIConfig("visibility","/"+m["team"]+"/missile/"+missileModel+"/"+missileMode+"/symbol"):
                    symbolType = self.getUIConfig("type","/"+m["team"]+"/missile/"+missileModel+"/"+missileMode+"/symbol")
                    self.drawMissile[symbolType](m, frame)

                #目標
                if self.getUIConfig("visibility","/"+m["team"]+"/missile/"+missileModel+"/"+missileMode+"/target"):
                    glColor4f(*self.getUIConfig("color","/"+m["team"]+"/missile/"+missileModel+"/"+missileMode+"/target"))
                    pm=self.regToSur(self.simToReg(m["pos"]))
                    tm=self.regToSur(self.simToReg(m["estTPos"]))
                    if self.draw_xy:
                        fillCircle2D(tm[0],2,8)
                        drawLine2D(pm[0][0],pm[0][1],tm[0][0],tm[0][1])
                    if self.draw_yz:
                        fillCircle2D(tm[1],2,8)
                        drawLine2D(pm[1][0],pm[1][1],tm[1][0],tm[1][1])

        # 前線の中心
        if self.getUIConfig("visibility","/Default/frontline"):
            colorF=self.getUIConfig("color","/Default/frontline")
            colorI=self.intColor(colorF)
            glColor4f(*colorF)
            glLineWidth(3.0)
            center=(leading[ruler["westSider"]]-leading[ruler["eastSider"]])/2
            n=self.regToSur(self.simToReg(np.array([+ruler["dOut"],center,0]),False))
            s=self.regToSur(self.simToReg(np.array([-ruler["dOut"],center,0]),False))
            if self.draw_xy:
                drawLine2D(n[0][0],n[0][1],s[0][0],s[0][1])
                txt=str(round(center/1000.0))
                drawText2D(txt,self.font,(n[0][0],s[0][1]-20),colorI)
            glLineWidth(1.0)
