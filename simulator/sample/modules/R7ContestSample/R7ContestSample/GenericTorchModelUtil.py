# Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
"""torch.nn.Moduleをjsonベースで柔軟に定義できるようにするためのユーティリティ
"""
from collections import defaultdict
import copy
import torch
import torch.nn as nn

from .SetTransformer import SAB,ISAB,PMA,PMAWithQuery

class GenericLayers(torch.nn.Module):
    """jsonベースのconfigから様々なNNを生成するための基本クラス。
    configは以下の形式とする。
    {
        "input_shape": list, #入力テンソルの形(バッチサイズを除く)
        "layers": list[tuple[str,dict]], #各構成要素を生成するためのタプルのリスト。
                                         #タプルの第1要素はクラス名。第2要素はコンストラクタ引数のうち、入力テンソルの形に関する引数を除いたもの。
    }
    途中の各構成要素の入出力テンソルの形状は自動計算される。
    """
    def __init__(self,config):
        super().__init__()
        self.config=copy.deepcopy(config)
        self.input_shape=config['input_shape']
        self.layerCount=defaultdict(lambda:0)
        self.layers=[]
        self.forwardConfigs=[]
        dummy=torch.zeros([2,]+list(self.input_shape))
        for layerType,layerConfig in config['layers']:
            layer, dummy = self.makeLayer(layerType, layerConfig, dummy)
            self.layerCount[layerType]+=1
            layerName=layerType+str(self.layerCount[layerType])
            setattr(self,layerName,layer)
            self.layers.append(layer)
        self.output_shape=dummy.shape

    def makeLayer(self, layerType: str , layerConfig: dict, dummy):
        if(layerType=='Conv1d'):
            layer=torch.nn.Conv1d(in_channels=dummy.shape[1],**layerConfig)
        elif(layerType=='BatchNorm1d'):
            layer=torch.nn.BatchNorm1d(num_features=dummy.shape[1],**layerConfig)
        elif(layerType=='transpose'):
            def gen(dim0,dim1):
                def func(x):
                    return torch.transpose(x,dim0,dim1)
                return func
            layer=gen(layerConfig[0],layerConfig[1])
        elif(layerType=='mean'):
            def gen(**layerConfig):
                def func(x):
                    return torch.mean(x,**layerConfig)
                return func
            layer=gen(**layerConfig)
        elif(layerType=='Conv2d'):
            layer=torch.nn.Conv2d(in_channels=dummy.shape[1],**layerConfig)
        elif(layerType=='AdaptiveAvgPool2d'):
            layer=torch.nn.AdaptiveAvgPool2d(**layerConfig)
        elif(layerType=='AdaptiveMaxPool2d'):
            layer=torch.nn.AdaptiveMaxPool2d(**layerConfig)
        elif(layerType=='AvgPool2d'):
            layer=torch.nn.AvgPool2d(**layerConfig)
        elif(layerType=='MaxPool2d'):
            layer=torch.nn.MaxPool2d(**layerConfig)
        elif(layerType=='BatchNorm2d'):
            layer=torch.nn.BatchNorm2d(num_features=dummy.shape[1],**layerConfig)
        elif(layerType=='Conv3d'):
            layer=torch.nn.Conv3d(in_channels=dummy.shape[1],**layerConfig)
        elif(layerType=='AvgPool3d'):
            layer=torch.nn.AvgPool3d(**layerConfig)
        elif(layerType=='MaxPool3d'):
            layer=torch.nn.MaxPool3d(**layerConfig)
        elif(layerType=='BatchNorm3d'):
            layer=torch.nn.BatchNorm3d(num_features=dummy.shape[1],**layerConfig)
        elif(layerType=='Flatten'):
            layer=torch.nn.Flatten(**layerConfig)
        elif(layerType=='Linear'):
            layer=torch.nn.Linear(dummy.shape[-1],**layerConfig)
        elif(layerType=='ReLU'):
            layer=torch.nn.ReLU()
        elif(layerType=='LeakyReLU'):
            layer=torch.nn.LeakyReLU()
        elif(layerType=='SiLU'):
            layer=torch.nn.SiLU()
        elif(layerType=='Tanh'):
            layer=torch.nn.Tanh()
        elif(layerType=='ResidualBlock'):
            layer=ResidualBlock(dummy.shape[1:],layerConfig)
        elif(layerType=='Concatenate'):
            layer=Concatenate(dummy.shape[1:],layerConfig)
        elif(layerType=='PositionalEncoding'):
            layer=PositionalEncoding(dummy.shape[1:],layerConfig)
        dummy=layer(dummy)
        return layer, dummy

    def forward(self,x):
        self.batch_size=x.shape[0]
        for layer in self.layers:
            x=layer(x)
        return x

class ResidualBlock(torch.nn.Module):
    """残差ブロック(y=x+h(x))。configは以下の形式とする。
    {
        "input_shape": list, #入力テンソルの形(バッチサイズを除く)
        "block": dict, #残差を計算するためのGenericLayersを生成するためのconfig。
    }
    """
    def __init__(self,input_shape,blockConfig):
        super().__init__()
        self.config=copy.deepcopy(blockConfig)
        self.config["input_shape"]=input_shape
        self.block=GenericLayers(self.config)
    def forward(self,x):
        return x+self.block(x)

class Concatenate(torch.nn.Module):
    """入力を複数のブロックに供給してその結果を結合するブロック。configは以下の形式とする。
    {
        "input_shape": list, #入力テンソルの形(バッチサイズを除く)
        "blocks": list[dict], #分岐・結合対象の各ブロック(GenericLayers)を生成するためのconfig。
        "dim": int, #結合対象の次元。省略時は-1(末尾)となる。

    }
    """
    def __init__(self,input_shape,config):
        super().__init__()
        self.config=copy.deepcopy(config)
        self.catDim=self.config.get("dim",-1)
        self.blocks=[]
        cnt=0
        for c in self.config["blocks"]:
            c["input_shape"]=input_shape
            block=GenericLayers(c)
            setattr(self,"Block"+str(cnt),block)
            self.blocks.append(block)
            cnt+=1
    def forward(self,x):
        return torch.cat([b(x) for b in self.blocks],dim=self.catDim)

class PositionalEncoding(torch.nn.Module):
    def __init__(self,dim,max_len=10000):
        super().__init__()
        pe=torch.zeros(max_len,dim)
        position=(torch.arange(0,max_len,dtype=torch.float).float()/max_len).unsqueeze(1)
        div_term=torch.exp(torch.arange(0,dim,2).float()/dim)
        pe[:,0::2]=torch.sin(position*div_term)
        pe[:,1::2]=torch.cos(position*div_term)
        pe=pe.unsqueeze(0).transpose(0,1)
        self.register_buffer('pe',pe)
    def forward(self,x):
        return x+self.pe[:x.size(0),:]

class TransformerEncoder(GenericLayers):
    # batch_firstは原則としてTrueとすること
    def __init__(self,config):
        super().__init__(config)

    def makeLayer(self, layerType: str , layerConfig: dict, dummy):
        if(layerType=='TransformerEncoderLayer'):
            if not 'batch_first' in layerConfig:
                layerConfig['batch_first']=True
            layer=torch.nn.TransformerEncoderLayer(d_model=dummy.shape[-1],**layerConfig)
            dummy=layer(dummy)
            return layer, dummy
        else:
            return super().makeLayer(layerType, layerConfig, dummy)

    def forward(self,src,src_mask=None,src_key_padding_mask=None):
        for layer in self.layers:
            if(isinstance(layer,torch.nn.TransformerEncoderLayer)):
                src=layer(src,src_mask,src_key_padding_mask)
            else:
                src=layer(src)
        return src

class TransformerDecoder(GenericLayers):
    # batch_firstは原則としてTrueとすること
    def __init__(self,config):
        super().__init__(config)

    def makeLayer(self, layerType: str , layerConfig: dict, dummy):
        if(layerType=='TransformerEncoderLayer'):
            if not 'batch_first' in layerConfig:
                layerConfig['batch_first']=True
            layer=torch.nn.TransformerEncoderLayer(d_model=dummy.shape[-1],**layerConfig)
            dummy=layer(dummy)
            return layer, dummy
        elif(layerType=='TransformerDecoderLayer'):
            if not 'batch_first' in layerConfig:
                layerConfig['batch_first']=True
            layer=torch.nn.TransformerDecoderLayer(d_model=dummy.shape[-1],**layerConfig)
            dummy=layer(dummy,dummy)
            return layer, dummy
        else:
            return super().makeLayer(layerType, layerConfig, dummy)

    def forward(self,tgt,memory,tgt_mask=None,memory_mask=None,tgt_key_padding_mask=None,memory_key_padding_mask=None):
        for layer in self.layers:
            if(isinstance(layer,torch.nn.TransformerEncoderLayer)):
                tgt=layer(tgt,tgt_mask,tgt_key_padding_mask)
            elif(isinstance(layer,torch.nn.TransformerDecoderLayer)):
                tgt=layer(tgt,memory,tgt_mask,memory_mask,tgt_key_padding_mask,memory_key_padding_mask)
            else:
                tgt=layer(tgt)
        return tgt

class SetTransformer(GenericLayers):
    def __init__(self,config):
        self.hasPMA=False
        self.num_seeds=None
        super().__init__(config)

    def makeLayer(self, layerType: str , layerConfig: dict, dummy):
        if(layerType=='SAB'):
            if not 'batch_first' in layerConfig:
                layerConfig['batch_first']=True
            layer=SAB(embed_dim=dummy.shape[-1],**layerConfig)
            dummy=layer(dummy)
            return layer, dummy
        elif(layerType=='ISAB'):
            if not 'batch_first' in layerConfig:
                layerConfig['batch_first']=True
            layer=ISAB(embed_dim=dummy.shape[-1],**layerConfig)
            dummy=layer(dummy)
            return layer, dummy
        elif(layerType=='PMA'):
            if not 'batch_first' in layerConfig:
                layerConfig['batch_first']=True
            if self.hasPMA:
                raise ValueError("SetTransformer block can have only one PMA.")
            layer=PMA(embed_dim=dummy.shape[-1],**layerConfig)
            self.num_seeds=layer.num_seeds
            dummy=layer(dummy)
            self.hasPMA=True
            return layer, dummy
        else:
            return super().makeLayer(layerType, layerConfig, dummy)

    def forward(self,src,src_key_padding_mask=None):
        for layer in self.layers:
            if(
                isinstance(layer,SAB) or
                isinstance(layer,ISAB) or
                isinstance(layer,PMA)
            ):
                src=layer(src,src_key_padding_mask)
            else:
                src=layer(src)
        return src

class SetTransformerWithQuery(GenericLayers):
    def __init__(self,config):
        self.hasPMA=False
        super().__init__(config)

    def makeLayer(self, layerType: str , layerConfig: dict, dummy):
        if(layerType=='SAB'):
            if not 'batch_first' in layerConfig:
                layerConfig['batch_first']=True
            layer=SAB(embed_dim=dummy.shape[-1],**layerConfig)
            dummy=layer(dummy)
            return layer, dummy
        elif(layerType=='ISAB'):
            if not 'batch_first' in layerConfig:
                layerConfig['batch_first']=True
            layer=ISAB(embed_dim=dummy.shape[-1],**layerConfig)
            dummy=layer(dummy)
            return layer, dummy
        elif(layerType=='PMAWithQuery'):
            if not 'batch_first' in layerConfig:
                layerConfig['batch_first']=True
            if self.hasPMA:
                raise ValueError("SetTransformer block can have only one PMAWithQuery.")
            layer=PMAWithQuery(embed_dim=dummy.shape[-1],**layerConfig)
            dummy=layer(dummy)
            self.hasPMA=True
            return layer, dummy
        else:
            return super().makeLayer(layerType, layerConfig, dummy)

    def forward(self,tgt,memory,memory_key_padding_mask=None):
        for layer in self.layers:
            if(
                isinstance(layer,SAB) or
                isinstance(layer,ISAB)
            ):
                memory=layer(memory,memory_key_padding_mask)
            elif(isinstance(layer,PMAWithQuery)):
                tgt=layer(tgt,memory,memory_key_padding_mask)
            else:
                tgt=layer(tgt)
        return tgt
