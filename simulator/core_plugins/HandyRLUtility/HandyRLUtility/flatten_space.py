# Copyright (c) 2021-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
#
# gymnasium.spacesに属する値を一次元のベクトルにflattenしたり、逆に一次元のベクトルからunflattenするユーティリティ
#
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn

def flattened_space_dim(space,one_hot=True):
    """
    Calculates size of flattened space for given space.

    Args:
        space (gym.spaces.Space):
        one_hot (bool): whether to count discrete component as one-hot vector or label
        
    Returns:
        int:
    """
    if(isinstance(space,gym.spaces.Discrete)):
        if(one_hot):
            return space.n
        else:
            return 1
    elif(isinstance(space,gym.spaces.MultiDiscrete)):
        if(one_hot):
            return sum(space.nvec)
        else:
            return len(space.nvec)
    elif(isinstance(space,gym.spaces.MultiBinary)):
        if(one_hot):
            return 2*space.n
        else:
            return space.n
    elif(isinstance(space,gym.spaces.Box)):
        return int(np.product(space.shape))
    elif(isinstance(space,gym.spaces.Tuple)):
        return sum([flattened_space_dim(sub,one_hot) for sub in space])
    elif(isinstance(space,gym.spaces.Dict)):
        return sum([flattened_space_dim(sub,one_hot) for sub in space.spaces.values()])

def flatten_value(value, space, one_hot=True):
    """
    Returns flattened value for given space.

    Args:
        value: Tensor or a container of Tensors to be flattened.
        space (gym.spaces.Space):
        one_hot (bool): whether to count discrete component as one-hot vector or label
        
    Returns:
        int:
    """
    # value = (..., B, T, P or 1, ...)
    if(isinstance(space,gym.spaces.Discrete)):
        # float -> one-hot, int -> label
        if(one_hot):
            if torch.is_floating_point(value):
                #already one-hot
                return value
            else:
                #label to one-hot
                return nn.functional.one_hot(value,space.n)
        else:
            if torch.is_floating_point(value):
                #one-hot input
                return torch.argmax(value,dim=-1)
            else:
                #already label
                return value.unsqueeze(-1)
    elif(isinstance(space,gym.spaces.MultiDiscrete)):
        if(one_hot):
            if torch.is_floating_point(value):
                #already one-hot
                return value
            else:
                #label to one-hot
                return torch.cat([nn.functional.one_hot(v,n).flatten(-2,-1) for v,n in zip(torch.split(value,1,dim=-1),space.nvec)],dim=-1)
        else:
            if torch.is_floating_point(value):
                #one-hot input
                return torch.stack([v.argmax(dim=-1) for v in torch.split(value,tuple(space.nvec),dim=-1)],dim=-1)
            else:
                #already label
                return value
    elif(isinstance(space,gym.spaces.MultiBinary)):
        if(one_hot):
            if torch.is_floating_point(value):
                #already one-hot
                return value
            else:
                #label to one-hot
                return nn.functional.one_hot(value,2).flatten(-2,-1)
        else:
            if torch.is_floating_point(value):
                #one-hot input
                return torch.stack([v.argmax(dim=-1) for v in torch.split(value,2,dim=-1)],dim=-1)
            else:
                #already label
                return value
    elif(isinstance(space,gym.spaces.Box)):
        return torch.flatten(value,-len(space.shape))
    elif(isinstance(space,gym.spaces.Tuple)):
        return torch.cat([flatten_value(a,s,one_hot) for a,s in zip(value,space)],dim=-1)
    elif(isinstance(space,gym.spaces.Dict)):
        return torch.cat([flatten_value(a,s,one_hot) for a,s in zip(value.values(),space.values())],dim=-1)

def unflatten_value(value, space, one_hot=True):
    """
    Returns unflattened value for given space.

    Args:
        value: Tensor or a container of Tensors to be flattened.
        space (gym.spaces.Space):
        one_hot (bool): whether to count discrete component as one-hot vector or label
        
    Returns:
        int:
    """
    # value = (..., B, T, P or 1, ...)
    dtype=torch.from_numpy(np.zeros(1,space.dtype)).dtype
    if(isinstance(space,gym.spaces.Discrete)):
        if(one_hot):
            # from one-hot to label
            return torch.argmax(value, dim=-1).to(dtype)
        else:
            # from label to label
            return value.squeeze(-1).to(dtype)
    elif(isinstance(space,gym.spaces.MultiDiscrete)):
        if(one_hot):
            # from one-hot to label
            return torch.stack([v.argmax(dim=-1) for v in torch.split(value,tuple(space.nvec),dim=-1)],dim=-1).to(dtype)
        else:
            # from label to label
            return value.to(dtype)
    elif(isinstance(space,gym.spaces.MultiBinary)):
        if(one_hot):
            # from one-hot to label
            return torch.stack([v.argmax(dim=-1) for v in torch.split(value,2,dim=-1)],dim=-1).to(dtype)
        else:
            # from label to label
            return value.to(dtype)
    elif(isinstance(space,gym.spaces.Box)):
        return value.unflatten(-1,space.shape).to(dtype)
    elif(isinstance(space,gym.spaces.Tuple)):
        splitted_value = torch.split(
            value,
            [flattened_space_dim(s, one_hot) for s in space.spaces],
            dim=-1
        )
        return tuple([
            unflatten_value(
                a,
                s,
                one_hot
            ) for a, s in zip(splitted_value, space.spaces)
        ])
    elif(isinstance(space,gym.spaces.Dict)):
        splitted_value = torch.split(
            value,
            [flattened_space_dim(s, one_hot) for s in space.spaces.values()],
            dim=-1
        )
        return {
            k: unflatten_value(
                a,
                s,
                one_hot
            ) for (k, s), a in zip(space.spaces.items(), splitted_value)
        }
