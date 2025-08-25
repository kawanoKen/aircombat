# Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
"""
順列不変性(permutation invariance)と順列同変性(permutation equivariance)をもったSet Transformer

[Lee 2019] Lee, J., et al. "Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks."

著者による実装例（https://github.com/juho-lee/set_transformer）も提供されているが、
maskに対応していないため、torch.nnのMultiHeadAttentionを用いて実装し直す。
SAB,ISAB,PMAのforward()の引数key_padding_maskとして無効トークンのmaskを与えられるようにしている。

なお、これらの性質は、torch.nnのTransformerEncoderLayerとTransformerDecoderLayerを用いても実現できる。

"""
import torch
import torch.nn as nn

class SAB(nn.Module):
    def __init__(self, embed_dim, num_heads, **kwargs):
        """
            kwargのうち特にbatch_first,add_bias_kv,add_zero_attnは用途に応じて適切に選択すること。
            特に、長さゼロ(全て無効トークン)のシーケンスが入力される可能性があるならば、add_zero_attnをTrueにする必要がある。
        """
        super(SAB, self).__init__()
        self.batch_first = kwargs.get('batch_first', True)
        kwargs['batch_first']=self.batch_first
        self.mab = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            **kwargs
        )
        #self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X, key_padding_mask=None):
        """
            key_padding_maskは、サイズ(batch_size, embed_dim)のテンソルで与える。
            bool型の場合はTrueのトークンが無視され、float型の場合はattention weightに値がそのまま加算される(-infを与えると無視)。
        """
        ret = self.mab(X, X, X,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            attn_mask=None,
            is_causal=False
        )[0]
        return ret

class ISAB(nn.Module):
    def __init__(self, embed_dim, num_heads, num_inds, **kwargs):
        """
            kwargのうち特にbatch_first,add_bias_kv,add_zero_attnは用途に応じて適切に選択すること。
            特に、長さゼロ(全て無効トークン)のシーケンスが入力される可能性があるならば、add_zero_attnをTrueにする必要がある。
        """
        super(ISAB, self).__init__()
        self.batch_first = kwargs.get('batch_first', True)
        kwargs['batch_first']=self.batch_first
        if self.batch_first:
            self.I = nn.Parameter(torch.Tensor(1, num_inds, embed_dim))
        else:
            self.I = nn.Parameter(torch.Tensor(num_inds, 1, embed_dim))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            **kwargs
        )
        self.mab1 = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            **kwargs
        )

    def forward(self, X, key_padding_mask=None):
        """
            key_padding_maskは、サイズ(batch_size, embed_dim)のテンソルで与える。
            bool型の場合はTrueのトークンが無視され、float型の場合はattention weightに値がそのまま加算される(-infを与えると無視)。
        """
        if self.batch_first:
            I = self.I.repeat(X.size(0), 1, 1)
        else:
            I = self.I.repeat(1, X.size(1), 1)
        H = self.mab0(
            I,
            X,
            X,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            attn_mask=None,
            is_causal=False
        )[0]
        ret = self.mab1(
            X,
            H,
            H,
            key_padding_mask=None,
            need_weights=False,
            attn_mask=None,
            is_causal=False
        )[0]
        return ret

class PMA(nn.Module):
    def __init__(self, embed_dim, num_heads, num_seeds, **kwargs):
        """
            kwargのうち特にbatch_first,add_bias_kv,add_zero_attnは用途に応じて適切に選択すること。
            特に、長さゼロ(全て無効トークン)のシーケンスが入力される可能性があるならば、add_zero_attnをTrueにする必要がある。
        """
        super(PMA, self).__init__()
        self.num_seeds=num_seeds
        self.batch_first = kwargs.get('batch_first', True)
        kwargs['batch_first']=self.batch_first
        if self.batch_first:
            self.S = nn.Parameter(torch.Tensor(1, num_seeds, embed_dim))
        else:
            self.S = nn.Parameter(torch.Tensor(num_seeds, 1, embed_dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            **kwargs
        )

    def forward(self, X, key_padding_mask=None):
        """
            key_padding_maskは、サイズ(batch_size, embed_dim)のテンソルで与える。
            bool型の場合はTrueのトークンが無視され、float型の場合はattention weightに値がそのまま加算される(-infを与えると無視)。
        """
        if self.batch_first:
            S = self.S.repeat(X.size(0), 1, 1)
        else:
            S = self.S.repeat(1, X.size(1), 1)
        ret = self.mab(
            S,
            X,
            X,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            attn_mask=None,
            is_causal=False
        )[0]
        return ret

class PMAWithQuery(nn.Module):
    """
        クエリを外から与えられるようにしたPMA亜種
    """
    def __init__(self, embed_dim, num_heads, **kwargs):
        """
            kwargのうち特にbatch_first,add_bias_kv,add_zero_attnは用途に応じて適切に選択すること。
            特に、長さゼロ(全て無効トークン)のシーケンスが入力される可能性があるならば、add_zero_attnをTrueにする必要がある。
        """
        super(PMAWithQuery, self).__init__()
        self.mab = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            **kwargs
        )

    def forward(self, Q, X, key_padding_mask=None):
        """
            key_padding_maskは、サイズ(batch_size, embed_dim)のテンソルで与える。
            bool型の場合はTrueのトークンが無視され、float型の場合はattention weightに値がそのまま加算される(-infを与えると無視)。
        """
        ret = self.mab(
            Q,
            X,
            X,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            attn_mask=None,
            is_causal=False
        )[0]
        return ret
