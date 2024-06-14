# File modified by authors of InstructPix2Pix from original (https://github.com/CompVis/stable-diffusion).
# See more details in LICENSE.

from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
import sys
sys.path.append('.')

from stable_diffusion.ldm.modules.diffusionmodules.util import checkpoint


def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):  # val ? val : d
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    """
    # The input x is first passed through the linear layer self.proj.
    The output of the linear layer is then divided into two equal chunks
    along the last dimension (dim=-1), which serve as the input x and a
    gate. The gating mechanism is applied using the GELU activation
    function on the gate and then multiplied element-wise with the x.
    """
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x+h_


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads  # head total dim
        # if context_dim is None, this is a self-attention,
        # and context_dim should be exactly the same as query_dim (input dim)
        context_dim = default(context_dim, query_dim)  # context_dim ? context_dim : query_dim
        self.scale = dim_head ** -0.5  # 1/\sqrt(d)
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

        self.prompt_to_prompt = False

    def forward(self, x, context=None, mask=None):
        is_self_attn = context is None
        # print("CrossAttention", "input x shape", x.shape)
        h = self.heads
        # print("CrossAttention", "h shape", h)
        q = self.to_q(x)
        # print("CrossAttention", "q shape", q.shape)

        # if context is None, then it is self-attention, otherwise cross-attention
        context = default(context, x)
        k = self.to_k(context)
        # print("CrossAttention", "k shape", k.shape)
        v = self.to_v(context)
        # print("CrossAttention", "v shape", v.shape)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        # print("After mapping", "q shape", q.shape, "k shape", k.shape, "v shape", v.shape)

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        # print("CrossAttention", "sim shape", sim.shape)

        """
        When self.prompt_to_prompt is set to True and the layer is
        performing self-attention, it duplicates the attention maps
        for the first half of the batch (effectively ignoring the
        second half of the batch). The code comment suggests this
        might be used in a scenario where you have 4 elements in the
        batch with a specific structure: {conditional, unconditional} x
        {prompt 1, prompt 2}. For self-attention, the model is essentially
        treating prompt 1 and prompt 2 as if they have the same attention map.
        """
        if is_self_attn and self.prompt_to_prompt:
            # Unlike the original Prompt-to-Prompt which uses cross-attention layers,
            # we copy attention maps for self-attention layers.
            # There must be 4 elements in the batch: {conditional, unconditional} x {prompt 1, prompt 2}
            assert x.size(0) == 4
            sims = sim.chunk(4)
            sim = torch.cat((sims[0], sims[0], sims[2], sims[2]))
        """
        In the context of attention mechanisms, a mask is often used to prevent 
        certain positions in the input from attending to other specific positions 
        in the input. This is usually done to enforce certain structural constraints, 
        like preventing future positions from being attended to in a sequence (to 
        ensure causality in autoregressive models), or masking out padding positions 
        in a sequence.
        """
        if exists(mask):
            """
            # mask is used to selectively ignore or "mask" certain parts of the input in 
            the attention calculation. This is done by setting the mask value to be False 
            at positions we want to ignore. Then, these positions get filled with a very 
            negative value (effectively negative infinity when used in a softmax function), 
            ensuring that they contribute almost nothing in the subsequent softmax operation 
            that calculates attention weights.
            """
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        attn = sim.softmax(dim=-1)
        # print("CrossAttention", "attn shape", attn.shape)

        out = einsum('b i j, b j d -> b i d', attn, v)
        # print("CrossAttention", "out shape", out.shape)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        # print("CrossAttention", "out shape", out.shape)
        out = self.to_out(out)
        # print("CrossAttention", "after out out shape", out.shape)
        return out


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True,
                 disable_self_attn=False):
        super().__init__()
        self.disable_self_attn = disable_self_attn
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
                                    context_dim=context_dim if self.disable_self_attn else None)  # is a self-attention if not self.disable_self_attn
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None):
        x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None,
                 disable_self_attn=False):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)  # GroupNormalize, by default 32 groups

        self.proj_in = nn.Conv2d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim,
                                   disable_self_attn=disable_self_attn)
                for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        # context: [bs, 77, 768]
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        x = self.proj_out(x)
        return x + x_in