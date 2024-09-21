
import comfy
import torch
from einops import rearrange
from torch import Tensor
from comfy.ldm.modules.attention import optimized_attention
import comfy.model_management


def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor) -> Tensor:
    q, k = apply_rope(q, k, pe)

    heads = q.shape[1]
    x = optimized_attention(q, k, v, heads, skip_reshape=True)
    return x


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    if comfy.model_management.is_device_mps(pos.device) or comfy.model_management.is_intel_xpu():
        device = torch.device("cpu")
    else:
        device = pos.device

    scale = torch.linspace(0, (dim - 2) / dim, steps=dim //
                           2, dtype=torch.float64, device=device)
    omega = 1.0 / (theta**scale)
    out = torch.einsum(
        "...n,d->...nd", pos.to(dtype=torch.float32, device=device), omega)
    out = torch.stack([torch.cos(out), -torch.sin(out),
                      torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.to(dtype=torch.float32, device=pos.device)


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor):
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    # print("freqs_cis: ", freqs_cis.shape)
    freqs_cis_q = freqs_cis[:, :, :xq_.shape[2], :, :]
    xq_out = freqs_cis_q[..., 0] * xq_[..., 0] + \
        freqs_cis_q[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


def single_block_forward(self, x: Tensor, vec: Tensor, pe: Tensor, layer_id: int, ctx) -> Tensor:
    txt_size = 512

    bs = x.shape[0]
    mod, _ = self.modulation(vec)
    x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
    qkv, mlp = torch.split(self.linear1(
        x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)

    q, k, v = qkv.view(qkv.shape[0], qkv.shape[1],
                       3, self.num_heads, -1).permute(2, 0, 3, 1, 4)

    if True:
        txt_k = k[:, :, :txt_size, :]
        img_k = k[:, :, txt_size:, :]
        img_k = img_k.permute(0, 2, 1, 3).reshape(
            1, bs * img_k.shape[2], img_k.shape[1], img_k.shape[3]).permute(0, 2, 1, 3).repeat(bs, 1, 1, 1)
        k = torch.cat((txt_k, img_k), dim=2)

        txt_v = v[:, :, :txt_size, :]
        img_v = v[:, :, txt_size:, :]
        img_v = img_v.permute(0, 2, 1, 3).reshape(
            1, bs * img_v.shape[2], img_v.shape[1], img_v.shape[3]).permute(0, 2, 1, 3).repeat(bs, 1, 1, 1)
        v = torch.cat((txt_v, img_v), dim=2)

        txt_pe = pe[:, :, :txt_size, :, :, :]
        img_pe = pe[:, :, txt_size:, :, :, :]
        img_pe = img_pe.permute(0, 2, 1, 3, 4, 5).reshape(
            1, bs * img_pe.shape[2], img_pe.shape[1], img_pe.shape[3], img_pe.shape[4], img_pe.shape[5]).permute(0, 2, 1, 3, 4, 5).repeat(bs, 1, 1, 1, 1, 1)
        pe = torch.cat((txt_pe, img_pe), dim=2)

    q, k = self.norm(q, k, v)

    attn = attention(q, k, v, pe=pe)
    # compute activation in mlp stream, cat again and run second linear layer
    output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
    x += mod.gate * output
    if x.dtype == torch.float16:
        x = torch.nan_to_num(x, nan=0.0, posinf=65504, neginf=-65504)

    return x


def double_blocks_forward(self, img: torch.Tensor, txt: torch.Tensor, vec: torch.Tensor, pe: torch.Tensor, layer_id: int):
    # print("input txt shape: ", txt.shape)
    bs = img.shape[0]

    img_mod1, img_mod2 = self.img_mod(vec)
    txt_mod1, txt_mod2 = self.txt_mod(vec)

    # prepare image for attention
    img_modulated = self.img_norm1(img)
    img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
    img_qkv = self.img_attn.qkv(img_modulated)
    img_q, img_k, img_v = img_qkv.view(
        img_qkv.shape[0], img_qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)

    img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

    # prepare txt for attention
    txt_modulated = self.txt_norm1(txt)
    txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
    txt_qkv = self.txt_attn.qkv(txt_modulated)
    txt_q, txt_k, txt_v = txt_qkv.view(
        txt_qkv.shape[0], txt_qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

    if True:
        img_k = img_k.permute(0, 2, 1, 3).reshape(
            1, bs * img_k.shape[2], img_k.shape[1], img_k.shape[3]).permute(0, 2, 1, 3).repeat(bs, 1, 1, 1)

        img_v = img_v.permute(0, 2, 1, 3).reshape(
            1, bs * img_v.shape[2], img_v.shape[1], img_v.shape[3]).permute(0, 2, 1, 3).repeat(bs, 1, 1, 1)

        txt_pe = pe[:, :, :txt_k.shape[2], :, :, :]
        img_pe = pe[:, :, txt_k.shape[2]:, :, :, :]

        img_pe = img_pe.permute(0, 2, 1, 3, 4, 5).reshape(
            1, bs * img_pe.shape[2], img_pe.shape[1], img_pe.shape[3], img_pe.shape[4], img_pe.shape[5]).permute(0, 2, 1, 3, 4, 5).repeat(bs, 1, 1, 1, 1, 1)

        pe = torch.cat((txt_pe, img_pe), dim=2)

    # run actual attention
    attn = attention(torch.cat((txt_q, img_q), dim=2),
                     torch.cat((txt_k, img_k), dim=2),
                     torch.cat((txt_v, img_v), dim=2), pe=pe)

    txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1]:]

    # calculate the img bloks
    img = img + img_mod1.gate * self.img_attn.proj(img_attn)
    img = img + img_mod2.gate * \
        self.img_mlp((1 + img_mod2.scale) *
                     self.img_norm2(img) + img_mod2.shift)

    # calculate the txt bloks
    txt += txt_mod1.gate * self.txt_attn.proj(txt_attn)
    txt += txt_mod2.gate * \
        self.txt_mlp((1 + txt_mod2.scale) *
                     self.txt_norm2(txt) + txt_mod2.shift)

    if txt.dtype == torch.float16:
        txt = torch.nan_to_num(txt, nan=0.0, posinf=65504, neginf=-65504)

    return img, txt
