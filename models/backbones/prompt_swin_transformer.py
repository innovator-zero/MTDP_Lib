import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from .swin_modules import (Mlp, PatchEmbed, PatchMerging, window_partition, window_reverse)

model_urls = {
    "swin_t":
    "https://github.com/SwinTransformer/storage/releases/download/v1.0.8/swin_tiny_patch4_window7_224_22k.pth",
    "swin_s":
    "https://github.com/SwinTransformer/storage/releases/download/v1.0.8/swin_small_patch4_window7_224_22k.pth",
    "swin_b":
    "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth",
    "swin_l":
    "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth"
}


class PromptedWindowAttention(nn.Module):
    """
    Prompted Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    :param dim (int): Number of input channels.
    :param window_size (tuple[int]): The height and width of the window.
    :param num_heads (int): Number of attention heads.
    :param qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
    :param qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
    :param attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
    :param proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    :param num_prompts (int, optional): Number of prompts to prepend to the input. Default: 0
    """

    def __init__(self,
                 dim,
                 window_size,
                 num_heads,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 num_prompts=0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.num_prompts = num_prompts

        # Define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # (2*w-1) * (2*w-1), nH

        # Get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # tuple => 2, w, w
        # coords[0 or 1]: row or col index of each element
        coords_flatten = torch.flatten(coords, 1)  # 2, w*w
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, w*w, w*w
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # w*w, w*w, 2
        # relative_coords[i, j, 0 or 1]: difference between row or col index of i-th and j-th element
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        # Range of difference: [0, 2*w-1] and [0, 2*w-1]
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        # Scale range of row difference to [0, 2*w-1, (2*w-1)*(2*w-1)], make row and col different
        relative_position_index = relative_coords.sum(-1)  # w*w, w*w
        # Range of elements [0, (2*w-1)*(2*w-1)]
        self.register_buffer("relative_position_index", relative_position_index)  # no training

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # put Q,K,V together
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape  # B_=nW*B, N=w*w
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # B_, N, 3*C => B_, N, 3, nH, head_dim => 3, B_, nH, N, head_dim
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B_, nH, N, N

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        # Use relative_position_index to fetch value in relative_position_bias_table
        # N*N, nH => N, N, nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, N, N

        if self.num_prompts > 0:
            _nH, _H, _W = relative_position_bias.shape

            # Padding to nH, nP+N, nP+N
            relative_position_bias = torch.cat(
                (torch.zeros(_nH, self.num_prompts, _W, device=attn.device), relative_position_bias), dim=1)
            relative_position_bias = torch.cat(
                (torch.zeros(_nH, _H + self.num_prompts, self.num_prompts, device=attn.device), relative_position_bias),
                dim=-1)

        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:  # nW, N, N
            nW, _H, _W = mask.shape
            if self.num_prompts > 0:
                # Padding prompts
                mask = torch.cat((torch.zeros(nW, self.num_prompts, _W, device=attn.device), mask), dim=1)
                mask = torch.cat((torch.zeros(nW, _H + self.num_prompts, self.num_prompts, device=attn.device), mask),
                                 dim=-1)

            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            # B, nW, nH, N, N + 1, nW, 1, N, N
            # Make attention between elements from different windows small negative value
            attn = attn.view(-1, self.num_heads, N, N)  # B_, nH, N, N
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        # B_, nH, N, head_dim => B_, N, nH, head_dim => B_, N, nH*head_dim=C
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class PromptedSwinTransformerBlock(nn.Module):
    """
    Prompted Swin Transformer Block.
    :param dim (int): Number of input channels.
    :param input_resolution (tuple[int]): Input resulotion.
    :param num_heads (int): Number of attention heads.
    :param window_size (int): Window size.
    :param shift_size (int): Shift size for SW-MSA.
    :param mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
    :param qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
    :param qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
    :param drop (float, optional): Dropout rate. Default: 0.0
    :param attn_drop (float, optional): Attention dropout rate. Default: 0.0
    :param drop_path (float, optional): Stochastic depth rate. Default: 0.0
    :param act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
    :param norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    :param num_prompts (int, optional): Number of prompts. Default: 0
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 num_heads,
                 window_size=7,
                 shift_size=0,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 num_prompts=0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.num_prompts = num_prompts

        if min(self.input_resolution) <= self.window_size:
            # If window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = PromptedWindowAttention(dim,
                                            window_size=to_2tuple(self.window_size),
                                            num_heads=num_heads,
                                            qkv_bias=qkv_bias,
                                            qk_scale=qk_scale,
                                            attn_drop=attn_drop,
                                            proj_drop=drop,
                                            num_prompts=num_prompts)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # Calculate attention mask for SW-MSA
            H, W = self.input_resolution
            # Padding size
            Hp = int(np.ceil(H / self.window_size)) * self.window_size
            Wp = int(np.ceil(W / self.window_size)) * self.window_size
            # Use 0-8 to mark 9 areas, area 1-8 are influenced by window shift
            img_mask = torch.zeros((1, Hp, Wp, 1))
            h_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                           -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                           -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, w, w, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)  # nW, w*w
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # nW, w*w, w*w
            # Value in attn_mask represent whether element pairs are from the same window, 0 = same window
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            # Element pairs from different windows equals -100
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape

        # Separate prompts and feature
        if self.num_prompts > 0:
            L = L - self.num_prompts
            prompt_emb = x[:, :self.num_prompts, :]
            x = x[:, self.num_prompts:, :]

        assert L == H * W, "Input feature has wrong size!"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # Cyclic shift
        if self.shift_size > 0:
            # Cycle shift feature map
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, w, w, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, w*w, C

        # W-MSA/SW-MSA
        if self.num_prompts > 0:
            # Expand prompts_embs
            nW = x_windows.shape[0] // B
            # B, nP, C => nW*B, nP, C
            prompt_emb = prompt_emb.expand(nW, -1, -1, -1)
            prompt_emb = prompt_emb.reshape(-1, self.num_prompts, C)
            x_windows = torch.cat((prompt_emb, x_windows), dim=1)

        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, nP+w*w, C

        # Discard prompts
        attn_windows = attn_windows[:, self.num_prompts:self.num_prompts + self.window_size * self.window_size, :]

        # Merge windows
        attn_windows = attn_windows.reshape(-1, self.window_size, self.window_size, C).contiguous()  # nW*B, w, w, C

        # Reverse cyclic shift
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B, Hp, Wp, C
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        # Reverse padding
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # Dropout of main stem
        x = shortcut + self.drop_path(x)

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PromptedBasicLayer(nn.Module):
    """
    A basic Prompted Swin Transformer layer.
    :param dim (int): Number of input channels.
    :param input_resolution (tuple[int]): Input resolution.
    :param depth (int): Number of blocks.
    :param num_heads (int): Number of attention heads.
    :param window_size (int): Local window size.
    :param mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
    :param qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
    :param qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
    :param drop (float, optional): Dropout rate. Default: 0.0
    :param attn_drop (float, optional): Attention dropout rate. Default: 0.0
    :param drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
    :param norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
    :param downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
    :param num_prompts (int, optional): Number of prompts. Default: 0
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 num_prompts=0):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.num_prompts = num_prompts

        # Build blocks
        self.blocks = nn.ModuleList([
            PromptedSwinTransformerBlock(dim=dim,
                                         input_resolution=input_resolution,
                                         num_heads=num_heads,
                                         window_size=window_size,
                                         shift_size=0 if (i % 2 == 0) else window_size // 2,
                                         mlp_ratio=mlp_ratio,
                                         qkv_bias=qkv_bias,
                                         qk_scale=qk_scale,
                                         drop=drop,
                                         attn_drop=attn_drop,
                                         drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                         norm_layer=norm_layer,
                                         num_prompts=num_prompts) for i in range(depth)
        ])

        # Patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, prompt_embeds=None):
        if self.num_prompts > 0:
            assert prompt_embeds is not None, "No prompt embeddings!"
            assert prompt_embeds.shape[1] == self.num_prompts
        B = x.shape[0]

        for i in range(len(self.blocks)):
            if self.num_prompts > 0:
                prompt_embed = prompt_embeds[i].expand(B, -1, -1)
                x = torch.cat((prompt_embed, x), dim=1)

            x = self.blocks[i](x)

        x_no_ds = x  # features before downsample

        if self.downsample is not None:
            x = self.downsample(x)

        return x, x_no_ds


class PromptedSwinTransformer(nn.Module):
    """
    Prompted Swin Transformer
    :param img_size (int | tuple(int)): Input image size.
    :param embed_dim (int): Patch embedding dimension. Default: 96
    :param depths (tuple(int)): Depth of each Swin Transformer layer.
    :param num_heads (tuple(int)): Number of attention heads in different layers.
    :param window_size (int): Window size. Default: 7
    :param mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
    :param qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
    :param qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
    :param drop_rate (float): Dropout rate. Default: 0
    :param attn_drop_rate (float): Attention dropout rate. Default: 0
    :param drop_path_rate (float): Stochastic depth rate. Default: 0.1
    :param norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
    :param patch_norm (bool): If True, add normalization after patch embedding. Default: True
    :param num_prompts (int): Number of prompts. Default: 0
    :param tasks (list(str)): List of tasks. Default: None
    """

    def __init__(self,
                 img_size,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.3,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 num_prompts=0,
                 tasks=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        self.num_prompts = num_prompts
        self.tasks = tasks

        # Split image into non-overlapping patches
        self.patch_embed = PatchEmbed(img_size=img_size,
                                      patch_size=4,
                                      in_chans=3,
                                      embed_dim=embed_dim,
                                      norm_layer=norm_layer if self.patch_norm else None)
        self.num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution

        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        # Have sum(depths) values in [0, drop_path_rate]

        # Build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = PromptedBasicLayer(
                dim=int(embed_dim * 2**i_layer),  # (2^i)*C
                input_resolution=(
                    patches_resolution[0] // (2**i_layer),  # H/4/(2^i)
                    patches_resolution[1] // (2**i_layer)),  # W/4/(2^i)
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                num_prompts=num_prompts)
            self.layers.append(layer)

        # Intialize prompts
        if num_prompts > 0:
            assert tasks is not None, "Prompts need tasks specified!"
            val = math.sqrt(6. / float(3 * 16 + embed_dim))

            self.all_prompt_embeddings = nn.ModuleDict()
            for task in tasks:
                prompt_embeddings = []
                for i in range(self.num_layers):
                    prompt = nn.Parameter(torch.zeros(depths[i], num_prompts, self.layers[i].dim))
                    nn.init.uniform_(prompt.data, -val, val)
                    prompt_embeddings.append(prompt)
                self.all_prompt_embeddings[task] = nn.ParameterList(prompt_embeddings)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, x, task=None):
        outs = []

        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for i in range(self.num_layers):
            if task is not None and self.num_prompts > 0:
                prompt_embeds = self.all_prompt_embeddings[task][i]
                x, fea = self.layers[i](x, prompt_embeds)
            else:
                x, fea = self.layers[i](x)
            outs.append(fea)

        return outs


def prompt_swin_t(pretrained: bool = False, img_size=None, num_prompts=0, tasks=None):
    my_swin = PromptedSwinTransformer(img_size=img_size,
                                      embed_dim=96,
                                      depths=(2, 2, 6, 2),
                                      num_heads=(3, 6, 12, 24),
                                      num_prompts=num_prompts,
                                      tasks=tasks)
    if pretrained:
        model_dict = my_swin.state_dict()
        pretrained_dict = torch.utils.model_zoo.load_url(model_urls['swin_t'])["model"]
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if "attn_mask" not in k and k in model_dict}
        model_dict.update(pretrained_dict)
        my_swin.load_state_dict(model_dict)

    return my_swin


def prompt_swin_s(pretrained: bool = False, img_size=None, num_prompts=0, tasks=None):
    my_swin = PromptedSwinTransformer(img_size=img_size,
                                      embed_dim=96,
                                      depths=(2, 2, 18, 2),
                                      num_heads=(3, 6, 12, 24),
                                      num_prompts=num_prompts,
                                      tasks=tasks)
    if pretrained:
        model_dict = my_swin.state_dict()
        pretrained_dict = torch.utils.model_zoo.load_url(model_urls['swin_s'])["model"]
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if "attn_mask" not in k and k in model_dict}
        model_dict.update(pretrained_dict)
        my_swin.load_state_dict(model_dict)

    return my_swin


def prompt_swin_b(pretrained: bool = False, img_size=None, num_prompts=0, tasks=None):
    my_swin = PromptedSwinTransformer(img_size=img_size,
                                      embed_dim=128,
                                      depths=(2, 2, 18, 2),
                                      num_heads=(4, 8, 16, 32),
                                      num_prompts=num_prompts,
                                      tasks=tasks)
    if pretrained:
        model_dict = my_swin.state_dict()
        pretrained_dict = torch.utils.model_zoo.load_url(model_urls['swin_b'])["model"]
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if "attn_mask" not in k and k in model_dict}
        model_dict.update(pretrained_dict)
        my_swin.load_state_dict(model_dict)

    return my_swin


def prompt_swin_l(pretrained: bool = False, img_size=None, num_prompts=0, tasks=None):
    my_swin = PromptedSwinTransformer(img_size=img_size,
                                      embed_dim=192,
                                      depths=(2, 2, 18, 2),
                                      num_heads=(6, 12, 24, 48),
                                      num_prompts=num_prompts,
                                      tasks=tasks)
    if pretrained:
        model_dict = my_swin.state_dict()
        pretrained_dict = torch.utils.model_zoo.load_url(model_urls['swin_l'])["model"]
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if "attn_mask" not in k and k in model_dict}
        model_dict.update(pretrained_dict)
        my_swin.load_state_dict(model_dict)

    return my_swin
