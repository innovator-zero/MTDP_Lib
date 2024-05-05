import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from .swin_modules import (Mlp, PatchEmbed, PatchMerging, WindowAttention, window_partition, window_reverse)

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


class Adapter(nn.Module):
    """
    Original Adapter module.
    :param int input_dim: Input dimension.
    :param int down_ratio: Adapter down ratio.
    :param list tasks: List of tasks.
    """

    def __init__(self, input_dim, down_ratio, tasks):
        super().__init__()
        assert tasks is not None, "Adapters need tasks specified!"

        hidden_dim = int(input_dim // down_ratio)
        self.down = nn.ModuleDict()
        self.up = nn.ModuleDict()
        for task in tasks:
            self.down[task] = nn.Linear(input_dim, hidden_dim)
            self.up[task] = nn.Linear(hidden_dim, input_dim)
        self.act = nn.GELU()

    def forward(self, x, task):
        x = self.down[task](x)
        x = self.up[task](self.act(x))
        return x


class MixTaskAdapter(nn.Module):
    """
    Mix Task Adapter module.
    :param int input_dim: Input dimension.
    :param int down_ratio: Adapter down ratio. d=n/donw_ratio
    :param int reduction_ratio: Reduction ratio. m=d/reduction_ratio
    :param list tasks: List of tasks.
    """

    def __init__(self, input_dim, down_ratio, reduction_ratio, tasks):
        super().__init__()
        assert tasks is not None, "Adapters need tasks specified!"

        hidden_dim = int(input_dim // down_ratio)
        reduced_dim = int(hidden_dim // reduction_ratio)
        self.down_shared = nn.Linear(input_dim, reduced_dim)

        self.down_task = nn.ModuleDict()
        for task in tasks:
            self.down_task[task] = nn.Linear(reduced_dim, hidden_dim)  # Task Indicating Matrix

        self.act = nn.GELU()
        self.up_1 = nn.Linear(hidden_dim, reduced_dim)
        self.up_2 = nn.Linear(reduced_dim, input_dim)

    def forward(self, x, task):
        x = self.down_shared(x)
        x = self.down_task[task](x)
        x = self.up_1(self.act(x))
        x = self.up_2(x)
        return x


class AdapterSwinTransformerBlock(nn.Module):
    """
    Swin Transformer block with Adapters after Attention and MLP layers.
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
    :param norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
    :param tasks (list[str], optional): List of tasks. Default: None
    """

    def __init__(
        self,
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
        tasks=None,
        **adapter_args,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        if min(self.input_resolution) <= self.window_size:
            # If window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim,
                                    window_size=to_2tuple(self.window_size),
                                    num_heads=num_heads,
                                    qkv_bias=qkv_bias,
                                    qk_scale=qk_scale,
                                    attn_drop=attn_drop,
                                    proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # Build adapters
        if "adapter_module" in adapter_args:
            self.adapter_module = adapter_args.pop("adapter_module")
            if self.adapter_module is not None:
                adapter_class = getattr(sys.modules[__name__], self.adapter_module)
                self.adapters_att = adapter_class(input_dim=dim, tasks=tasks, **adapter_args)
                self.adapters_mlp = adapter_class(input_dim=dim, tasks=tasks, **adapter_args)
        else:
            self.adapter_module = None

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

    def forward(self, x, task=None):
        H, W = self.input_resolution
        B, L, C = x.shape

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
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, nP+w*w, C

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
        if self.adapter_module is not None:
            x = x + self.adapters_att(x, task)
        x = shortcut + self.drop_path(x)

        # FFN
        shortcut2 = x
        x = self.mlp(self.norm2(x))
        if self.adapter_module is not None:
            x = x + self.adapters_mlp(x, task)
        x = shortcut2 + self.drop_path(x)

        return x


class AdapterBasicLayer(nn.Module):
    """
    A basic Swin Transformer layer with Adapters.
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
    :param tasks (list[str] | None, optional): Tasks. Default: None
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
                 tasks=None,
                 **adapter_args):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # Build blocks
        self.blocks = nn.ModuleList([
            AdapterSwinTransformerBlock(dim=dim,
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
                                        tasks=tasks,
                                        **adapter_args) for i in range(depth)
        ])

        # Patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, task=None):
        for i in range(len(self.blocks)):
            x = self.blocks[i](x, task)

        x_no_ds = x  # features before downsample

        if self.downsample is not None:
            x = self.downsample(x)

        return x, x_no_ds


class SwinAdapter(nn.Module):
    """
    Swin Transformer with Adapters.
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
                 tasks=None,
                 **adapter_args):
        super().__init__()
        img_size = to_2tuple(img_size)
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
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
            layer = AdapterBasicLayer(
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
                tasks=tasks,
                **adapter_args)
            self.layers.append(layer)

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
            if task is not None:
                x, fea = self.layers[i](x, task)
            else:
                x, fea = self.layers[i](x)
            outs.append(fea)

        return outs


def adapter_swin_t(pretrained: bool = False, img_size=None, tasks=None, **adapter_args):
    my_swin = SwinAdapter(img_size=img_size,
                          embed_dim=96,
                          depths=[2, 2, 6, 2],
                          num_heads=[3, 6, 12, 24],
                          tasks=tasks,
                          **adapter_args)

    if pretrained:
        model_dict = my_swin.state_dict()
        pretrained_dict = torch.utils.model_zoo.load_url(model_urls['swin_t'])["model"]
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if "attn_mask" not in k and k in model_dict}
        model_dict.update(pretrained_dict)
        my_swin.load_state_dict(model_dict)

    return my_swin


def adapter_swin_s(pretrained: bool = False, img_size=None, tasks=None, **adapter_args):
    my_swin = SwinAdapter(img_size=img_size,
                          embed_dim=96,
                          depths=[2, 2, 18, 2],
                          num_heads=[3, 6, 12, 24],
                          tasks=tasks,
                          **adapter_args)

    if pretrained:
        model_dict = my_swin.state_dict()
        pretrained_dict = torch.utils.model_zoo.load_url(model_urls['swin_s'])["model"]
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if "attn_mask" not in k and k in model_dict}
        model_dict.update(pretrained_dict)
        my_swin.load_state_dict(model_dict)

    return my_swin


def adapter_swin_b(pretrained: bool = False, img_size=None, tasks=None, **adapter_args):
    my_swin = SwinAdapter(img_size=img_size,
                          embed_dim=128,
                          depths=[2, 2, 18, 2],
                          num_heads=[4, 8, 16, 32],
                          tasks=tasks,
                          **adapter_args)

    if pretrained:
        model_dict = my_swin.state_dict()
        pretrained_dict = torch.utils.model_zoo.load_url(model_urls['swin_b'])["model"]
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if "attn_mask" not in k and k in model_dict}
        model_dict.update(pretrained_dict)
        my_swin.load_state_dict(model_dict)

    return my_swin


def adapter_swin_l(pretrained: bool = False, img_size=None, tasks=None, **adapter_args):
    my_swin = SwinAdapter(img_size=img_size,
                          embed_dim=192,
                          depths=[2, 2, 18, 2],
                          num_heads=[6, 12, 24, 48],
                          tasks=tasks,
                          **adapter_args)

    if pretrained:
        model_dict = my_swin.state_dict()
        pretrained_dict = torch.utils.model_zoo.load_url(model_urls['swin_l'])["model"]
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if "attn_mask" not in k and k in model_dict}
        model_dict.update(pretrained_dict)
        my_swin.load_state_dict(model_dict)

    return my_swin
