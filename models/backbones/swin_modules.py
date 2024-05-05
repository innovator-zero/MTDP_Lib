import torch
import torch.nn as nn
from timm.models.layers import to_2tuple, trunc_normal_


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        """
        MLP
        :param int in_features: Input feature dimension
        :param int hidden_features: Hidden feature dimension
        :param int out_features: Output feature dimension
        :param nn.Module act_layer: Activation layer
        :param float drop: Dropout rate
        """

        super().__init__()
        out_features = out_features or in_features  # remain same if not given
        hidden_features = hidden_features or in_features  # remain same if not given

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Partition x into windows
    :param Tensor x: [B, H, W, C]
    :param int window_size: Window size
    :return: Tensor [B*(H/w)*(W/w), w, w, C]
    """
    B, H, W, C = x.shape
    # B, H, W, C => B, H/w, w, W/w, w, C => B, H/w, W/w, w, w, C => B*(H/w)*(W/w), w, w, C
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)

    return windows


def window_reverse(windows, window_size, H, W):
    """
    Reverse function of window_partition
    :param Tensor windows: [B*(H/w)*(W/w), w, w, C]
    :param int window_size: Window size
    :param int H: Height of image
    :param int W: Width of image
    :return: Tensor [B, H, W, C]
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    # B*(H/w)*(W/w), w, w, C => B, H/w, W/w, w, w, C => B, H/w, w, W/w, w, C => B, H, W, C
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)

    return x


class PatchMerging(nn.Module):
    """
    Patch Merging Layer
    :param tuple input_resolution: Resolution of input feature
    :param int dim: Number of input channels
    :param nn.Module norm_layer: Normalization layer
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)
        # Merge 2x2 pixels into 1 pixel
        x0 = x[:, 0::2, 0::2, :]  # B, H/2, W/2, C
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)  # B, H/2, W/2, 4*C
        x = x.view(B, -1, 4 * C)  # B, H/2*W/2, 4*C

        x = self.norm(x)
        x = self.reduction(x)  # B, H/2*W/2, 4*C => B, H/2*W/2, 2*C

        return x


class PatchEmbed(nn.Module):
    """
    Patch Embedding Layer
    :param int/tuple img_size: Image size
    :param int/tuplepatch_size: Patch size
    :param int in_chans: Number of input image channels
    :param int embed_dim: Embedding dimension
    :param nn.Module norm_layer: Normalization layer
    """

    def __init__(self, img_size, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])  # H/4, W/4
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)  # Non-overlap Conv
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B,C,H/4,W/4 => B,C,H/4*W/4 => B,H/4*W/4,C
        if self.norm is not None:
            x = self.norm(x)
        return x


class WindowAttention(nn.Module):
    """
    Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    :param dim (int): Number of input channels.
    :param window_size (tuple[int]): The height and width of the window.
    :param num_heads (int): Number of attention heads.
    :param qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
    :param qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
    :param attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
    :param proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

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

        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:  # nW, N, N
            nW, _H, _W = mask.shape
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
