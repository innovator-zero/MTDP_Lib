import math

import torch
import torch.nn as nn
from timm.models.layers import to_2tuple, trunc_normal_

from ..backbones.prompt_swin_transformer import PromptedBasicLayer

from .decoder_modules import Transform


class LightPromptedDecoder(nn.Module):
    """
    Lightweight PGT Decoder.
    :param input_size (int | tuple(int)): Input feature size.
    :param encoder_dims (list(int)): Input feature dimensions.
    :param embed_dim (int): Patch embedding dimension.
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
    :param num_prompts (int): Number of prompts. Default: 0
    :param tasks (list(str)): List of tasks. Default: None
    """

    def __init__(self,
                 input_size,
                 encoder_dims,
                 embed_dim=96,
                 depths=[2, 2],
                 num_heads=[6, 3],
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 num_prompts=0,
                 tasks=None):
        super().__init__()
        input_size = to_2tuple(input_size)
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self.num_prompts = num_prompts
        self.tasks = tasks

        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        dpr = dpr[::-1]  # reverse dpr

        # Transform features
        self.transform = Transform(input_size=input_size, in_dims=encoder_dims, embed_dim=embed_dim)
        # Build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = PromptedBasicLayer(dim=embed_dim,
                                       input_resolution=(input_size[0] * 8, input_size[1] * 8),
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
                                       num_prompts=num_prompts)
            self.layers.append(layer)

        # Initialize prompts
        if num_prompts > 0:
            assert tasks is not None, "Prompts need tasks specified!"
            val = math.sqrt(6. / float(3 * 16 + embed_dim))

            self.all_prompt_embeddings = nn.ModuleDict()
            for task in tasks:
                prompt_embeddings = []
                for i in range(self.num_layers):
                    prompt = nn.Parameter(torch.zeros(depths[i], num_prompts, embed_dim))
                    nn.init.uniform_(prompt.data, -val, val)
                    prompt_embeddings.append(prompt)
                self.all_prompt_embeddings[task] = nn.ParameterList(prompt_embeddings)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, encoder_output, task=None):
        x = self.transform(encoder_output)

        # B, C, H/4, W/4 => B, H/4*W/4, C
        x = x.reshape(x.shape[0], x.shape[1], -1).transpose(1, 2).contiguous()

        for i in range(self.num_layers):
            if task is not None and self.num_prompts > 0:
                prompt_embeds = self.all_prompt_embeddings[task][i]
                x, _ = self.layers[i](x, prompt_embeds)
            else:
                x, _ = self.layers[i](x)

        h, w = self.layers[-1].input_resolution
        x = x.transpose(1, 2).reshape(x.size(0), x.size(2), h, w).contiguous()

        return x
