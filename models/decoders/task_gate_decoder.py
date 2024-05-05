import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .decoder_modules import Transform


class EmbeddingGenerator(nn.Module):

    def __init__(self, input_size, task_embed_dim, task_ind_dim):
        super().__init__()
        self.input_size = input_size
        self.task_embed_dim = task_embed_dim
        self.mlp = nn.Linear(task_ind_dim, task_embed_dim * input_size[0] * input_size[1])

    def forward(self, task_ind):
        x = self.mlp(task_ind)
        x = x.reshape(self.task_embed_dim, self.input_size[0], self.input_size[1])
        task_embedding = F.interpolate(x.unsqueeze(0),
                                       size=(self.input_size[0] * 8, self.input_size[1] * 8),
                                       mode='bilinear',
                                       align_corners=False)
        return task_embedding


class TaskGateDecoder(nn.Module):
    """
    Task Gate Decoder
    :param tuple input_size: Input feature size
    :param list encoder_dims: List of encoder feature dimensions
    :param int embed_dim: Embedding dimension
    :param list tasks: List of tasks
    :param int task_embed_dim: Dimension C_T of task embedding
    :param int task_ind_dim: Dimension v^t of Task Indicating Vector
    """

    def __init__(self, input_size, encoder_dims, embed_dim, tasks, task_embed_dim, task_ind_dim):
        super().__init__()
        # Transform features
        # B, C, H/4, W/4
        self.transform = Transform(input_size=input_size, in_dims=encoder_dims, embed_dim=embed_dim)

        self.reset_gate = nn.Conv2d(embed_dim + task_embed_dim, embed_dim, kernel_size=3, padding=1)
        self.update_gate = nn.Conv2d(embed_dim + task_embed_dim, embed_dim, kernel_size=3, padding=1)
        self.out_gate = nn.Conv2d(embed_dim + task_embed_dim, embed_dim, kernel_size=3, padding=1)

        self.task_emb_gens = nn.ModuleDict()
        for task in tasks:
            self.task_emb_gens[task] = EmbeddingGenerator(input_size, task_embed_dim, task_ind_dim)

        # Task Indicating Vector
        self.all_task_inds = nn.ParameterDict()
        val = math.sqrt(6. / float(3 * 16 + task_ind_dim))
        for task in tasks:
            # Task_C, H/4, W/4
            ind = nn.Parameter(torch.zeros(task_ind_dim))
            nn.init.uniform_(ind.data, -val, val)
            self.all_task_inds[task] = ind

    def forward(self, fea, task):
        fea = self.transform(fea)

        task_ind = self.all_task_inds[task]
        task_emb = self.task_emb_gens[task](task_ind)
        task_emb = task_emb.expand(fea.shape[0], -1, -1, -1)

        stacked_inputs = torch.cat([task_emb, fea], dim=1)
        update = F.sigmoid(self.update_gate(stacked_inputs))
        reset = F.sigmoid(self.reset_gate(stacked_inputs))
        out_inputs = F.tanh(self.out_gate(torch.cat([task_emb, reset * fea], dim=1)))
        output = (1 - update) * fea + update * out_inputs

        return output
