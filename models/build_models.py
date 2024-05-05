import torch.nn as nn
import torch.nn.functional as F

from datasets.utils.configs import TRAIN_SCALE, get_output_num

from models.heads import BaseHead, TransposeHead


def build_model(arch, tasks, dataname, backbone_args, decoder_args, head_args):
    '''
    Initialize model from encoder backbone, decoder, and head
    '''

    backbone, backbone_channels = get_backbone(tasks=tasks, dataname=dataname, **backbone_args)
    decoders, heads = get_decoder_head(arch=arch,
                                       tasks=tasks,
                                       dataname=dataname,
                                       backbone_channels=backbone_channels,
                                       **decoder_args,
                                       **head_args)

    if arch == 'md':
        model = MultiDecoderModel(backbone, decoders, heads, tasks)
    elif arch == 'tc':
        model = TaskConditionalModel(backbone, decoders, heads, tasks)
    else:
        raise NotImplementedError

    return model


def get_backbone(tasks, dataname, backbone_type, **args):
    """
    Return the backbone
    """

    if backbone_type == 'prompt_swin_t':
        from models.backbones.prompt_swin_transformer import prompt_swin_t
        backbone = prompt_swin_t(**args, img_size=TRAIN_SCALE[dataname], tasks=tasks)
        backbone_channels = 96
    elif backbone_type == 'prompt_swin_s':
        from models.backbones.prompt_swin_transformer import prompt_swin_s
        backbone = prompt_swin_s(**args, img_size=TRAIN_SCALE[dataname], tasks=tasks)
        backbone_channels = 96
    elif backbone_type == 'prompt_swin_b':
        from models.backbones.prompt_swin_transformer import prompt_swin_b
        backbone = prompt_swin_b(**args, img_size=TRAIN_SCALE[dataname], tasks=tasks)
        backbone_channels = 128
    elif backbone_type == 'prompt_swin_l':
        from models.backbones.prompt_swin_transformer import prompt_swin_l
        backbone = prompt_swin_l(**args, img_size=TRAIN_SCALE[dataname], tasks=tasks)
        backbone_channels = 192
    elif backbone_type == 'adapter_swin_t':
        from models.backbones.adapter_swin_transformer import adapter_swin_t
        backbone = adapter_swin_t(**args, img_size=TRAIN_SCALE[dataname], tasks=tasks)
        backbone_channels = 96
    elif backbone_type == 'adapter_swin_s':
        from models.backbones.adapter_swin_transformer import adapter_swin_s
        backbone = adapter_swin_s(**args, img_size=TRAIN_SCALE[dataname], tasks=tasks)
        backbone_channels = 96
    elif backbone_type == 'adapter_swin_b':
        from models.backbones.adapter_swin_transformer import adapter_swin_b
        backbone = adapter_swin_b(**args, img_size=TRAIN_SCALE[dataname], tasks=tasks)
        backbone_channels = 128
    elif backbone_type == 'adapter_swin_l':
        from models.backbones.adapter_swin_transformer import adapter_swin_l
        backbone = adapter_swin_l(**args, img_size=TRAIN_SCALE[dataname], tasks=tasks)
        backbone_channels = 192
    else:
        raise NotImplementedError

    return backbone, backbone_channels


def get_decoder_module(decoder_type, enc_out_size, backbone_channels, tasks, **decoder_args):
    """
    Return a single decoder module
    """

    encoder_dims = [(backbone_channels * 2**i) for i in range(4)]

    if decoder_type == 'light_prompt_decoder':
        from .decoders.light_prompt_decoder import LightPromptedDecoder
        decoder = LightPromptedDecoder(input_size=enc_out_size,
                                       encoder_dims=encoder_dims,
                                       embed_dim=backbone_channels,
                                       tasks=tasks,
                                       **decoder_args)
    elif decoder_type == 'task_gate_decoder':
        from .decoders.task_gate_decoder import TaskGateDecoder
        decoder = TaskGateDecoder(input_size=enc_out_size,
                                  encoder_dims=encoder_dims,
                                  embed_dim=backbone_channels,
                                  tasks=tasks,
                                  **decoder_args)
    elif decoder_type == 'fusion':
        from .decoders.decoder_modules import Transform
        decoder = Transform(input_size=enc_out_size, in_dims=encoder_dims, embed_dim=backbone_channels)
    else:
        raise NotImplementedError

    return decoder


def get_decoder_head(arch, tasks, dataname, backbone_channels, decoder_type, head_type, **decoder_args):
    """
    Return decoders and heads
    """

    input_size = TRAIN_SCALE[dataname]
    enc_out_size = (int(input_size[0] / 32), int(input_size[1] / 32))

    decoders = nn.ModuleDict()
    heads = nn.ModuleDict()

    if arch == 'md':
        for task in tasks:
            decoders[task] = get_decoder_module(decoder_type, enc_out_size, backbone_channels, tasks, **decoder_args)
    elif arch == 'tc':
        decoders['all'] = get_decoder_module(decoder_type, enc_out_size, backbone_channels, tasks, **decoder_args)
    else:
        raise ValueError

    for task in tasks:
        if head_type == 'transpose':
            heads[task] = TransposeHead(dim=backbone_channels, out_ch=get_output_num(task, dataname))
        elif head_type == 'base':
            heads[task] = BaseHead(dim=backbone_channels, out_ch=get_output_num(task, dataname))
        else:
            raise NotImplementedError

    return decoders, heads


class MultiDecoderModel(nn.Module):
    """
    Multi-decoder model with shared encoder + task-specific decoders + task-specific heads
    """

    def __init__(self, backbone, decoders, heads, tasks):
        super().__init__()
        assert (set(decoders.keys()) == set(tasks))
        self.backbone = backbone
        self.decoders = decoders
        self.heads = heads
        self.tasks = tasks

    def forward(self, x):
        out = {}
        img_size = x.size()[2:]

        encoder_output = self.backbone(x)
        for task in self.tasks:
            out[task] = F.interpolate(self.heads[task](self.decoders[task](encoder_output)), img_size, mode='bilinear')
        return out


class TaskConditionalModel(nn.Module):
    """
    Task-conditional model with shared encoder + shared decoder + task-specific heads
    """

    def __init__(self, backbone, decoders, heads, tasks):
        super().__init__()
        self.backbone = backbone
        self.decoders = decoders
        self.heads = heads
        self.tasks = tasks

    def forward(self, x, task):
        assert task in self.tasks
        out = {}
        img_size = x.size()[2:]

        encoder_output = self.backbone(x, task)
        out[task] = F.interpolate(self.heads[task](self.decoders['all'](encoder_output, task)),
                                  img_size,
                                  mode='bilinear')
        return out
