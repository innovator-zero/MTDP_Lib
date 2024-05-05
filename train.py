import argparse
import datetime
import os
import shutil

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP

from datasets.custom_dataset import get_dataloader, get_dataset
from datasets.custom_transforms import get_transformations
from datasets.utils.configs import TEST_SCALE, TRAIN_SCALE
from losses import get_criterion
from models.build_models import build_model
from train_utils import eval_metric, get_optimizer_scheduler, train_one_epoch
from utils import RunningMeter, create_results_dir, get_loss_metric


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True, help="Config file path")
    parser.add_argument('--exp', type=str, required=True, help="Experiment name")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--wandb_name', type=str, help="Wandb project name")
    parser.add_argument('--checkpoint', default=None, help="Resume from checkpoint")
    parser.add_argument('--fp16', action='store_true', help="Whether to use fp16")

    args = parser.parse_args()

    with open(args.config_path, 'r') as stream:
        configs = yaml.safe_load(stream)

    # Join args and configs
    configs = {**configs, **vars(args)}

    # Set seed and ddp
    set_seed(args.seed)
    dist.init_process_group('nccl', timeout=datetime.timedelta(0, 3600 * 2))
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    cudnn.benchmark = True
    cv2.setNumThreads(0)

    # Setup logger and output folders
    if local_rank == 0:
        os.makedirs(configs['results_dir'], exist_ok=True)
        configs['exp_dir'], configs['checkpoint_dir'] = create_results_dir(configs['results_dir'], args.exp)
        shutil.copy(args.config_path, os.path.join(configs['exp_dir'], 'config.yml'))
        if args.wandb_name is not None:
            import wandb
            wandb.init(project=args.wandb_name, id=args.exp, name=args.exp, config=configs)
    dist.barrier()

    # Setup dataset and dataloader
    dataname = configs['dataset']
    task_dict = configs['task_dict']
    task_list = []
    for task_name in task_dict:
        task_list += [task_name] * task_dict[task_name]

    train_transforms = get_transformations(TRAIN_SCALE[dataname], train=True)
    val_transforms = get_transformations(TEST_SCALE[dataname], train=False)

    train_ds = get_dataset(dataname, train=True, tasks=task_list, transform=train_transforms)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds, drop_last=True)
    train_dl = get_dataloader(train=True, configs=configs, dataset=train_ds, sampler=train_sampler)

    val_ds = get_dataset(dataname, train=False, tasks=task_list, transform=val_transforms)
    val_dl = get_dataloader(train=False, configs=configs, dataset=val_ds)

    # Setup model
    arch = configs['arch']
    model = build_model(arch,
                        task_list,
                        dataname,
                        backbone_args=configs['backbone'],
                        decoder_args=configs['decoder'],
                        head_args=configs['head']).cuda()
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # Find total parameters and trainable parameters
    if local_rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Total parameters: %.2fM, Trainable: %.2fM" % (total_params / 1e6, total_trainable_params / 1e6))

    # Setup optimizer and scheduler
    optimizer, scheduler = get_optimizer_scheduler(configs, model)

    # Setup loss function
    criterion = get_criterion(dataname, task_list).cuda()

    # Setup scaler for amp
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    # Setup loss meters
    train_loss = {}
    val_loss = {}
    for task in task_list:
        train_loss[task] = RunningMeter()
        val_loss[task] = RunningMeter()

    # Determine max epochs and iterations
    max_epochs = configs['max_epochs']
    max_iter = configs['max_iters']

    if max_epochs > 0:
        max_iter = 1000000
        if local_rank == 0:
            print("Training for %d epochs" % max_epochs)
    else:
        assert max_iter > 0
        max_epochs = 1000000
        if local_rank == 0:
            print("Training for %d iterations" % max_iter)

    # Resume from checkpoint
    if args.checkpoint is not None:
        if local_rank == 0:
            print("Resume from checkpoint %s" % args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        # Add .module to keys as model is wrapped by DDP
        checkpoint['model'] = {'module.' + k: v for k, v in checkpoint['model'].items()}
        model.load_state_dict(checkpoint['model'])
        if 'optimizer' in checkpoint.keys():
            optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint.keys():
            scheduler.load_state_dict(checkpoint['scheduler'])
        if 'epoch' in checkpoint.keys():
            start_epoch = checkpoint['epoch'] + 1
        else:
            start_epoch = 0
        if 'iter_count' in checkpoint.keys():
            iter_count = checkpoint['iter_count']
        else:
            iter_count = 0
    else:
        start_epoch = 0
        iter_count = 0

    for epoch in range(start_epoch, max_epochs):
        logs = {}
        end_signal, iter_count = train_one_epoch(arch, epoch, iter_count, max_iter, task_list, train_dl, model,
                                                 optimizer, scheduler, criterion, scaler, configs['grad_clip'],
                                                 train_loss, local_rank, args.fp16)

        train_stats = get_loss_metric(train_loss, task_list, 'train')
        logs.update(train_stats)

        # Validation
        if local_rank == 0:
            if (epoch + 1) % configs['eval_freq'] == 0 or epoch == max_epochs - 1 or end_signal:
                print("Validation at epoch %d." % epoch)
                val_logs = eval_metric(arch, task_list, dataname, val_dl, model)
                print(val_logs)
                if args.wandb_name is not None:
                    wandb.log({**logs, **val_logs})

                # Save checkpoint
                save_ckpt_temp = {
                    'model': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                    'iter_count': iter_count - 1
                }
                torch.save(save_ckpt_temp, os.path.join(configs['checkpoint_dir'], 'checkpoint.pth'))
                print('Checkpoint saved.')

            else:
                if args.wandb_name is not None:
                    wandb.log(logs)

        if end_signal:
            break

    if local_rank == 0:
        print('Training finished.')

    dist.destroy_process_group()


if __name__ == '__main__':
    main()
