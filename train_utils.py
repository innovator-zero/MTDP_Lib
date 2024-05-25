import numpy as np
import torch
from timm.scheduler.cosine_lr import CosineLRScheduler
from tqdm import tqdm

from evaluation.evaluate_utils import PerformanceMeter
from utils import get_output, to_cuda


class PolynomialLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, max_iterations, gamma=0.9, min_lr=0., last_epoch=-1):
        self.max_iterations = max_iterations
        self.gamma = gamma
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # slight abuse: last_epoch refers to last iteration
        factor = (1 - self.last_epoch / float(self.max_iterations))**self.gamma
        return [(base_lr - self.min_lr) * factor + self.min_lr for base_lr in self.base_lrs]


def get_optimizer_scheduler(config, model):
    """
    Get optimizer and scheduler for model
    """
    params = model.parameters()

    if config['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(params,
                                    lr=float(config['lr']),
                                    momentum=0.9,
                                    weight_decay=float(config['weight_decay']))

    elif config['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(params, lr=float(config['lr']), weight_decay=float(config['weight_decay']))

    elif config['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(params, lr=float(config['lr']), weight_decay=float(config['weight_decay']))

    else:
        raise NotImplementedError("Invalid optimizer %s!" % config['optimizer'])

    if config['scheduler'] == 'poly':
        # Operate in each iteration
        assert config['max_iters'] is not None
        scheduler = PolynomialLR(optimizer=optimizer, max_iterations=int(config['max_iters']), gamma=0.9, min_lr=0)

    elif config['scheduler'] == 'cosine':
        # Operate in each epoch
        assert config['max_epochs'] is not None
        assert config['warmup_epochs'] is not None
        max_epochs = int(config['max_epochs'])
        warmup_epochs = int(config['warmup_epochs'])
        scheduler = CosineLRScheduler(optimizer=optimizer,
                                      t_initial=max_epochs - warmup_epochs,
                                      lr_min=1.25e-6,
                                      warmup_t=warmup_epochs,
                                      warmup_lr_init=1.25e-7,
                                      warmup_prefix=True)

    else:
        raise NotImplementedError("Invalid scheduler %s!" % config['scheduler'])

    return optimizer, scheduler


def train_one_iter_multi_decoder(tasks, batch, model, optimizer, criterion, scaler, grad_clip, train_loss, fp16):
    optimizer.zero_grad()
    batch = to_cuda(batch)
    images = batch['image']

    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=fp16):
        outputs = model(images)
        loss_dict = criterion(outputs, batch, tasks)

    # Log loss values
    for task in tasks:
        loss_value = loss_dict[task].detach().item()
        batch_size = outputs[task].size(0)
        train_loss[task].update(loss_value / batch_size, batch_size)

    scaler.scale(loss_dict['total']).backward()

    if grad_clip > 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

    scaler.step(optimizer)
    scaler.update()


def train_one_iter_task_conditional(tasks, order, batch, model, optimizer, criterion, scaler, grad_clip, train_loss,
                                    fp16):
    optimizer.zero_grad()
    batch = to_cuda(batch)
    images = batch['image']

    for i in range(len(tasks)):
        task = tasks[order[i]]
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=fp16):
            outputs = model(images, task)
            loss_dict = criterion(outputs, batch, [task])

        # Log loss values
        loss_value = loss_dict[task].detach().item()
        batch_size = outputs[task].size(0)
        train_loss[task].update(loss_value / batch_size, batch_size)

        scaler.scale(loss_dict['total']).backward()

    if grad_clip > 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

    scaler.step(optimizer)
    scaler.update()


def train_one_epoch(arch, epoch, iter_count, max_iters, tasks, train_dl, model, optimizer, scheduler, criterion, scaler,
                    grad_clip, train_loss, local_rank, fp16):
    """
    Train one batch
    """

    model.train()
    train_dl.sampler.set_epoch(epoch)

    # Random shuffle tasks for task-conditional model
    order = np.arange(len(tasks))
    np.random.shuffle(order)

    with tqdm(total=len(train_dl), disable=(local_rank != 0)) as t:
        for batch in train_dl:
            t.set_description("Epoch: %d Iter: %d" % (epoch, iter_count))
            t.update(1)

            if arch == 'md':
                train_one_iter_multi_decoder(tasks, batch, model, optimizer, criterion, scaler, grad_clip, train_loss,
                                             fp16)
            elif arch == 'tc':
                train_one_iter_task_conditional(tasks, order, batch, model, optimizer, criterion, scaler, grad_clip,
                                                train_loss, fp16)
            else:
                raise ValueError

            if scheduler.__class__.__name__ == 'PolynomialLR':
                scheduler.step()

            iter_count += 1

            if iter_count >= max_iters:
                end_signal = True
                break
            else:
                end_signal = False

    if scheduler.__class__.__name__ == 'CosineLRScheduler':
        scheduler.step(epoch)

    return end_signal, iter_count


def eval_metric(arch, tasks, dataname, val_dl, model):
    """
    Evaluate the model
    """

    performance_meter = PerformanceMeter(dataname, tasks)

    model.eval()
    with torch.no_grad():
        for batch in tqdm(val_dl, desc="Evaluating"):
            batch = to_cuda(batch)
            images = batch['image']

            if arch == 'md':
                outputs = model.module(images)  # IMPORTANT
            elif arch == 'tc':
                outputs = {}
                for task in tasks:
                    outputs.update(model.module(images, task))
            performance_meter.update({t: get_output(outputs[t], t) for t in tasks}, batch)

    eval_results = performance_meter.get_score()

    results_dict = {}
    for task in tasks:
        for key in eval_results[task]:
            results_dict['eval/' + task + '_' + key] = eval_results[task][key]

    return results_dict
