import argparse
import os

import torch
import yaml
from tqdm import tqdm

from datasets.custom_dataset import get_dataloader, get_dataset
from datasets.custom_transforms import get_transformations
from datasets.utils.configs import TEST_SCALE
from evaluation.evaluate_utils import PerformanceMeter, predict
from models.build_models import build_model
from utils import create_pred_dir, get_output, to_cuda


def eval_metric(arch, tasks, dataname, test_dl, model, evaluate, save, pred_dir):
    if evaluate:
        performance_meter = PerformanceMeter(dataname, tasks)

    if save:
        # Save all tasks
        tasks_to_save = tasks
    else:
        # Save only edge
        tasks_to_save = ['edge'] if 'edge' in tasks else []

    assert evaluate or len(tasks_to_save) > 0

    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_dl, desc='Evaluating'):
            batch = to_cuda(batch)
            images = batch['image']
            if arch == 'md':
                outputs = model(images)
            elif arch == 'tc':
                outputs = {}
                for task in tasks:
                    outputs.update(model(images, task))
            else:
                raise ValueError

            if evaluate:
                performance_meter.update({t: get_output(outputs[t], t) for t in tasks}, batch)

            for task in tasks_to_save:
                predict(dataname, batch['meta'], outputs, task, pred_dir)

    if evaluate:
        # Get evaluation results
        eval_results = performance_meter.get_score()

        results_dict = {}
        for t in tasks:
            for key in eval_results[t]:
                results_dict[t + '_' + key] = eval_results[t][key]

        return results_dict


def test(arch, tasks, dataname, test_dl, model, evaluate, save, pred_dir):
    res = eval_metric(arch, tasks, dataname, test_dl, model, evaluate, save, pred_dir)
    # Print and log results
    if evaluate:
        test_results = {key: "%.5f" % res[key] for key in res}
        print(test_results)
        results_file = os.path.join(args.results_dir, args.exp, 'test_results.txt')
        with open(results_file, 'w') as f:
            f.write(str(test_results))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, required=True, help="Experiment name")
    parser.add_argument('--results_dir', type=str, required=True, help="Directory of results")
    parser.add_argument('--evaluate', action='store_true', help="Whether to evaluate on all tasks")
    parser.add_argument('--save', action='store_true', help="Whether to save predictions")
    parser.add_argument('--gpu_id', type=int, default=0, help="GPU id")

    args = parser.parse_args()

    with open(os.path.join(args.results_dir, args.exp, 'config.yml'), 'r') as stream:
        configs = yaml.safe_load(stream)

    # Join args and configs
    configs = {**configs, **vars(args)}

    torch.cuda.set_device(args.gpu_id)

    # Get dataset and tasks
    dataname = configs['dataset']
    task_dict = configs['task_dict']
    task_list = []
    for task_name in task_dict:
        task_list += [task_name] * task_dict[task_name]

    test_transforms = get_transformations(TEST_SCALE[dataname], train=False)
    test_ds = get_dataset(dataname, train=False, tasks=task_list, transform=test_transforms)
    test_dl = get_dataloader(train=False, configs=configs, dataset=test_ds)

    # Setup output folders
    checkpoint_dir, pred_dir = create_pred_dir(args.results_dir, args.exp, task_list)

    # Setup model
    arch = configs['arch']
    model = build_model(arch,
                        task_list,
                        dataname,
                        backbone_args=configs['backbone'],
                        decoder_args=configs['decoder'],
                        head_args=configs['head']).cuda()

    # load model from checkpoint
    checkpoint_file = os.path.join(checkpoint_dir, 'checkpoint.pth')
    if not os.path.exists(checkpoint_file):
        raise ValueError('Checkpoint %s not found!' % (checkpoint_file))

    checkpoint = torch.load(checkpoint_file, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    test(arch, task_list, dataname, test_dl, model, args.evaluate, args.save, pred_dir)
