import argparse
import os.path as osp
import os
import shutil
import tempfile

import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import load_checkpoint, get_dist_info
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

from mmdet.apis import init_dist
from mmdet.core import results2json, coco_eval
from mmdet.datasets import build_dataloader, get_dataset
from mmdet.models import build_detector
import time

def get_time_str():
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())

def single_gpu_test(model, data_loader, show=False, log_dir=None):
    model.eval()
    results = []
    dataset = data_loader.dataset
    if log_dir != None:
        filename = 'inference{}.log'.format(get_time_str())
        log_file = osp.join(log_dir, filename)
        f = open(log_file, 'w')
        prog_bar = mmcv.ProgressBar(len(dataset), file=f)
    else:
        prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=not show, **data)
        results.append(result)

        if show:
            model.module.show_result(data, result, dataset.img_norm_cfg)

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()

    return results


def multi_gpu_test(model, data_loader, tmpdir=None):
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        results.append(result)

        if rank == 0:
            batch_size = data['img'][0].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    results = collect_results(results, len(dataset), tmpdir)

    return results


def collect_results(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        choices=['proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'],
        help='eval types')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument('--log_dir', help='log the inference speed')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    return args


def before_run(taget_dir):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except:
        from tensorboardX import SummaryWriter
    log_dir = osp.join(taget_dir, 'tf_logs_epoch')
    if not osp.isdir(log_dir):
        os.mkdir(log_dir)
        
    writer = SummaryWriter(log_dir)
    
    return writer


def log(stats, stats_category, writer, epoch_num):
    # tags = All_Category_mAP
    metrics = ['mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l']
    res_type = 'bbox'
    log_buffer = {}
    log_buffer_category = stats_category
    for i in range(len(metrics)):
        key = '{}_{}'.format(res_type, metrics[i])
        val = float('{:.3f}'.format(stats[i]))
        log_buffer[key] = val
    
    for tag, val in log_buffer.items():
        if isinstance(val, str):
            writer.add_text("All Categories/" + tag, val, epoch_num)
        else:
            writer.add_scalar("All Categories/" + tag, val, epoch_num)

    for tag, val in log_buffer_category.items():
        if isinstance(val, str):
            writer.add_text("One Category/" + tag, val, epoch_num)
        else:
            writer.add_scalar("One Category/" + tag, float('{:.3f}'.format(val)), epoch_num)

    print("logged")


def after_run(writer):
    writer.close()


def main():
    args = parse_args()

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = get_dataset(cfg.data.val)
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    epoch_num = 12
    writer = before_run(args.checkpoint)
    while True:
        f = os.listdir(args.checkpoint)
        if 'epoch_{}.pth'.format(epoch_num) in f:
            print('epoch_{}.pth'.format(epoch_num), "evaluation")
            checkpoint_path = os.path.join(args.checkpoint, 'epoch_{}.pth'.format(epoch_num))

            model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
            checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')
            # old versions did not save class info in checkpoints, this walkaround is
            # for backward compatibility
            if 'CLASSES' in checkpoint['meta']:
                model.CLASSES = checkpoint['meta']['CLASSES']
                print("checkpoint['meta']['CLASSES']: ", model.CLASSES)
                print(dataset.CLASSES)

            else:
                model.CLASSES = dataset.CLASSES
                print("dataset.CLASSES: ", dataset.CLASSES)
                print(checkpoint['meta']['CLASSES'])

            if not distributed:
                model = MMDataParallel(model, device_ids=[0])
                outputs = single_gpu_test(model, data_loader, args.show, args.log_dir)
            else:
                model = MMDistributedDataParallel(model.cuda())
                outputs = multi_gpu_test(model, data_loader, args.tmpdir)

            rank, _ = get_dist_info()
            if args.out and rank == 0:
                print('\nwriting results to {}'.format(args.out))
                mmcv.dump(outputs, args.out)
                eval_types = args.eval
                if eval_types:
                    print('Starting evaluate {}'.format(' and '.join(eval_types)))
                    if eval_types == ['proposal_fast']:
                        result_file = args.out
                        stats, stats_category = coco_eval(result_file, eval_types, dataset.coco)
                        log(stats, stats_category, writer, epoch_num)
                    else:
                        if not isinstance(outputs[0], dict):
                            result_file = args.out + '.json'
                            results2json(dataset, outputs, result_file)
                            stats, stats_category = coco_eval(result_file, eval_types, dataset.coco)
                            log(stats, stats_category, writer, epoch_num)
                        else:
                            for name in outputs[0]:
                                print('\nEvaluating {}'.format(name))
                                outputs_ = [out[name] for out in outputs]
                                result_file = args.out + '.{}.json'.format(name)
                                results2json(dataset, outputs_, result_file)
                                stats, stats_category = coco_eval(result_file, eval_types, dataset.coco)
                                log(stats, stats_category, writer, epoch_num)
            epoch_num += 1
            if epoch_num >= 101:
                break

    after_run(writer) 

if __name__ == '__main__':
    main()
