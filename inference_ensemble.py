"""PyTorch Inference Script
An example inference script that outputs top-k class ids for images in a folder into a csv.
Hacked together by Ross Wightman (https://github.com/rwightman)
"""
import os
import time
import argparse
import logging
import numpy as np
import torch
from ensemble import ensemble
from collections import OrderedDict
import torch.nn as nn

try:
    from apex import amp
    has_apex = True
except ImportError:
    has_apex = False

from timm.models import create_model, apply_test_time_pool, is_model, list_models,\
    set_scriptable, set_no_jit
from timm.data import CocoDataset, Dataset, DatasetTar, create_loader, resolve_data_config
from timm.utils import bulk_multi_label_metrics, accuracy, AverageMeter, natural_key, setup_default_logging

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='PyTorch ImageNet Inference')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--output_dir', metavar='DIR', default='./',
                    help='path to output files')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--img-size', default=None, type=int,
                    metavar='N', help='Input image dimension')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--log-freq', default=10, type=int,
                    metavar='N', help='batch logging frequency (default: 10)')
parser.add_argument('--checkpoints', default='', type=str, metavar='PATH',
                    help='comma seperated paths to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--no-test-pool', dest='no_test_pool', action='store_true',
                    help='disable test time pool')
parser.add_argument('--topk', default=5, type=int,
                    metavar='N', help='Top-k to output to CSV')


def load_state_dict(checkpoint, use_ema=False):
    state_dict_key = 'state_dict'
    if isinstance(checkpoint, dict):
        if use_ema and 'state_dict_ema' in checkpoint:
            print('using model ema')
            state_dict_key = 'state_dict_ema'
    if state_dict_key and state_dict_key in checkpoint:
        new_state_dict = OrderedDict()
        for k, v in checkpoint[state_dict_key].items():
            # strip `module.` prefix
            name = k[7:] if k.startswith('module') else k
            new_state_dict[name] = v
        state_dict = new_state_dict
    else:
        state_dict = checkpoint
    logging.info("Loaded {} from checkpoint".format(state_dict_key))
    return state_dict


def load_checkpoint(model, checkpoint, use_ema=False, strict=True):
    state_dict = load_state_dict(checkpoint, use_ema)
    model.load_state_dict(state_dict, strict=strict)


def inference(args, checkpoint=None):
    args.pretrained = args.pretrained or not args.checkpoints

    # create model
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        in_chans=3,
    )

    if checkpoint:
        load_checkpoint(model, checkpoint, args.use_ema)

    logging.info('Model %s created, param count: %d' %
                 (args.model, sum([m.numel() for m in model.parameters()])))

    config = resolve_data_config(vars(args), model=model)
    model, test_time_pool = apply_test_time_pool(model, config, args)

    if args.amp:
        model = amp.initialize(model.cuda(), opt_level='O1')
    else:
        model = model.cuda()

    if args.num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu)))

    if args.use_multi_label:
        criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss().cuda()

    crop_pct = 1.0 if test_time_pool else config['crop_pct']
    loader = create_loader(
        Dataset(args.data),
        input_size=config['input_size'],
        batch_size=args.batch_size,
        use_prefetcher=True,
        interpolation=config['interpolation'],
        mean=config['mean'],
        std=config['std'],
        num_workers=args.workers,
        crop_pct=crop_pct)

    model.eval()

    k = min(args.topk, args.num_classes)
    batch_time = AverageMeter()
    end = time.time()

    logits = []
    with torch.no_grad():
        for batch_idx, (input, _) in enumerate(loader):
            input = input.cuda()
            output = model(input)
            logits.append(torch.sigmoid(output).cpu().numpy())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % args.log_freq == 0:
                logging.info('Predict: [{0}/{1}] Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
                        batch_idx, len(loader), batch_time=batch_time))

    logits = np.concatenate(logits, axis=0)

    filenames = loader.dataset.filenames()

    return logits, filenames

def main():
    setup_default_logging()
    args = parser.parse_args()

    logits_list = []
    for checkpoint in args.checkpoints.split(','):
        data = torch.load(checkpoint, map_location='cpu')
        args.model = data['args'].model
        args.amp   = data['args'].amp
        args.use_coco_dataset = data['args'].use_coco_dataset
        args.use_multi_label = data['args'].use_multi_label
        args.num_classes = data['args'].num_classes
        args.use_ema = data['args'].model_ema

        logits, filenames = inference(args, data)
        logits_list.append(logits)

    _, predictions = ensemble(logits_list, thresholds=np.array([0.35, 0.45, 0.3, 0.6, 0.3, 0.6, 0.5, 0.55]))
    labels = ["can", "plastic", "paper", "vinyl", "normal", "food", "glass", "styrofoam"]
    with open(os.path.join(args.output_dir, './predictions.csv'), 'w') as out_file:
        out_file.write('filename' + ',' + ','.join(labels) + '\n')
        for filename, prediction in zip(filenames, predictions):
            filename = os.path.basename(filename)

            label = ','.join(map(str, map(int, prediction)))
            out_file.write('{0},{1}\n'.format(filename, label))


if __name__ == '__main__':
    main()
