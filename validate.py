#!/usr/bin/env python
""" ImageNet Validation Script

This is intended to be a lean and easily modifiable ImageNet validation script for evaluating pretrained
models or training checkpoints against ImageNet or similarly organized image datasets. It prioritizes
canonical PyTorch, standard Python style, and good performance. Repurpose as you see fit.

Hacked together by Ross Wightman (https://github.com/rwightman)
"""
import argparse
import os
import csv
import glob
import time
import logging
import torch
import torch.nn as nn
import torch.nn.parallel
import numpy as np
import json
from collections import OrderedDict
from contextlib import suppress
from ensemble import calibrate

from timm.models import create_model, apply_test_time_pool, load_checkpoint, is_model, list_models
from timm.data import MultiLabelDataset, CocoDataset, Dataset, DatasetTar, create_loader, resolve_data_config, RealLabelsImagenet
from timm.utils import bulk_multi_label_metrics, accuracy, AverageMeter, natural_key, setup_default_logging, set_jit_legacy

has_apex = False
try:
    from apex import amp
    has_apex = True
except ImportError:
    pass

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('validate')


parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--img-size', default=None, type=int,
                    metavar='N', help='Input image dimension, uses model default if empty')
parser.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='Input image center crop pct')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float,  nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                    help='path to class to idx mapping file (default: "")')
parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
parser.add_argument('--log-freq', default=10, type=int,
                    metavar='N', help='batch logging frequency (default: 10)')
parser.add_argument('--checkpoints', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--no-test-pool', dest='no_test_pool', action='store_true',
                    help='disable test time pool')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--amp', action='store_true', default=False,
                    help='Use AMP mixed precision. Defaults to Apex, fallback to native Torch AMP.')
parser.add_argument('--apex-amp', action='store_true', default=False,
                    help='Use NVIDIA Apex AMP mixed precision')
parser.add_argument('--native-amp', action='store_true', default=False,
                    help='Use Native Torch AMP mixed precision')
parser.add_argument('--tf-preprocessing', action='store_true', default=False,
                    help='Use Tensorflow preprocessing pipeline (require CPU TF installed')
parser.add_argument('--use-ema', dest='use_ema', action='store_true',
                    help='use ema version of weights if present')
parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                    help='convert model torchscript for inference')
parser.add_argument('--legacy-jit', dest='legacy_jit', action='store_true',
                    help='use legacy jit mode for pytorch 1.5/1.5.1/1.6 to get back fusion performance')
parser.add_argument('--results-file', default='', type=str, metavar='FILENAME',
                    help='Output csv file for validation results (summary)')
parser.add_argument('--real-labels', default='', type=str, metavar='FILENAME',
                    help='Real labels JSON file for imagenet evaluation')
parser.add_argument('--valid-labels', default='', type=str, metavar='FILENAME',
                    help='Valid label indices txt file for validation of partial label space')

import torch
import torch.nn as nn
from copy import deepcopy
import torch.utils.model_zoo as model_zoo
import os
import logging
from collections import OrderedDict
from timm.models.layers.conv2d_same import Conv2dSame


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
    
def get_model(args, checkpoint):
    # create model
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        in_chans=3,
        global_pool=args.gp,
        scriptable=args.torchscript)

    if checkpoint:
        load_checkpoint(model, checkpoint, args.use_ema)

    param_count = sum([m.numel() for m in model.parameters()])
    _logger.info('Model %s created, param count: %d' % (args.model, param_count))

    data_config = resolve_data_config(vars(args), model=model)
    model, test_time_pool = (model, False) if args.no_test_pool else apply_test_time_pool(model, data_config)
    print('test_time_pool:', test_time_pool)

    if args.torchscript:
        torch.jit.optimized_execution(True)
        model = torch.jit.script(model)

    model = model.cuda()
    if args.apex_amp:
        model = amp.initialize(model, opt_level='O1')

    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    if args.num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu)))
    model.eval()
    
    return model, data_config
    
def validate(args, checkpoints=list()):
    # might as well try to validate something
    args.pretrained = False
    args.prefetcher = not args.no_prefetcher
    amp_autocast = suppress  # do nothing
    if args.amp:
        if has_apex:
            args.apex_amp = True
        elif has_native_amp:
            args.native_amp = True
        else:
            _logger.warning("Neither APEX or Native Torch AMP is available, using FP32.")
    assert not args.apex_amp or not args.native_amp, "Only one AMP mode should be set."
    if args.native_amp:
        amp_autocast = torch.cuda.amp.autocast

    if args.legacy_jit:
        set_jit_legacy()
        
    
    models = []
    for checkpoint in checkpoints:
        data = torch.load(checkpoint, map_location='cpu')

        args.model = data['args'].model
        args.amp   = data['args'].amp
        args.num_classes = data['args'].num_classes
        args.use_ema = data['args'].model_ema

        model, data_config = get_model(args, data)
        models.append(model)

    args.dataset_type = data['args'].dataset_type
    args.use_multi_label = data['args'].use_multi_label

        
    if args.use_multi_label:
        criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss().cuda()

    #from torchvision.datasets import ImageNet
    #dataset = ImageNet(args.data, split='val')
    if args.dataset_type == 'default':
        dataset = Dataset(args.data)
    elif args.dataset_type == 'coco':
        dataset = CocoDataset(args.data)
    elif args.dataset_type == 'multi':
        dataset = MultiLabelDataset(args.data, feed_filename=True)
    else:
        raise NotImplemented

    if args.valid_labels:
        with open(args.valid_labels, 'r') as f:
            valid_labels = {int(line.rstrip()) for line in f}
            valid_labels = [i in valid_labels for i in range(args.num_classes)]
    else:
        valid_labels = None

    if args.real_labels:
        real_labels = RealLabelsImagenet(dataset.filenames(basename=True), real_json=args.real_labels)
    else:
        real_labels = None

#     crop_pct = 1.0 if test_time_pool else data_config['crop_pct']
    print('crop_pct:', data_config['crop_pct'])
    crop_pct = data_config['crop_pct']
    
    data_config['input_size'] = (720, 1280)
    loader = create_loader(
        dataset,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        crop_pct=1.0, # crop_pct,
        pin_memory=args.pin_mem,
        tf_preprocessing=args.tf_preprocessing)

    batch_time = AverageMeter()

    import ttach as tta
    
    tta_model = tta.ClassificationTTAWrapper(
        models=[torch.nn.Sequential(
            model,
            torch.nn.Sigmoid(),
        ) for model in models],
        transforms=tta.Compose(
            [
                tta.FiveCrops(args.img_size, args.img_size),
                tta.HorizontalFlip(),
#                 tta.Rotate90(angles=[0, 90, 180, 270]),
#                 tta.Scale(scales=[1, 2, 4]),
#                 tta.Multiply(factors=[0.9, 1, 1.1]),        
            ]
        ),
        merge_mode='max',
    )
    
    with torch.no_grad():
        # warmup, reduce variability of first batch time, especially for comparing torchscript vs non
        input = torch.randn((args.batch_size, 3) + data_config['input_size']).cuda()
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)
        model(input)
        
        preds, targets, filenames = [], [], []
        end = time.time()
        for batch_idx, ((input, target), filename) in enumerate(loader):
            if args.no_prefetcher:
                target = target.cuda()
                input = input.cuda()
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

#             input = input[:, :, :, 280:280+720]
            
#             from IPython.display import display
#             from PIL import Image
#             display(Image.fromarray(((input[0].cpu().permute(1, 2, 0)*torch.FloatTensor(data_config['std'])+torch.FloatTensor(data_config['mean']))*255).byte().numpy()))
            
            # compute output
            with amp_autocast():
#                 output = model(input)
                output = tta_model(input)

            if valid_labels is not None:
                output = output[:, valid_labels]
            loss = criterion(output, target)

            if real_labels is not None:
                real_labels.add_result(output)

            # measure accuracy and record loss
#             pred = torch.sigmoid(output)
            pred = output
            preds.append(pred.cpu().numpy())
            targets.append(target.cpu().numpy())
            filenames.extend(filename)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % args.log_freq == 0:
                _logger.info(
                    'Test: [{0:>4d}/{1}]  '
                    'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '.format(
                        batch_idx, len(loader), batch_time=batch_time,
                        rate_avg=input.size(0) / batch_time.avg
                    )
                )
        
        preds = np.concatenate(preds)
        targets = np.concatenate(targets)

    return preds, targets, filenames

def get_metric(predictions, targets):
    return bulk_multi_label_metrics(predictions, targets, 0.5)

from sklearn.metrics import f1_score, precision_score, recall_score

def main():
    setup_default_logging()
    args, _ = parser.parse_known_args()

    data_dir = args.data

    preds_list = []
    args.data = os.path.join(data_dir, 'val')

    preds, targets, filenames = validate(args, args.checkpoints.split(','))

    thresholds = calibrate(preds, targets)
    print('thresholds: ', thresholds)

    predicts = preds > thresholds

    f1 = f1_score(targets, predicts, average='macro')
    print('calibrated_f1(macro): ', f1)
    f1 = f1_score(targets, predicts, average=None)
    print('calibrated_f1(None): ', f1)

    metric = get_metric(preds, targets)
    print(metric)
    
    annotations = [
        {
            'id': idx,
            'file_name': filename,
            'object': [
                {
                    'box': [0, 0, 1, 1],
                    'label': 'c{}'.format(label+1)
                } for label in range(7) if predict[label]
            ]
        } for idx, (filename, predict) in enumerate(sorted(zip(filenames, predicts)))
    ]
    with open('/aichallenge/t3_res_caffeinism.json', 'w') as f:
        f.write(json.dumps({'annotations': annotations}))

def write_results(results_file, results):
    with open(results_file, mode='w') as cf:
        dw = csv.DictWriter(cf, fieldnames=results[0].keys())
        dw.writeheader()
        for r in results:
            dw.writerow(r)
        cf.flush()


if __name__ == '__main__':
    main()
