#!/usr/bin/env python
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
from collections import OrderedDict
import torch.nn as nn
import json
import ttach as tta

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
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--img-size', default=912, type=int,
                    metavar='N', help='Input image dimension')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--log-freq', default=10, type=int,
                    metavar='N', help='batch logging frequency (default: 10)')
parser.add_argument('--checkpoints', default='/checkpoints/checkpoint-1.pth.tar,/checkpoints/checkpoint-2.pth.tar', type=str, metavar='PATH',
                    help='comma seperated paths to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--no-test-pool', dest='no_test_pool', action='store_true',
                    help='disable test time pool')
parser.add_argument('--topk', default=5, type=int,
                    metavar='N', help='Top-k to output to CSV')
parser.add_argument('--crop-pct', default=1.0, type=float,
                    metavar='N', help='Input image center crop percent (for validation only)')


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
    )

    if checkpoint:
        load_checkpoint(model, checkpoint, args.use_ema)

    logging.info('Model %s created, param count: %d' %
                 (args.model, sum([m.numel() for m in model.parameters()])))

    config = resolve_data_config(vars(args), model=model)
    model, test_time_pool = (model, False) if args.no_test_pool else apply_test_time_pool(model, config)
    
    model = model.cuda()
    model.eval()
    
    return model, config
def inference(args, checkpoints=list()):
    args.pretrained = False

    models = []
    for checkpoint in checkpoints:
        data = torch.load(checkpoint, map_location='cpu')
        args.model = data['args'].model
        args.use_multi_label = data['args'].use_multi_label
        args.num_classes = data['args'].num_classes
        args.use_ema = data['args'].model_ema

        model, config = get_model(args, data)
        models.append(model)
        
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


    if args.use_multi_label:
        criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss().cuda()

    config['input_size'] = (720, 1280)
    loader = create_loader(
        Dataset(args.data),
        input_size=config['input_size'],
        batch_size=args.batch_size,
        use_prefetcher=True,
        interpolation=config['interpolation'],
        mean=config['mean'],
        std=config['std'],
        num_workers=args.workers,
        crop_pct=1)

    k = min(args.topk, args.num_classes)
    batch_time = AverageMeter()
    end = time.time()

    preds = []
    with torch.no_grad():
        for batch_idx, (input, _) in enumerate(loader):
            input = input.cuda()
            output = model(input)
            preds.append(output.cpu().numpy())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % args.log_freq == 0:
                logging.info('Predict: [{0}/{1}] Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
                        batch_idx, len(loader), batch_time=batch_time))

    preds = np.concatenate(preds, axis=0)

    filenames = loader.dataset.filenames()

    return preds, filenames

def main():
    setup_default_logging()
    args = parser.parse_args()
    
    preds, filenames = inference(args, args.checkpoints.split(','))    
    thresholds=np.array([[0.65, 0.15, 0.75, 0.6,  0.45, 0.75, 0.85]])
    
    predictions = preds > thresholds

    annotations = [
        {
            'id': idx,
            'file_name': filename,
            'object': [
                {
                    'box': [0, 0, 1, 1],
                    'label': 'c{}'.format(label+1)
                } for label in range(7) if prediction[label]
            ]
        } for idx, (filename, prediction) in enumerate(zip(filenames, predictions))
    ]
    with open('/aichallenge/t3_res_caffeinism.json', 'w') as f:
        f.write(json.dumps({'annotations': annotations}))

if __name__ == '__main__':
    main()
