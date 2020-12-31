""" Quick n Simple Image Folder, Tarfile based DataSet

Hacked together by / Copyright 2020 Ross Wightman
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data

import os
import re
import torch
import tarfile
from PIL import Image


IMG_EXTENSIONS = ['.png', '.jpg', '.jpeg']


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def find_images_and_targets(folder, types=IMG_EXTENSIONS, class_to_idx=None, leaf_name_only=True, sort=True):
    labels = []
    filenames = []
    for root, subdirs, files in os.walk(folder, topdown=False):
        rel_path = os.path.relpath(root, folder) if (root != folder) else ''
        label = os.path.basename(rel_path) if leaf_name_only else rel_path.replace(os.path.sep, '_')
        for f in files:
            base, ext = os.path.splitext(f)
            if ext.lower() in types:
                filenames.append(os.path.join(root, f))
                labels.append(label)
    if class_to_idx is None:
        # building class index
        unique_labels = set(labels)
        sorted_labels = list(sorted(unique_labels, key=natural_key))
        class_to_idx = {c: idx for idx, c in enumerate(sorted_labels)}
    images_and_targets = [(f, class_to_idx[l]) for f, l in zip(filenames, labels) if l in class_to_idx]
    if sort:
        images_and_targets = sorted(images_and_targets, key=lambda k: natural_key(k[0]))
    return images_and_targets, class_to_idx


def load_class_map(filename, root=''):
    class_map_path = filename
    if not os.path.exists(class_map_path):
        class_map_path = os.path.join(root, filename)
        assert os.path.exists(class_map_path), 'Cannot locate specified class map file (%s)' % filename
    class_map_ext = os.path.splitext(filename)[-1].lower()
    if class_map_ext == '.txt':
        with open(class_map_path) as f:
            class_to_idx = {v.strip(): k for k, v in enumerate(f)}
    else:
        assert False, 'Unsupported class map extension'
    return class_to_idx


class Dataset(data.Dataset):

    def __init__(
            self,
            root,
            load_bytes=False,
            transform=None,
            class_map=''):

        class_to_idx = None
        if class_map:
            class_to_idx = load_class_map(class_map, root)
        images, class_to_idx = find_images_and_targets(root, class_to_idx=class_to_idx)
        if len(images) == 0:
            raise RuntimeError(f'Found 0 images in subfolders of {root}. '
                               f'Supported image extensions are {", ".join(IMG_EXTENSIONS)}')
        self.root = root
        self.samples = images
        self.imgs = self.samples  # torchvision ImageFolder compat
        self.class_to_idx = class_to_idx
        self.load_bytes = load_bytes
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = open(path, 'rb').read() if self.load_bytes else Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if target is None:
            target = torch.zeros(1).long()
        return img, target

    def __len__(self):
        return len(self.samples)

    def filename(self, index, basename=False, absolute=False):
        filename = self.samples[index][0]
        if basename:
            filename = os.path.basename(filename)
        elif not absolute:
            filename = os.path.relpath(filename, self.root)
        return filename

    def filenames(self, basename=False, absolute=False):
        fn = lambda x: x
        if basename:
            fn = os.path.basename
        elif not absolute:
            fn = lambda x: os.path.relpath(x, self.root)
        return [fn(x[0]) for x in self.samples]


def _extract_tar_info(tarfile, class_to_idx=None, sort=True):
    files = []
    labels = []
    for ti in tarfile.getmembers():
        if not ti.isfile():
            continue
        dirname, basename = os.path.split(ti.path)
        label = os.path.basename(dirname)
        ext = os.path.splitext(basename)[1]
        if ext.lower() in IMG_EXTENSIONS:
            files.append(ti)
            labels.append(label)
    if class_to_idx is None:
        unique_labels = set(labels)
        sorted_labels = list(sorted(unique_labels, key=natural_key))
        class_to_idx = {c: idx for idx, c in enumerate(sorted_labels)}
    tarinfo_and_targets = [(f, class_to_idx[l]) for f, l in zip(files, labels) if l in class_to_idx]
    if sort:
        tarinfo_and_targets = sorted(tarinfo_and_targets, key=lambda k: natural_key(k[0].path))
    return tarinfo_and_targets, class_to_idx


class DatasetTar(data.Dataset):

    def __init__(self, root, load_bytes=False, transform=None, class_map=''):

        class_to_idx = None
        if class_map:
            class_to_idx = load_class_map(class_map, root)
        assert os.path.isfile(root)
        self.root = root
        with tarfile.open(root) as tf:  # cannot keep this open across processes, reopen later
            self.samples, self.class_to_idx = _extract_tar_info(tf, class_to_idx)
        self.imgs = self.samples
        self.tarfile = None  # lazy init in __getitem__
        self.load_bytes = load_bytes
        self.transform = transform

    def __getitem__(self, index):
        if self.tarfile is None:
            self.tarfile = tarfile.open(self.root)
        tarinfo, target = self.samples[index]
        iob = self.tarfile.extractfile(tarinfo)
        img = iob.read() if self.load_bytes else Image.open(iob).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if target is None:
            target = torch.zeros(1).long()
        return img, target

    def __len__(self):
        return len(self.samples)

    def filename(self, index, basename=False):
        filename = self.samples[index][0].name
        if basename:
            filename = os.path.basename(filename)
        return filename

    def filenames(self, basename=False):
        fn = os.path.basename if basename else lambda x: x
        return [fn(x[0].name) for x in self.samples]


class AugMixDataset(torch.utils.data.Dataset):
    """Dataset wrapper to perform AugMix or other clean/augmentation mixes"""

    def __init__(self, dataset, num_splits=2):
        self.augmentation = None
        self.normalize = None
        self.dataset = dataset
        if self.dataset.transform is not None:
            self._set_transforms(self.dataset.transform)
        self.num_splits = num_splits

    def _set_transforms(self, x):
        assert isinstance(x, (list, tuple)) and len(x) == 3, 'Expecting a tuple/list of 3 transforms'
        self.dataset.transform = x[0]
        self.augmentation = x[1]
        self.normalize = x[2]

    @property
    def transform(self):
        return self.dataset.transform

    @transform.setter
    def transform(self, x):
        self._set_transforms(x)

    def _normalize(self, x):
        return x if self.normalize is None else self.normalize(x)

    def __getitem__(self, i):
        x, y = self.dataset[i]  # all splits share the same dataset base transform
        x_list = [self._normalize(x)]  # first split only normalizes (this is the 'clean' split)
        # run the full augmentation on the remaining splits
        for _ in range(self.num_splits - 1):
            x_list.append(self._normalize(self.augmentation(x)))
        return tuple(x_list), y

    def __len__(self):
        return len(self.dataset)

import torch.utils.data as data
import os.path
import json
from collections import defaultdict
from PIL import Image  
import numpy as np
    
class CocoDataset(data.Dataset):
    def _get_class_to_idx_from_annotation(self, annotation):
        assert 'categories' in annotation, 'wrong coco annotation format'

        return {
            it['name']: it['id'] for it in annotation['categories']
        }


    def _get_samples_from_annotation(self, annotation, base_path=''):
        assert 'images' in annotation, 'wrong coco annotation format'
        assert 'annotations' in annotation, 'wrong coco annotation format'

        image_label_dict = defaultdict(list)

        for it in annotation['annotations']:
            category_id = it['category_id']
            image_id = it['image_id']

            image_label_dict[image_id].append(category_id)

        return [
            (
                os.path.join(base_path, it['file_name']), tuple(set(image_label_dict[it['id']]))
            ) for it in annotation['images']
        ]

    def __init__(
        self,
        root,
        load_bytes=False,
        transform=None,
        **_,
    ):
        self.root = root
        self.load_bytes = load_bytes
        self.transform = transform
        
        
        data_base_path = os.path.join(root, 'data/')
        annotation_file_path = os.path.join(root, 'annotations.json')
        
        with open(annotation_file_path, 'r') as fp:
            annotation = json.loads(fp.read())
            
        self.class_to_idx = self._get_class_to_idx_from_annotation(annotation)
        self.samples = self._get_samples_from_annotation(annotation, base_path=data_base_path)
    
    def __getitem__(self, index):
        path, target = self.samples[index]
        img = open(path, 'rb').read() if self.load_bytes else Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
            
        label = np.zeros(len(self.class_to_idx))
        for t in target:
            label[t] = 1.0
        
        return img, label
    
    def __len__(self):
        return len(self.samples)
    
    def filenames(self, indices=[], basename=False):
        if indices:
            if basename:
                return [os.path.basename(self.samples[i][0]) for i in indices]
            else:
                return [self.samples[i][0] for i in indices]
        else:
            if basename:
                return [os.path.basename(x[0]) for x in self.samples]
            else:
                return [x[0] for x in self.samples]
            
class MultiLabelDataset(data.Dataset):
    def _get_class_to_idx_from_annotation(self, annotation):
        assert 'classes' in annotation, 'wrong format'

        return {
            it['classname']: it['class_idx'] for it in annotation['classes']
        }


    def _get_samples_from_annotation(self, annotation, base_path='', shy_pct=.0):
        assert 'images' in annotation, 'wrong format'
        assert 'annotations' in annotation, 'wrong format'

        def is_crawled(image):
            return 'crawled' in image and image['crawled']
        
        images = {image['image_idx']: {'filename': image['filename'], 'label': [shy_pct if is_crawled(image) else 0] * len(annotation['classes'])} for image in annotation['images']}
        for it in annotation['annotations']:
            images[it['image_idx']]['label'][it['class_idx']] = 1
#             images[it['image_idx']]['label'][it['class_idx']] = 1 - shy_pct
            
        return [(it['filename'], it['label']) for it in images.values()]

    def __init__(
        self,
        root,
        transform=None,
        shy_pct=.0,
        feed_filename=False,
        **_,
    ):
        self.root = root
        self.transform = transform
        
        
        data_base_path = os.path.join(root, 'data/')
        annotation_file_path = os.path.join(root, 'annotations.json')
        
        with open(annotation_file_path, 'r') as fp:
            annotation = json.loads(fp.read())
            
        self.class_to_idx = self._get_class_to_idx_from_annotation(annotation)
        self.samples = self._get_samples_from_annotation(annotation, base_path=data_base_path, shy_pct=shy_pct)
        self.feed_filename = feed_filename
    
    def __getitem__(self, index):
        path, label = self.samples[index]
        img = Image.open(os.path.join(self.root, 'data', path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
                   
        if self.feed_filename:
            return (img, np.array(label, dtype=np.float)), path
        else:
            return img, np.array(label, dtype=np.float)
    
    def __len__(self):
        return len(self.samples)
    
    def filenames(self, indices=[], basename=False):
        if indices:
            if basename:
                return [os.path.basename(self.samples[i][0]) for i in indices]
            else:
                return [self.samples[i][0] for i in indices]
        else:
            if basename:
                return [os.path.basename(x[0]) for x in self.samples]
            else:
                return [x[0] for x in self.samples]
