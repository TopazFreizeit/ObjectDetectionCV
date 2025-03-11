import os
import torch
import torchvision.transforms.v2
from torch.utils.data.dataset import Dataset
import xml.etree.ElementTree as ET
from torchvision import tv_tensors
from torchvision.io import read_image

import os
import xml.etree.ElementTree as ET

def load_images_and_anns(im_sets, label2idx, split, chosen_class='person'):
    im_infos = []
    for im_set in im_sets:
        im_names = [line.strip() for line in open(os.path.join(
            im_set, 'ImageSets', 'Main', '{}.txt'.format(split)))]
        ann_dir = os.path.join(im_set, 'Annotations')
        im_dir = os.path.join(im_set, 'JPEGImages')

        for im_name in im_names:
            ann_file = os.path.join(ann_dir, '{}.xml'.format(im_name))
            im_info = {}
            ann_info = ET.parse(ann_file)
            root = ann_info.getroot()
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            im_info['img_id'] = os.path.basename(ann_file).split('.xml')[0]
            im_info['filename'] = os.path.join(im_dir, '{}.jpg'.format(im_info['img_id']))
            im_info['width'] = width
            im_info['height'] = height
            detections = []

            # Only process objects of the chosen class
            for obj in ann_info.findall('object'):
                class_name = obj.find('name').text
                if class_name == chosen_class:
                    det = {}
                    det['label'] = label2idx[chosen_class]  # Maps to 1 for 'person'
                    difficult = int(obj.find('difficult').text)
                    bbox_info = obj.find('bndbox')
                    bbox = [
                        int(bbox_info.find('xmin').text) - 1,
                        int(bbox_info.find('ymin').text) - 1,
                        int(bbox_info.find('xmax').text) - 1,
                        int(bbox_info.find('ymax').text) - 1
                    ]
                    det['bbox'] = bbox
                    det['difficult'] = difficult
                    detections.append(det)

            # Include image if it has one or more objects of the chosen class
            if len(detections) >= 1:
                im_info['detections'] = detections
                im_infos.append(im_info)

    print(f"Total {len(im_infos)} images found after filtering")
    return im_infos

class VOCDataset(Dataset):
    def __init__(self, split, im_sets, im_size=300, chosen_class='person'):
        self.split = split
        self.im_sets = im_sets
        self.fname = 'trainval' if self.split == 'train' else 'test'
        self.im_size = im_size
        self.im_mean = [123.0, 117.0, 104.0]
        self.imagenet_mean = [0.485, 0.456, 0.406]
        self.imagenet_std = [0.229, 0.224, 0.225]
        self.chosen_class = chosen_class

        # Transformations remain the same
        self.transforms = {
            'train': torchvision.transforms.v2.Compose([
                torchvision.transforms.v2.RandomPhotometricDistort(),
                torchvision.transforms.v2.RandomZoomOut(fill=self.im_mean),
                torchvision.transforms.v2.RandomIoUCrop(),
                torchvision.transforms.v2.RandomHorizontalFlip(p=0.5),
                torchvision.transforms.v2.Resize(size=(self.im_size, self.im_size)),
                torchvision.transforms.v2.SanitizeBoundingBoxes(
                    labels_getter=lambda transform_input:
                    (transform_input[1]["labels"], transform_input[1]["difficult"])),
                torchvision.transforms.v2.ToPureTensor(),
                torchvision.transforms.v2.ToDtype(torch.float32, scale=True),
                torchvision.transforms.v2.Normalize(mean=self.imagenet_mean,
                                                    std=self.imagenet_std)
            ]),
            'test': torchvision.transforms.v2.Compose([
                torchvision.transforms.v2.Resize(size=(self.im_size, self.im_size)),
                torchvision.transforms.v2.ToPureTensor(),
                torchvision.transforms.v2.ToDtype(torch.float32, scale=True),
                torchvision.transforms.v2.Normalize(mean=self.imagenet_mean,
                                                    std=self.imagenet_std)
            ]),
        }

        # Define classes (unchanged)
        classes = ['person']
        classes = sorted(classes)
        classes = ['background'] + classes
        self.label2idx = {'background': 0, self.chosen_class: 1}
        self.idx2label = {0: 'background', 1: self.chosen_class}
        print(self.idx2label)

        # Load filtered images
        self.images_info = load_images_and_anns(self.im_sets, self.label2idx, self.split, self.chosen_class)

    def __len__(self):
        return len(self.images_info)

    def __getitem__(self, index):
        im_info = self.images_info[index]
        im = read_image(im_info['filename'])

        targets = {}
        targets['bboxes'] = tv_tensors.BoundingBoxes(
            [detection['bbox'] for detection in im_info['detections']],
            format='XYXY', canvas_size=im.shape[-2:])
        # Create labels for all objects (all are the same class with label 1)
        targets['labels'] = torch.as_tensor([detection['label'] for detection in im_info['detections']])
        targets['difficult'] = torch.as_tensor(
            [detection['difficult'] for detection in im_info['detections']])

        transformed_info = self.transforms[self.split](im, targets)
        im_tensor, targets = transformed_info

        h, w = im_tensor.shape[-2:]
        wh_tensor = torch.as_tensor([[w, h, w, h]]).expand_as(targets['bboxes'])
        targets['bboxes'] = targets['bboxes'] / wh_tensor
        return im_tensor, targets, im_info['filename']