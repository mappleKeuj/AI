
import os

from pycocotools.coco import COCO
from torch.utils.data import DataLoader

from ssd.utils import SSDTransformer
from ssd.utils import dboxes300_coco, OpenImagesDataset, COCODetection


def get_coco_ground_truth(val_annotate_path):
    cocoGt = COCO(annotation_file=val_annotate_path)
    return cocoGt


def get_dataset(root, annotate, labels):
    dboxes = dboxes300_coco()
    transforms = SSDTransformer(dboxes, (300, 300), val=False)

    dataset = OpenImagesDataset(root, annotate, labels, transforms)
    return dataset


def get_val_dataset(val_annotate, val_coco_root):
    dboxes = dboxes300_coco()
    val_trans = SSDTransformer(dboxes, (300, 300), val=False)

    val_coco = COCODetection(val_coco_root, val_annotate, val_trans)
    return val_coco


def get_dataloader(dataset, batch_size, shuffle, num_workers):
    dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=shuffle, 
                                num_workers=num_workers, 
                                pin_memory=True,
                                drop_last=True)

    return dataloader
