#!/usr/bin/python
""" 🪐 AI for KOSMOS - Ocean Hackathon Brest 2021

File used to train the detection model with detectron2 framework. Inspired from `here <https://wendeehsu.medium.com/instance-segmentation-with-detectron2-127fbe01b20b>`_

authors:
  @Thomas Chaigneau
  @Julien Furiga
"""

import os
import cv2
import json
import numpy as np

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer
from detectron2.data import DatasetCatalog, MetadataCatalog

def get_dicts(img_dir: str):
    """
    Function used to get the dicts used to train the model.

    Parameters
    ----------
    img_dir: str
        Path to the directory containing the images and the json annotations.

    Returns
    -------
    dicts: list
        List of dicts used to train the model.
    """

    json_file = os.path.join(img_dir, "via_export_json.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
        record = {}
        
        filename = os.path.join(img_dir, v["filename"])
        print(filename)
        height, width = cv2.imread(filename).shape[:2]
        
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
      
        annos = v["regions"]
        objs = []
        for anno in annos:
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]
            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
                "iscrowd": 0
                }
            objs.append(obj)

        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts

path = "dataset" # path to your image folder
for d in ["train"]:
    DatasetCatalog.register("fish" + d, lambda d=d: get_dicts(path + "/" +  d))
    MetadataCatalog.get("fish" + d).set(thing_classes=["fish"])

cfg = get_cfg()
cfg.OUTPUT_DIR = "output"
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

cfg.DATASETS.TRAIN = ("fishtrain")     # Our training dataset
cfg.DATASETS.TEST = () # You can add some data to test dataset
cfg.DATALOADER.NUM_WORKERS = 2     # Number of parallel data loading workers
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")     # use pretrained weights
cfg.SOLVER.IMS_PER_BATCH = 2     # In 1 iteration the model sees 2 images
cfg.SOLVER.BASE_LR = 0.00025     # Learning rate
cfg.SOLVER.MAX_ITER = 450        # Number of iteration
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128     # Number of proposals to sample for training
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Only has one class (fish)

# Training part
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
