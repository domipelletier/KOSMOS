#!/usr/bin/python
""" 🪐 AI for KOSMOS - Ocean Hackathon Brest 2021

File used to run detection and classification models on a picture.

authors:
  @Ludivine Maintier
  @Aude Pertron
  @Thomas Chaigneau
  @Julien Furiga
"""

import cv2
import os
import numpy as np
import tensorflow as tf

from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.engine.defaults import DefaultPredictor
import matplotlib.pyplot as plt
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor


model = tf.keras.models.load_model('classification/classifier_fish3')
LABELS = {
    0: 'Acanthurus olivaceus',
    1: 'Carcharhinus amblyrhynchos', 
    2: 'Gymnocranius euanus', 
    3: 'Lethrinus miniatus', 
    4: 'Parupeneus barberinus'
}

cfg = get_cfg()
cfg.OUTPUT_DIR = "detection/output"
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("fishtrain")     # our training dataset
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2     # number of parallel data loading workers
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")     # use pretrained weights
cfg.SOLVER.IMS_PER_BATCH = 2     # in 1 iteration the model sees 2 images
cfg.SOLVER.BASE_LR = 0.00025     # learning rate
cfg.SOLVER.MAX_ITER = 450        # number of iteration
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128     # number of proposals to sample for training
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (mango)

cfg.MODEL.DEVICE = "cpu"
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
predictor = DefaultPredictor(cfg)



def resize_and_pad(im):
    desired_size = 96
    old_size = im.shape[:2]

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    im = cv2.resize(im, (new_size[1], new_size[0]), interpolation =cv2.INTER_CUBIC)

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT)

    return new_im

def make_prediction(img):
    img = resize_and_pad(img)
    img = np.expand_dims(img, 0)
    pred = model.predict(img)
    pp = np.argmax(pred)
    proba = np.round(np.max(pred) * 100,2)

    return LABELS[pp], proba

im = cv2.imread('detection/output/test2.png')
pred, proba = make_prediction(im)
outputs = predictor(im)
for i in range(len(outputs["instances"])):
    xmin, ymin, xmax, ymax = outputs["instances"].pred_boxes[i].tensor.cpu().numpy()[0]
    fish = np.array(im[int(ymin):int(ymax), int(xmin):int(xmax)])
    pred, proba = make_prediction(fish)
    im = cv2.putText(im, pred, (int(xmin), int(ymin - 20)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)
v = Visualizer(im[:, :, ::-1],
                scale=1, 
                instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
)
v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
plt.figure(figsize = (14, 10))
plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
plt.show()
