#!/usr/bin/python
""" 🪐 AI for KOSMOS - Ocean Hackathon Brest 2021

File used to run detection and classification models on a video stream. Partly inspired from `here <https://wendeehsu.medium.com/instance-segmentation-with-detectron2-127fbe01b20b>`_


authors:
  @Ludivine Maintier
  @Aude Pertron
  @Thomas Chaigneau
  @Julien Furiga
"""

import os
import cv2
import numpy as np
import tensorflow as tf

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.engine.defaults import DefaultPredictor

# Load classification model
model = tf.keras.models.load_model('classification/fish_classifier')
# These are the labels we stored in the training process
LABELS = {
  0: 'Acanthurus olivaceus',
  1: 'Carcharhinus amblyrhynchos', 
  2: 'Gymnocranius euanus', 
  3: 'Lethrinus miniatus', 
  4: 'Parupeneus barberinus'
}

# Load detection model with training config
cfg = get_cfg()
cfg.OUTPUT_DIR = "detection/output"
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("fishtrain")     # Our training dataset
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2     # Number of parallel data loading workers
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")     # use pretrained weights
cfg.SOLVER.IMS_PER_BATCH = 2     # In 1 iteration the model sees 2 images
cfg.SOLVER.BASE_LR = 0.00025     # Learning rate
cfg.SOLVER.MAX_ITER = 450        # Number of iteration
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128     # Number of proposals to sample for training
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Only has one class (fish)

cfg.MODEL.DEVICE = "cpu"
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth") # Trained model path
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.75 # Treshold for detection
predictor = DefaultPredictor(cfg)

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
video = cv2.VideoCapture("output/out3.mp4") # Add your own video path here
 
# Check if camera opened successfully
if (video.isOpened() == False):
  print("Error opening video stream or file")

frame_width = int(video.get(3))
frame_height = int(video.get(4))
   
size = (frame_width, frame_height)
   
# Below VideoWriter object will create
# a frame of above defined The output 
# is stored in 'filename.avi' file.
result = cv2.VideoWriter('final.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size)

def resize_and_pad(im):
    desired_size = 96
    old_size = im.shape[:2]

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    im = cv2.resize(im, (new_size[1], new_size[0]), interpolation =cv2.INTER_CUBIC)

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT)

    return new_im

def make_prediction(img):
    img = resize_and_pad(img)
    img = np.expand_dims(img, 0)
    pred = model.predict(img)
    pp = np.argmax(pred)
    proba = np.round(np.max(pred) * 100,2)

    return LABELS[pp], proba

# Read until video is completed
while(video.isOpened()):
  # Capture frame-by-frame
  ret, frame = video.read()
  outputs = predictor(frame)
  frame = cv2.putText(
    frame, f"Nombre de poissons: {len(outputs['instances'])}", 
    (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3, cv2.LINE_AA
  )
  for i in range(len(outputs["instances"])):
    xmin, ymin, xmax, ymax = outputs["instances"].pred_boxes[i].tensor.cpu().numpy()[0]
    xmin = xmin - 30 if xmin > 30 else xmin
    xmax = xmin + 30 if xmax < frame.shape[1] - 30 else xmax
    ymin = ymin - 30 if ymin > 30 else ymin
    ymax = ymin + 30 if ymax < frame.shape[0] - 30 else ymax
    fish = np.array(frame[int(ymin):int(ymax), int(xmin):int(xmax)])
    pred, proba = make_prediction(fish)
    im = cv2.putText(frame, pred, (int(xmin), int(ymin)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
  v = Visualizer(im[:, :, ::-1],
                  scale=1 # remove the colors of unsegmented pixels
  )
  v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
  result.write(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.IMREAD_COLOR))

# When everything done, release the video capture object
video.release()
result.release()

# Closes all the frames
cv2.destroyAllWindows()
