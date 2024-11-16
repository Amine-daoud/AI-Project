
# coding: utf-8


import tensorflow as tf
from keras import backend as K
from mrcnn.model import MaskRCNN

# Verifier si un GPU est disponible dans TensorFlow 1.x
if tf.test.is_gpu_available():
    print("GPU detecte et pret a l'emploi.")
    # Specifier le GPU comme peripherique d'execution
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    K.set_session(session)
else:
    print("Aucun GPU detecte. Utilisation du CPU.")




import tensorflow as tf
import skimage
import keras

print("TensorFlow version:", tf.__version__)
print("scikit-image version:", skimage.__version__)
print("Keras version:", keras.__version__)

gpus = tf.GPUOptions()

# Check for available devices
gpu_devices = tf.config.experimental.list_physical_devices('GPU') if hasattr(tf, 'config') else tf.test.gpu_device_name()

# Print the number of available GPUs
if gpu_devices:
    print("Num GPUs Available: ", len(gpu_devices))
else:
    print("No GPUs available")


DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0



import warnings
import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import cv2
import random
import math
import re
import time
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg

from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
from mrcnn.visualize import display_instances
import mrcnn.model as modellib
from mrcnn.model import log
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Root directory of the project
#ROOT_DIR = "D:\MRCNN_tensorflow2.7_env\Mask-RCNN"
ROOT_DIR = os.getcwd()

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library


# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")



print(ROOT_DIR)



from mrcnn.config import Config

class WatchConfig(Config):
    NAME = "Watch_detection"
    GPU_COUNT = 1  # Pas de GPU
    IMAGES_PER_GPU = 1  # Une image a la fois sur le CPU
    NUM_CLASSES = 1 + 1  # Background + classes "drone"
    STEPS_PER_EPOCH = 100
    DETECTION_MIN_CONFIDENCE = 0.9
    CLASS_NAMES = ["bg", "watch"]
    IMAGE_MAX_DIM = 256

config = WatchConfig()



import os
import json
import numpy as np
from skimage.io import imread
from mrcnn.utils import Dataset
from sklearn.model_selection import train_test_split
from skimage.draw import polygon as draw_polygon  # Correctly import the draw_polygon function

class WatchDataset(Dataset):
    def load_watch(self, dataset_dir, subset):
        """
        Load a subset of the watch dataset from a directory structure with images and JSON annotations.
        Args:
            dataset_dir: Root directory of the dataset.
            subset: Subset to load (train/val).
        """
        # Add a single class for "watch"
        self.add_class("watch", 1, "watch")

        # Assert that the subset is either 'train' or 'val'
        assert subset in ["train", "val"], "Subset must be 'train' or 'val'"
        subset_dir = os.path.join(dataset_dir, subset)

        # List JSON files in the subset directory
        json_files = [f for f in os.listdir(subset_dir) if f.endswith('.json')]
        for json_file in json_files:
            # Open the JSON file
            with open(os.path.join(subset_dir, json_file)) as f:
                annotation = json.load(f)

            # Image path and image dimensions
            image_id = json_file.replace('.json', '')
            image_path = os.path.join(subset_dir, annotation['imagePath'])
            image = imread(image_path)
            height, width = image.shape[:2]

            # Collect all polygons (shape-based annotations)
            polygons = []
            #print("ANNOTATION : ", annotation["shapes"])
            for shape in annotation.get('shapes', []):
                #print("SHAPE : ", shape)
                if shape['shape_type'] == 'polygon':
                    points = shape['points']
                    all_points_x = [point[0] for point in points]
                    all_points_y = [point[1] for point in points]
                    #print("ALL X : ", all_points_x)
                    #print("ALL Y : ", all_points_y)
                    polygons.append({'all_points_x': all_points_x, 'all_points_y': all_points_y})

            #print("POLYGONS : ", polygons)
            # Add the image and its polygons to the dataset
            self.add_image("watch", image_id=image_id, path=image_path, width=width, height=height, polygons=polygons)
    def load_mask(self, image_id):
        """Generate instance masks for an image."""
        image_info = self.image_info[image_id]
        if image_info["source"] != "watch":
            return super(WatchDataset, self).load_mask(image_id)

        # Get the height and width of the image
        height = image_info["height"]
        width = image_info["width"]
        polygons = image_info["polygons"]

        # Initialize the mask array
        mask = np.zeros([height, width, len(polygons)], dtype=np.uint8)

        # Create a binary mask for each polygon
        for i, p in enumerate(polygons):
            polygon = np.array(list(zip(p['all_points_x'], p['all_points_y'])))
            rr, cc = draw_polygon(polygon[:, 1], polygon[:, 0], shape=(height, width))
            mask[rr, cc, i] = 1

        # Return the mask and the class IDs
        return mask.astype(bool), np.ones([mask.shape[-1]], dtype=np.int32)


import os
print("Current Working Directory:", os.getcwd())

# Import the model
from mrcnn.model import MaskRCNN

# Create a new Mask R-CNN model in training mode
model = MaskRCNN(mode="training", config=config, model_dir="./logs")

# Load pre-trained weights (exclude layers that don't match our classes)
model.load_weights("mask_rcnn_coco.h5", by_name=True, exclude=[
    "mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])



import os
# Prepare the training dataset (80% of the data)
train_dataset = WatchDataset()
train_dataset.load_watch(dataset_dir="dataset_watch", subset="train")
train_dataset.prepare()

# Prepare the validation dataset (20% of the data)
val_dataset = WatchDataset()
val_dataset.load_watch(dataset_dir="dataset_watch", subset="val")
val_dataset.prepare()


from keras.callbacks import Callback
import time

class EpochProgressCallback(Callback):
    def __init__(self, update_interval=300):  # Intervalle de mise a jour en secondes (5 minutes = 300 secondes)
        super(EpochProgressCallback, self).__init__()
        self.update_interval = update_interval
        self.last_update_time = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.epoch = epoch
        print("\nDebut de l'epoch {}...".format(epoch + 1))

    def on_batch_end(self, batch, logs=None):
        current_time = time.time()
        if current_time - self.last_update_time >= self.update_interval:
            self.last_update_time = current_time
            total_batches = self.params['steps']
            percentage = ((batch + 1) / total_batches) * 100
            print("Epoch {}: {:.2f}% termine.".format(self.epoch + 1, percentage))

# Utiliser le callback lors de l'entraînement
epoch_progress_callback = EpochProgressCallback(update_interval=300)  # Met a jour toutes les 5 minutes

# Lancer l'entraînement avec le callback
model.train(
    train_dataset, 
    val_dataset, 
    learning_rate=config.LEARNING_RATE, 
    epochs=20, 
    layers='heads', 
    custom_callbacks=[epoch_progress_callback]
)

# Sauvegarder le modèle après l'entrainement
model.keras_model.save_weights("model.h5")
