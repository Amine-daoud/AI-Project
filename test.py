import os
import numpy as np
import mrcnn.model as modellib
from mrcnn.config import Config
from mrcnn import utils
import matplotlib.pyplot as plt

class WatchConfig(Config):
    NAME = "watch_model"
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1
    STEPS_PER_EPOCH = 100
    DETECTION_MIN_CONFIDENCE = 0.7
    LEARNING_RATE = 1e-3

class WatchDataset(utils.Dataset):
    def load_watch(self, dataset_dir):
        self.add_class("watch", 1, "watch")
        for i, filename in enumerate(os.listdir(dataset_dir)):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(dataset_dir, filename)
                self.add_image("watch", image_id=i, path=image_path)

config = WatchConfig()
inference_config = config
inference_config.GPU_COUNT = 1
inference_config.IMAGES_PER_GPU = 1

model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=".")
trained_model_path = "mask_rcnn_watch_model.h5"
model.load_weights(trained_model_path, by_name=True)

dataset_val = WatchDataset()
dataset_val.load_watch("dataset_watch/val")
dataset_val.prepare()

image_id = dataset_val.image_ids[4]
image = dataset_val.load_image(image_id)

results = model.detect([image], verbose=1)
r = results[0]

plt.imshow(image)
plt.title("Résultat de la détection")
plt.axis("off")

for i, bbox in enumerate(r['rois']):
    y1, x1, y2, x2 = bbox
    class_id = r['class_ids'][i]
    score = r['scores'][i]
    label = dataset_val.class_names[class_id]

    plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="red", linewidth=2))
    plt.text(x1, y1, "{}: {:.2f}%".format(label, score * 100), color="white", backgroundcolor="red", fontsize=8)

    if class_id == 1:
        print("Montre détectée avec une confiance de {:.2f}%".format(score * 100))

plt.show()
