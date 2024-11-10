import os
import sys
import json
import numpy as np
import skimage.io
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from mrcnn import model as modellib, utils
from mrcnn.config import Config
from mrcnn import visualize
import tensorflow as tf
import keras

# Configuration pour Mask R-CNN
class WatchConfig(Config):
    NAME = "watch"
    IMAGES_PER_GPU = 2  # Nombre d'images traitées par GPU pendant l'entraînement
    NUM_CLASSES = 1 + 1  # 1 classe (montre) + 1 classe (arrière-plan)
    STEPS_PER_EPOCH = 100  # Nombre de steps par époque
    DETECTION_MIN_CONFIDENCE = 0.9  # Confiance minimale pour les prédictions
    LEARNING_RATE = 0.001  # Taux d'apprentissage
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    GPU_COUNT = 1
    BACKBONE = "resnet50"  # Utilisation de ResNet50 comme backbone

# Dataset personnalisé (LabelMe)
class WatchDataset(utils.Dataset):
    def load_watch(self, dataset_dir, subset):
        """Charge un sous-ensemble du dataset de montres."""
        self.add_class("watch", 1, "watch")
        
        # Chargement des annotations
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)
        
        json_files = [f for f in os.listdir(dataset_dir) if f.endswith('.json')]
        for json_file in json_files:
            with open(os.path.join(dataset_dir, json_file)) as f:
                annotation = json.load(f)
            image_path = os.path.join(dataset_dir, json_file.replace('.json', '.jpg'))
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]
            
            # Récupération des polygones depuis le fichier JSON
            polygons = []
            for region in annotation.get('regions', {}).values():
                polygons.append(region['shape_attributes'])
            
            self.add_image("watch", image_id=json_file, path=image_path, width=width, height=height, polygons=polygons)

    def load_mask(self, image_id):
        """Génère les masques d'instances pour une image donnée."""
        image_info = self.image_info[image_id]
        mask = np.zeros([image_info["height"], image_info["width"], len(image_info["polygons"])], dtype=np.uint8)
        
        for i, p in enumerate(image_info["polygons"]):
            # Convertir les coordonnées du polygone en tableau NumPy
            polygon = np.array(list(zip(p['all_points_x'], p['all_points_y'])))
            rr, cc = skimage.draw.polygon(polygon[:, 1], polygon[:, 0])  # Polygon (y, x) -> (row, col)
            mask[rr, cc, i] = 1
            
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)


# Fonction principale pour l'entraînement et le test
def main(args):
    # Définir la configuration
    config = WatchConfig()
    
    # Créer le modèle Mask R-CNN
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config, model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config, model_dir=args.logs)
    
    # Charger les poids COCO ou personnalisés
    if args.weights.lower() == "coco":
        # Exclure les couches de tête pour les réentraîner
        model.load_weights(args.weights_path, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(args.weights_path, by_name=True)
    
    # Charger le dataset
    dataset_train = WatchDataset()
    dataset_train.load_watch(args.dataset, "train")
    dataset_train.prepare()
    
    dataset_val = WatchDataset()
    dataset_val.load_watch(args.dataset, "val")
    dataset_val.prepare()
    
    # Entraîner le modèle (en réentraînant seulement les têtes)
    if args.command == "train":
        model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=20, layers='heads')

    # Tester et visualiser une image
    if args.command == "test":
        # Exemple de test
        image_path = args.test_image
        image = skimage.io.imread(image_path)
        results = model.detect([image], verbose=1)
        
        # Visualiser les résultats
        r = results[0]
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], dataset_train.class_names, r['scores'])
    
    # Sauvegarder et afficher les logs
    if args.command == "train":
        # Sauvegarde des graphiques de perte
        plt.plot(model.history["loss"], label="Training Loss")
        plt.plot(model.history["val_loss"], label="Validation Loss")
        plt.title('Loss during training')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(args.logs, "loss_graph.png"))
        plt.show()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("command", metavar="<command>", help="'train' or 'test'")
    parser.add_argument('--dataset', required=True, help="Chemin vers le répertoire du dataset")
    parser.add_argument('--weights', required=True, help="Poids du modèle : 'coco' ou chemin vers un fichier de poids personnalisé")
    parser.add_argument('--weights_path', required=True, help="Chemin vers le fichier de poids (par exemple mask_rcnn_coco.h5)")
    parser.add_argument('--logs', required=True, help="Répertoire où les logs et modèles seront sauvegardés")
    parser.add_argument('--test_image', help="Chemin vers l'image à tester (seulement pour la commande 'test')")
    
    args = parser.parse_args()
    main(args)
