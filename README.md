# AI-Project

# Entraînement et Test du modèle Mask R-CNN pour la détection de montres

Ce projet utilise **Mask R-CNN** pour entraîner un modèle de détection d'objets et de segmentation d'instances pour des images de montres.

## Prérequis

Avant de commencer, assurez-vous de disposer des éléments suivants :

-Cloner le git de Mask R-CNN
-Vous avez le fichier mask_rcnn_coco.h5 a la racine du répéertoire Mask R-CNN
- Python 3.x installé sur votre machine.
- Les dépendances nécessaires installées. Vous pouvez installer les bibliothèques requises avec :

  ```bash
  pip install -r requirements.txt


## Commande d'entrainement 

python train.py \
    --dataset /chemin/vers/le/répertoire/dataset_watch \
    --weights coco \
    --weights_path /chemin/vers/le/fichier/mask_rcnn_coco.h5 \
    --logs /chemin/vers/le/répertoire/logs

# Commande de test 

python test.py \
    --dataset /chemin/vers/le/répertoire/dataset_watch \
    --weights coco \
    --weights_path /chemin/vers/le/fichier/mask_rcnn_coco.h5 \
    --logs /chemin/vers/le/répertoire/logs \
    --test_image /chemin/vers/mon/image.jpg
