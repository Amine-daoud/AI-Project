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

-Décompressez l'archive du dataset : le dataset doit être organisé avec des dossiers train et val contenant les images et leurs annotations au format JSON.
Structure attendue :

dataset_watch/
├── train/
│   ├── image1.jpg
│   ├── image1.json
│   └── ...
└── val/
    ├── image2.jpg
    ├── image2.json
    └── ...
-Construire le dataset : Assurez-vous que le dataset contient au moins 150 images (100 pour l'entraînement et 50 pour la validation) avec les annotations appropriées.

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

# Hyperparamètres Utilisés

Les hyperparamètres choisis pour ce projet sont les suivants :

-Taux d'apprentissage (LEARNING_RATE) : 0.001
-Époques d'entraînement : 20
-Backbone : ResNet-50
-Nombre d'images par GPU : 2
-Mode de redimensionnement des images : square
-Dimensions minimales et maximales des images : 512 x 512

# Pistes d'Amélioration
Augmentation des données : L'ajout de techniques d'augmentation des données pourrait diversifier l'ensemble d'entraînement et améliorer la robustesse du modèle.
Ajustement des hyperparamètres : Tester des valeurs différentes de LEARNING_RATE, changer le BACKBONE ou utiliser des optimizers différents comme Adam ou SGD.

# Conclusion

Ce projet démontre la mise en œuvre de Mask R-CNN pour la détection et la segmentation d'instances d'images de montres. La préparation minutieuse des données et l'ajustement des hyperparamètres sont essentiels pour obtenir des résultats précis. Le code est conçu pour être facilement adaptable à d'autres projets de détection d'objets.
