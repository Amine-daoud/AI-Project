# Pour consulter les logs et captures, dirigez vous vers : log/Steps100 pour les logs des 100 steps et log/Steps100 pour les logs de 25 steps. 
# la detection d'image se trouve dans le répertoire images et les graphs et racines se trouve dans la racine du projet.
# AI-Project

# Entraînement et Test du modèle Mask R-CNN pour la détection de montres

Ce projet utilise **Mask R-CNN** pour entraîner un modèle de détection d'objets et de segmentation d'instances pour des images de montres.

## Prérequis

Avant de commencer, assurez-vous de disposer des éléments suivants :

-Cloner le git de Mask R-CNN
-Vous avez le fichier mask_rcnn_coco.h5 a la racine du répéertoire Mask R-CNN
- Python 3.7 installé sur votre machine.
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

python train.py 
# Commande de test 

Lancer juoyter notebook et executer le notebook test.ipynb

# Entraînement 
Dasn ce projet nous avons entraîné deux models

## Models 1 :
### Hyperparamètres Utilisés

Les hyperparamètres choisis pour ce model sont les suivants :

-Taux d'apprentissage (LEARNING_RATE) : 0.001
-Époques d'entraînement : 20
-Backbone : ResNet-50
-Nombre d'images par GPU : 2
-Mode de redimensionnement des images : square
-Dimensions minimales et maximales des images : 256 x 256
-STEPS_PER_EPOCH = 100


## Models 2 :
### Hyperparamètres Utilisés

Les hyperparamètres choisis pour ce model sont les suivants :

-Taux d'apprentissage (LEARNING_RATE) : 0.001
-Époques d'entraînement : 20
-Backbone : ResNet-50
-Nombre d'images par GPU : 2
-Mode de redimensionnement des images : square
-Dimensions minimales et maximales des images : 256 x 256
-STEPS_PER_EPOCH = 25

# Résultats
Des logs sont disponibles dans le répértoire logs pour les deux modèls
## Entraînement avec 100 steps par epoch :

Les résultats de l'entraînement avec 100 steps par epoch sont les suivants :

100/100 [==============================] - 1079s 11s/step - loss: 0.1594 - rpn_class_loss: 0.0021 - rpn_bbox_loss: 0.0490 - mrcnn_class_loss: 0.0155 - mrcnn_bbox_loss: 0.0267 - mrcnn_mask_loss: 0.0662 - val_loss: 0.1388 - val_rpn_class_loss: 0.0026 - val_rpn_bbox_loss: 0.0356 - val_mrcnn_class_loss: 0.0074 - val_mrcnn_bbox_loss: 0.0260 - val_mrcnn_mask_loss: 0.0671

Les résultats du modèle entraîné avec 100 steps par epoch sont enregistrés dans le répertoire logs.
## Entraînement avec 25 steps par epoch :
Les résultats de l'entraînement avec 100 steps par epoch sont les suivants :
25/25 [==============================] - 392s 16s/step - loss: 0.32 - rpn_class_loss: 0.0054 - rpn_bbox_loss: 0.1050 - mrcnn_class_loss: 0.0199 - mrcnn_bbox_loss: 0.12039 - mrcnn_mask_loss: 0.0917 - val_loss: 0.3346 - val_rpn_class_loss: 0.0584 - val_rpn_bbox_loss: 0.0272 - val_mrcnn_class_loss: 0.0074 - val_mrcnn_bbox_loss: 0.0260 - val_mrcnn_mask_loss: 0.1148

Les résultats du modèle entraîné avec 25 steps par epoch sont enregistrés dans le répertoire logs.

## Matrices de confusion

Les matrices de confusion pour les deux configurations d'entraînement (25 steps et 100 steps) sont disponibles dans les fichiers suivants :

    matrice_de_confusion_25_steps.png
    matrice_de_confusion_100_steps.png

Les matrices de confusion montrent la précision du modèle pour classer les objets en deux catégories : "watch" et "background".

## Graphiques de Loss

Les courbes de loss pour chaque modèle (25 et 100 steps par epoch) sont affichées dans un graphique montrant la diminution de la perte au fil des epochs. Le graphique est disponible sous le nom loss_graph.png.
