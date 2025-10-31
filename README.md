# ChestXray-COVID-Detection

# Classification des Radiographies Pulmonaires par Deep Learning

## Description du Projet
Système de classification automatique des radiographies pulmonaires en quatre catégories : **COVID**, **Normal**, **Lung Opacity** et **Viral Pneumonia**. Ce projet utilise le deep learning pour assister le diagnostic médical grâce à l'analyse d'images radiographiques.

## Objectifs
- Développer un classifieur robuste de radiographies pulmonaires
- Gérer le déséquilibre important des données médicales
- Comparer les performances de différentes architectures de deep learning
- Fournir une base pour l'assistance au diagnostic médical

## Dataset et Préparation des Données

### Distribution Initiale
| Classe | Nombre d'images |
|--------|-----------------|
| COVID | 3,616 |
| Viral Pneumonia | 1,345 |
| Normal | 10,192 |
| Lung Opacity | 6,012 |

### Problème de Déséquilibre
- Classe **Normal** dominante (48%)
- Classe **Viral Pneumonia** sous-représentée (6%)
- Risque de sur-apprentissage

### Stratégie d'Équilibrage
- **Sous-échantillonnage** : Normal et Lung Opacity réduites à 5000 images
- **Augmentation de données** : COVID et Viral Pneumonia augmentées à 5000 images
- **Distribution finale** : 5000 images par classe (total 20,000 images)

### Data Augmentation
```python
transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor()
])
```

# Split des Données
Ensemble	Nombre d'images	Pourcentage
Entraînement	14,000	70%
Validation	3,000	15%
Test	3,000	15%

# Architectures des Modèles
1. CNN_MLP

    Architecture développée from scratch

    Combinaison de couches convolutives et fully-connected
2. EfficientNetV2-S

    Modèle pré-entraîné de torchvision

    Transfer learning avec fine-tuning

3. ResNet-18

    Architecture résiduelle classique

    Transfer learning avec adaptation

# Points Forts

    Résultats excellents avec les modèles par transfer learning (>96% accuracy)

    Gestion efficace du déséquilibre des données

    Robustesse grâce à la data augmentation


# Conclusions
## Réussites

    Pipeline complet de classification d'images médicales

    Gestion robuste du déséquilibre des données

    Performances excellentes (>96% accuracy)

    Comparaison systématique d'architectures

## Applications Médicales

    Assistance au diagnostic de la COVID-19

    Détection précoce des pneumonies virales

    Identification des opacités pulmonaires

    Support décisionnel pour les radiologues

# Auteur

Ahmed Bennasser
*Projet de Deep Learning - 12 octobre 2025*

# Perspectives

    Analyse des erreurs et confusion matrix détaillée

    Optimisation des hyperparamètres

    Explicabilité (Grad-CAM, SHAP)

    Déploiement en environnement clinique

    Extension à d'autres pathologies pulmonaires

    
