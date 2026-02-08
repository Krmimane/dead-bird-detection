# Dead Bird Detection – YOLOv11

Ce projet vise la **détection d’oiseaux morts sur images haute résolution** à l’aide de modèles **YOLOv11**.  
Il couvre tout le pipeline : **préparation des données, découpage en tuiles 512×512, augmentation, entraînement et évaluation**.

---

## ⚙️ Description des fichiers

### 🔹 `data_augmentation.py`  
Script dédié à l’**augmentation des données** afin d’enrichir le dataset (flip, rotation, variations, etc.) et améliorer la robustesse du modèle.

---

### 🔹 `test_split.py`  
Script pour le **découpage des images en tuiles de 512×512 pixels avec chevauchement**.  
Il :
- segmente les grandes images,  
- recalcule les labels YOLO pour chaque tuile,  
- génère un nouveau dataset prêt pour l’entraînement.

---

### 🔹 `dataprep.ipynb`  
Notebook de **préparation des données**.  
Il permet :
- d’analyser le dataset,  
- nettoyer et équilibrer les labels,  
- vérifier les annotations.
---

### 🔹 `entrainement&eval.ipynb`  
Notebook **principal et final** du projet pour :
- l’**entraînement des modèles YOLO**,  
- l’**évaluation des performances**,  
- le calcul des métriques (Precision, Recall, mAP),  
- la visualisation des résultats,  
- et la sélection du modèle final.

👉 C’est ce notebook qui contient la **version finale du pipeline d’entraînement**.

---

## 🤖 Notebooks YOLOv11 (Tests de modèles)

Les notebooks suivants ont servi uniquement à **tester différentes variantes de YOLOv11** avant de fixer la version finale utilisée dans `entrainement&eval.ipynb`.

---

### 🔹 `NOTEBOOK_A__YOLO11n(FAST).ipynb`  
Tests avec une version légère pour des expérimentations rapides.

---

### 🔹 `NOTEBOOK_B__YOLO11s_(BASELINE).ipynb`  
Tests avec un modèle intermédiaire servant de baseline.

---

### 🔹 `NOTEBOOK_C__YOLO11m_(STRONG).ipynb`  
Tests avec un modèle plus puissant pour comparer les performances.

---

### 🔹 `PIC_PROJECT_YOLO11s_ancien_test(...).ipynb`  
Notebook expérimental ancien utilisé pour valider certaines étapes du pipeline.

---

## 🚀 Pipeline global

1. **Data preparation** → `dataprep.ipynb`  
2. **Data augmentation** → `data_augmentation.py`  
3. **Split en tuiles 512×512** → `test_split.py`  
4. **Entraînement final & évaluation** → `entrainement&eval.ipynb`  
5. **Tests de modèles** → Notebooks YOLOv11 A / B / C  
