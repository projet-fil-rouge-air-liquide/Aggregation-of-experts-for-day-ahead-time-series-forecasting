# 📊 Aggregation of Experts for Day-Ahead Time Series Forecasting

Projet réalisé dans le cadre du **Projet Fil Rouge** des **Mastères Spécialisés IA et Data** de **Télécom Paris**.

Ce projet vise à mettre en œuvre et comparer plusieurs modèles experts de prévision de séries temporelles à horizon J+1, puis à les agréger à l’aide d’une approche de **Mixture of Experts (MOE)**.

---

## 👥 Contributeurs

* Alexandre Donnat
* Ambroise Laroye
* Héloïse Lordez
* Oscar De La Cruz
* William Jan

---

## 📁 Structure générale du projet

* `src/Experts/` : construction et prédiction des modèles experts
* `src/opera/` : implémentation de la méthode d’agrégation (MOE)
* `src/Data_cleaning.py` : récupération et nettoyage des données
* `API_ERA5.py` : script de téléchargement des données météorologiques
* `data/` : stockage des jeux de données (générés automatiquement)

---

## 📥 Chargement des données

### 1. Données météorologiques (ERA5)

Les données météorologiques doivent être chargées **en premier**.
Elles nécessitent un compte personnel sur la plateforme **Copernicus ERA5**.

#### Étapes à suivre :

1. Créer un compte :  
   https://cds.climate.copernicus.eu  
2. Générer une API key personnelle 
3. Exécuter le script météo : API_ERA5.py dans config/API


### 2. Données ELIA
Les données ELIA sont récupérées sur le site ELIA (fichier csv):
https://opendata.elia.be/explore/dataset/ods086/export/

### 3. Traitement des données et construction des features
Exécuter le pipe: src/data_pipe.py
Les données traitées et les features créées - sont stockée au format csv (data_engineering_belgique.csv) dans data/processed_data.

### 4. Données d'entrainement - Features des experts
- Les données d'entrainement sont stockées dans src/config/data_train_valid_test.py
- les features associées aux classes sont stockées dans src/config/features

### 5. Experts - agrégateurs
Les experts et les agrégateurs sont créés sous forme de classes. base_expert et base_agg sont sont les classes mères des experts/agrégateurs.
Les experts/agrégateurs sont instanciés/entrainés dans agg_pipe.py.

### 6. Fonctionnement du repository & Workflow Git
Chaque contributeur possède des droits de lecture et d'écriture sur le repository.
Il est recommandé de créer une branche personnelle DevOps après avoir cloné la branche main.