# Projet : Aggregation of Experts for Day-Ahead Time Series Forecasting

Projet réalisé dans le cadre du Projet Fil Rouge des MS IA et Data de Telecom Paris

---

## Contributeurs

- Alexandre Donnat
- Ambroise Laroye
- Héloïse Lordez
- Oscar De La Cruz
- William Jan

---

## Chargement des données

### 1. Données Météo (ERA5)

Les données météorologiques doivent être chargées en premier.  
Elles nécessitent un compte personnel sur la plateforme ERA5.

Étapes pour récupérer les données ERA5 :

1. Créer un compte :  
   https://cds.climate.copernicus.eu  
2. Générer une API key personnelle 
3. Exécuter le script météo : API_ERA5.py


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