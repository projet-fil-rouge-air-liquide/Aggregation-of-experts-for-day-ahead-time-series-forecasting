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


### 2. Données RTE
Les données RTE sont automatiquement téléchargées/extraites/renommées lors du premier lancement du script suivant: src/Data_cleaning.py.
Aucun téléchargement manuel n'est donc nécessaire.

### 3.Fonctionnement du repository & Workflow Git
Chaque contributeur possède des droits de lecture et d'écriture sur le repository.
Il est recommandé de créer une branche personnelle DevOps après avoir cloné la branche main.