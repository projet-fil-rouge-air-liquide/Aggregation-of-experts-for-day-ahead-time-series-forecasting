# 📊 Aggregation of Experts for Day-Ahead Time Series Forecasting

Projet réalisé dans le cadre du **Projet Fil Rouge** des **Mastères Spécialisés IA et Data** de **Télécom Paris**.

Ce projet vise à mettre en œuvre et comparer plusieurs modèles experts. de prévision de séries temporelles à horizon J+1, puis à les agréger à l’aide d’une approche de **Mixture of Experts (MOE)**.

---

## 👥 Contributeurs

* Alexandre Donnat
* Ambroise Laroye
* Héloïse Lordez
* Oscar De La Cruz
* William Jan

---

## 📁 Structure générale du projet

* `src/experts/` : construction et prédiction des modèles experts
* `src/opera/` : implémentation de la méthode d’agrégation (MOE)
* `src/data_cleaning.py` : récupération et nettoyage des données
* `API_ERA5.py` : script de téléchargement des données météorologiques
* `data/` : stockage des jeux de données (générés automatiquement)

---

## 📥 Chargement des données

### 1. Données météorologiques (ERA5)

Les données météorologiques doivent être chargées **en premier**.
Elles nécessitent un compte personnel sur la plateforme **Copernicus ERA5**.

#### Étapes à suivre :

1. Créer un compte sur :
   👉 [https://cds.climate.copernicus.eu](https://cds.climate.copernicus.eu)
2. Générer une **clé API personnelle**
3. Lancer le script de récupération des données :

   ```bash
   python API_ERA5.py
   ```

---

### 2. Données RTE

Les données RTE sont **automatiquement téléchargées, extraites et renommées** lors du premier lancement du script suivant :

```bash
python src/data_cleaning.py
```

👉 Aucun téléchargement manuel n’est requis.

---

## 🔧 Fonctionnement du repository & Workflow Git

* Chaque contributeur dispose de droits de lecture et d’écriture sur le repository.
* Il est fortement recommandé de :

  * Cloner la branche `main`
  * Créer une branche personnelle de développement (`dev/<prenom>` ou équivalent)
  * Effectuer les pull requests vers `main` une fois les fonctionnalités validées

---

## ⚙️ Exécution du projet

### 1. Construction des modèles experts

```bash
python -m src.experts.build_experts
```

**Sorties :**

* `expert.csv` : prédictions des experts
* Graphique de comparaison *Expert vs Vérité terrain*

---

### 2. Prédictions à 24h

```bash
python -m src.expertsprediction_for_24h
```

**Sortie :**

* `pred_24h.csv` : prédictions à J+1 des experts

---

### 3. Agrégation des experts (MOE)

```bash
python -m src.opera.moe
```

**Sorties :**

* Graphique des **poids attribués aux experts**
* Comparaison **Experts vs MOE vs Vérité terrain** sur 24h

