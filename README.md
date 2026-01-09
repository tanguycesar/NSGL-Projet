# Projet NSGL - Network Science and Graph Learning

**Auteur** : Tanguy CESAR  
**Cours** : Network Science and Graph Learning  
**Année** : 2025-2026

---

## 📁 Structure du projet

### Fichiers principaux

- **`load_data.py`** : Chargement des graphes depuis les fichiers `.gml` du dossier `data/`
- **`cache_manager.py`** : Système de mise en cache pour éviter de recharger les graphes à chaque exécution
- **`create_cache.py`** : Script pour initialiser le cache (à exécuter une fois)

### Scripts par question

Chaque question du TP possède son propre script Python :

1. **`question1_stats.py`** : Analyse descriptive des réseaux (degrés, clustering, assortativité)
2. **`question2_analysis.py`** : Analyse détaillée de 3 réseaux (Caltech, MIT, Johns Hopkins)
3. **`question3_assortativity.py`** : Calcul de l'assortativité pour 5 attributs (student_fac, dorm, major, degree, gender)
4. **`question4_link_prediction.py`** : Prédiction de liens avec 3 métriques (Common Neighbors, Jaccard, Adamic-Adar)
5. **`question5_label_propagation.py`** : Propagation de labels semi-supervisée (PyTorch)
6. **`question6_communities.py`** : Détection de communautés (Louvain, Greedy Modularity)

### Script d'exécution global

- **`run_analysis.py`** : Lance toutes les analyses de manière séquentielle

### Données et résultats

- **`data/`** : 100 fichiers `.gml` des réseaux Facebook universitaires
- **`report/`** : Rapport LaTeX et figures générées
  - `rapport_NSGL_CESAR.tex` : Rapport complet
  - `figures/` : Tous les graphiques et tableaux CSV

---

## 🚀 Ordre d'exécution

### 1. Initialisation (première fois uniquement)

```bash
python create_cache.py
```

Cela crée un fichier `graph_cache.pkl` contenant tous les graphes chargés, ce qui accélère considérablement les analyses suivantes.

### 2. Exécution des analyses

**Option A** : Exécuter toutes les questions d'un coup
```bash
python run_analysis.py
```

**Option B** : Exécuter question par question
```bash
python question1_stats.py
python question2_analysis.py
python question3_assortativity.py
python question4_link_prediction.py
python question5_label_propagation.py
python question6_communities.py
```

### 3. Génération du rapport PDF

```bash
cd report
pdflatex rapport_NSGL_CESAR.tex
pdflatex rapport_NSGL_CESAR.tex  # Deux fois pour les références croisées
```

---

## 📊 Résultats

Tous les résultats (figures PNG et tableaux CSV) sont automatiquement sauvegardés dans `report/figures/`.

Le rapport final (`rapport_NSGL_CESAR.pdf`) contient :
- L'analyse descriptive des 100 réseaux
- L'étude d'assortativité pour 5 attributs
- Les performances de prédiction de liens
- Les résultats de propagation de labels
- La détection de communautés et leur correspondance avec les attributs

---

## 🛠️ Dépendances

- Python 3.8+
- NetworkX
- NumPy, Pandas
- Matplotlib
- PyTorch (pour la propagation de labels)
- scikit-learn

---

## 📝 Notes

- Le cache permet de charger instantanément les 100 graphes (gain de temps considérable)
- Chaque script génère ses propres figures et CSV dans `report/figures/`
- Les analyses peuvent prendre quelques minutes selon la machine (surtout Q4 et Q5)
