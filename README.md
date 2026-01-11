# Projet NSGL - Network Science and Graph Learning

**Auteur** : Tanguy CESAR  
**Cours** : Network Science and Graph Learning  
**Année** : 2025-2026

---

## Structure du projet

```
NSGL-Projet/
├── data/                   # 100 réseaux Facebook universitaires (.gml)
├── docs/                   # Documentation et énoncé du TP
│   ├── Homework_Network_Analysis.pdf
│   ├── notebook_net_homework.ipynb
│   └── notebook_net_homework.py
├── src/                    # Scripts d'analyse par question
│   ├── question1_stats.py
│   ├── question2_analysis.py
│   ├── question3_assortativity.py
│   ├── question4_link_prediction.py
│   ├── question5_label_propagation.py
│   ├── question6_communities.py
│   └── run_analysis.py
├── utils/                  # Utilitaires (chargement et cache)
│   ├── load_data.py
│   ├── cache_manager.py
│   └── create_cache.py
├── report/                 # Rapport LaTeX et figures
│   ├── rapport_NSGL_CESAR.tex
│   ├── rapport_NSGL_CESAR.pdf
│   └── figures/
└── README.md
```

### Description des dossiers

#### **`utils/`** - Utilitaires de chargement
- **`load_data.py`** : Fonctions pour charger les graphes depuis les fichiers `.gml`
- **`cache_manager.py`** : Système de mise en cache pour éviter de recharger les graphes
- **`create_cache.py`** : Script d'initialisation du cache (à exécuter une fois)

#### **`src/`** - Scripts d'analyse
Chaque question du TP possède son propre script Python :

1. **`question1_stats.py`** : Analyse descriptive des 100 réseaux (degrés, clustering, assortativité)
2. **`question2_analysis.py`** : Analyse détaillée de 3 réseaux (Caltech, MIT, Johns Hopkins)
3. **`question3_assortativity.py`** : Assortativité pour 5 attributs (student_fac, dorm, major, degree, gender)
4. **`question4_link_prediction.py`** : Prédiction de liens (Common Neighbors, Jaccard, Adamic-Adar)
5. **`question5_label_propagation.py`** : Propagation de labels semi-supervisée avec PyTorch
6. **`question6_communities.py`** : Détection de communautés (Louvain, Greedy Modularity)
7. **`run_analysis.py`** : Exécution de toutes les analyses séquentiellement

#### **`data/`** - Données
100 fichiers `.gml` des réseaux sociaux Facebook universitaires (Facebook100 dataset)

#### **`report/`** - Rapport et résultats
- Rapport LaTeX complet (`rapport_NSGL_CESAR.tex` et `.pdf`)
- Dossier `figures/` contenant tous les graphiques PNG et tableaux CSV générés

#### **`docs/`** - Documentation
- Énoncé du TP (`Homework_Network_Analysis.pdf`)
- Notebooks exploratoires

---

## Ordre d'exécution

### 1. Initialisation du cache (première fois uniquement)

```bash
python utils/create_cache.py
```

Cela crée un fichier `graph_cache.pkl` à la racine contenant tous les graphes chargés (gain de temps considérable).

### 2. Exécution des analyses

**Option A** : Exécuter toutes les questions d'un coup
```bash
python src/run_analysis.py
```

**Option B** : Exécuter question par question
```bash
python src/question1_stats.py
python src/question2_analysis.py
python src/question3_assortativity.py
python src/question4_link_prediction.py
python src/question5_label_propagation.py
python src/question6_communities.py
```

### 3. Génération du rapport PDF

```bash
cd report
pdflatex rapport_NSGL_CESAR.tex
pdflatex rapport_NSGL_CESAR.tex  # Deux fois pour les références croisées
```

---

## Résultats

Tous les résultats (figures PNG et tableaux CSV) sont automatiquement sauvegardés dans `report/figures/`.

Le rapport final contient :
-  Analyse descriptive des 100 réseaux (Q1)
-  Étude détaillée de 3 réseaux spécifiques (Q2)
-  Assortativité et homophilie pour 5 attributs (Q3)
-  Prédiction de liens avec 3 métriques (Q4)
-  Propagation de labels semi-supervisée (Q5)
-  Détection de communautés et correspondance avec attributs (Q6)

---

## Dépendances

```
Python 3.8+
networkx
numpy
pandas
matplotlib
pytorch
scikit-learn
```

Installation :
```bash
pip install networkx numpy pandas matplotlib torch scikit-learn
```

