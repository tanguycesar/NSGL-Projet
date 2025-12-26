# Projet NSGL — Network Science and Graph Learning

Ce dépôt contient le travail réalisé dans le cadre du projet **NSGL (Network Science and Graph Learning)**.
L’objectif est d’analyser des réseaux sociaux issus du jeu de données *Facebook100* à l’aide d’outils de network science et de graph learning.

---

## Structure du dépôt

Projet/
├── data/
│ └── (graphes .mat / .gml)
│
├── notebook_net_homework.ipynb
│
├── report/
│ ├── rapport_NSGL_CESAR.tex
│ ├── rapport_NSGL_CESAR.pdf
│ └── figures/
│ └── *.png
│
├── Homework_Network_Analysis.pdf
├── requirements.txt
└── README.md


---

## Contenu

- **Notebook Jupyter**  
  `notebook_net_homework.ipynb` contient l’ensemble des analyses :
  - analyse structurelle des graphes
  - assortativité et homophilie
  - prédiction de liens
  - propagation de labels
  - détection de communautés

- **`report/`**  
  Rapport final du projet :
  - version LaTeX (`.tex`)
  - version PDF prête à être rendue
  - figures utilisées dans le rapport

- **`data/`**  
  Données utilisées pour le projet (Facebook100, fichiers `.mat` et `.gml`).

- **`Homework_Network_Analysis.pdf`**  
  Énoncé officiel du projet.

---

## Environnement Python

Le projet a été développé à l’aide d’un **environnement virtuel Python situé à la racine du semestre** (`.venv`), commun à plusieurs projets.

Pour reproduire l’environnement :

```bash
python -m venv .venv
source .venv/bin/activate    # Linux / Mac
.venv\Scripts\activate       # Windows
pip install -r requirements.txt
