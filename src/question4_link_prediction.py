"""
Question 4: Link Prediction

Implémentation manuelle (SANS NetworkX) de 3 métriques:
1. Common Neighbors (CN): |N(u) ∩ N(v)|
2. Jaccard: |N(u) ∩ N(v)| / |N(u) ∪ N(v)|
3. Adamic/Adar: Σ(1/log(|N(w)|)) pour w ∈ N(u) ∩ N(v)

Évaluation:
- Fractions d'arêtes retirées: f ∈ {0.05, 0.1, 0.15, 0.2}
- Valeurs de k: {50, 100, 200, 300, 400}
- Métriques: Precision@k, Recall@k
"""

import sys
import os
from pathlib import Path
import math
import random
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Set, Tuple
from abc import ABC, abstractmethod

# Se placer dans le répertoire du script
script_dir = Path(__file__).parent.absolute()
os.chdir(script_dir)

# Importer le module de chargement avec cache
from cache_manager import load_graphs_with_cache

# Configuration matplotlib
plt.style.use('seaborn-v0_8-darkgrid')


class LinkPredictor(ABC):
    """Classe abstraite pour les prédicteurs de liens"""
    
    def __init__(self, graph: nx.Graph):
        """
        Args:
            graph: Graphe NetworkX
        """
        self.graph = graph
        self.N = graph.number_of_nodes()
        # Précalculer les voisins pour chaque nœud
        self._neighbors = {u: set(graph.neighbors(u)) for u in graph.nodes()}
    
    def neighbors(self, node: int) -> Set[int]:
        """Retourne l'ensemble des voisins d'un nœud"""
        return self._neighbors.get(node, set())
    
    @abstractmethod
    def fit(self):
        """Entraîne le modèle (si nécessaire)"""
        pass
    
    @abstractmethod
    def score(self, u: int, v: int) -> float:
        """Calcule le score pour une paire de nœuds"""
        pass


class CommonNeighbors(LinkPredictor):
    """Common Neighbors: |N(u) ∩ N(v)|"""
    
    def fit(self):
        return self
    
    def score(self, u: int, v: int) -> float:
        Nu = self.neighbors(u)
        Nv = self.neighbors(v)
        return float(len(Nu & Nv))


class Jaccard(LinkPredictor):
    """Jaccard: |N(u) ∩ N(v)| / |N(u) ∪ N(v)|"""
    
    def fit(self):
        return self
    
    def score(self, u: int, v: int) -> float:
        Nu = self.neighbors(u)
        Nv = self.neighbors(v)
        intersection = len(Nu & Nv)
        union = len(Nu | Nv)
        return float(intersection / union) if union > 0 else 0.0


class AdamicAdar(LinkPredictor):
    """Adamic/Adar: Σ(1/log(|N(w)|)) pour w ∈ N(u) ∩ N(v)"""
    
    def fit(self):
        # Précalculer les poids pour chaque nœud
        self._weights = {}
        for w in self.graph.nodes():
            degree_w = len(self.neighbors(w))
            if degree_w <= 1:
                self._weights[w] = 0.0
            else:
                self._weights[w] = 1.0 / math.log(degree_w)
        return self
    
    def score(self, u: int, v: int) -> float:
        Nu = self.neighbors(u)
        Nv = self.neighbors(v)
        common = Nu & Nv
        return float(sum(self._weights[w] for w in common))


def train_test_split_edges(G: nx.Graph, frac_remove: float, 
                           seed: int = 0) -> Tuple[nx.Graph, Set[Tuple[int, int]]]:
    """
    Retire aléatoirement une fraction des arêtes
    
    Args:
        G: Graphe NetworkX
        frac_remove: Fraction d'arêtes à retirer
        seed: Graine aléatoire
        
    Returns:
        (G_train, removed_edges)
    """
    rng = random.Random(seed)
    
    edges = list(G.edges())
    rng.shuffle(edges)
    
    n_remove = int(frac_remove * len(edges))
    removed_edges = set(tuple(sorted(e)) for e in edges[:n_remove])
    
    G_train = G.copy()
    G_train.remove_edges_from(removed_edges)
    
    return G_train, removed_edges


def get_candidate_pairs(G: nx.Graph, max_candidates: int = 50000) -> Set[Tuple[int, int]]:
    """
    Génère les paires candidates ayant au moins un voisin commun
    VERSION OPTIMISÉE: limite le nombre de candidats et utilise un échantillonnage
    
    Args:
        G: Graphe NetworkX
        max_candidates: Nombre maximum de candidats à générer
        
    Returns:
        Ensemble de paires (u, v) avec u < v
    """
    neighbors = {u: set(G.neighbors(u)) for u in G.nodes()}
    edges_set = set(G.edges())
    candidates = set()
    
    # Créer un ensemble d'arêtes normalisées pour vérification rapide
    edges_normalized = set()
    for u, v in edges_set:
        edges_normalized.add((min(u, v), max(u, v)))
    
    nodes = list(G.nodes())
    random.shuffle(nodes)
    
    for w in nodes:
        neigh_w = list(neighbors[w])
        n_neigh = len(neigh_w)
        
        if n_neigh < 2:
            continue
        
        # Pour les nœuds avec beaucoup de voisins, échantillonner
        if n_neigh > 100:
            # Prendre un sous-ensemble aléatoire
            neigh_w = random.sample(neigh_w, 100)
            n_neigh = 100
        
        # Générer les paires
        for i in range(n_neigh):
            u = neigh_w[i]
            for j in range(i + 1, n_neigh):
                v = neigh_w[j]
                
                edge = (min(u, v), max(u, v))
                
                # Vérifier si pas déjà une arête
                if edge not in edges_normalized:
                    candidates.add(edge)
                    
                    # Limiter le nombre de candidats
                    if len(candidates) >= max_candidates:
                        return candidates
    
    return candidates


def evaluate_predictor(G: nx.Graph, 
                       predictor_class,
                       frac_remove_list: List[float] = [0.05, 0.1, 0.15, 0.2],
                       k_list: List[int] = [50, 100, 200, 300, 400],
                       seed: int = 0,
                       max_candidates: int = 50000) -> pd.DataFrame:
    """
    Évalue un prédicteur de liens
    
    Args:
        G: Graphe NetworkX
        predictor_class: Classe du prédicteur
        frac_remove_list: Fractions d'arêtes à retirer
        k_list: Valeurs de k pour precision@k et recall@k
        seed: Graine aléatoire
        max_candidates: Nombre maximum de paires candidates
        
    Returns:
        DataFrame avec les résultats
    """
    results = []
    
    for frac in frac_remove_list:
        # Split train/test
        G_train, removed_edges = train_test_split_edges(G, frac, seed=seed)
        
        # Entraîner le prédicteur
        predictor = predictor_class(G_train).fit()
        
        # Générer les candidats (limité pour performance)
        candidates = get_candidate_pairs(G_train, max_candidates=max_candidates)
        
        # Calculer les scores
        scored_pairs = []
        for (u, v) in candidates:
            score = predictor.score(u, v)
            if score > 0:
                scored_pairs.append((u, v, score))
        
        # Trier par score décroissant
        scored_pairs.sort(key=lambda x: x[2], reverse=True)
        
        # Évaluer pour chaque k
        for k in k_list:
            # Top-k prédictions
            topk = scored_pairs[:k]
            predicted_edges = set((min(u, v), max(u, v)) for u, v, _ in topk)
            
            # Métriques
            true_positives = len(predicted_edges & removed_edges)
            precision = true_positives / k if k > 0 else 0.0
            recall = true_positives / len(removed_edges) if len(removed_edges) > 0 else 0.0
            
            results.append({
                "méthode": predictor_class.__name__,
                "f_removed": frac,
                "k": k,
                "TP": true_positives,
                "precision@k": precision,
                "recall@k": recall,
                "n_candidates": len(scored_pairs),
                "n_removed": len(removed_edges)
            })
    
    return pd.DataFrame(results)


def plot_link_prediction_results(df: pd.DataFrame, output_dir: str = "report/figures"):
    """Visualise les résultats de link prediction"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Une figure par fraction
    for frac in sorted(df['f_removed'].unique()):
        df_frac = df[df['f_removed'] == frac]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Precision@k
        for method in df_frac['méthode'].unique():
            df_method = df_frac[df_frac['méthode'] == method].sort_values('k')
            axes[0].plot(df_method['k'], df_method['precision@k'], 
                        marker='o', label=method, linewidth=2, markersize=6)
        
        axes[0].set_xlabel('k (nombre de prédictions)', fontsize=11)
        axes[0].set_ylabel('Precision@k', fontsize=11)
        axes[0].set_title(f'Precision@k (fraction retirée = {frac})', 
                         fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Recall@k
        for method in df_frac['méthode'].unique():
            df_method = df_frac[df_frac['méthode'] == method].sort_values('k')
            axes[1].plot(df_method['k'], df_method['recall@k'], 
                        marker='s', label=method, linewidth=2, markersize=6)
        
        axes[1].set_xlabel('k (nombre de prédictions)', fontsize=11)
        axes[1].set_ylabel('Recall@k', fontsize=11)
        axes[1].set_title(f'Recall@k (fraction retirée = {frac})', 
                         fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        # Sauvegarder
        filename = f"question4_link_prediction_f{frac:.2f}.png"
        filepath = output_path / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"  {filename}")
        plt.close()


def analyze_question4_multiple(graphs: Dict[str, nx.Graph], output_dir: str = "report/figures"):
    """
    Analyse de link prediction sur plusieurs graphes
    
    Args:
        graphs: Dictionnaire {nom: graphe NetworkX}
        output_dir: Dossier de sortie
    """
    print("QUESTION 4: LINK PREDICTION")
    
    all_results = []
    
    for graph_name, G in graphs.items():
        print(f"\n{'='*80}")
        print(f"Réseau: {graph_name}")
        print(f"Nœuds: {G.number_of_nodes():,}, Arêtes: {G.number_of_edges():,}")
        print("="*80)
        
        # Évaluer les 3 métriques
        predictors = [
            ("CommonNeighbors", CommonNeighbors),
            ("Jaccard", Jaccard),
            ("AdamicAdar", AdamicAdar)
        ]
        
        for name, predictor_class in predictors:
            print(f"\nÉvaluation: {name}...")
            df = evaluate_predictor(G, predictor_class, seed=0)
            df['réseau'] = graph_name
            all_results.append(df)
            print(f"  Terminé")
    
    # Combiner tous les résultats
    df_all = pd.concat(all_results, ignore_index=True)
    
    # Statistiques globales par méthode
    print("PERFORMANCES MOYENNES GLOBALES (tous réseaux confondus)")
    summary = df_all.groupby(['méthode', 'k'], as_index=False)[['precision@k', 'recall@k']].mean()
    print(summary.to_string(index=False))
    
    # Statistiques par réseau
    print("PERFORMANCES PAR RÉSEAU (moyennées sur k et f)")
    summary_network = df_all.groupby(['réseau', 'méthode'], as_index=False)[['precision@k', 'recall@k']].mean()
    print(summary_network.to_string(index=False))
    
    # Sauvegarder
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    csv_path = output_path / "question4_link_prediction_results_all.csv"
    df_all.to_csv(csv_path, index=False)
    print(f"\nRésultats sauvegardés: {csv_path}")
    
    # Visualisation agrégée
    print("\nGénération du graphique agrégé...")
    plot_link_prediction_aggregated(df_all, output_dir)
    
    print("\nAnalyse Question 4 terminée")


def plot_link_prediction_aggregated(df: pd.DataFrame, output_dir: str):
    """Visualise les résultats agrégés sur tous les réseaux"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Moyennes par méthode et k (toutes fractions confondues)
    df_agg = df.groupby(['méthode', 'k'], as_index=False)[['precision@k', 'recall@k']].mean()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    methods = df_agg['méthode'].unique()
    for method in methods:
        sub = df_agg[df_agg['méthode'] == method]
        axes[0].plot(sub['k'], sub['precision@k'], marker='o', label=method, linewidth=2)
        axes[1].plot(sub['k'], sub['recall@k'], marker='o', label=method, linewidth=2)
    
    axes[0].set_xlabel('k (nombre de prédictions)', fontsize=11)
    axes[0].set_ylabel('Precision@k', fontsize=11)
    axes[0].set_title('Précision moyenne (tous réseaux)', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    axes[1].set_xlabel('k (nombre de prédictions)', fontsize=11)
    axes[1].set_ylabel('Recall@k', fontsize=11)
    axes[1].set_title('Rappel moyen (tous réseaux)', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    filepath = output_path / "question4_link_prediction_aggregated.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  question4_link_prediction_aggregated.png")
    plt.close()


def analyze_question4(G: nx.Graph, graph_name: str, output_dir: str = "report/figures"):
    """Analyse complète pour la Question 4"""
    
    print("\n" + "="*80)
    print("QUESTION 4: LINK PREDICTION")
    print("="*80)
    print(f"\nRéseau: {graph_name}")
    print(f"Nœuds: {G.number_of_nodes():,}, Arêtes: {G.number_of_edges():,}")
    print("\nCette analyse peut prendre plusieurs minutes...\n")
    
    # Évaluer les 3 méthodes
    methods = [
        (CommonNeighbors, "Common Neighbors"),
        (Jaccard, "Jaccard"),
        (AdamicAdar, "Adamic/Adar")
    ]
    
    all_results = []
    
    for predictor_class, name in methods:
        print(f"Évaluation: {name}...")
        df = evaluate_predictor(G, predictor_class, seed=0)
        all_results.append(df)
        print(f"  ✓ Terminé")
    
    # Combiner les résultats
    df_all = pd.concat(all_results, ignore_index=True)
    
    # Afficher un aperçu
    print("Aperçu des résultats:")
    print(df_all.head(15).to_string(index=False))
    
    # Statistiques moyennes
    print("Performances moyennes (sur toutes les fractions):")
    summary = df_all.groupby(['méthode', 'k'], as_index=False)[['precision@k', 'recall@k']].mean()
    print(summary.to_string(index=False))
    
    # Sauvegarder
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    csv_path = output_path / "question4_link_prediction_results.csv"
    df_all.to_csv(csv_path, index=False)
    print(f"\nRésultats sauvegardés: {csv_path}")
    
    # Visualisations
    print("\nGénération des graphiques...")
    plot_link_prediction_results(df_all, output_dir)
    
    print("\nAnalyse Question 4 terminée")


def main():
    """Fonction principale - Link prediction sur 15 réseaux aléatoires"""
    DATA_DIR = "data"
    NUM_NETWORKS = 15
    SEED = 42
    
    print(f"QUESTION 4: LINK PREDICTION SUR {NUM_NETWORKS} RÉSEAUX")
    
    try:
        # Charger les données avec cache
        all_graphs = load_graphs_with_cache(data_dir=DATA_DIR, verbose=True)
        
        if not all_graphs:
            print("Aucun graphe chargé")
            return
        
        # Échantillonner NUM_NETWORKS graphes aléatoires
        random.seed(SEED)
        if len(all_graphs) > NUM_NETWORKS:
            print(f"\nÉchantillonnage de {NUM_NETWORKS} réseaux aléatoires...\n")
            sampled_names = random.sample(list(all_graphs.keys()), NUM_NETWORKS)
            selected_graphs = {name: all_graphs[name] for name in sampled_names}
        else:
            selected_graphs = all_graphs
        
        print(f"✓ {len(selected_graphs)} réseaux sélectionnés pour l'analyse\n")
        
        # Analyser
        analyze_question4_multiple(selected_graphs, output_dir="report/figures")
        
    except FileNotFoundError as e:
        print(f"\nErreur: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nErreur inattendue: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
