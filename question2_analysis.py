"""
Question 2: Analyse détaillée des réseaux sociaux

Focus sur 3 universités: Caltech36, MIT8, Johns Hopkins55

Analyses:
2a) Distribution des degrés (histogramme + CCDF log-log)
2b) Clustering global, clustering local moyen, densité
2c) Relation degré vs clustering local
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List

# Se placer dans le répertoire du script
script_dir = Path(__file__).parent.absolute()
os.chdir(script_dir)

# Importer le module de chargement avec cache
from cache_manager import load_graphs_with_cache

# Configuration matplotlib
plt.style.use('seaborn-v0_8-darkgrid')


def get_degree_array(G: nx.Graph) -> np.ndarray:
    """Retourne un tableau des degrés (strictement positifs)"""
    degrees = [d for _, d in G.degree() if d > 0]
    return np.array(degrees, dtype=int)


def compute_degree_statistics(G: nx.Graph) -> Dict:
    """Calcule les statistiques de degré"""
    degrees = get_degree_array(G)
    
    if len(degrees) == 0:
        return {
            "mean": 0, "median": 0, "max": 0, 
            "std": 0, "min": 0
        }
    
    return {
        "mean": np.mean(degrees),
        "median": np.median(degrees),
        "max": np.max(degrees),
        "std": np.std(degrees),
        "min": np.min(degrees)
    }


def plot_degree_distribution(graphs: Dict[str, nx.Graph], output_dir: str = "report/figures"):
    """
    Trace les distributions de degrés:
    - Histogrammes (échelle linéaire)
    - CCDF en log-log
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    n_graphs = len(graphs)
    
    # Figure 1: Histogrammes des degrés
    fig, axes = plt.subplots(1, n_graphs, figsize=(6*n_graphs, 5))
    if n_graphs == 1:
        axes = [axes]
    
    for idx, (name, G) in enumerate(graphs.items()):
        degrees = get_degree_array(G)
        
        axes[idx].hist(degrees, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        axes[idx].set_xlabel('Degré k', fontsize=11)
        axes[idx].set_ylabel('Fréquence', fontsize=11)
        axes[idx].set_title(f'{name}\n(n={G.number_of_nodes():,})', fontsize=12, fontweight='bold')
        axes[idx].grid(alpha=0.3)
    
    plt.tight_layout()
    hist_path = output_path / "question2a_degree_histograms.png"
    plt.savefig(hist_path, dpi=300, bbox_inches='tight')
    print(f"✓ Sauvegardé: {hist_path}")
    plt.close()
    
    # Figure 2: CCDF log-log (comparaison)
    fig, ax = plt.subplots(figsize=(10, 7))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for idx, (name, G) in enumerate(graphs.items()):
        degrees = get_degree_array(G)
        k_sorted = np.sort(degrees)
        ccdf = 1.0 - np.arange(1, len(k_sorted) + 1) / len(k_sorted)
        
        ax.plot(k_sorted, ccdf, marker='o', linestyle='none', 
                markersize=4, alpha=0.6, label=name, color=colors[idx % len(colors)])
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Degré k (échelle log)', fontsize=12)
    ax.set_ylabel('P(K ≥ k) (échelle log)', fontsize=12)
    ax.set_title('Comparaison des CCDF des degrés', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3, which='both')
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    ccdf_path = output_path / "question2a_degree_ccdf.png"
    plt.savefig(ccdf_path, dpi=300, bbox_inches='tight')
    print(f"✓ Sauvegardé: {ccdf_path}")
    plt.close()


def compute_clustering_density(G: nx.Graph) -> Dict:
    """Calcule les métriques de clustering et densité"""
    clustering_coeffs = nx.clustering(G)
    
    return {
        "n": G.number_of_nodes(),
        "m": G.number_of_edges(),
        "densité": nx.density(G),
        "clustering_global": nx.transitivity(G),
        "clustering_local_moyen": np.mean(list(clustering_coeffs.values()))
    }


def plot_degree_vs_clustering(graphs: Dict[str, nx.Graph], output_dir: str = "report/figures"):
    """
    Trace la relation degré vs clustering local
    Utilise des bins (quantiles) pour lisser le bruit
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    n_graphs = len(graphs)
    fig, axes = plt.subplots(1, n_graphs, figsize=(7*n_graphs, 6))
    if n_graphs == 1:
        axes = [axes]
    
    for idx, (name, G) in enumerate(graphs.items()):
        degrees = dict(G.degree())
        clustering = nx.clustering(G)
        
        # Créer un DataFrame pour faciliter le binning
        data = [(degrees[v], clustering[v]) for v in G.nodes() if degrees[v] > 0]
        df = pd.DataFrame(data, columns=['k', 'C'])
        
        # Binning par quantiles
        n_bins = 12
        try:
            df['k_bin'] = pd.qcut(df['k'], q=n_bins, duplicates='drop')
            grouped = df.groupby('k_bin', observed=True).mean()
        except:
            # Si trop peu de données, utiliser moins de bins
            n_bins = min(5, len(df) // 10)
            if n_bins > 1:
                df['k_bin'] = pd.qcut(df['k'], q=n_bins, duplicates='drop')
                grouped = df.groupby('k_bin', observed=True).mean()
            else:
                grouped = df  # Pas de binning
        
        # Scatter plot (données brutes)
        axes[idx].scatter(df['k'], df['C'], s=10, alpha=0.2, color='gray', label='Nœuds')
        
        # Tendance moyenne
        if len(grouped) > 1:
            axes[idx].plot(grouped['k'], grouped['C'], color='crimson', 
                          marker='o', linewidth=2, markersize=8, label='Moyenne par classe')
        
        axes[idx].set_xscale('log')
        axes[idx].set_xlabel('Degré k (échelle log)', fontsize=11)
        axes[idx].set_ylabel('Clustering local', fontsize=11)
        axes[idx].set_title(f'{name}', fontsize=12, fontweight='bold')
        axes[idx].grid(alpha=0.3)
        axes[idx].legend()
    
    plt.tight_layout()
    filepath = output_path / "question2c_degree_vs_clustering.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ Sauvegardé: {filepath}")
    plt.close()


def analyze_question2(graphs: Dict[str, nx.Graph], output_dir: str = "report/figures"):
    """Analyse complète pour la Question 2"""
    
    print("\n" + "="*80)
    print("QUESTION 2a: DISTRIBUTION DES DEGRÉS")
    print("="*80)
    
    # Statistiques de degré
    degree_stats = []
    for name, G in graphs.items():
        stats = compute_degree_statistics(G)
        stats['réseau'] = name
        degree_stats.append(stats)
        
        print(f"\n{name}:")
        print(f"  Degré moyen   : {stats['mean']:.2f}")
        print(f"  Degré médian  : {stats['median']:.0f}")
        print(f"  Degré max     : {stats['max']}")
        print(f"  Écart-type    : {stats['std']:.2f}")
    
    df_degrees = pd.DataFrame(degree_stats)
    df_degrees = df_degrees[['réseau', 'mean', 'median', 'max', 'std', 'min']]
    
    # Visualisations
    plot_degree_distribution(graphs, output_dir)
    
    print("\n" + "="*80)
    print("QUESTION 2b: CLUSTERING ET DENSITÉ")
    print("="*80)
    
    # Métriques de clustering
    clustering_stats = []
    for name, G in graphs.items():
        stats = compute_clustering_density(G)
        stats['réseau'] = name
        clustering_stats.append(stats)
    
    df_clustering = pd.DataFrame(clustering_stats)
    df_clustering = df_clustering[['réseau', 'n', 'm', 'densité', 
                                    'clustering_global', 'clustering_local_moyen']]
    
    print("\n" + df_clustering.to_string(index=False))
    
    # Sauvegarder les résultats
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    df_degrees.to_csv(output_path / "question2a_degree_stats.csv", index=False)
    df_clustering.to_csv(output_path / "question2b_clustering_stats.csv", index=False)
    
    print("\n" + "="*80)
    print("QUESTION 2c: RELATION DEGRÉ VS CLUSTERING")
    print("="*80)
    
    plot_degree_vs_clustering(graphs, output_dir)
    
    print("\n✓ Analyse Question 2 terminée")


def main():
    """Fonction principale"""
    # Configuration
    DATA_DIR = "data"
    SELECTED_NETWORKS = ["Caltech36", "MIT8", "Johns Hopkins55"]
    
    print("="*80)
    print("QUESTION 2: ANALYSE DÉTAILLÉE DES RÉSEAUX SOCIAUX")
    print("="*80 + "\n")
    
    try:
        # Charger les données avec cache
        all_graphs = load_graphs_with_cache(data_dir=DATA_DIR, verbose=True)
        
        # Sélectionner les réseaux demandés
        selected_graphs = {}
        missing = []
        
        for name in SELECTED_NETWORKS:
            if name in all_graphs:
                selected_graphs[name] = all_graphs[name]
            else:
                missing.append(name)
        
        if missing:
            print(f"\n⚠ Réseaux manquants: {', '.join(missing)}")
        
        if not selected_graphs:
            print("❌ Aucun des réseaux sélectionnés n'est disponible")
            return
        
        print(f"\n✓ {len(selected_graphs)} réseaux sélectionnés pour l'analyse\n")
        
        # Analyser
        analyze_question2(selected_graphs, output_dir="report/figures")
        
    except FileNotFoundError as e:
        print(f"\n❌ Erreur: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erreur inattendue: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
