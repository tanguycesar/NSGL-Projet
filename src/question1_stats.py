"""
Statistiques globales des réseaux Facebook100

Calcule les métriques standards pour caractériser les réseaux:
- n: nombre de nœuds
- m: nombre d'arêtes  
- degré moyen
- densité
- clustering global (transitivity)
- clustering local moyen
- assortativité par degré
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
import random
from typing import Dict

# Se placer dans le répertoire du script
script_dir = Path(__file__).parent.absolute()
os.chdir(script_dir)

# Importer le module de chargement avec cache
from cache_manager import load_graphs_with_cache


def compute_global_statistics(G: nx.Graph) -> Dict:
    """
    Calcule les statistiques globales d'un réseau
    
    Args:
        G: Graphe NetworkX
        
    Returns:
        Dictionnaire avec les métriques
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()
    
    if n == 0:
        return {
            "n": 0,
            "m": 0,
            "degré_moyen": 0,
            "densité": 0,
            "clustering_global": 0,
            "clustering_local_moyen": 0,
            "assortativité_degré": 0
        }
    
    # Métriques de base
    avg_degree = 2 * m / n
    density = nx.density(G)
    
    # Clustering
    transitivity = nx.transitivity(G)  # Clustering global
    clustering_coeffs = nx.clustering(G)
    avg_clustering = np.mean(list(clustering_coeffs.values()))
    
    # Assortativité par degré
    try:
        degree_assortativity = nx.degree_assortativity_coefficient(G)
    except:
        degree_assortativity = np.nan
    
    return {
        "n": n,
        "m": m,
        "degré_moyen": avg_degree,
        "densité": density,
        "clustering_global": transitivity,
        "clustering_local_moyen": avg_clustering,
        "assortativité_degré": degree_assortativity
    }


def analyze_all_networks(graphs: Dict[str, nx.Graph], 
                         selected_networks: list = None) -> pd.DataFrame:
    """
    Analyse tous les réseaux ou une sélection
    
    Args:
        graphs: Dictionnaire {nom: graphe}
        selected_networks: Liste de noms de réseaux à analyser (None = tous)
        
    Returns:
        DataFrame avec les statistiques
    """
    if selected_networks is None:
        selected_networks = list(graphs.keys())
    
    results = []
    
    print("Calcul des statistiques globales...")
    print(f"{'Réseau':<30} {'Nœuds':>8} {'Arêtes':>10}")
    print("-" * 50)
    
    for name in selected_networks:
        if name not in graphs:
            print(f"[ATTENTION] Réseau '{name}' non trouvé")
            continue
        
        G = graphs[name]
        stats = compute_global_statistics(G)
        stats["réseau"] = name
        results.append(stats)
        
        print(f"{name:<30} {stats['n']:>8,} {stats['m']:>10,}")
    
    if not results:
        return pd.DataFrame()
    
    # Créer le DataFrame et trier par taille
    df = pd.DataFrame(results)
    df = df[["réseau", "n", "m", "degré_moyen", "densité", 
             "clustering_global", "clustering_local_moyen", "assortativité_degré"]]
    df = df.sort_values("n")
    
    return df


def display_summary(df: pd.DataFrame):
    """Affiche un résumé formaté des résultats"""
    if df.empty:
        print("Aucun résultat à afficher")
        return
    
    print("STATISTIQUES GLOBALES DES RÉSEAUX")
        
    # Formater l'affichage
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', lambda x: f'{x:.6f}')
    
    print("\n" + df.to_string(index=False))
    
    # Statistiques agrégées
    print("STATISTIQUES AGRÉGÉES (sur tous les réseaux)")
    
    numeric_cols = ["n", "m", "degré_moyen", "densité", 
                    "clustering_global", "clustering_local_moyen", "assortativité_degré"]
    
    summary = df[numeric_cols].describe().T[["mean", "std", "min", "max"]]
    print(summary.to_string())


def save_results(df: pd.DataFrame, output_dir: str = "results"):
    """Sauvegarde les résultats dans un fichier CSV"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    filepath = output_path / "question1_statistics.csv"
    df.to_csv(filepath, index=False)
    print(f"\nRésultats sauvegardés: {filepath}")


def main():
    """Fonction principale - Analyse statistiques globales des 100 réseaux Facebook"""
    DATA_DIR = "data"
    
    print("="*80)
    print("STATISTIQUES GLOBALES DES RÉSEAUX FACEBOOK100")
    print("="*80 + "\n")
    
    try:
        # Charger tous les graphes avec cache
        graphs = load_graphs_with_cache(data_dir=DATA_DIR, verbose=True)
        
        if not graphs:
            print("Aucun graphe chargé. Vérifiez le dossier data/")
            return
        
        print(f"\n{len(graphs)} réseaux chargés\n")
        
        # Analyser tous les réseaux disponibles
        df_all = analyze_all_networks(graphs, selected_networks=None)
        display_summary(df_all)
        save_results(df_all, output_dir="report/figures")
        
    except FileNotFoundError as e:
        print(f"\nErreur: {e}")
        print("\nAssurez-vous que:")
        print("  1. Le dossier 'data/' existe")
        print("  2. Les fichiers .gml Facebook100 sont présents")
        sys.exit(1)
    except Exception as e:
        print(f"\nErreur inattendue: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
