"""
Question 3: Assortativité (Homophilie)

Calcule l'assortativité pour 5 attributs:
1. student_fac (statut étudiant/faculté)
2. major_index (discipline)
3. degree (degré des nœuds)
4. dorm (résidence)
5. gender (genre)

Pour chaque attribut:
- Scatter: assortativité vs taille du réseau (log)
- Histogramme: distribution des assortativités
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import random
from typing import Dict, Optional

# Se placer dans le répertoire du script
script_dir = Path(__file__).parent.absolute()
os.chdir(script_dir)

# Importer le module de chargement avec cache
from cache_manager import load_graphs_with_cache

# Configuration matplotlib
plt.style.use('seaborn-v0_8-darkgrid')

# Valeurs considérées comme manquantes
MISSING_VALUES = {0, -1, None, '', '0', '-1'}


def compute_assortativity(G: nx.Graph, attribute: str) -> Optional[float]:
    """
    Calcule l'assortativité pour un attribut donné
    
    Args:
        G: Graphe NetworkX
        attribute: Nom de l'attribut ('student_fac', 'major_index', 'degree', 'dorm', 'gender')
        
    Returns:
        Coefficient d'assortativité (ou None si impossible à calculer)
    """
    # Cas spécial: assortativité par degré
    if attribute == "degree":
        try:
            return nx.degree_assortativity_coefficient(G)
        except:
            return None
    
    # Récupérer les attributs des nœuds
    node_attrs = nx.get_node_attributes(G, attribute)
    
    if len(node_attrs) == 0:
        return None
    
    # Filtrer les nœuds avec des valeurs valides
    valid_nodes = []
    for node in G.nodes():
        if node in node_attrs:
            value = node_attrs[node]
            # Exclure les valeurs manquantes
            if value not in MISSING_VALUES and value != -1:
                valid_nodes.append(node)
    
    # Nécessite au moins 10 nœuds avec attribut valide
    if len(valid_nodes) < 10:
        return None
    
    # Créer sous-graphe avec nœuds valides
    H = G.subgraph(valid_nodes).copy()
    
    if H.number_of_edges() == 0:
        return None
    
    # Calculer l'assortativité
    try:
        return nx.attribute_assortativity_coefficient(H, attribute)
    except:
        return None


def compute_assortativity_all_graphs(graphs: Dict[str, nx.Graph], 
                                     attribute: str) -> pd.DataFrame:
    """
    Calcule l'assortativité pour tous les graphes
    
    Args:
        graphs: Dictionnaire {nom: graphe}
        attribute: Nom de l'attribut
        
    Returns:
        DataFrame avec les résultats
    """
    results = []
    
    for name, G in graphs.items():
        r = compute_assortativity(G, attribute)
        
        if r is not None and not np.isnan(r):
            results.append({
                "graphe": name,
                "n": G.number_of_nodes(),
                "assortativité": float(r)
            })
    
    if not results:
        return pd.DataFrame(columns=["graphe", "n", "assortativité"])
    
    df = pd.DataFrame(results).sort_values("n")
    return df


def plot_assortativity(df: pd.DataFrame, attribute: str, output_dir: str = "report/figures"):
    """
    Visualise l'assortativité pour un attribut
    
    Args:
        df: DataFrame avec les résultats
        attribute: Nom de l'attribut
        output_dir: Dossier de sortie
    """
    if df.empty:
        print(f"⚠ Aucune valeur d'assortativité calculable pour '{attribute}'")
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scatter: assortativité vs taille
    axes[0].scatter(df["n"], df["assortativité"], s=50, alpha=0.6, color='darkviolet')
    axes[0].axhline(0.0, linestyle="--", color='red', linewidth=1.5, 
                    alpha=0.7, label='r=0 (pas d\'assortativité)')
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Taille du réseau (n)", fontsize=12)
    axes[0].set_ylabel("Assortativité r", fontsize=12)
    axes[0].set_title(f"Assortativité ({attribute}) vs Taille", 
                     fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)
    
    # Histogramme: distribution
    axes[1].hist(df["assortativité"], bins=20, density=True, 
                edgecolor='black', alpha=0.7, color='teal')
    axes[1].axvline(0.0, linestyle="--", color='red', linewidth=1.5, 
                   alpha=0.7, label='r=0')
    axes[1].set_xlabel("Assortativité r", fontsize=12)
    axes[1].set_ylabel("Densité", fontsize=12)
    axes[1].set_title(f"Distribution — {attribute}", 
                     fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Sauvegarder
    filename = f"question3_assortativity_{attribute}.png"
    filepath = output_path / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  ✓ {filename}")
    plt.close()
    
    # Statistiques
    print(f"\n  Statistiques ({attribute}):")
    print(f"    Réseaux analysés : {len(df)}")
    print(f"    Moyenne          : {df['assortativité'].mean():.4f}")
    print(f"    Médiane          : {df['assortativité'].median():.4f}")
    print(f"    Min              : {df['assortativité'].min():.4f}")
    print(f"    Max              : {df['assortativité'].max():.4f}")


def analyze_question3(graphs: Dict[str, nx.Graph], output_dir: str = "report/figures"):
    """Analyse complète pour la Question 3"""
    
    print("\n" + "="*80)
    print("QUESTION 3: ASSORTATIVITÉ (HOMOPHILIE)")
    print("="*80)
    
    # Les 5 attributs demandés (avec les vrais noms dans les fichiers .gml)
    attributes = [
        ("student_fac", "Statut étudiant/faculté"),
        ("major_index", "Discipline principale"),
        ("degree", "Degré des nœuds"),
        ("dorm", "Résidence"),
        ("gender", "Genre")
    ]
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    all_results = []
    
    for attr, description in attributes:
        print(f"\n{'-'*80}")
        print(f"Analyse: {description} ({attr})")
        print(f"{'-'*80}")
        
        # Calculer l'assortativité
        df = compute_assortativity_all_graphs(graphs, attr)
        
        if not df.empty:
            # Visualiser
            plot_assortativity(df, attr, output_dir)
            
            # Sauvegarder les résultats
            csv_path = output_path / f"question3_assortativity_{attr}.csv"
            df.to_csv(csv_path, index=False)
            
            # Ajouter pour le tableau récapitulatif
            all_results.append({
                "attribut": attr,
                "description": description,
                "n_réseaux": len(df),
                "assort_moyenne": df['assortativité'].mean(),
                "assort_médiane": df['assortativité'].median(),
                "assort_min": df['assortativité'].min(),
                "assort_max": df['assortativité'].max(),
                "assort_std": df['assortativité'].std()
            })
        else:
            print(f"  ⚠ Aucune donnée utilisable pour {attr}")
    
    # Tableau récapitulatif
    if all_results:
        print("\n" + "="*80)
        print("RÉCAPITULATIF")
        print("="*80 + "\n")
        
        df_summary = pd.DataFrame(all_results)
        df_summary = df_summary[["attribut", "description", "n_réseaux", 
                                 "assort_moyenne", "assort_médiane", 
                                 "assort_min", "assort_max", "assort_std"]]
        
        print(df_summary.to_string(index=False))
        
        # Sauvegarder
        summary_path = output_path / "question3_assortativity_summary.csv"
        df_summary.to_csv(summary_path, index=False)
        print(f"\n✓ Résumé sauvegardé: {summary_path}")
    
    print("\n✓ Analyse Question 3 terminée")


def main():
    """Fonction principale"""
    DATA_DIR = "data"
    
    print("="*80)
    print("QUESTION 3: ASSORTATIVITÉ SUR LES RÉSEAUX FACEBOOK100")
    print("="*80 + "\n")
    
    try:
        # Charger tous les graphes avec cache
        graphs = load_graphs_with_cache(data_dir=DATA_DIR, verbose=True)
        
        if not graphs:
            print("❌ Aucun graphe chargé")
            return
        
        print(f"\n✓ {len(graphs)} réseaux chargés\n")
        
        # Analyser
        analyze_question3(graphs, output_dir="report/figures")
        
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
