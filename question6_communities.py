"""
Question 6: Détection de Communautés (Question de Recherche)

Démarche:
1. Proposer une hypothèse de recherche sur la formation de groupes
2. Choisir des universités pour tester l'hypothèse
3. Appliquer des méthodes de détection de communautés
4. Analyser la correspondance avec les attributs
5. Conclure

Méthodes:
- Louvain (si disponible)
- Greedy Modularity

Métriques de comparaison:
- NMI (Normalized Mutual Information)
- ARI (Adjusted Rand Index)

Attributs à comparer:
- dorm, major_index, gender, year
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

# Se placer dans le répertoire du script
script_dir = Path(__file__).parent.absolute()
os.chdir(script_dir)

# Importer le module de chargement avec cache
from cache_manager import load_graphs_with_cache

# Configuration matplotlib
plt.style.use('seaborn-v0_8-darkgrid')

# Valeurs considérées comme manquantes
MISSING_VALUES = {0, -1, None, '', '0', '-1'}


def detect_communities(G: nx.Graph) -> Dict[str, List]:
    """
    Applique plusieurs algorithmes de détection de communautés
    
    Args:
        G: Graphe NetworkX
        
    Returns:
        Dictionnaire {méthode: liste_de_communautés}
    """
    results = {}
    
    # Méthode 1: Louvain
    try:
        communities = nx.algorithms.community.louvain_communities(G, seed=0)
        results["Louvain"] = list(communities)
    except Exception as e:
        print(f"  ⚠ Louvain indisponible: {e}")
    
    # Méthode 2: Greedy Modularity
    try:
        communities = nx.algorithms.community.greedy_modularity_communities(G)
        results["Greedy Modularity"] = list(communities)
    except Exception as e:
        print(f"  ⚠ Greedy Modularity échec: {e}")
    
    return results


def communities_to_labels(communities: List[set], G: nx.Graph) -> np.ndarray:
    """
    Convertit une liste de communautés en tableau de labels
    
    Args:
        communities: Liste de sets de nœuds
        G: Graphe NetworkX
        
    Returns:
        Array avec les labels de communauté pour chaque nœud
    """
    n = G.number_of_nodes()
    labels = -np.ones(n, dtype=int)
    
    # Créer un mapping nœud -> index
    node_to_idx = {node: idx for idx, node in enumerate(G.nodes())}
    
    for comm_id, community in enumerate(communities):
        for node in community:
            idx = node_to_idx.get(node)
            if idx is not None:
                labels[idx] = comm_id
    
    return labels


def get_node_labels(G: nx.Graph, attribute: str) -> np.ndarray:
    """
    Extrait les labels d'un attribut
    
    Args:
        G: Graphe NetworkX
        attribute: Nom de l'attribut
        
    Returns:
        Array numpy avec les labels (-1 pour manquant)
    """
    n = G.number_of_nodes()
    labels = -np.ones(n, dtype=int)
    
    node_attrs = nx.get_node_attributes(G, attribute)
    
    # Créer un mapping nœud -> index
    node_to_idx = {node: idx for idx, node in enumerate(G.nodes())}
    
    for node in G.nodes():
        if node in node_attrs:
            value = node_attrs[node]
            if value not in MISSING_VALUES and value != -1:
                idx = node_to_idx[node]
                labels[idx] = int(value)
    
    return labels


def compare_communities_with_attribute(comm_labels: np.ndarray,
                                      attr_labels: np.ndarray) -> Optional[Dict]:
    """
    Compare les communautés détectées avec un attribut
    
    Args:
        comm_labels: Labels des communautés
        attr_labels: Labels de l'attribut
        
    Returns:
        Dictionnaire avec NMI et ARI (ou None si impossible)
    """
    # Filtrer les nœuds avec labels valides dans les deux
    valid_idx = (comm_labels >= 0) & (attr_labels >= 0)
    
    if valid_idx.sum() < 10:
        return None
    
    comm_valid = comm_labels[valid_idx]
    attr_valid = attr_labels[valid_idx]
    
    nmi = normalized_mutual_info_score(attr_valid, comm_valid)
    ari = adjusted_rand_score(attr_valid, comm_valid)
    
    return {
        "n_évalués": valid_idx.sum(),
        "NMI": nmi,
        "ARI": ari
    }


def analyze_communities(G: nx.Graph, graph_name: str) -> pd.DataFrame:
    """
    Analyse complète des communautés pour un graphe
    
    Args:
        G: Graphe NetworkX
        graph_name: Nom du réseau
        
    Returns:
        DataFrame avec les résultats
    """
    print(f"\n{'-'*80}")
    print(f"Analyse: {graph_name}")
    print(f"{'-'*80}")
    print(f"Nœuds: {G.number_of_nodes():,}, Arêtes: {G.number_of_edges():,}")
    
    # Détecter les communautés
    print("\nDétection de communautés...")
    communities_dict = detect_communities(G)
    
    if not communities_dict:
        print("  ⚠ Aucune méthode de détection n'a fonctionné")
        return pd.DataFrame()
    
    print(f"  ✓ {len(communities_dict)} méthode(s) appliquée(s)")
    
    # Attributs à comparer
    attributes = [
        ("dorm", "Résidence"),
        ("major_index", "Discipline"),
        ("gender", "Genre"),
        ("year", "Année")
    ]
    
    results = []
    
    for method, communities in communities_dict.items():
        n_comm = len(communities)
        comm_labels = communities_to_labels(communities, G)
        
        print(f"\n{method}: {n_comm} communautés")
        
        for attr, attr_desc in attributes:
            attr_labels = get_node_labels(G, attr)
            
            comparison = compare_communities_with_attribute(comm_labels, attr_labels)
            
            if comparison is not None:
                print(f"  {attr_desc:15} - NMI: {comparison['NMI']:.4f}, ARI: {comparison['ARI']:.4f}")
                
                results.append({
                    "réseau": graph_name,
                    "méthode": method,
                    "n_communautés": n_comm,
                    "attribut": attr,
                    "description": attr_desc,
                    **comparison
                })
            else:
                print(f"  {attr_desc:15} - Données insuffisantes")
    
    return pd.DataFrame(results)


def plot_community_comparison(df: pd.DataFrame, output_dir: str = "report/figures"):
    """Visualise les comparaisons communautés vs attributs"""
    
    if df.empty:
        print("Aucun résultat à visualiser")
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Une figure par réseau
    for network in df['réseau'].unique():
        df_net = df[df['réseau'] == network]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Préparer les données pour le plot
        methods = df_net['méthode'].unique()
        attributes = df_net['attribut'].unique()
        
        x_labels = []
        nmi_values = {method: [] for method in methods}
        ari_values = {method: [] for method in methods}
        
        for attr in attributes:
            x_labels.append(attr)
            for method in methods:
                row = df_net[(df_net['méthode'] == method) & (df_net['attribut'] == attr)]
                if not row.empty:
                    nmi_values[method].append(row['NMI'].values[0])
                    ari_values[method].append(row['ARI'].values[0])
                else:
                    nmi_values[method].append(0)
                    ari_values[method].append(0)
        
        # Plot NMI
        x = np.arange(len(x_labels))
        width = 0.35 if len(methods) == 2 else 0.25
        
        for idx, method in enumerate(methods):
            offset = (idx - len(methods)/2 + 0.5) * width
            axes[0].bar(x + offset, nmi_values[method], width, label=method, alpha=0.8)
        
        axes[0].set_xlabel('Attribut', fontsize=11)
        axes[0].set_ylabel('NMI', fontsize=11)
        axes[0].set_title(f'NMI — {network}', fontsize=12, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(x_labels, rotation=45, ha='right')
        axes[0].legend()
        axes[0].grid(alpha=0.3, axis='y')
        
        # Plot ARI
        for idx, method in enumerate(methods):
            offset = (idx - len(methods)/2 + 0.5) * width
            axes[1].bar(x + offset, ari_values[method], width, label=method, alpha=0.8)
        
        axes[1].set_xlabel('Attribut', fontsize=11)
        axes[1].set_ylabel('ARI', fontsize=11)
        axes[1].set_title(f'ARI — {network}', fontsize=12, fontweight='bold')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(x_labels, rotation=45, ha='right')
        axes[1].legend()
        axes[1].grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        filename = f"question6_communities_{network}.png"
        filepath = output_path / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"  ✓ {filename}")
        plt.close()


def analyze_question6(graphs: Dict[str, nx.Graph], output_dir: str = "report/figures"):
    """Analyse complète pour la Question 6"""
    
    print("\n" + "="*80)
    print("QUESTION 6: DÉTECTION DE COMMUNAUTÉS (QUESTION DE RECHERCHE)")
    print("="*80)
    
    print("\nHYPOTHÈSE DE RECHERCHE:")
    print("-" * 80)
    print("Les communautés détectées reflètent-elles principalement la structure")
    print("résidentielle (dorm), la discipline (major), ou d'autres attributs sociaux ?")
    print("-" * 80)
    
    all_results = []
    
    for name, G in graphs.items():
        df = analyze_communities(G, name)
        if not df.empty:
            all_results.append(df)
    
    if not all_results:
        print("\n❌ Aucun résultat obtenu")
        return
    
    # Combiner tous les résultats
    df_all = pd.concat(all_results, ignore_index=True)
    
    # Afficher le tableau récapitulatif
    print("\n" + "="*80)
    print("RÉSULTATS RÉCAPITULATIFS")
    print("="*80 + "\n")
    
    # Tableau simplifié
    summary = df_all[['réseau', 'méthode', 'attribut', 'n_communautés', 'NMI', 'ARI']]
    print(summary.to_string(index=False))
    
    # Statistiques par attribut
    print("\n" + "-"*80)
    print("STATISTIQUES MOYENNES PAR ATTRIBUT")
    print("-"*80 + "\n")
    
    stats_by_attr = df_all.groupby('attribut', as_index=False)[['NMI', 'ARI']].mean()
    stats_by_attr = stats_by_attr.sort_values('NMI', ascending=False)
    print(stats_by_attr.to_string(index=False))
    
    # Sauvegarder
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    csv_path = output_path / "question6_communities_results.csv"
    df_all.to_csv(csv_path, index=False)
    print(f"\n✓ Résultats sauvegardés: {csv_path}")
    
    # Visualisations
    print("\nGénération des graphiques...")
    plot_community_comparison(df_all, output_dir)
    
    # Conclusion
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    
    best_attr = stats_by_attr.iloc[0]['attribut']
    best_nmi = stats_by_attr.iloc[0]['NMI']
    
    print(f"\nL'attribut avec la meilleure correspondance moyenne est: {best_attr}")
    print(f"NMI moyen: {best_nmi:.4f}")
    
    if best_nmi > 0.3:
        print("\n→ FORTE correspondance: Les communautés reflètent principalement cet attribut")
    elif best_nmi > 0.1:
        print("\n→ MODÉRÉE: Les communautés sont partiellement influencées par cet attribut")
    else:
        print("\n→ FAIBLE: Les communautés ne correspondent pas fortement aux attributs sociaux")
    
    print("\n✓ Analyse Question 6 terminée")


def main():
    """Fonction principale"""
    DATA_DIR = "data"
    # Réseaux à analyser (peut être ajusté selon l'hypothèse)
    SELECTED_NETWORKS = ["American75", "Caltech36", "MIT8"]
    
    print("="*80)
    print("QUESTION 6: DÉTECTION DE COMMUNAUTÉS SUR RÉSEAUX FACEBOOK100")
    print("="*80 + "\n")
    
    try:
        # Charger les données avec cache
        all_graphs = load_graphs_with_cache(data_dir=DATA_DIR, verbose=True)
        
        if not all_graphs:
            print("❌ Aucun graphe chargé")
            return
        
        # Sélectionner les réseaux
        selected_graphs = {}
        for name in SELECTED_NETWORKS:
            if name in all_graphs:
                selected_graphs[name] = all_graphs[name]
            else:
                print(f"⚠ Réseau '{name}' non trouvé")
        
        if not selected_graphs:
            # Prendre le premier disponible
            first_name = next(iter(all_graphs))
            selected_graphs[first_name] = all_graphs[first_name]
            print(f"\nUtilisation de '{first_name}' par défaut")
        
        print(f"\n✓ {len(selected_graphs)} réseau(x) sélectionné(s)")
        
        # Analyser
        analyze_question6(selected_graphs, output_dir="report/figures")
        
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
