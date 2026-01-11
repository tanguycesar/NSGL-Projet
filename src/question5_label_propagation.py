"""
Question 5: Label Propagation

Algorithme semi-supervisé pour prédire les attributs manquants:
- dorm (résidence)
- major_index (discipline)
- gender (genre)

Protocole:
- Retirer aléatoirement 10%, 20%, 30% des labels
- Prédire avec label propagation
- Évaluer: Accuracy, F1-score (macro), MAE
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import torch
import random
from typing import Dict, List, Tuple
from sklearn.metrics import accuracy_score, f1_score

# Se placer dans le répertoire du script
script_dir = Path(__file__).parent.absolute()
os.chdir(script_dir)

# Importer le module de chargement avec cache
from cache_manager import load_graphs_with_cache

# Configuration matplotlib
plt.style.use('seaborn-v0_8-darkgrid')

# Valeurs considérées comme manquantes
MISSING_VALUES = {0, -1, None, '', '0', '-1'}


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
    
    for node in G.nodes():
        if node in node_attrs:
            value = node_attrs[node]
            if value not in MISSING_VALUES and value != -1:
                labels[node] = int(value)
    
    return labels


def build_normalized_adjacency(G: nx.Graph, device: str = "cpu") -> torch.Tensor:
    """
    Construit la matrice d'adjacence normalisée par ligne (sparse PyTorch)
    
    Args:
        G: Graphe NetworkX
        device: Device PyTorch ('cpu' ou 'cuda')
        
    Returns:
        Matrice sparse normalisée
    """
    n = G.number_of_nodes()
    edges = np.array(list(G.edges()), dtype=int)
    
    if len(edges) == 0:
        raise ValueError("Graphe sans arêtes")
    
    # Créer les indices (bidirectionnel)
    src = np.concatenate([edges[:, 0], edges[:, 1]])
    dst = np.concatenate([edges[:, 1], edges[:, 0]])
    
    indices = torch.tensor(np.vstack([src, dst]), dtype=torch.long, device=device)
    values = torch.ones(indices.shape[1], dtype=torch.float32, device=device)
    
    # Matrice d'adjacence sparse
    A = torch.sparse_coo_tensor(indices, values, size=(n, n))
    
    # Normalisation par ligne: S[i,j] = A[i,j] / deg(i)
    degrees = torch.sparse.sum(A, dim=1).to_dense().clamp(min=1.0)
    inv_degrees = 1.0 / degrees
    
    # Appliquer la normalisation
    values_norm = inv_degrees[src]
    S = torch.sparse_coo_tensor(indices, values_norm, size=(n, n))
    
    return S


def label_propagation(G: nx.Graph,
                     y: np.ndarray,
                     mask_labeled: np.ndarray,
                     alpha: float = 0.9,
                     max_iter: int = 200,
                     tol: float = 1e-6,
                     device: str = "cpu") -> np.ndarray:
    """
    Algorithme de label propagation semi-supervisé
    
    Args:
        G: Graphe NetworkX
        y: Labels initiaux (-1 pour inconnu)
        mask_labeled: Booléen indiquant les nœuds avec labels connus
        alpha: Paramètre de mixage (0.9 = 90% voisins, 10% labels initiaux)
        max_iter: Nombre maximum d'itérations
        tol: Tolérance pour la convergence
        device: Device PyTorch
        
    Returns:
        Labels prédits
    """
    n = len(y)
    
    # Classes présentes
    classes = np.unique(y[mask_labeled])
    classes = classes[classes >= 0]
    
    if len(classes) == 0:
        raise ValueError("Aucune classe trouvée dans les labels")
    
    n_classes = int(classes.max() + 1)
    
    # Matrice d'adjacence normalisée
    S = build_normalized_adjacency(G, device=device)
    
    # Matrice des labels initiaux (one-hot)
    Y = torch.zeros((n, n_classes), dtype=torch.float32, device=device)
    for i in range(n):
        if mask_labeled[i] and y[i] >= 0 and y[i] < n_classes:
            Y[i, y[i]] = 1.0
    
    # Initialisation
    F = Y.clone()
    
    # Itérations
    for iteration in range(max_iter):
        # F_new = alpha * S @ F + (1 - alpha) * Y
        F_new = alpha * torch.sparse.mm(S, F) + (1.0 - alpha) * Y
        
        # Fixer les labels connus
        F_new[mask_labeled] = Y[mask_labeled]
        
        # Vérifier la convergence
        delta = torch.norm(F_new - F).item()
        F = F_new
        
        if delta < tol:
            break
    
    # Prédiction: argmax
    y_pred = torch.argmax(F, dim=1).detach().cpu().numpy().astype(int)
    
    return y_pred


def evaluate_label_propagation(G: nx.Graph,
                               attribute: str,
                               frac_remove_list: List[float] = [0.1, 0.2, 0.3],
                               seed: int = 0,
                               device: str = "cpu") -> pd.DataFrame:
    """
    Évalue la label propagation en retirant une fraction des labels
    
    Args:
        G: Graphe NetworkX
        attribute: Nom de l'attribut à prédire
        frac_remove_list: Fractions de labels à retirer
        seed: Graine aléatoire
        device: Device PyTorch
        
    Returns:
        DataFrame avec les résultats
    """
    rng = np.random.default_rng(seed)
    
    # Récupérer les labels complets
    y_full = get_node_labels(G, attribute)
    known_idx = np.where(y_full >= 0)[0]
    
    if len(known_idx) < 50:
        raise ValueError(f"Pas assez de labels connus pour '{attribute}' (n={len(known_idx)})")
    
    results = []
    
    for frac in frac_remove_list:
        n_remove = int(frac * len(known_idx))
        removed_idx = rng.choice(known_idx, size=n_remove, replace=False)
        
        # Créer labels observés (masquer les labels retirés)
        y_obs = y_full.copy()
        y_obs[removed_idx] = -1
        
        mask_labeled = (y_obs >= 0)
        
        # Prédire
        try:
            y_pred = label_propagation(G, y_obs, mask_labeled, device=device)
        except Exception as e:
            print(f"    ⚠ Échec pour f={frac}: {e}")
            continue
        
        # Évaluer
        y_true = y_full[removed_idx]
        y_hat = y_pred[removed_idx]
        
        acc = accuracy_score(y_true, y_hat)
        f1 = f1_score(y_true, y_hat, average='macro', zero_division=0)
        mae = float(np.mean(np.abs(y_true - y_hat)))
        
        results.append({
            "attribut": attribute,
            "fraction_retirée": frac,
            "n_test": len(removed_idx),
            "accuracy": acc,
            "f1_macro": f1,
            "mae": mae
        })
    
    return pd.DataFrame(results)


def plot_label_propagation_results(df: pd.DataFrame, output_dir: str = "report/figures"):
    """Visualise les résultats de label propagation"""
    
    if df.empty:
        print("Aucun résultat à visualiser")
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = ['accuracy', 'f1_macro', 'mae']
    titles = ['Accuracy', 'F1-Score (macro)', 'MAE']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        for attr in df['attribut'].unique():
            df_attr = df[df['attribut'] == attr].sort_values('fraction_retirée')
            axes[idx].plot(df_attr['fraction_retirée'], df_attr[metric],
                          marker='o', label=attr, linewidth=2, markersize=8)
        
        axes[idx].set_xlabel('Fraction de labels retirés', fontsize=11)
        axes[idx].set_ylabel(title, fontsize=11)
        axes[idx].set_title(title, fontsize=12, fontweight='bold')
        axes[idx].legend()
        axes[idx].grid(alpha=0.3)
    
    plt.tight_layout()
    
    filepath = output_path / "question5_label_propagation.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  {filepath.name}")
    plt.close()


def analyze_question5(G: nx.Graph, graph_name: str, output_dir: str = "report/figures"):
    """Analyse complète pour la Question 5"""
    
    print("\n" + "="*80)
    print("QUESTION 5: LABEL PROPAGATION")
    print("="*80)
    print(f"\nRéseau: {graph_name}")
    print(f"Nœuds: {G.number_of_nodes():,}, Arêtes: {G.number_of_edges():,}\n")
    
    # Attributs à évaluer
    attributes = ["dorm", "major_index", "gender"]
    
    all_results = []
    
    # Détecter si PyTorch peut utiliser CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")
    
    for attr in attributes:
        print(f"Évaluation: {attr}")
        
        try:
            df = evaluate_label_propagation(G, attr, 
                                           frac_remove_list=[0.1, 0.2, 0.3],
                                           seed=0, device=device)
            all_results.append(df)
            print(f"  Terminé")
            
        except Exception as e:
            print(f"  ✗ Échec: {e}")
    
    if not all_results:
        print("\n❌ Aucun résultat obtenu")
        return
    
    # Combiner les résultats
    df_all = pd.concat(all_results, ignore_index=True)
    
    # Afficher
    print("\n" + "-"*80)
    print("Résultats de Label Propagation:")
    print("-"*80)
    print(df_all.to_string(index=False))
    
    # Sauvegarder
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    csv_path = output_path / "question5_label_propagation_results.csv"
    df_all.to_csv(csv_path, index=False)
    print(f"\nRésultats sauvegardés: {csv_path}")
    
    # Visualiser
    print("\nGénération des graphiques...")
    plot_label_propagation_results(df_all, output_dir)
    
    print("\nAnalyse Question 5 terminée")
    
    return df_all


def analyze_question5_multiple(graphs: Dict[str, nx.Graph], output_dir: str = "report/figures"):
    """
    Analyse Label Propagation sur plusieurs graphes et agrège les résultats
    
    Args:
        graphs: Dictionnaire {nom_graphe: graphe_NetworkX}
        output_dir: Répertoire de sortie
    """
    print(f"\n{'='*80}")
    print(f"Analyse de {len(graphs)} graphes pour Label Propagation")
    print(f"{'='*80}\n")
    
    all_results = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")
    
    for i, (graph_name, G) in enumerate(graphs.items(), 1):
        print(f"\n[{i}/{len(graphs)}] Graphe: {graph_name}")
        print(f"  Nœuds: {G.number_of_nodes():,}, Arêtes: {G.number_of_edges():,}")
        
        attributes = ["dorm", "major_index", "gender"]
        
        for attr in attributes:
            print(f"  Attribut: {attr}...", end=" ")
            try:
                df = evaluate_label_propagation(G, attr, 
                                               frac_remove_list=[0.1, 0.2, 0.3],
                                               seed=0, device=device)
                df["network"] = graph_name
                all_results.append(df)
                print("✓")
            except Exception as e:
                print(f"✗ ({e})")
    
    if not all_results:
        print("\n❌ Aucun résultat obtenu")
        return
    
    # Combiner tous les résultats
    df_all = pd.concat(all_results, ignore_index=True)
    
    # Statistiques agrégées par attribut et fraction
    print("\n" + "="*80)
    print("Résultats agrégés (moyenne sur tous les graphes):")
    print("="*80)
    
    agg_stats = df_all.groupby(["attribut", "fraction_retirée"]).agg({
        "accuracy": ["mean", "std"],
        "f1_macro": ["mean", "std"],
        "mae": ["mean", "std"]
    }).round(4)
    
    print(agg_stats)
    
    # Sauvegarder les résultats détaillés et agrégés
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    csv_detailed = output_path / "question5_label_propagation_detailed.csv"
    df_all.to_csv(csv_detailed, index=False)
    print(f"\n✓ Résultats détaillés: {csv_detailed}")
    
    csv_aggregated = output_path / "question5_label_propagation_aggregated.csv"
    agg_stats.to_csv(csv_aggregated)
    print(f"✓ Résultats agrégés: {csv_aggregated}")
    
    # Visualisation agrégée
    print("\nGénération des graphiques...")
    plot_label_propagation_aggregated(df_all, output_dir)
    
    print("\n✓ Analyse Question 5 (multi-graphes) terminée")


def plot_label_propagation_aggregated(df: pd.DataFrame, output_dir: str):
    """
    Visualise les résultats agrégés de label propagation sur plusieurs graphes
    
    Args:
        df: DataFrame avec colonnes [network, attribut, fraction_retirée, accuracy, f1_macro, mae]
        output_dir: Répertoire de sortie
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Calculer les moyennes par attribut et fraction
    agg = df.groupby(["attribut", "fraction_retirée"]).agg({
        "accuracy": ["mean", "std"],
        "f1_macro": ["mean", "std"]
    }).reset_index()
    
    agg.columns = ["attribut", "fraction_retirée", "acc_mean", "acc_std", "f1_mean", "f1_std"]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    attributes = df["attribut"].unique()
    colors = plt.cm.Set2.colors
    
    # Accuracy
    ax = axes[0]
    for i, attr in enumerate(attributes):
        data = agg[agg["attribut"] == attr]
        ax.errorbar(data["fraction_retirée"], data["acc_mean"], 
                   yerr=data["acc_std"], 
                   marker='o', label=attr, color=colors[i], capsize=5)
    
    ax.set_xlabel("Fraction de labels retirés", fontsize=12)
    ax.set_ylabel("Accuracy (moyenne)", fontsize=12)
    ax.set_title("Performance moyenne sur tous les graphes", fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # F1-score
    ax = axes[1]
    for i, attr in enumerate(attributes):
        data = agg[agg["attribut"] == attr]
        ax.errorbar(data["fraction_retirée"], data["f1_mean"], 
                   yerr=data["f1_std"], 
                   marker='o', label=attr, color=colors[i], capsize=5)
    
    ax.set_xlabel("Fraction de labels retirés", fontsize=12)
    ax.set_ylabel("F1-score macro (moyenne)", fontsize=12)
    ax.set_title("Performance moyenne sur tous les graphes", fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    png_path = output_path / "question5_label_propagation_aggregated.png"
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Figure sauvegardée: {png_path}")
    plt.close()


def main():
    """Fonction principale"""
    DATA_DIR = "data"
    NUM_GRAPHS = 15  # Nombre de graphes à échantillonner
    SEED = 42
    
    print("="*80)
    print("QUESTION 5: LABEL PROPAGATION SUR RÉSEAUX FACEBOOK100")
    print(f"Analyse sur {NUM_GRAPHS} réseaux aléatoires")
    print("="*80 + "\n")
    
    try:
        # Charger les données avec cache
        all_graphs = load_graphs_with_cache(data_dir=DATA_DIR, verbose=True)
        
        if not all_graphs:
            print("❌ Aucun graphe chargé")
            return
        
        # Échantillonner NUM_GRAPHS graphes aléatoirement
        random.seed(SEED)
        available_names = list(all_graphs.keys())
        
        if len(available_names) < NUM_GRAPHS:
            print(f"⚠ Seulement {len(available_names)} graphes disponibles, analyse de tous")
            sampled_names = available_names
        else:
            sampled_names = random.sample(available_names, NUM_GRAPHS)
        
        sampled_graphs = {name: all_graphs[name] for name in sampled_names}
        
        print(f"\n✓ {len(sampled_graphs)} graphes sélectionnés aléatoirement:")
        for name in sampled_names:
            print(f"  - {name}")
        
        # Analyser tous les graphes sélectionnés
        analyze_question5_multiple(sampled_graphs, output_dir="report/figures")
        
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
