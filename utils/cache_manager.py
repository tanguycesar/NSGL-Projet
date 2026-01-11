"""
Module de cache pour les graphes Facebook100

Ce module permet de charger les graphes une seule fois et de les sauvegarder
dans un fichier pickle pour un accès rapide lors des exécutions suivantes.
"""

import pickle
from pathlib import Path
from typing import Dict
import networkx as nx
from load_data import Facebook100Loader


CACHE_FILE = Path("data") / "graphs_cache.pkl"


def load_graphs_with_cache(data_dir: str = "data", 
                           force_reload: bool = False,
                           verbose: bool = True) -> Dict[str, nx.Graph]:
    """
    Charge les graphes avec système de cache
    
    Args:
        data_dir: Répertoire contenant les fichiers .gml
        force_reload: Si True, force le rechargement depuis les .gml
        verbose: Si True, affiche les informations
        
    Returns:
        Dictionnaire {nom_réseau: graphe}
    """
    cache_path = Path(data_dir) / "graphs_cache.pkl"
    
    # Vérifier si le cache existe et n'est pas forcé à recharger
    if cache_path.exists() and not force_reload:
        if verbose:
            print(f"Chargement depuis le cache: {cache_path}")
            print("   (pour forcer le rechargement: force_reload=True)")
        
        try:
            with open(cache_path, 'rb') as f:
                graphs = pickle.load(f)
            
            if verbose:
                print(f"✓ {len(graphs)} réseaux chargés depuis le cache")
                # Afficher un aperçu
                for name, G in list(graphs.items())[:3]:
                    print(f"  - {name}: {G.number_of_nodes():,} nœuds, {G.number_of_edges():,} arêtes")
                if len(graphs) > 3:
                    print(f"  ... et {len(graphs)-3} autres réseaux")
            
            return graphs
            
        except Exception as e:
            if verbose:
                print(f"⚠ Erreur lors du chargement du cache: {e}")
                print("  → Rechargement depuis les fichiers .gml")
    
    # Charger depuis les fichiers .gml
    if verbose:
        print(f"Chargement depuis les fichiers .gml dans {data_dir}/")
    
    loader = Facebook100Loader(data_dir=data_dir)
    graphs = loader.load_all_graphs(extract_lcc=True, verbose=verbose)
    
    # Sauvegarder le cache
    try:
        cache_path.parent.mkdir(exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(graphs, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        if verbose:
            size_mb = cache_path.stat().st_size / (1024 * 1024)
            print(f"\n✓ Cache sauvegardé: {cache_path} ({size_mb:.1f} MB)")
            print(f"  Les prochaines exécutions seront beaucoup plus rapides !")
    
    except Exception as e:
        if verbose:
            print(f"⚠ Impossible de sauvegarder le cache: {e}")
    
    return graphs


def clear_cache(data_dir: str = "data", verbose: bool = True):
    """Supprime le fichier cache"""
    cache_path = Path(data_dir) / "graphs_cache.pkl"
    
    if cache_path.exists():
        cache_path.unlink()
        if verbose:
            print(f"✓ Cache supprimé: {cache_path}")
    else:
        if verbose:
            print(f"Aucun cache à supprimer")


def main():
    """Test du système de cache"""
    import time
    
    print("="*80)
    print("TEST DU SYSTÈME DE CACHE")
    print("="*80 + "\n")
    
    # Premier chargement (depuis .gml)
    print("Premier chargement (depuis .gml)...")
    start = time.time()
    graphs1 = load_graphs_with_cache(force_reload=True)
    time1 = time.time() - start
    print(f"\nTemps: {time1:.2f}s")
    
    # Deuxième chargement (depuis cache)
    print("\n" + "-"*80)
    print("Deuxième chargement (depuis cache)...")
    start = time.time()
    graphs2 = load_graphs_with_cache()
    time2 = time.time() - start
    print(f"\nTemps: {time2:.2f}s")
    
    # Comparaison
    print("\n" + "="*80)
    print("RÉSULTATS")
    print("="*80)
    print(f"Temps depuis .gml : {time1:.2f}s")
    print(f"Temps depuis cache: {time2:.2f}s")
    print(f"Accélération: {time1/time2:.1f}x plus rapide")
    
    # Vérifier que les graphes sont identiques
    assert len(graphs1) == len(graphs2), "Nombre de graphes différent"
    for name in graphs1:
        assert name in graphs2, f"Graphe {name} manquant"
        assert graphs1[name].number_of_nodes() == graphs2[name].number_of_nodes()
        assert graphs1[name].number_of_edges() == graphs2[name].number_of_edges()
    
    print(f"\n✓ Vérification: Les données sont identiques")


if __name__ == "__main__":
    main()
