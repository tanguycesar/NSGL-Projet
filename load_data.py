"""
Module de chargement des graphes Facebook100 (.gml)

Ce module charge les fichiers .gml, gère les attributs spécifiques,
et extrait la plus grande composante connexe (LCC).

Structure des attributs dans les fichiers .gml:
- student_fac: statut étudiant/faculté (pas 'status')
- gender: genre
- major_index: discipline principale (pas 'major')
- second_major: majeure secondaire
- dorm: résidence
- year: année d'études
- high_school: lycée d'origine

Valeurs manquantes typiques: 0, -1, None, ''
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import networkx as nx
import numpy as np


class Facebook100Loader:
    """Classe pour charger et prétraiter les graphes Facebook100"""
    
    # Mapping des noms d'attributs (ancien -> nouveau)
    ATTRIBUTE_MAPPING = {
        'status': 'student_fac',
        'major': 'major_index'
    }
    
    # Valeurs considérées comme manquantes
    MISSING_VALUES = {0, -1, None, '', '0', '-1'}
    
    def __init__(self, data_dir: str = "data"):
        """
        Args:
            data_dir: Répertoire contenant les fichiers .gml
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Le dossier {data_dir} n'existe pas")
    
    def list_graph_files(self) -> List[Path]:
        """Liste tous les fichiers .gml dans le répertoire de données"""
        files = sorted(self.data_dir.glob("*.gml"))
        if not files:
            raise FileNotFoundError(f"Aucun fichier .gml trouvé dans {self.data_dir}")
        return files
    
    def load_graph(self, filepath: Path) -> nx.Graph:
        """
        Charge un graphe depuis un fichier .gml
        
        Args:
            filepath: Chemin vers le fichier .gml
            
        Returns:
            Graphe NetworkX simple non orienté
        """
        # Charger le graphe avec label='id' pour utiliser les IDs du fichier
        G = nx.read_gml(filepath, label='id')
        
        # Convertir en graphe simple non orienté
        if not isinstance(G, nx.Graph):
            G = nx.Graph(G)
        
        # Convertir les labels en entiers séquentiels
        G = nx.convert_node_labels_to_integers(G, ordering="sorted")
        
        # Supprimer les boucles
        G.remove_edges_from(nx.selfloop_edges(G))
        
        # Normaliser les attributs
        G = self._normalize_attributes(G)
        
        return G
    
    def _normalize_attributes(self, G: nx.Graph) -> nx.Graph:
        """
        Normalise les noms d'attributs et convertit les valeurs
        
        Args:
            G: Graphe NetworkX
            
        Returns:
            Graphe avec attributs normalisés
        """
        # Créer un nouveau dictionnaire d'attributs
        for node in G.nodes():
            attrs = G.nodes[node]
            new_attrs = {}
            
            for key, value in attrs.items():
                # Appliquer le mapping de noms
                new_key = self.ATTRIBUTE_MAPPING.get(key, key)
                
                # Convertir en type approprié
                if value in self.MISSING_VALUES:
                    new_value = -1  # Valeur standardisée pour "manquant"
                else:
                    try:
                        new_value = int(value)
                    except (ValueError, TypeError):
                        new_value = -1
                
                new_attrs[new_key] = new_value
            
            # Mettre à jour les attributs du nœud
            nx.set_node_attributes(G, {node: new_attrs})
        
        return G
    
    def extract_lcc(self, G: nx.Graph) -> nx.Graph:
        """
        Extrait la plus grande composante connexe (LCC)
        
        Args:
            G: Graphe NetworkX
            
        Returns:
            Sous-graphe correspondant à la LCC
        """
        if G.number_of_nodes() == 0:
            return G.copy()
        
        if nx.is_connected(G):
            return G.copy()
        
        # Trouver la plus grande composante connexe
        largest_cc = max(nx.connected_components(G), key=len)
        lcc = G.subgraph(largest_cc).copy()
        
        # Réindexer les nœuds de 0 à n-1
        lcc = nx.convert_node_labels_to_integers(lcc, ordering="sorted")
        
        return lcc
    
    def load_all_graphs(self, extract_lcc: bool = True, verbose: bool = True) -> Dict[str, nx.Graph]:
        """
        Charge tous les graphes du répertoire
        
        Args:
            extract_lcc: Si True, extrait la LCC de chaque graphe
            verbose: Si True, affiche les informations de chargement
            
        Returns:
            Dictionnaire {nom_réseau: graphe}
        """
        files = self.list_graph_files()
        graphs = {}
        
        if verbose:
            print(f"Chargement de {len(files)} fichiers .gml depuis {self.data_dir}")
            print(f"{'Fichier':<30} {'Nœuds':>8} {'Arêtes':>10} {'LCC':>8}")
            print("-" * 60)
        
        for filepath in files:
            try:
                G = self.load_graph(filepath)
                
                if extract_lcc:
                    original_nodes = G.number_of_nodes()
                    G = self.extract_lcc(G)
                    lcc_nodes = G.number_of_nodes()
                else:
                    lcc_nodes = G.number_of_nodes()
                
                graphs[filepath.stem] = G
                
                if verbose:
                    lcc_pct = f"{lcc_nodes}" if not extract_lcc else f"{lcc_nodes} ({100*lcc_nodes/original_nodes:.1f}%)" if original_nodes > 0 else f"{lcc_nodes}"
                    print(f"{filepath.name:<30} {G.number_of_nodes():>8,} {G.number_of_edges():>10,} {lcc_pct:>8}")
                    
            except Exception as e:
                if verbose:
                    print(f"[ERREUR] {filepath.name}: {e}")
        
        if verbose:
            print(f"\n{len(graphs)} réseaux chargés avec succès")
        
        return graphs
    
    def get_attribute_statistics(self, G: nx.Graph, attribute: str) -> Dict:
        """
        Calcule des statistiques sur un attribut
        
        Args:
            G: Graphe NetworkX
            attribute: Nom de l'attribut
            
        Returns:
            Dictionnaire avec statistiques
        """
        attrs = nx.get_node_attributes(G, attribute)
        
        # Filtrer les valeurs manquantes
        valid_values = [v for v in attrs.values() if v not in self.MISSING_VALUES and v != -1]
        
        if not valid_values:
            return {
                "attribute": attribute,
                "total_nodes": G.number_of_nodes(),
                "nodes_with_value": 0,
                "coverage": 0.0,
                "unique_values": 0,
                "min": None,
                "max": None,
                "mean": None
            }
        
        return {
            "attribute": attribute,
            "total_nodes": G.number_of_nodes(),
            "nodes_with_value": len(valid_values),
            "coverage": len(valid_values) / G.number_of_nodes(),
            "unique_values": len(set(valid_values)),
            "min": min(valid_values),
            "max": max(valid_values),
            "mean": np.mean(valid_values)
        }
    
    def print_graph_summary(self, name: str, G: nx.Graph):
        """Affiche un résumé d'un graphe"""
        print(f"\n{'='*60}")
        print(f"Réseau: {name}")
        print(f"{'='*60}")
        print(f"Nœuds: {G.number_of_nodes():,}")
        print(f"Arêtes: {G.number_of_edges():,}")
        print(f"Densité: {nx.density(G):.6f}")
        print(f"Connexe: {nx.is_connected(G)}")
        
        # Statistiques sur les attributs
        print(f"\nAttributs disponibles:")
        if G.number_of_nodes() > 0:
            first_node = list(G.nodes())[0]
            attrs = G.nodes[first_node]
            for attr in sorted(attrs.keys()):
                stats = self.get_attribute_statistics(G, attr)
                print(f"  - {attr:<15} : {stats['nodes_with_value']:>6} nœuds ({stats['coverage']*100:>5.1f}%), "
                      f"{stats['unique_values']:>4} valeurs uniques")


def main():
    """Fonction de test"""
    # Charger tous les graphes
    loader = Facebook100Loader(data_dir="data")
    
    try:
        graphs = loader.load_all_graphs(extract_lcc=True, verbose=True)
        
        # Afficher un exemple de graphe
        if graphs:
            first_name = next(iter(graphs))
            loader.print_graph_summary(first_name, graphs[first_name])
            
    except FileNotFoundError as e:
        print(f"Erreur: {e}")
        print("\nAssurez-vous que le dossier 'data/' existe et contient des fichiers .gml")


if __name__ == "__main__":
    main()
