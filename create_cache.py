"""
Script pour créer le cache des graphes

Exécuter ce script UNE SEULE FOIS au début pour créer le cache.
Toutes les autres analyses utiliseront ensuite ce cache et seront beaucoup plus rapides.
"""

from cache_manager import load_graphs_with_cache, clear_cache
from pathlib import Path
import os

def main():
    print("="*80)
    print("CRÉATION DU CACHE DES GRAPHES FACEBOOK100")
    print("="*80 + "\n")
    
    print("Ce script va charger tous les fichiers .gml et créer un cache.")
    print("Cela peut prendre quelques minutes la première fois.")
    print("Les prochaines exécutions seront instantanées !\n")
    
    # Se placer dans le répertoire du script
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    print(f"Répertoire de travail: {script_dir}\n")
    
    data_dir = Path("data")
    
    # Vérifier que le dossier data existe
    if not data_dir.exists():
        print(f"❌ Erreur: Le dossier '{data_dir}' n'existe pas")
        print("\nCréation du dossier...")
        data_dir.mkdir(exist_ok=True)
        print("✓ Dossier créé")
        print("\nPlacez vos fichiers .gml dans ce dossier, puis relancez ce script.")
        return
    
    # Vérifier qu'il y a des fichiers .gml
    gml_files = list(data_dir.glob("*.gml"))
    if not gml_files:
        print(f"❌ Erreur: Aucun fichier .gml trouvé dans '{data_dir}'")
        print("\nPlacez vos fichiers .gml dans ce dossier, puis relancez ce script.")
        return
    
    print(f"✓ {len(gml_files)} fichiers .gml trouvés\n")
    
    # Demander si on veut recréer le cache
    cache_file = data_dir / "graphs_cache.pkl"
    if cache_file.exists():
        print("⚠ Un cache existe déjà.")
        response = input("Voulez-vous le recréer ? (o/n): ").strip().lower()
        if response == 'o':
            clear_cache(verbose=True)
            print()
        else:
            print("✓ Conservation du cache existant")
            return
    
    # Créer le cache
    print("Chargement et création du cache...\n")
    graphs = load_graphs_with_cache(force_reload=True, verbose=True)
    
    if graphs:
        print("\n" + "="*80)
        print("✅ CACHE CRÉÉ AVEC SUCCÈS !")
        print("="*80)
        print(f"\nVous pouvez maintenant exécuter les analyses :")
        print("  python run_analysis.py      # Toutes les questions")
        print("  python question1_stats.py   # Question spécifique")
        print("\nLes chargements seront maintenant instantanés !")
    else:
        print("\n❌ Erreur lors de la création du cache")


if __name__ == "__main__":
    main()
