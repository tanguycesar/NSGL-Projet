"""
Script principal pour exécuter toutes les analyses

Usage:
    python run_analysis.py [question_number]
    
    Sans argument: exécute toutes les questions (1-6)
    Avec argument: exécute uniquement la question spécifiée
    
Exemples:
    python run_analysis.py         # Toutes les questions
    python run_analysis.py 1       # Question 1 uniquement
    python run_analysis.py 2       # Question 2 uniquement
"""

import sys
import time
from pathlib import Path
from datetime import datetime

# Importer les modules des questions
import question1_stats
import question2_analysis
import question3_assortativity
import question4_link_prediction
import question5_label_propagation
import question6_communities


def print_header(text: str):
    """Affiche un en-tête formaté"""
    width = 80
    print("\n" + "="*width)
    print(f" {text}")
    print("="*width + "\n")


def print_section(text: str):
    """Affiche une section formatée"""
    print("\n" + "-"*80)
    print(f" {text}")
    print("-"*80 + "\n")


def run_question(question_num: int):
    """
    Exécute une question spécifique
    
    Args:
        question_num: Numéro de la question (1-6)
    """
    start_time = time.time()
    
    try:
        if question_num == 1:
            print_header("QUESTION 1: STATISTIQUES GLOBALES")
            question1_stats.main()
            
        elif question_num == 2:
            print_header("QUESTION 2: ANALYSE DES RÉSEAUX SOCIAUX")
            question2_analysis.main()
            
        elif question_num == 3:
            print_header("QUESTION 3: ASSORTATIVITÉ")
            question3_assortativity.main()
            
        elif question_num == 4:
            print_header("QUESTION 4: LINK PREDICTION")
            question4_link_prediction.main()
            
        elif question_num == 5:
            print_header("QUESTION 5: LABEL PROPAGATION")
            question5_label_propagation.main()
            
        elif question_num == 6:
            print_header("QUESTION 6: DÉTECTION DE COMMUNAUTÉS")
            question6_communities.main()
            
        else:
            print(f"❌ Question {question_num} n'existe pas (1-6 uniquement)")
            return False
        
        elapsed = time.time() - start_time
        print(f"\n✓ Question {question_num} terminée en {elapsed:.1f}s")
        return True
        
    except Exception as e:
        print(f"\n❌ Erreur lors de l'exécution de la question {question_num}:")
        print(f"   {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_questions():
    """Exécute toutes les questions séquentiellement"""
    
    print_header("ANALYSE COMPLÈTE DES RÉSEAUX FACEBOOK100")
    print("Ce script va exécuter toutes les analyses (Questions 1-6)")
    print("Temps estimé: 10-30 minutes selon la taille des données\n")
    
    start_time = time.time()
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Créer un fichier log
    log_file = results_dir / f"analysis_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    # Rediriger stdout vers le fichier log ET la console
    class DualOutput:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, 'w', encoding='utf-8')
        
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
        
        def flush(self):
            self.terminal.flush()
            self.log.flush()
    
    sys.stdout = DualOutput(log_file)
    
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Log sauvegardé: {log_file}\n")
    
    # Exécuter toutes les questions
    questions = [1, 2, 3, 4, 5, 6]
    success_count = 0
    
    for q in questions:
        success = run_question(q)
        if success:
            success_count += 1
        else:
            print(f"\n⚠ Poursuite de l'analyse malgré l'échec de la question {q}...\n")
    
    # Résumé
    total_time = time.time() - start_time
    
    print_header("RÉSUMÉ DE L'ANALYSE")
    print(f"Questions réussies: {success_count}/{len(questions)}")
    print(f"Temps total: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"\nTous les résultats sont dans le dossier: {results_dir.absolute()}")
    print(f"Log complet: {log_file}")
    
    # Liste des fichiers générés
    print("\nFichiers générés:")
    for file in sorted(results_dir.glob("*")):
        if file.is_file() and file != log_file:
            size_kb = file.stat().st_size / 1024
            print(f"  - {file.name} ({size_kb:.1f} KB)")
    
    # Restaurer stdout
    sys.stdout.log.close()
    sys.stdout = sys.stdout.terminal


def print_usage():
    """Affiche l'aide d'utilisation"""
    print(__doc__)


def main():
    """Fonction principale"""
    
    # Traiter les arguments
    if len(sys.argv) > 2:
        print("Trop d'arguments")
        print_usage()
        sys.exit(1)
    
    # Vérifier que le dossier data/ existe
    data_dir = Path("data")
    if not data_dir.exists():
        print("❌ Erreur: Le dossier 'data/' n'existe pas")
        print("\nAssurez-vous que:")
        print("  1. Le dossier 'data/' est présent dans le répertoire courant")
        print("  2. Les fichiers .gml Facebook100 sont dans ce dossier")
        sys.exit(1)
    
    # Compter les fichiers .gml
    gml_files = list(data_dir.glob("*.gml"))
    if len(gml_files) == 0:
        print("⚠ Attention: Aucun fichier .gml trouvé dans le dossier 'data/'")
        print("Les analyses risquent d'échouer.\n")
    else:
        print(f"✓ {len(gml_files)} fichiers .gml trouvés dans 'data/'\n")
    
    # Exécuter
    if len(sys.argv) == 1:
        # Pas d'argument: toutes les questions
        run_all_questions()
    else:
        # Argument: question spécifique
        try:
            question_num = int(sys.argv[1])
            if question_num < 1 or question_num > 6:
                print(f"❌ Numéro de question invalide: {question_num}")
                print("   Valeurs acceptées: 1-6")
                sys.exit(1)
            
            run_question(question_num)
            
        except ValueError:
            print(f"❌ Argument invalide: '{sys.argv[1]}'")
            print("   Utilisez un nombre entre 1 et 6")
            print_usage()
            sys.exit(1)


if __name__ == "__main__":
    main()
