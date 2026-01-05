#!/usr/bin/env python3
"""
Script principal pour exécuter tous les exercices du TP2
"""
import sys
import os

# Ajouter le répertoire src au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from train_model_tp2 import (
    exercise_1_baseline,
    exercise_2_regularization,
    exercise_3_optimizers,
    exercise_4_batch_norm
)

def main():
    """Exécute tous les exercices du TP2"""
    print("\n" + "="*70)
    print(" TP2: Improving Deep Neural Networks - ENSPY 2025")
    print("="*70 + "\n")
    
    print("Ce script va exécuter tous les exercices du TP2:")
    print("  - Exercise 1: Bias/Variance Analysis")
    print("  - Exercise 2: Regularization (L2 + Dropout)")
    print("  - Exercise 3: Optimizer Comparison (SGD, RMSprop, Adam)")
    print("  - Exercise 4: Batch Normalization")
    print("\nLes résultats seront trackés dans MLflow.\n")
    
    response = input("Continuer? (y/n): ")
    if response.lower() != 'y':
        print("Exécution annulée.")
        return
    
    try:
        # Exercise 1
        print("\n" + "-"*70)
        print("Executing Exercise 1: Bias/Variance Analysis")
        print("-"*70)
        exercise_1_baseline()
        
        # Exercise 2
        print("\n" + "-"*70)
        print("Executing Exercise 2: Regularization")
        print("-"*70)
        exercise_2_regularization()
        
        # Exercise 3
        print("\n" + "-"*70)
        print("Executing Exercise 3: Optimizer Comparison")
        print("-"*70)
        exercise_3_optimizers()
        
        # Exercise 4
        print("\n" + "-"*70)
        print("Executing Exercise 4: Batch Normalization")
        print("-"*70)
        exercise_4_batch_norm()
        
        print("\n" + "="*70)
        print("✓ Tous les exercices du TP2 sont terminés avec succès!")
        print("="*70)
        print("\nConsultez l'interface web MLflow pour voir les résultats détaillés:")
        print("  - Exercise 1: Expérience 'TP2-Exercise1-BiasVariance'")
        print("  - Exercise 2: Expérience 'TP2-Exercise2-Regularization'")
        print("  - Exercise 3: Expérience 'TP2-Exercise3-Optimizers'")
        print("  - Exercise 4: Expérience 'TP2-Exercise4-BatchNorm'")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Erreur lors de l'exécution: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()