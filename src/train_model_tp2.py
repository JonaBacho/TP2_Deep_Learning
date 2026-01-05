import mlflow
import mlflow.tensorflow
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import json
import os
from config.mlflow_config import MLflowConfig

# Configuration MLflow
mlflow_client = MLflowConfig.setup_mlflow()

def load_and_split_data():
    """Charge et divise les données MNIST en train/val/test"""
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Utiliser 90% pour l'entraînement et 10% pour la validation
    x_val = x_train[54000:]
    y_val = y_train[54000:]
    x_train = x_train[:54000]
    y_train = y_train[:54000]
    
    # Normalisation et reshape
    x_train = x_train.astype("float32") / 255.0
    x_val = x_val.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    
    x_train = x_train.reshape(54000, 784)
    x_val = x_val.reshape(6000, 784)
    x_test = x_test.reshape(10000, 784)
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

# ==================== EXERCISE 1: Bias/Variance Analysis ====================
def exercise_1_baseline():
    """Exercise 1: Modèle baseline sans régularisation"""
    mlflow.set_experiment("TP2-Exercise1-BiasVariance")
    
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_and_split_data()
    
    with mlflow.start_run(run_name="Ex1_Baseline_NoRegularization"):
        mlflow.log_param("exercise", "1")
        mlflow.log_param("model_type", "baseline")
        mlflow.log_param("regularization", "none")
        mlflow.log_param("epochs", 5)
        mlflow.log_param("batch_size", 128)
        
        # Modèle simple sans régularisation
        model = keras.Sequential([
            keras.layers.Dense(512, activation='relu', input_shape=(784,)),
            keras.layers.Dense(10, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        history = model.fit(
            x_train, y_train,
            epochs=5,
            batch_size=128,
            validation_data=(x_val, y_val),
            verbose=1
        )
        
        # Log des métriques par epoch
        for epoch in range(len(history.history['loss'])):
            mlflow.log_metric("train_loss", history.history['loss'][epoch], step=epoch)
            mlflow.log_metric("train_accuracy", history.history['accuracy'][epoch], step=epoch)
            mlflow.log_metric("val_loss", history.history['val_loss'][epoch], step=epoch)
            mlflow.log_metric("val_accuracy", history.history['val_accuracy'][epoch], step=epoch)
        
        # Évaluation finale
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", test_acc)
        
        # Diagnostic Bias/Variance
        final_train_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]
        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        
        gap_loss = abs(final_val_loss - final_train_loss)
        gap_acc = abs(final_train_acc - final_val_acc)
        
        mlflow.log_metric("loss_gap", gap_loss)
        mlflow.log_metric("accuracy_gap", gap_acc)
        
        # Diagnostic textuel
        diagnosis = ""
        if final_train_acc < 0.90:
            diagnosis = "HIGH_BIAS (underfitting)"
        elif gap_acc > 0.05:
            diagnosis = "HIGH_VARIANCE (overfitting)"
        else:
            diagnosis = "GOOD_FIT"
        
        mlflow.set_tag("diagnosis", diagnosis)
        
        print(f"\n{'='*60}")
        print(f"EXERCISE 1: Bias/Variance Analysis")
        print(f"{'='*60}")
        print(f"Train Accuracy: {final_train_acc:.4f}")
        print(f"Val Accuracy: {final_val_acc:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Accuracy Gap: {gap_acc:.4f}")
        print(f"DIAGNOSIS: {diagnosis}")
        print(f"{'='*60}\n")

# ==================== EXERCISE 2: Regularization ====================
def exercise_2_regularization():
    """Exercise 2: Application de la régularisation L2 et Dropout"""
    mlflow.set_experiment("TP2-Exercise2-Regularization")
    
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_and_split_data()
    
    with mlflow.start_run(run_name="Ex2_With_L2_Dropout"):
        mlflow.log_param("exercise", "2")
        mlflow.log_param("model_type", "regularized")
        mlflow.log_param("l2_regularization", 0.001)
        mlflow.log_param("dropout_rate", 0.2)
        mlflow.log_param("epochs", 5)
        mlflow.log_param("batch_size", 128)
        
        # Modèle avec L2 et Dropout
        model = keras.Sequential([
            keras.layers.Dense(512, activation='relu', input_shape=(784,),
                             kernel_regularizer=keras.regularizers.l2(0.001)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(10, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        history = model.fit(
            x_train, y_train,
            epochs=5,
            batch_size=128,
            validation_data=(x_val, y_val),
            verbose=1
        )
        
        # Log des métriques
        for epoch in range(len(history.history['loss'])):
            mlflow.log_metric("train_loss", history.history['loss'][epoch], step=epoch)
            mlflow.log_metric("train_accuracy", history.history['accuracy'][epoch], step=epoch)
            mlflow.log_metric("val_loss", history.history['val_loss'][epoch], step=epoch)
            mlflow.log_metric("val_accuracy", history.history['val_accuracy'][epoch], step=epoch)
        
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", test_acc)
        
        # Comparaison
        gap_acc = abs(history.history['accuracy'][-1] - history.history['val_accuracy'][-1])
        mlflow.log_metric("accuracy_gap", gap_acc)
        
        print(f"\n{'='*60}")
        print(f"EXERCISE 2: Regularization (L2 + Dropout)")
        print(f"{'='*60}")
        print(f"Train Accuracy: {history.history['accuracy'][-1]:.4f}")
        print(f"Val Accuracy: {history.history['val_accuracy'][-1]:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Accuracy Gap: {gap_acc:.4f}")
        print(f"{'='*60}\n")

# ==================== EXERCISE 3: Optimizer Comparison ====================
def exercise_3_optimizers():
    """Exercise 3: Comparaison des optimiseurs"""
    mlflow.set_experiment("TP2-Exercise3-Optimizers")
    
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_and_split_data()
    
    optimizers = {
        'SGD_with_momentum': keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
        'RMSprop': 'rmsprop',
        'Adam': 'adam'
    }
    
    results = {}
    
    for opt_name, optimizer in optimizers.items():
        with mlflow.start_run(run_name=f"Ex3_Optimizer_{opt_name}"):
            mlflow.log_param("exercise", "3")
            mlflow.log_param("optimizer", opt_name)
            mlflow.log_param("epochs", 5)
            mlflow.log_param("batch_size", 128)
            
            # Modèle identique pour tous les optimiseurs
            model = keras.Sequential([
                keras.layers.Dense(512, activation='relu', input_shape=(784,)),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(10, activation='softmax')
            ])
            
            model.compile(
                optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            history = model.fit(
                x_train, y_train,
                epochs=5,
                batch_size=128,
                validation_data=(x_val, y_val),
                verbose=1
            )
            
            # Log des métriques
            for epoch in range(len(history.history['loss'])):
                mlflow.log_metric("train_loss", history.history['loss'][epoch], step=epoch)
                mlflow.log_metric("train_accuracy", history.history['accuracy'][epoch], step=epoch)
                mlflow.log_metric("val_loss", history.history['val_loss'][epoch], step=epoch)
                mlflow.log_metric("val_accuracy", history.history['val_accuracy'][epoch], step=epoch)
            
            test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
            mlflow.log_metric("final_test_loss", test_loss)
            mlflow.log_metric("final_test_accuracy", test_acc)
            
            results[opt_name] = {
                'test_accuracy': test_acc,
                'test_loss': test_loss,
                'convergence_speed': len(history.history['loss'])
            }
            
            print(f"\n{opt_name}: Test Accuracy = {test_acc:.4f}")
    
    print(f"\n{'='*60}")
    print(f"EXERCISE 3: Optimizer Comparison Summary")
    print(f"{'='*60}")
    for opt_name, metrics in results.items():
        print(f"{opt_name:20s}: Accuracy={metrics['test_accuracy']:.4f}, Loss={metrics['test_loss']:.4f}")
    print(f"{'='*60}\n")

# ==================== EXERCISE 4: Batch Normalization ====================
def exercise_4_batch_norm():
    """Exercise 4: Ajout de Batch Normalization"""
    mlflow.set_experiment("TP2-Exercise4-BatchNorm")
    
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_and_split_data()
    
    # Modèle SANS Batch Normalization
    with mlflow.start_run(run_name="Ex4_Without_BatchNorm"):
        mlflow.log_param("exercise", "4")
        mlflow.log_param("batch_normalization", False)
        mlflow.log_param("epochs", 5)
        
        model = keras.Sequential([
            keras.layers.Dense(512, activation='relu', input_shape=(784,)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(10, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        history_no_bn = model.fit(
            x_train, y_train,
            epochs=5,
            batch_size=128,
            validation_data=(x_val, y_val),
            verbose=1
        )
        
        for epoch in range(len(history_no_bn.history['loss'])):
            mlflow.log_metric("train_loss", history_no_bn.history['loss'][epoch], step=epoch)
            mlflow.log_metric("val_loss", history_no_bn.history['val_loss'][epoch], step=epoch)
            mlflow.log_metric("val_accuracy", history_no_bn.history['val_accuracy'][epoch], step=epoch)
        
        test_loss_no_bn, test_acc_no_bn = model.evaluate(x_test, y_test, verbose=0)
        mlflow.log_metric("final_test_accuracy", test_acc_no_bn)
    
    # Modèle AVEC Batch Normalization
    with mlflow.start_run(run_name="Ex4_With_BatchNorm"):
        mlflow.log_param("exercise", "4")
        mlflow.log_param("batch_normalization", True)
        mlflow.log_param("epochs", 5)
        
        model = keras.Sequential([
            keras.layers.Dense(512, activation='relu', input_shape=(784,)),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(10, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        history_bn = model.fit(
            x_train, y_train,
            epochs=5,
            batch_size=128,
            validation_data=(x_val, y_val),
            verbose=1
        )
        
        for epoch in range(len(history_bn.history['loss'])):
            mlflow.log_metric("train_loss", history_bn.history['loss'][epoch], step=epoch)
            mlflow.log_metric("val_loss", history_bn.history['val_loss'][epoch], step=epoch)
            mlflow.log_metric("val_accuracy", history_bn.history['val_accuracy'][epoch], step=epoch)
        
        test_loss_bn, test_acc_bn = model.evaluate(x_test, y_test, verbose=0)
        mlflow.log_metric("final_test_accuracy", test_acc_bn)
    
    print(f"\n{'='*60}")
    print(f"EXERCISE 4: Batch Normalization Comparison")
    print(f"{'='*60}")
    print(f"Without BatchNorm: Test Accuracy = {test_acc_no_bn:.4f}")
    print(f"With BatchNorm:    Test Accuracy = {test_acc_bn:.4f}")
    print(f"Improvement: {(test_acc_bn - test_acc_no_bn):.4f}")
    print(f"{'='*60}\n")

# ==================== MAIN ====================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("TP2: Improving Deep Neural Networks")
    print("="*60 + "\n")
    
    # Exécuter tous les exercices
    print("Executing Exercise 1: Bias/Variance Analysis...")
    exercise_1_baseline()
    
    print("\nExecuting Exercise 2: Regularization...")
    exercise_2_regularization()
    
    print("\nExecuting Exercise 3: Optimizer Comparison...")
    exercise_3_optimizers()
    
    print("\nExecuting Exercise 4: Batch Normalization...")
    exercise_4_batch_norm()
    
    print("\n" + "="*60)
    print("✓ All TP2 exercises completed successfully!")
    print("Check MLflow UI for detailed results")
    print("="*60 + "\n")