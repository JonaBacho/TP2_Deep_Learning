import mlflow
import mlflow.tensorflow
import tensorflow as tf
import os
from tensorflow import keras
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import json
from config.mlflow_config import MLflowConfig

# Configuration MLflow
mlflow_client = MLflowConfig.setup_mlflow()

# Définir l'expérience
mlflow.set_experiment("mnist-improved-training")

# Hyperparamètres
EPOCHS = int(os.getenv("EPOCHS", "10"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "128"))
DROPOUT_RATE = float(os.getenv("DROPOUT_RATE", "0.2"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.001"))
L2_FACTOR = float(os.getenv("L2_FACTOR", "0.001"))

def load_and_preprocess_data():
    """Charge et prétraite les données MNIST avec split explicite"""
    (x_train_full, y_train_full), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Normalisation
    x_train_full = x_train_full.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Reshape
    x_train_full = x_train_full.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)

    # Split explicite
    x_train = x_train_full[:54000]
    y_train = y_train_full[:54000]

    x_val = x_train_full[54000:]
    y_val = y_train_full[54000:]

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def build_model(dropout_rate=0.2, learning_rate=0.001, l2_factor=0.001):
    """Construit et compile le modèle"""
    model = keras.Sequential([
        keras.layers.Dense(512, activation='relu', 
            kernel_regularizer=keras.regularizers.l2(l2_factor), 
            input_shape=(784,)),
            keras.layers.BatchNormalization(),
        keras.layers.Dropout(dropout_rate),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(dropout_rate),
        keras.layers.Dense(10, activation='softmax')
    ])
    
    """ L'Optimisation sera utilisé dans la boucle de l'entrainement pour comparaison
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    """
    return model

def build_model_batchnorm(dropout_rate=0.2):
    model = keras.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(784,)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(dropout_rate),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def evaluate_model_detailed(model, x_test, y_test):
    """Évaluation détaillée du modèle"""
    # Prédictions
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Métriques globales
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    
    # Rapport de classification
    report = classification_report(y_test, y_pred_classes, output_dict=True)
    
    # Matrice de confusion
    conf_matrix = confusion_matrix(y_test, y_pred_classes)
    
    return {
        'test_loss': float(test_loss),
        'test_accuracy': float(test_acc),
        'precision': float(report['weighted avg']['precision']),
        'recall': float(report['weighted avg']['recall']),
        'f1_score': float(report['weighted avg']['f1-score']),
        'classification_report': report,
        'confusion_matrix': conf_matrix.tolist()
    }

def train_and_log_model():
    """Entraîne le modèle et log tout dans MLflow"""
    
    # Charger les données
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_and_preprocess_data()

    # optimizers
    optimizers = {
        "SGD_with_momentum": keras.optimizers.SGD(
            learning_rate=0.01,
            momentum=0.9
        ),
        "RMSprop": keras.optimizers.RMSprop(),
        "Adam": keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    }

    results = []

    for opt_name, optimizer in optimizers.items():
    
        # Démarrer une run MLflow
        with mlflow.start_run(run_name=f"mnist-improved-train-{opt_name}-{EPOCHS}epochs") as run:
            
            # Logger les paramètres
            mlflow.log_param("epochs", EPOCHS)
            mlflow.log_param("batch_size", BATCH_SIZE)
            mlflow.log_param("dropout_rate", DROPOUT_RATE)
            mlflow.log_param("learning_rate", LEARNING_RATE)
            mlflow.log_param("optimizer", opt_name)
            mlflow.log_param("architecture", "dense-512-256")
            
            # Logger les tags
            mlflow.set_tag("model_type", "neural_network")
            mlflow.set_tag("dataset", "mnist")
            mlflow.set_tag("framework", "tensorflow")
            
            # Construire le modèle
            model = build_model(DROPOUT_RATE, LEARNING_RATE, L2_FACTOR)

            model.compile(
                optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Callbacks
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            )
            
            # Entraînement
            print("Début de l'entraînement...")
            history = model.fit(
                x_train, y_train,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                #validation_split=0.1,
                validation_data=(x_val, y_val),
                callbacks=[early_stopping],
                verbose=1
            )
            
            # Logger les métriques d'entraînement par epoch
            for epoch in range(len(history.history['loss'])):
                mlflow.log_metric("train_loss", history.history['loss'][epoch], step=epoch)
                mlflow.log_metric("train_accuracy", history.history['accuracy'][epoch], step=epoch)
                mlflow.log_metric("val_loss", history.history['val_loss'][epoch], step=epoch)
                mlflow.log_metric("val_accuracy", history.history['val_accuracy'][epoch], step=epoch)
            
            # Évaluation détaillée
            print("Évaluation du modèle...")
            eval_metrics = evaluate_model_detailed(model, x_test, y_test)
            
            # Logger toutes les métriques d'évaluation
            mlflow.log_metric("test_loss", eval_metrics['test_loss'])
            mlflow.log_metric("test_accuracy", eval_metrics['test_accuracy'])
            mlflow.log_metric("precision", eval_metrics['precision'])
            mlflow.log_metric("recall", eval_metrics['recall'])
            mlflow.log_metric("f1_score", eval_metrics['f1_score'])
            
            # Logger le rapport de classification comme artifact
            with open("classification_report.json", "w") as f:
                json.dump(eval_metrics['classification_report'], f, indent=2)
            mlflow.log_artifact("classification_report.json")
            
            # Logger la matrice de confusion
            with open("confusion_matrix.json", "w") as f:
                json.dump(eval_metrics['confusion_matrix'], f, indent=2)
            mlflow.log_artifact("confusion_matrix.json")
            
            run_id = run.info.run_id

            results.append({
                "optimizer": opt_name,
                "run_id": run.info.run_id,
                "metrics": eval_metrics,
                "model": model
            })
            
            print(f"\n{'='*50}")
            print(f"✓ Entraînement avec le premier optimiseur terminé avec succès!")
            print(f"{'='*50}")
            print(f"Run ID: {run_id}")
            print(f"Accuracy: {eval_metrics['test_accuracy']:.4f}")
            print(f"F1 Score: {eval_metrics['f1_score']:.4f}")
            print(f"Precision: {eval_metrics['precision']:.4f}")
            print(f"Recall: {eval_metrics['recall']:.4f}")
            print(f"{'='*50}\n")
        
        
    best_run = max(
        results,
        key=lambda r: r["metrics"]["test_accuracy"]
    )

    with mlflow.start_run(run_name="mnist-final-best-model") as final_run:

        mlflow.log_param("best_optimizer", best_run["optimizer"])
        mlflow.log_metric("test_accuracy", best_run["metrics"]["test_accuracy"])
        mlflow.log_metric("f1_score", best_run["metrics"]["f1_score"])

        # Logger le modèle
        print("Enregistrement du modèle...")
        mlflow.tensorflow.log_model(
            model=best_run["model"],
            artifact_path="model",
            registered_model_name=MLflowConfig.MODEL_NAME
        )

        final_run_id = final_run.info.run_id

        return final_run_id, best_run["metrics"]

if __name__ == "__main__":
    train_and_log_model()