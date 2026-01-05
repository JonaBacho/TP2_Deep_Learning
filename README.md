# 🧠 TP2: Improving Deep Neural Networks - MLOps Pipeline

[![TP2 Experiments](https://github.com/<YOUR-USERNAME>/<YOUR-REPO>/actions/workflows/tp2-train-experiments.yml/badge.svg)](https://github.com/<YOUR-USERNAME>/<YOUR-REPO>/actions/workflows/tp2-train-experiments.yml)
[![MLflow](https://img.shields.io/badge/MLflow-2.9.2-blue.svg)](https://mlflow.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**École Nationale Supérieure Polytechnique de Yaoundé**  
Département de Génie Informatique - 5GI  
Instructeurs: Louis Fippo Fitime, Claude Tinku, Kerolle Sonfack

Ce projet implémente le **TP2 sur l'amélioration des réseaux de neurones profonds** avec un pipeline MLOps complet. Il couvre le diagnostic bias/variance, la régularisation, la batch normalization, la comparaison d'optimiseurs, le tracking MLflow et l'automatisation CI/CD via GitHub Actions.

---

## 🎯 Objectifs d'Apprentissage

- **Diagnostiquer** les problèmes de high bias (underfitting) et high variance (overfitting)
- **Maîtriser** les techniques de régularisation : L2 et Dropout
- **Utiliser** la Batch Normalization pour stabiliser et accélérer l'entraînement
- **Comparer** les algorithmes d'optimisation : SGD with Momentum, RMSprop, Adam
- **Automatiser** l'entraînement et le tracking via GitHub Actions et MLflow

---

## 📁 Structure du Projet

```text
├── config/                          # Configuration centralisée
│   └── mlflow_config.py             # Configuration MLflow, S3/MinIO
├── src/                             # Code source principal
│   ├── app.py                       # API Flask (Service de prédiction)
│   ├── train_model.py               # TP1 - Entraînement baseline
│   ├── train_model_tp2.py           # TP2 - 4 exercices complets
│   ├── auto_promote.py              # Promotion automatique basée sur seuils
│   ├── promote_model.py             # Gestion manuelle des stages
│   └── evaluate_model.py            # Évaluation détaillée des modèles
├── tests/                           # Tests unitaires
│   └── test_model.py                # Tests API et chargement modèle
├── .github/workflows/               # Pipelines CI/CD
│   ├── tp2-train-experiments.yml    # Workflow principal (4 exercices)
│   ├── tp2-quick-test.yml           # Tests rapides sur PR
│   ├── train-register.yml           # TP1 - Entraînement baseline
│   └── deploy.yml                   # Déploiement API Flask
├── run_tp2.py                       # Script d'exécution local
├── Dockerfile                       # Image Docker de production
├── requirements.txt                 # Dépendances Python
└── .env.example                     # Template configuration
```

---

## 🚀 Exercices du TP2

### Exercise 1: Bias/Variance Analysis
Diagnostic du modèle baseline pour identifier underfitting ou overfitting.
- **Expérience MLflow**: `TP2-Exercise1-BiasVariance`
- **Métriques**: `train_accuracy`, `val_accuracy`, `accuracy_gap`
- **Tag**: `diagnosis` (HIGH_BIAS, HIGH_VARIANCE, GOOD_FIT)

### Exercise 2: Regularization
Application de L2 regularization et Dropout pour réduire l'overfitting.
- **Expérience MLflow**: `TP2-Exercise2-Regularization`
- **Techniques**: L2 (0.001), Dropout (0.2)
- **Comparaison**: Avant/Après régularisation

### Exercise 3: Optimizer Comparison
Comparaison de 3 optimiseurs sur la même architecture.
- **Expérience MLflow**: `TP2-Exercise3-Optimizers`
- **Optimiseurs**: SGD with Momentum, RMSprop, Adam
- **Métriques**: `final_test_accuracy`, vitesse de convergence

### Exercise 4: Batch Normalization
Mesure de l'impact de la Batch Normalization sur la stabilité et la vitesse.
- **Expérience MLflow**: `TP2-Exercise4-BatchNorm`
- **Comparaison**: Sans vs Avec BatchNorm
- **Architecture**: Dense(512) → BatchNorm → Dropout → Dense(10)

---

## 🛠️ Installation

### Prérequis
- Python 3.10+
- Accès à un serveur MLflow (distant ou local)
- GitHub repository avec Actions activées

### Installation Locale

```bash
# Cloner le projet
git clone <votre-repo-url>
cd TP2_Deep_Learning

# Créer environnement virtuel
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Installer dépendances
pip install -r requirements.txt

# Configurer environnement
cp .env.example .env
# Éditer .env avec vos credentials MLflow
```

---

## 🔌 Utilisation

### Exécution via GitHub Actions (Recommandé)

```bash
# 1. Push sur main → déclenche automatiquement tous les exercices
git add .
git commit -m "feat: run TP2 experiments"
git push origin main

# 2. Ou exécution manuelle depuis GitHub UI
# Actions → "TP2 - Run All Experiments" → Run workflow
# Choisir: all, exercise1, exercise2, exercise3, ou exercise4
```

**Durée d'exécution**: 25-30 minutes (4 jobs en parallèle)

### Exécution Locale

```bash
# Tous les exercices
python run_tp2.py

# Exercice spécifique
python -c "from src.train_model_tp2 import exercise_1_baseline; exercise_1_baseline()"
python -c "from src.train_model_tp2 import exercise_2_regularization; exercise_2_regularization()"
python -c "from src.train_model_tp2 import exercise_3_optimizers; exercise_3_optimizers()"
python -c "from src.train_model_tp2 import exercise_4_batch_norm; exercise_4_batch_norm()"
```

---

## 🤖 Pipeline CI/CD

### 1. TP2 - Run All Experiments (`tp2-train-experiments.yml`)
- **Déclenchement**: Push sur main/dev, modifications des fichiers TP2, workflow_dispatch
- **4 Jobs Parallèles**:
  - `exercise1-bias-variance` (7-10 min)
  - `exercise2-regularization` (7-10 min)
  - `exercise3-optimizers` (12-15 min)
  - `exercise4-batchnorm` (10-12 min)
- **Job Summary**: Génère rapport consolidé avec tous les résultats
- **Artifacts**: Logs de chaque exercice + rapport markdown

### 2. TP2 - Quick Test (`tp2-quick-test.yml`)
- **Déclenchement**: Pull Request vers main/dev
- **Durée**: 2-3 minutes
- **Objectif**: Validation rapide (1 epoch, 1000 échantillons)

### 3. MLflow Train and Register (`train-register.yml`)
- TP1 - Entraînement baseline avec promotion automatique

### 4. Deploy Application (`deploy.yml`)
- Build Docker + Déploiement API Flask sur serveur distant

---

## 📊 Visualisation des Résultats

### Dans MLflow UI

Accédez à votre serveur MLflow configuré dans `.env`:

```bash
# Les expériences créées automatiquement:
- TP2-Exercise1-BiasVariance     (1 run)
- TP2-Exercise2-Regularization   (1 run)
- TP2-Exercise3-Optimizers       (3 runs)
- TP2-Exercise4-BatchNorm        (2 runs)
```

**Métriques trackées**:
- `train_loss`, `train_accuracy` (par epoch)
- `val_loss`, `val_accuracy` (par epoch)
- `test_loss`, `test_accuracy` (final)
- `loss_gap`, `accuracy_gap` (diagnostic)

**Tags spéciaux**:
- `exercise`: 1, 2, 3, ou 4
- `diagnosis`: HIGH_BIAS, HIGH_VARIANCE, GOOD_FIT
- `optimizer`: SGD_with_momentum, RMSprop, Adam
- `batch_normalization`: true/false

### Dans GitHub Actions

```
Actions → TP2 - Run All Experiments → Latest run
→ Artifacts: exercise1-logs, exercise2-logs, exercise3-logs, exercise4-logs, tp2-summary-report
→ Summary: Résumé consolidé avec extraits des logs
```

---

## ⚙️ Configuration

### Variables d'Environnement (`.env`)

```bash
# MLflow Tracking
MLFLOW_TRACKING_URI=http://your-mlflow-server:5000
MLFLOW_TRACKING_USERNAME=your_username
MLFLOW_TRACKING_PASSWORD=your_password

# Model Registry
MODEL_NAME=mnist-classifier-tp2

# S3/MinIO (si utilisé)
MLFLOW_S3_ENDPOINT_URL=https://your-s3-endpoint
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
```

### Secrets GitHub

Configurez dans **Settings → Secrets and variables → Actions**:

```
MLFLOW_TRACKING_URI
MLFLOW_TRACKING_USERNAME
MLFLOW_TRACKING_PASSWORD
MLFLOW_S3_ENDPOINT_URL
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
```

---

## 🔌 API REST (Optionnel)

L'API Flask sert le modèle en production pour les prédictions.

### Endpoints

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| GET | `/health` | État de santé et version du modèle |
| GET | `/model/info` | Informations détaillées sur le modèle |
| POST | `/predict` | Prédiction sur image (784 pixels) |
| POST | `/model/reload` | Rechargement du modèle depuis MLflow |

### Exemple `/predict`

```json
{
  "image": [0, 0, 0.5, 0.8, ..., 0]
}
```

### Lancement Local

```bash
python -m src.app
# API accessible sur http://localhost:5000
```

### Déploiement Docker

```bash
docker build -t mnist-mlflow-tp2-app .
docker run -p 5000:5000 --env-file .env mnist-mlflow-tp2-app
```

---

## 🧪 Tests

```bash
# Exécuter tous les tests
pytest tests/test_model.py

# Avec couverture
pytest --cov=src tests/
```

---

## 📈 Résultats Attendus

### Exercise 1: Bias/Variance
```
Train Accuracy: 0.9856
Val Accuracy: 0.9778
Accuracy Gap: 0.0078
DIAGNOSIS: GOOD_FIT
```

### Exercise 2: Regularization
```
Sans régularisation → Gap: 0.0078
Avec régularisation → Gap: 0.0022 ✓ Amélioration
```

### Exercise 3: Optimizers
```
SGD_with_momentum : 0.9756
RMSprop          : 0.9801
Adam             : 0.9823 ✓ Meilleur
```

### Exercise 4: Batch Normalization
```
Sans BatchNorm : 0.9778
Avec BatchNorm : 0.9812 ✓ Plus rapide + stable
```

---

## 📚 Documentation
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)

---

## 👥 Auteurs

**ENSPY - Université de Yaoundé I**  
FOMEKONG TAMDJI JONATHAN BACHELARD 21P021 - Département de Génie Informatique - Promotion 5GI 2025

**Instructeurs**:
- Louis Fippo Fitime - louis.fippo@univ-yaounde1.cm
---

## ⚖️ Licence

Distribué sous la licence MIT. Voir [LICENSE](LICENSE) pour plus d'informations.