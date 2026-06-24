"""
DeepSurv (CoxPH) model training with Optuna hyperparameter optimization.

Adapted for multi-model-pipeline:
- Uses CSV input format (not parquet)
- Consumes selected features from feature selection step
- Outputs model with scikit-survival compatible wrapper
- Includes preprocessing pipeline (imputation, scaling)
- Saves all artifacts for wrapper to load

DeepSurv is a Cox proportional hazards deep neural network for survival analysis.
Reference: https://github.com/havakv/pycox
"""

import argparse
import os
import sys
import json
import numpy as np
import pandas as pd
import mlflow
import optuna
import joblib
from pathlib import Path
import torch
import torchtuples as tt
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import RobustScaler

# Fix pandas compatibility with pycox (pandas >= 1.5 removed is_monotonic)
if not hasattr(pd.Series, 'is_monotonic'):
    pd.Series.is_monotonic = property(lambda self: self.is_monotonic_increasing)

# Import pycox
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv

# Import sksurv for data preparation
from sksurv.util import Surv


def objective_with_cv(trial, X, y_time, y_event, cv, input_dim, n_epochs, early_stop_patience, device):
    """
    Optuna objective function with cross-validation.
    
    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial object
    X : np.ndarray
        All features
    y_time : np.ndarray
        All durations
    y_event : np.ndarray
        All events
    cv : RepeatedStratifiedKFold
        Cross-validation splitter
    input_dim : int
        Number of input features
    n_epochs : int
        Number of training epochs
    early_stop_patience : int
        Early stopping patience
    device : str
        'cpu' or 'cuda'
        
    Returns
    -------
    mean_ci : float
        Mean C-index across all folds (to maximize)
    """
    # Hyperparameter search space
    n_layers = trial.suggest_int('n_layers', 1, 3)
    
    num_nodes = []
    for i in range(n_layers):
        num_nodes.append(trial.suggest_int(f'num_nodes_l{i}', 16, 128, step=16))
    
    batch_norm = trial.suggest_categorical('batch_norm', [True, False])
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    
    # Store C-index for each fold
    fold_scores = []
    
    # Iterate over CV folds
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y_event)):
        x_train_fold = X[train_idx].astype('float32')
        y_train_fold = (y_time[train_idx].astype('float32'), y_event[train_idx].astype('float32'))
        
        x_val_fold = X[val_idx].astype('float32')
        y_val_fold = (y_time[val_idx].astype('float32'), y_event[val_idx].astype('float32'))
        
        # Create network
        net = tt.practical.MLPVanilla(
            in_features=input_dim,
            num_nodes=num_nodes,
            out_features=1,
            batch_norm=batch_norm,
            dropout=dropout,
            output_bias=False
        )
        
        # Create model
        model = CoxPH(net, tt.optim.Adam(lr=learning_rate))
        
        # Early stopping callback
        callbacks = [tt.callbacks.EarlyStopping(patience=early_stop_patience)]
        
        val_data = (x_val_fold, y_val_fold)
        
        try:
            log = model.fit(
                x_train_fold, y_train_fold,
                batch_size=batch_size,
                epochs=n_epochs,
                callbacks=callbacks,
                verbose=False,
                val_data=val_data,
                val_batch_size=batch_size
            )
            
            # Compute baseline hazards
            model.compute_baseline_hazards()
            
            # Evaluate on validation fold
            surv_val = model.predict_surv_df(x_val_fold)
            ev_val = EvalSurv(surv_val, y_val_fold[0], y_val_fold[1], censor_surv='km')
            val_ci = ev_val.concordance_td()
            
            fold_scores.append(val_ci)
            
            # Log to MLflow
            mlflow.log_metric(f"cv_fold_{fold_idx}_ci", val_ci, step=trial.number)
            
        except Exception as e:
            print(f"Fold {fold_idx} failed with error: {e}")
            fold_scores.append(0.0)
    
    # Return mean C-index across folds
    mean_ci = np.mean(fold_scores)
    mlflow.log_metric(f"cv_mean_ci", mean_ci, step=trial.number)
    
    return mean_ci


def train_final_model(x_train, y_train, x_val, y_val, best_params, n_epochs, device):
    """
    Train final model with best hyperparameters.
    
    Parameters
    ----------
    x_train : np.ndarray
        Training features
    y_train : tuple
        Training labels (durations, events)
    x_val : np.ndarray
        Validation features
    y_val : tuple
        Validation labels (durations, events)
    best_params : dict
        Best hyperparameters from Optuna
    n_epochs : int
        Number of training epochs
    device : str
        'cpu' or 'cuda'
        
    Returns
    -------
    model : CoxPH
        Trained model
    training_history : dict
        Training history
    """
    print("\n============================================================")
    print("Training Final Model with Best Hyperparameters")
    print("============================================================\n")
    
    # Extract hyperparameters
    n_layers = best_params['n_layers']
    num_nodes = [best_params[f'num_nodes_l{i}'] for i in range(n_layers)]
    batch_norm = best_params['batch_norm']
    dropout = best_params['dropout']
    learning_rate = best_params['learning_rate']
    batch_size = best_params['batch_size']
    
    print(f"Network architecture: {num_nodes}")
    print(f"Batch norm: {batch_norm}, Dropout: {dropout}")
    print(f"Learning rate: {learning_rate}, Batch size: {batch_size}")
    
    # Create network
    net = tt.practical.MLPVanilla(
        in_features=x_train.shape[1],
        num_nodes=num_nodes,
        out_features=1,
        batch_norm=batch_norm,
        dropout=dropout,
        output_bias=False
    )
    
    # Create model
    model = CoxPH(net, tt.optim.Adam(lr=learning_rate))
    
    # Training with early stopping
    callbacks = [tt.callbacks.EarlyStopping(patience=20)]
    
    val_data = (x_val, y_val)
    
    print(f"\nTraining for up to {n_epochs} epochs...")
    log = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=n_epochs,
        callbacks=callbacks,
        verbose=True,
        val_data=val_data,
        val_batch_size=batch_size
    )
    
    # Compute baseline hazards
    print("\nComputing baseline hazards...")
    model.compute_baseline_hazards()
    
    # Final evaluation
    print("\nFinal Evaluation:")
    surv_train = model.predict_surv_df(x_train)
    surv_val = model.predict_surv_df(x_val)
    
    ev_train = EvalSurv(surv_train, y_train[0], y_train[1], censor_surv='km')
    ev_val = EvalSurv(surv_val, y_val[0], y_val[1], censor_surv='km')
    
    train_ci = ev_train.concordance_td()
    val_ci = ev_val.concordance_td()
    
    print(f"  Training C-index: {train_ci:.4f}")
    print(f"  Validation C-index: {val_ci:.4f}")
    
    # Create training history
    training_history = {
        'train_loss': log.monitors['train_'].scores['loss']['score'],
        'val_loss': log.monitors['val_'].scores['loss']['score'],
        'epochs': list(range(1, len(log.monitors['train_'].scores['loss']['score']) + 1)),
        'final_train_ci': float(train_ci),
        'final_val_ci': float(val_ci)
    }
    
    return model, training_history


def save_model(model, output_path, best_params, training_metrics, baseline_hazards, 
               imputer, scaler, feature_cols=None):
    """
    Save trained DeepSurv model and all artifacts for wrapper.
    
    Parameters
    ----------
    model : CoxPH
        Trained model
    output_path : Path
        Output directory
    best_params : dict
        Best hyperparameters
    training_metrics : dict
        Training metrics
    baseline_hazards : pd.DataFrame
        Baseline hazard estimates
    imputer : IterativeImputer
        Fitted imputer
    scaler : RobustScaler
        Fitted scaler
    feature_cols : list, optional
        List of feature names used in training
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save model weights
    model.save_net(str(output_path / 'deepsurv_net.pt'))
    
    # Save baseline hazards
    baseline_hazards.to_pickle(str(output_path / 'baseline_hazards.pkl'))
    
    # Save hyperparameters with input dimension
    params_with_dim = best_params.copy()
    params_with_dim['input_dim'] = len(feature_cols) if feature_cols else model.net.in_features
    with open(output_path / 'hyperparameters.json', 'w') as f:
        json.dump(params_with_dim, f, indent=2)
    
    # Save training metrics
    with open(output_path / 'training_metrics.json', 'w') as f:
        json.dump(training_metrics, f, indent=2)
    
    # Save preprocessing components
    joblib.dump(imputer, output_path / 'imputer.pkl')
    joblib.dump(scaler, output_path / 'scaler.pkl')
    
    # Save selected features if provided
    if feature_cols is not None:
        with open(output_path / 'selected_features.txt', 'w') as f:
            for feat in feature_cols:
                f.write(f"{feat}\n")
        print(f"Saved {len(feature_cols)} feature names to selected_features.txt")
    
    # Save wrapper for easy loading
    from deepsurv_wrapper import DeepSurvWrapper
    wrapper = DeepSurvWrapper(output_path)
    joblib.dump(wrapper, output_path / 'deepsurv_model.pkl')
    
    print(f"\nModel saved to {output_path}")


def main(args):
    """Main training function."""
    
    mlflow.start_run()
    
    print("============================================================")
    print("DeepSurv Model Training - Multi-Model Pipeline")
    print("============================================================")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Load data from CSV
    print("Loading data...")
    csv_files = [f for f in os.listdir(args.training_data) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError("No CSV file found in training_data folder.")
    csv_path = os.path.join(args.training_data, csv_files[0])
    print(f"Loading dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Number of events: {df['event'].sum()} / {len(df)}")
    
    # Prepare features
    X = df.drop(columns=["event", "tte"]).select_dtypes(include=[np.number])
    X.set_index("id", inplace=True)
    y = Surv.from_arrays(event=df["event"].astype(bool), time=df["tte"].astype(float))
    
    # Load selected features if provided
    if args.selected_features:
        selected_features_path = os.path.join(args.selected_features, "selected_features.txt")
        print(f"\nLoading selected features from {selected_features_path}")
        with open(selected_features_path, 'r') as f:
            selected_features = [line.strip() for line in f if line.strip()]
        
        print(f"Using {len(selected_features)} RSF-selected features out of {X.shape[1]} total")
        X = X[selected_features]
        feature_cols = selected_features
    else:
        print(f"Using all {X.shape[1]} features")
        feature_cols = X.columns.tolist()
    
    print(f"\nFinal feature matrix: {X.shape}")
    
    # Preprocessing pipeline
    print("\n=== Preprocessing ===")
    print("Applying imputation and scaling...")
    imputer = IterativeImputer(random_state=42, max_iter=10)
    scaler = RobustScaler()
    
    X_imputed = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imputed)
    X_processed = X_scaled.astype('float64')  # Keep float64 for now, convert to float32 in training
    
    # Extract time and event
    y_time = y['time'].astype('float64')
    y_event = y['event'].astype('int32')
    
    input_dim = X_processed.shape[1]
    
    print(f"Preprocessed data shape: {X_processed.shape}")
    print(f"Number of features: {input_dim}")
    
    # Create CV splitter
    cv_stratify = args.cv_stratify_by_event.lower() == 'true'
    print(f"\nSetting up {args.cv_n_repeats}-Repeated {args.cv_n_splits}-Fold CV")
    print(f"Stratify by event: {cv_stratify}")
    
    cv = RepeatedStratifiedKFold(
        n_splits=args.cv_n_splits,
        n_repeats=args.cv_n_repeats,
        random_state=42
    )
    
    # Log parameters
    mlflow.log_param('n_features', input_dim)
    mlflow.log_param('n_samples', len(X_processed))
    mlflow.log_param('n_events', int(y_event.sum()))
    mlflow.log_param('cv_n_splits', args.cv_n_splits)
    mlflow.log_param('cv_n_repeats', args.cv_n_repeats)
    mlflow.log_param('cv_total_folds', args.cv_n_splits * args.cv_n_repeats)
    mlflow.log_param('n_optuna_trials', args.n_optuna_trials)
    mlflow.log_param('n_epochs', args.n_epochs)
    mlflow.log_param('selected_features', args.selected_features is not None)
    
    # Hyperparameter optimization with Optuna using CV
    print("\n============================================================")
    print("Starting Optuna Hyperparameter Optimization with CV")
    print("============================================================\n")
    
    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda trial: objective_with_cv(
            trial, X_processed, y_time, y_event, cv,
            input_dim, args.n_epochs, args.early_stop_patience, device
        ),
        n_trials=args.n_optuna_trials,
        show_progress_bar=True
    )
    
    print("\n============================================================")
    print("Optuna Optimization Complete")
    print("============================================================\n")
    
    best_trial = study.best_trial
    best_params = best_trial.params
    best_ci = best_trial.value
    
    print(f"Best CV mean C-index: {best_ci:.4f}")
    print(f"Best hyperparameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
        mlflow.log_param(f'best_{key}', value)
    
    mlflow.log_metric('best_cv_mean_ci', best_ci)
    
    # Train final model on train/val split for early stopping
    print("\n============================================================")
    print("Training Final Model on Train/Val Split")
    print("============================================================\n")
    
    # Create train/validation split (80/20)
    from sklearn.model_selection import train_test_split
    train_idx, val_idx = train_test_split(
        np.arange(len(X_processed)),
        test_size=0.2,
        random_state=42,
        stratify=y_event
    )
    
    x_train = X_processed[train_idx].astype('float32')
    y_train = (y_time[train_idx].astype('float32'), y_event[train_idx].astype('float32'))
    
    x_val = X_processed[val_idx].astype('float32')
    y_val = (y_time[val_idx].astype('float32'), y_event[val_idx].astype('float32'))
    
    print(f"Training samples: {len(x_train)}")
    print(f"Validation samples: {len(x_val)}")
    
    model, training_history = train_final_model(
        x_train, y_train, x_val, y_val,
        best_params, args.n_epochs * 2, device
    )
    
    # Log final metrics
    mlflow.log_metric('final_train_ci', training_history['final_train_ci'])
    mlflow.log_metric('final_val_ci', training_history['final_val_ci'])
    
    # Save model and artifacts
    training_metrics = {
        'best_hyperparameters': best_params,
        'best_cv_mean_ci': float(best_ci),
        'training_history': training_history,
        'cv_config': {
            'n_splits': args.cv_n_splits,
            'n_repeats': args.cv_n_repeats,
            'stratify_by_event': cv_stratify
        }
    }
    
    # Get baseline hazards
    baseline_hazards = model.compute_baseline_hazards()
    
    save_model(
        model,
        Path(args.trained_model),
        best_params,
        training_metrics,
        baseline_hazards,
        imputer,
        scaler,
        feature_cols=feature_cols
    )
    
    print("\n============================================================")
    print("Training Complete!")
    print("============================================================")
    
    mlflow.end_run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DeepSurv model for multi-model-pipeline')
    parser.add_argument('--training_data', type=str, required=True,
                        help='Path to training data folder (CSV format)')
    parser.add_argument('--trained_model', type=str, required=True,
                        help='Path to save trained model')
    parser.add_argument('--selected_features', type=str, default=None,
                        help='Path to selected features folder (optional)')
    parser.add_argument('--cv_n_splits', type=int, default=5,
                        help='Number of CV splits')
    parser.add_argument('--cv_n_repeats', type=int, default=3,
                        help='Number of CV repeats')
    parser.add_argument('--cv_stratify_by_event', type=str, default='true',
                        help='Stratify CV by event status')
    parser.add_argument('--n_optuna_trials', type=int, default=30,
                        help='Number of Optuna trials')
    parser.add_argument('--n_epochs', type=int, default=100,
                        help='Number of training epochs per trial')
    parser.add_argument('--early_stop_patience', type=int, default=10,
                        help='Early stopping patience')
    
    args = parser.parse_args()
    main(args)
