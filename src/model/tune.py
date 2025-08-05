import pandas as pd
import yaml
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import joblib
import optuna
import mlflow
import mlflow.lightgbm
import mlflow.optuna
import numpy as np

# load the config
def load_config(config_path):
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"Configuration file not found: {config_path}")
        return None

def load_loan_data(config):
    try:
        train_data = pd.read_csv(config['data']['file_paths']['train_data'], low_memory=False)
        valid_data = pd.read_csv(config['data']['file_paths']['valid_data'], low_memory=False)
        test_data = pd.read_csv(config['data']['file_paths']['test_data'], low_memory=False)
        return train_data, valid_data, test_data
    except FileNotFoundError:
        print(f"Loan data file not found: {config['data']['file_paths']['train_data']}")
        print(f"Loan data file not found: {config['data']['file_paths']['valid_data']}")
        print(f"Loan data file not found: {config['data']['file_paths']['test_data']}")
        return None

# Prepare the data for LightGBM
def prepare_lgb_data(train_data, val_data, test_data, target_col='isRisky'):
    """
    Prepare data for LightGBM training with proper categorical feature handling
    """
    # Separate features and target
    X_train = train_data.drop(columns=[target_col])
    y_train = train_data[target_col]
    
    X_val = val_data.drop(columns=[target_col])
    y_val = val_data[target_col]
    
    X_test = test_data.drop(columns=[target_col])
    y_test = test_data[target_col]
    
    # Identify categorical columns (object and category types)
    categorical_columns = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"Categorical columns found: {len(categorical_columns)}")
    for col in categorical_columns:
        # print(f"  - {col}: {X_train[col].nunique()} unique values")
        
        # Convert to category dtype for LightGBM's native support
        X_train[col] = X_train[col].astype('category')
        X_val[col] = X_val[col].astype('category')
        X_test[col] = X_test[col].astype('category')
    
    print(f"\nFinal data shapes:")
    print(f"Train: {X_train.shape}, Target: {y_train.shape}")
    print(f"Validation: {X_val.shape}, Target: {y_val.shape}")
    print(f"Test: {X_test.shape}, Target: {y_test.shape}")
    
    # Verify data types
    print(f"\nData types check:")
    object_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    if object_cols:
        print(f"WARNING: Object columns still present: {object_cols}")
    else:
        print("✅ No object columns remaining")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, categorical_columns

# Define hyperparameter optimization with Optuna
def objective(trial, X_train, y_train, categorical_columns):
    """
    Objective function for Optuna hyperparameter optimization
    """
    # Suggest hyperparameters
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 10, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'verbosity': -1,
        'random_state': 42
    }
    
    # Log parameters to MLflow
    with mlflow.start_run(nested=True):
        mlflow.log_params(params)
        
        # Use Stratified K-Fold for robust validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        auc_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_fold_train = X_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            y_fold_train = y_train.iloc[train_idx]
            y_fold_val = y_train.iloc[val_idx]
            
            # Create LightGBM datasets
            train_dataset = lgb.Dataset(
                X_fold_train, 
                label=y_fold_train,
                categorical_feature=categorical_columns
            )
            val_dataset = lgb.Dataset(
                X_fold_val, 
                label=y_fold_val,
                categorical_feature=categorical_columns,
                reference=train_dataset
            )
            
            # Train model
            model = lgb.train(
                params,
                train_dataset,
                num_boost_round=1000,
                valid_sets=[val_dataset],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )
            
            # Predict and calculate AUC
            y_pred_proba = model.predict(X_fold_val, num_iteration=model.best_iteration)
            auc_score = roc_auc_score(y_fold_val, y_pred_proba)
            auc_scores.append(auc_score)
            
            # Log fold metrics
            mlflow.log_metric(f"fold_{fold}_auc", auc_score)
        
        mean_auc = np.mean(auc_scores)
        std_auc = np.std(auc_scores)
        
        # Log aggregated metrics
        mlflow.log_metric("mean_cv_auc", mean_auc)
        mlflow.log_metric("std_cv_auc", std_auc)
        mlflow.log_metric("trial_number", trial.number)
    
    return mean_auc

def tune_hyperparameters(X_train, y_train, categorical_columns, config):
    number_of_trials = config['model_params']['n_trials']
    
    # Set MLflow experiment
    experiment_name = "/Shared/lightgbm_hyperparameter_tuning"
    mlflow.set_experiment(experiment_name)

    print("Starting hyperparameter optimization with Optuna...")
    
    with mlflow.start_run(run_name="optuna_optimization"):
        # Log general experiment info
        mlflow.log_param("n_trials", number_of_trials)
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_categorical_features", len(categorical_columns))
        
        study = optuna.create_study(
            direction='maximize', 
            sampler=optuna.samplers.TPESampler(seed=42),
            study_name="loan_risk_prediction_tuning",
            )
        study.optimize(lambda trial: objective(trial, X_train, y_train, categorical_columns), n_trials=number_of_trials)

        print(f"Best parameters: {study.best_params}")
        print(f"Best cross-validation AUC: {study.best_value:.4f}")
        
        # Log best results
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_cv_auc", study.best_value)
        
        # Log optimization history
        for i, trial in enumerate(study.trials):
            mlflow.log_metric("trial_auc", trial.value, step=i)

        # save the params
        best_params_path = config['data']['model_paths']['best_params']
        joblib.dump(study.best_params, best_params_path)
        
        # Log the best parameters file as artifact
        mlflow.log_artifact(best_params_path, "model_artifacts")
        
        print(f"Best parameters saved to: {best_params_path}")
        print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
        print(f"Experiment URL: {mlflow.get_experiment_by_name(experiment_name)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Merging the data.')
    parser.add_argument('--config', required=True, help='Path to the configuration file.')
    args = parser.parse_args()

    config = load_config(args.config)
    train_data, valid_data, test_data = load_loan_data(config)
    X_train, X_val, X_test, y_train, y_val, y_test, categorical_columns = prepare_lgb_data(
    train_data, valid_data, test_data
    )

    # Start hyperparameter tuning
    tune_hyperparameters(X_train, y_train, categorical_columns, config)

# how to run
# py -3.12 src/model/tune.py --config "config/config.yaml"