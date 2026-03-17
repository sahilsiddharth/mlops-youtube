import joblib,os
import optuna
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

import mlflow
import mlflow.xgboost


def Objective(trial,X_train,y_train,X_test,y_test):
    params={
        "n_estimators": trial.suggest_int("n_estimators",100,500),
        "max_depth": trial.suggest_int("max_depth",3,10),
        "learning_rate": trial.suggest_float("learning_rate",0.01,0.2),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "random_state": 42,
        "n_jobs": -1
    }

    with mlflow.start_run(nested=True):
        mlflow.log_params(params)
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, probs)
        mlflow.log_metric("roc_auc", roc_auc)

        
    return roc_auc



def run_optuna(X_train,X_test,y_train,y_test):
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "file:mlruns/")
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("Fraud Detection")
    with mlflow.start_run(run_name="optuna_study"):
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: Objective(trial, X_train, y_train, X_test, y_test),
            n_trials=20
        )

        best_params = study.best_trial.params
        best_params.update({"random_state": 42, "n_jobs": -1})

        best_model = XGBClassifier(**best_params)
        best_model.fit(X_train, y_train)

        preds = best_model.predict(X_test)
        probs = best_model.predict_proba(X_test)[:, 1]

        final_metrics = {
            "final_accuracy": accuracy_score(y_test, preds),
            "final_precision": precision_score(y_test, preds),
            "final_recall": recall_score(y_test, preds),
            "final_f1": f1_score(y_test, preds),
            "final_roc_auc": roc_auc_score(y_test, probs)
        }

        mlflow.log_params(best_params)
        mlflow.log_metrics(final_metrics)
        mlflow.xgboost.log_model(
            best_model,
            "fraud_model",
            registered_model_name="fraud-detection-best"
        )

        # ── Key Change 2: Save to /app/models inside container ──
        #os.makedirs("/app/models", exist_ok=True)
        joblib.dump(best_model, "models/fraud_model.pkl")

        print(f"Best params: {best_params}")
        print(f"Final ROC-AUC: {final_metrics['final_roc_auc']:.4f}")

    return best_params, best_model
