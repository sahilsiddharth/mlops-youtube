import joblib
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
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Fraud Detection")

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

    with mlflow.start_run():
        mlflow.log_params(params)
        model = XGBClassifier(**params)
        
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds)
        recall = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        roc_auc = roc_auc_score(y_test, probs)

        print("\nModel Evaluation\n")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")
        print(f"ROC-AUC: {roc_auc}")

        

        ## log metrics
        mlflow.log_metric("accuracy",accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        # log model
        mlflow.xgboost.log_model(model, "fraud_model")

    return roc_auc



def run_optuna(X_train,X_test,y_train,y_test):
    study=optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial:Objective(trial,X_train,y_train,X_test,y_test),
        n_trials=20
    )
    print("Best trial:", study.best_trial.params)

    return study.best_trial.params