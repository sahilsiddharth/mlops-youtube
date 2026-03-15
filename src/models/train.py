import joblib
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
def train_model(X_train,y_train,X_test,y_test):
    with mlflow.start_run():
        model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )

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

        ## log parameter
        mlflow.log_param("n_estimator",200)
        mlflow.log_param("max_depth",6)
        mlflow.log_param("learning_rate",0.05)

        ## log metrics
        mlflow.log_metric("accuracy",accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        # log model
        mlflow.xgboost.log_model(model, "fraud_model")

    return model



def save_model(model, path="models/fraud_model.pkl"):

    joblib.dump(model, path)

    print(f"\nModel saved at {path}")