import sys
import os
from src.data.load_data import load_data
from src.data.preprocess import preprocess_data

from src.models.train import run_optuna





def main():

    print("Loading dataset...")
    df = load_data("data/raw/creditcard.csv")

    print("Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data(df)
    print(y_test)

    print("Running Optuna hyperparameter tuning...")
    best_params = run_optuna(X_train, X_test, y_train, y_test)

    print("Best parameters:", best_params)



   


if __name__ == "__main__":
    main()