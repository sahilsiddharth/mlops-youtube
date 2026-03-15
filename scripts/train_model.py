import sys
import os
from src.data.load_data import load_data
from src.data.preprocess import preprocess_data

from src.models.train import train_model, save_model


def main():

    print("Loading dataset...")

    df = load_data("data/raw/creditcard.csv")

    print("Preprocessing data...")

    X_train, X_test, y_train, y_test = preprocess_data(df)

    print("Training model...")

    model = train_model(X_train, y_train,X_test, y_test)

    print("Evaluating model...")

    

    print("Saving model...")

    save_model(model)


if __name__ == "__main__":
    main()