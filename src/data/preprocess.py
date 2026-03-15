import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE


def preprocess_data(df):

    """
    Preprocess fraud dataset
    """

    # Separate features and target
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Scale Amount and Time
    scaler = StandardScaler()

    X_train[["Amount", "Time"]] = scaler.fit_transform(
        X_train[["Amount", "Time"]]
    )

    X_test[["Amount", "Time"]] = scaler.transform(
        X_test[["Amount", "Time"]]
    )

    # Handle class imbalance
    smote = SMOTE(random_state=42)

    X_train_resampled, y_train_resampled = smote.fit_resample(
        X_train,
        y_train
    )

    return X_train_resampled, X_test, y_train_resampled, y_test