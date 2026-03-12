import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE


def load_data():
    df = pd.read_csv("data/creditcard.csv")
    return df


def train_model():
    df = load_data()

    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Before SMOTE:")
    print(y_train.value_counts())

    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print("\nAfter SMOTE:")
    print(pd.Series(y_train_resampled).value_counts())

    # Train model
    model = LogisticRegression(max_iter=1000)

    print("\nTraining model with balanced data...")
    model.fit(X_train_resampled, y_train_resampled)

    predictions = model.predict(X_test)

    print("\nModel Accuracy:", accuracy_score(y_test, predictions))
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    # Save model
    joblib.dump(model, "models/fraud_detection_model.pkl")
    print("\nModel saved successfully in models/fraud_detection_model.pkl")

    return model


if __name__ == "__main__":
    train_model()