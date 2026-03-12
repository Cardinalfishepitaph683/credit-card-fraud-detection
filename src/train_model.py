import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier


def load_data() -> pd.DataFrame:
    """Load credit card fraud dataset."""
    df = pd.read_csv("data/creditcard.csv")
    return df


def prepare_data(df: pd.DataFrame):
    """Split dataset into features and target."""
    X = df.drop("Class", axis=1)
    y = df["Class"]
    return X, y


def apply_smote(X_train, y_train):
    """Apply SMOTE to balance the training dataset."""
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    return X_train_resampled, y_train_resampled


def train_models(X_train, y_train, X_test, y_test):
    """Train multiple models and compare performance."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss"
        )
    }

    best_model = None
    best_model_name = None
    best_f1 = 0.0

    for name, model in models.items():
        print(f"\n{'=' * 50}")
        print(f"Training {name}...")
        print(f"{'=' * 50}")

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)

        print(f"\n{name} Accuracy: {accuracy:.4f}")
        print(f"{name} F1 Score: {f1:.4f}")

        print(f"\n{name} Classification Report:")
        print(classification_report(y_test, predictions))

        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_model_name = name

    print(f"\n{'=' * 50}")
    print(f"Best Model: {best_model_name}")
    print(f"Best F1 Score: {best_f1:.4f}")
    print(f"{'=' * 50}")

    return best_model, best_model_name


def main() -> None:
    # Load dataset
    df = load_data()

    # Prepare features and target
    X, y = prepare_data(df)

    print("Dataset shape:", df.shape)
    print("Features shape:", X.shape)
    print("Target shape:", y.shape)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nBefore SMOTE:")
    print(y_train.value_counts())

    # Apply SMOTE
    X_train_resampled, y_train_resampled = apply_smote(X_train, y_train)

    print("\nAfter SMOTE:")
    print(pd.Series(y_train_resampled).value_counts())

    # Train and compare models
    best_model, best_model_name = train_models(
        X_train_resampled,
        y_train_resampled,
        X_test,
        y_test
    )

    # Save best model
    model_file_name = f"models/{best_model_name.lower().replace(' ', '_')}_fraud_model.pkl"
    joblib.dump(best_model, model_file_name)

    print(f"\nBest model saved successfully: {model_file_name}")


if __name__ == "__main__":
    main()