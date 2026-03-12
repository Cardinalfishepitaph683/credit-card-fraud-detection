import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier


def load_data():
    df = pd.read_csv("data/creditcard.csv")
    return df


def evaluate_models():
    df = load_data()

    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

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

    for name, model in models.items():
        print(f"\n{'=' * 60}")
        print(f"Evaluating {name}")
        print(f"{'=' * 60}")

        model.fit(X_train_resampled, y_train_resampled)
        y_pred = model.predict(X_test)

        print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
        print("F1 Score:", round(f1_score(y_test, y_pred), 4))

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    evaluate_models()