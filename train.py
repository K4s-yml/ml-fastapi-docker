
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("pulsar_classification")
    df = pd.read_csv("data/HTRU_2.csv", header=None)

    df.columns = [
    "mean_integrated_profile",
    "std_integrated_profile",
    "kurtosis_integrated_profile",
    "skewness_integrated_profile",
    "mean_dm_snr_curve",
    "std_dm_snr_curve",
    "kurtosis_dm_snr_curve",
    "skewness_dm_snr_curve",
    "target_class"
]

    X = df.drop(columns=["target_class"])
    y = df["target_class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    with mlflow.start_run(run_name="random_forest"):
        mlflow.log_params({
            "n_estimators": 200,
            "class_weight": "balanced",
            "random_state": 42,
            "test_size": 0.2
        })

        model = RandomForestClassifier(
            n_estimators=200,
            n_jobs=-1,
            class_weight="balanced",
            random_state=42
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        print("Accuracy:", acc)
        print("F1-score:", f1)

        print(classification_report(y_test, y_pred))

        joblib.dump(model, "model.joblib")

        mlflow.log_artifact("model.joblib")
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name="pulsar-rf-model",
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )

if __name__ == "__main__":
    main()


