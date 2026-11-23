
import pandas as pd
import joblib 

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)


def main():
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


    model = RandomForestClassifier(
        n_estimators=200,
        n_jobs=-1,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train, y_train)


    y_pred = model.predict(X_test)  


    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1-score:", f1_score(y_test, y_pred))

    print(classification_report(y_test, y_pred))

    joblib.dump(model, "model.joblib")

if __name__ == "__main__":
    main()


