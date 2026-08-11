from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "wine.data"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "wine_model.pkl"


# =========================================================
# DATASET COLUMNS
# =========================================================

COLUMNS = [
    "target",
    "alcohol",
    "malic_acid",
    "ash",
    "alcalinity_of_ash",
    "magnesium",
    "total_phenols",
    "flavanoids",
    "nonflavanoid_phenols",
    "proanthocyanins",
    "color_intensity",
    "hue",
    "od280_od315",
    "proline"
]


FEATURES = [
    "alcohol",
    "malic_acid",
    "ash",
    "alcalinity_of_ash",
    "magnesium",
    "total_phenols",
    "flavanoids",
    "nonflavanoid_phenols",
    "proanthocyanins",
    "color_intensity",
    "hue",
    "od280_od315",
    "proline"
]


# =========================================================
# CHECK DATASET
# =========================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )


# =========================================================
# LOAD DATASET
# =========================================================

df = pd.read_csv(
    DATA_PATH,
    header=None,
    names=COLUMNS
)


print("=" * 60)
print("WINE CLASSIFICATION MODEL TRAINING")
print("=" * 60)

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())


# =========================================================
# TARGET INFORMATION
# =========================================================

print("\nTarget classes:")
print(sorted(df["target"].unique()))

print("\nClass distribution:")
print(df["target"].value_counts().sort_index())


# =========================================================
# FEATURES AND TARGET
# =========================================================

X = df[FEATURES]
y = df["target"]


# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining samples:")
print(len(X_train))

print("\nTesting samples:")
print(len(X_test))


# =========================================================
# MACHINE LEARNING PIPELINE
# =========================================================

print("\nTraining Logistic Regression model...")

model_pipeline = Pipeline(
    [
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            LogisticRegression(
                max_iter=2000,
                random_state=42
            )
        )
    ]
)


# =========================================================
# TRAIN MODEL
# =========================================================

model_pipeline.fit(
    X_train,
    y_train
)


# =========================================================
# PREDICTION
# =========================================================

y_pred = model_pipeline.predict(
    X_test
)


# =========================================================
# MODEL EVALUATION
# =========================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\nTest Accuracy:")
print(f"{accuracy * 100:.2f}%")


print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)


# =========================================================
# SAVE MODEL
# =========================================================

MODEL_DIR.mkdir(
    exist_ok=True
)

joblib.dump(
    {
        "model": model_pipeline,
        "features": FEATURES,
        "classes": sorted(df["target"].unique())
    },
    MODEL_PATH
)


print("\nModel saved successfully at:")
print(MODEL_PATH)

print("\nTraining completed successfully.")