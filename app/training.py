import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def split_data(df):
    X = df.drop(columns=["Class"])
    y = df["Class"]
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)

    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)
    return X_train, X_valid, X_test, y_train, y_valid, y_test

def build_pipeline():
    preprocessing = ColumnTransformer(transformers=[("scaler", StandardScaler(), ["Time", "Amount"])],
        remainder="passthrough")

    pipeline = Pipeline(steps=[("preprocessing", preprocessing),("model",
                LogisticRegression(max_iter=1000,class_weight="balanced",random_state=42))])
    return pipeline

def train_model(pipeline, X_train, y_train):
    pipeline.fit(X_train, y_train)
    return pipeline

def choose_threshold(pipeline, X_valid, y_valid):
    probabilities = pipeline.predict_proba(X_valid)[:, 1]
    thresholds = np.arange(0.01, 1.00, 0.01)
    valid_thresholds = []
    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_valid, predictions, labels=[0, 1]).ravel()
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        if recall >= 0.80 and fpr <= 0.01:
            valid_thresholds.append((threshold, precision, recall, fpr))
    if valid_thresholds:
        best = max(valid_thresholds, key=lambda x: x[1])
        return float(best[0])
    best_threshold = 0.5
    best_f1 = -1
    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_valid, predictions, labels=[0, 1]).ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return float(best_threshold)

def save_pipeline(pipeline, threshold, filepath):
    artifact = {"pipeline": pipeline, "threshold": threshold, "feature_columns": list(pipeline.feature_names_in_)}
    joblib.dump(artifact, filepath)
