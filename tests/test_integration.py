import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from app.preprocessing import validate_data
from app.training import (split_data, build_pipeline, train_model, choose_threshold, save_pipeline)
from app.prediction import (load_pipeline, predict_transactions)
from app.evaluation import evaluate_predictions

def create_test_data(n=200):
    rng = np.random.default_rng(42)
    data = {"Time": rng.uniform(0, 100000, n)}
    for i in range(1, 29):
        data[f"V{i}"] = rng.normal(0, 1, n)
    data["Amount"] = rng.uniform(1, 1000, n)
    data["Class"] = [0] * (n - 20) + [1] * 20
    return pd.DataFrame(data)

# séparation des données
def test_data_split():
    df = create_test_data()
    result = split_data(df)
    X_train, X_valid, X_test, y_train, y_valid, y_test = result
    assert len(X_train) + len(X_valid) + len(X_test) == len(df)
    assert len(X_train) > 0
    assert len(X_valid) > 0
    assert len(X_test) > 0

# entraînement du pipeline
def test_pipeline_training():
    df = create_test_data()
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(df)
    pipeline = build_pipeline()
    train_model(pipeline, X_train, y_train)
    predictions = pipeline.predict(X_test)
    assert len(predictions) == len(X_test)

# cohérence des dimensions
def test_dimensions_consistency():
    df = create_test_data()
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(df)
    pipeline = build_pipeline()
    train_model(pipeline, X_train, y_train)
    transformed = pipeline.named_steps["preprocessing"].transform(X_test)
    assert transformed.shape[0] == X_test.shape[0]
    assert transformed.shape[1] == X_test.shape[1]

# absence de valeurs non finies après transformation
def test_no_non_finite_values_after_preprocessing():
    df = create_test_data()
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(df)
    pipeline = build_pipeline()
    train_model(pipeline, X_train, y_train)
    transformed = pipeline.named_steps["preprocessing"].transform(X_test)
    assert np.isfinite(transformed).all()

# scaler est ajusté sur le train
def test_scaler_fitted_only_on_training_data():
    df = create_test_data()
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(df)
    pipeline = build_pipeline()
    train_model(pipeline, X_train, y_train)
    scaler = (pipeline.named_steps["preprocessing"].named_transformers_["scaler"])
    
    expected_mean = X_train[["Time", "Amount"]].mean().to_numpy()
    np.testing.assert_allclose(scaler.mean_, expected_mean, rtol=1e-7, atol=1e-7)

# sauvegarde, rechargement et reproductibilité
def test_save_load_and_prediction_reproducibility(tmp_path):
    df = create_test_data()
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(df)
    pipeline = build_pipeline()
    train_model(pipeline, X_train, y_train)
    threshold = choose_threshold(pipeline, X_valid, y_valid)
    probabilities_before = pipeline.predict_proba(X_test)[:, 1]
    predictions_before = (probabilities_before >= threshold).astype(int)
    
    model_path = tmp_path / "pipeline.joblib"
    save_pipeline(pipeline, threshold, model_path)
    assert model_path.exists()
    artifact = load_pipeline(model_path)
    predictions_after, probabilities_after = predict_transactions(artifact, X_test)

    np.testing.assert_allclose(probabilities_before, probabilities_after, rtol=1e-7, atol=1e-7)
    np.testing.assert_array_equal(predictions_before, predictions_after)

# gestion d'un fichier inexistant
def test_missing_model_file():
    with pytest.raises(FileNotFoundError):
        load_pipeline("modele_inexistant.joblib")

# rejet des données invalides
def test_invalid_input_rejected():
    df = create_test_data()
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(df)
    pipeline = build_pipeline()
    train_model(pipeline, X_train, y_train)
    threshold = choose_threshold(pipeline, X_valid, y_valid)
    artifact = {"pipeline": pipeline, "threshold": threshold, "feature_columns": list(X_train.columns)}
    invalid_data = X_test.copy()
    invalid_data.iloc[0, 0] = np.inf
    with pytest.raises(ValueError):
        predict_transactions(artifact, invalid_data)

# production du rapport de métriques
def test_evaluation_report():
    y_true = np.array([0, 0, 1, 1])
    predictions = np.array([0, 0, 1, 0])
    probabilities = np.array([0.1, 0.2, 0.9, 0.4])
    results = evaluate_predictions(y_true, predictions, probabilities)
    assert "Accuracy" in results
    assert "Precision" in results
    assert "Recall" in results
    assert "F1-score" in results
    assert "ROC-AUC" in results