
import numpy as np
import pandas as pd
import pytest

from app.prediction import load_pipeline, predict_transactions
from app.preprocessing import validate_data
from app.training import (
    build_pipeline,
    choose_threshold,
    save_pipeline,
    split_data,
    train_model,
)

# ==========================================================
# 1. CREATION D'UN JEU DE DONNEES POUR LES TESTS
# ==========================================================

def create_test_data(n=200):

    rng = np.random.default_rng(42)

    data = {
        "Time": rng.uniform(0, 100000, n)
    }

    for i in range(1, 29):
        data[f"V{i}"] = rng.normal(0, 1, n)

    data["Amount"] = rng.uniform(1, 1000, n)
    data["Class"] = [0] * (n - 20) + [1] * 20

    return pd.DataFrame(data)


# ==========================================================
# 2. PREPARATION DU MODELE POUR LES TESTS
# ==========================================================

@pytest.fixture
def trained_artifact(tmp_path):

    df = create_test_data()

    validate_data(df)

    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(df)

    pipeline = build_pipeline()

    train_model(pipeline, X_train, y_train)

    threshold = choose_threshold(pipeline, X_valid, y_valid)

    model_path = tmp_path / "pipeline_robustesse.joblib"

    save_pipeline(pipeline, threshold, model_path)

    artifact = load_pipeline(model_path)

    return artifact, X_test


# ==========================================================
# 3. TEST 1 : VALEURS MANQUANTES
# ==========================================================

def test_missing_values_rejected(trained_artifact):

    artifact, X_test = trained_artifact

    X_invalid = X_test.copy()

    X_invalid.iloc[0, 0] = np.nan

    with pytest.raises(ValueError):

        predict_transactions(artifact, X_invalid)


# ==========================================================
# 4. TEST 2 : VALEURS INFINIES
# ==========================================================

def test_infinite_values_rejected(trained_artifact):

    artifact, X_test = trained_artifact

    X_invalid = X_test.copy()

    X_invalid.iloc[0, 0] = np.inf

    with pytest.raises(ValueError):

        predict_transactions(artifact, X_invalid)


# ==========================================================
# 5. TEST 3 : VALEURS NON NUMERIQUES
# ==========================================================

def test_non_numeric_values_rejected(trained_artifact):

    artifact, X_test = trained_artifact

    X_invalid = X_test.copy()

    X_invalid["Amount"] = X_invalid["Amount"].astype(object)

    X_invalid.iloc[0, X_invalid.columns.get_loc("Amount")] = "inconnu"

    with pytest.raises(TypeError):

        predict_transactions(artifact, X_invalid)


# ==========================================================
# 6. TEST 4 : MONTANT EXTREMEMENT ELEVE
# ==========================================================

def test_extreme_amount_is_processed(trained_artifact):

    artifact, X_test = trained_artifact

    X_extreme = X_test.copy()

    # Montant très élevé, mais toujours numérique et fini
    X_extreme.iloc[0, X_extreme.columns.get_loc("Amount")] = 1_000_000_000

    predictions, probabilities = predict_transactions(
        artifact,
        X_extreme
    )

    assert len(predictions) == len(X_extreme)

    assert np.isfinite(probabilities).all()

    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)

    print("\nMontant extrême traité avec succès.")


# ==========================================================
# 7. TEST 5 : PERTURBATION LIMITEE DES VARIABLES
# ==========================================================

def test_limited_perturbation(trained_artifact):

    artifact, X_test = trained_artifact

    # Probabilités sur les données originales
    _, probabilities_before = predict_transactions(
        artifact,
        X_test
    )

    # Petite perturbation du montant : augmentation de 1 %
    X_perturbed = X_test.copy()

    X_perturbed["Amount"] = X_perturbed["Amount"] * 1.01

    predictions_after, probabilities_after = predict_transactions(
        artifact,
        X_perturbed
    )

    # Mesurer la variation des probabilités
    variation = np.abs(
        probabilities_after - probabilities_before
    )

    # Mesurer le nombre de changements de classe
    _, predictions_before = predict_transactions(
        artifact,
        X_test
    )

    changed_classes = np.sum(
        predictions_before != predictions_after
    )

    assert np.isfinite(probabilities_after).all()

    print("\nPerturbation de Amount : +1 %")
    print("Variation moyenne des probabilités :", variation.mean())
    print("Variation maximale :", variation.max())
    print("Nombre de classes modifiées :", changed_classes)


# ==========================================================
# 8. TEST 6 : DISTRIBUTION DIFFERENTE
# ==========================================================

def test_distribution_shift(trained_artifact):

    artifact, X_test = trained_artifact

    # Probabilités sur les données originales
    _, probabilities_before = predict_transactions(
        artifact,
        X_test
    )

    # Créer un décalage de distribution
    X_shifted = X_test.copy()

    # Décalage temporel d'une journée
    X_shifted["Time"] = X_shifted["Time"] + 86400

    # Augmentation de 100 % des montants
    X_shifted["Amount"] = X_shifted["Amount"] * 2

    predictions, probabilities_after = predict_transactions(
        artifact,
        X_shifted
    )

    variation = np.abs(
        probabilities_after - probabilities_before
    )

    assert np.isfinite(probabilities_after).all()

    assert np.all(probabilities_after >= 0)
    assert np.all(probabilities_after <= 1)

    print("\nTest de décalage de distribution")
    print("Variation moyenne des probabilités :", variation.mean())
    print("Variation maximale :", variation.max())
    print("Nombre de fraudes prédites :", predictions.sum())


# ==========================================================
# 9. TEST 7 : SCHEMA INVALIDE
# ==========================================================

def test_invalid_schema_rejected(trained_artifact):

    artifact, X_test = trained_artifact

    # Supprimer une colonne obligatoire
    X_invalid = X_test.drop(columns=["V1"])

    with pytest.raises(ValueError):

        predict_transactions(artifact, X_invalid)


# ==========================================================
# 10. TEST 8 : PROBABILITES VALIDES
# ==========================================================

def test_prediction_probabilities_are_valid(trained_artifact):

    artifact, X_test = trained_artifact

    predictions, probabilities = predict_transactions(
        artifact,
        X_test
    )

    assert np.isfinite(probabilities).all()

    assert np.all(probabilities >= 0)

    assert np.all(probabilities <= 1)

    assert len(predictions) == len(X_test)
