
import numpy as np
import pandas as pd
import pytest

from app.prediction import load_pipeline, predict_transactions


# ============================================================
# 1. FIXTURE : CHARGEMENT DU MODELE
# ============================================================

@pytest.fixture(scope="module")
def model_artifact():
    """
    Charge le modèle de détection de fraude sauvegardé.
    """
    return load_pipeline("reports/pipeline_fraude.joblib")


# ============================================================
# 2. FIXTURE : DONNEES DE TEST
# ============================================================

@pytest.fixture
def sample_transactions(model_artifact):
    """
    Construit deux transactions avec les colonnes attendues
    par le modèle.
    """
    columns = model_artifact["feature_columns"]

    data = {
        "Time": [10000.0, 20000.0],
        "Amount": [50.0, 150.0],
    }

    for column in columns:
        if column not in data:
            data[column] = [0.0, 0.0]

    return pd.DataFrame(data)[columns]


# ============================================================
# 3. TEST : REPRODUCTIBILITE DES PREDICTIONS
# ============================================================

def test_predictions_are_reproducible(
    model_artifact,
    sample_transactions
):
    """
    Vérifie que les mêmes données produisent les mêmes
    prédictions lors de plusieurs exécutions.
    """
    predictions1, probabilities1 = predict_transactions(
        model_artifact,
        sample_transactions
    )

    predictions2, probabilities2 = predict_transactions(
        model_artifact,
        sample_transactions
    )

    np.testing.assert_array_equal(
        predictions1,
        predictions2
    )

    np.testing.assert_allclose(
        probabilities1,
        probabilities2,
        rtol=0,
        atol=1e-12
    )


# ============================================================
# 4. TEST : STABILITE DES PROBABILITES
# ============================================================

def test_probabilities_are_reproducible(
    model_artifact,
    sample_transactions
):
    """
    Vérifie que les probabilités de fraude restent identiques
    pour des données identiques.
    """
    _, probabilities1 = predict_transactions(
        model_artifact,
        sample_transactions
    )

    _, probabilities2 = predict_transactions(
        model_artifact,
        sample_transactions
    )

    np.testing.assert_allclose(
        probabilities1,
        probabilities2,
        rtol=0,
        atol=1e-12
    )


# ============================================================
# 5. TEST : VALIDITE DU SEUIL DE DECISION
# ============================================================

def test_saved_threshold_is_valid(model_artifact):
    """
    Vérifie que le seuil de décision sauvegardé est valide.
    """
    threshold = model_artifact["threshold"]

    assert isinstance(
        threshold,
        (int, float, np.number)
    )

    assert not isinstance(threshold, (bool, np.bool_))
    assert np.isfinite(threshold)
    assert 0 <= threshold <= 1


# ============================================================
# 6. TEST : PREDICTIONS BINAIRES
# ============================================================

def test_predictions_are_binary(
    model_artifact,
    sample_transactions
):
    """
    Vérifie que les prédictions correspondent uniquement
    aux classes 0 et 1.
    """
    predictions, _ = predict_transactions(
        model_artifact,
        sample_transactions
    )

    assert set(np.asarray(predictions).ravel()).issubset({0, 1})


# ============================================================
# 7. TEST : VALIDITE DES PROBABILITES
# ============================================================

def test_probabilities_remain_between_zero_and_one(
    model_artifact,
    sample_transactions
):
    """
    Vérifie que les probabilités sont finies et comprises
    entre 0 et 1.
    """
    _, probabilities = predict_transactions(
        model_artifact,
        sample_transactions
    )

    probabilities = np.asarray(probabilities)

    assert np.isfinite(probabilities).all()

    assert (
        (probabilities >= 0) &
        (probabilities <= 1)
    ).all()


# ============================================================
# 8. TEST : STABILITE APRES RECHARGEMENT DU MODELE
# ============================================================

def test_predictions_stable_after_reloading(
    model_artifact,
    sample_transactions
):
    """
    Vérifie que le rechargement de l'artefact ne modifie
    pas les prédictions ni les probabilités.
    """
    predictions1, probabilities1 = predict_transactions(
        model_artifact,
        sample_transactions
    )

    reloaded_artifact = load_pipeline(
        "reports/pipeline_fraude.joblib"
    )

    predictions2, probabilities2 = predict_transactions(
        reloaded_artifact,
        sample_transactions
    )

    np.testing.assert_array_equal(
        predictions1,
        predictions2
    )

    np.testing.assert_allclose(
        probabilities1,
        probabilities2,
        rtol=0,
        atol=1e-12
    )
