
import joblib
import numpy as np
import pandas as pd
import pytest

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.prediction import load_pipeline, predict_transactions


# ============================================================
# 1. FIXTURES : CRÉATION DU MODÈLE ET DES DONNÉES DE TEST
# ============================================================

@pytest.fixture
def sample_data():
    """Crée un petit jeu de données pour les tests."""

    X = pd.DataFrame({
        "Time": [10, 20, 30, 40, 50, 60, 70, 80],
        "V1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        "Amount": [100, 200, 150, 300, 250, 400, 350, 500]
    })

    y = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])

    return X, y


@pytest.fixture
def sample_pipeline(sample_data):
    """Construit et entraîne un pipeline de test."""

    X, y = sample_data

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(random_state=42))
    ])

    pipeline.fit(X, y)

    return pipeline


@pytest.fixture
def sample_artifact(sample_pipeline):
    """Crée un dictionnaire représentant le modèle sauvegardé."""

    return {
        "pipeline": sample_pipeline,
        "threshold": 0.5,
        "feature_columns": ["Time", "V1", "Amount"]
    }


@pytest.fixture
def sample_X(sample_data):
    """Retourne une copie des données pour les prédictions."""

    X, _ = sample_data

    return X.copy()


@pytest.fixture
def saved_model(tmp_path, sample_artifact):
    """Sauvegarde un modèle temporaire."""

    filepath = tmp_path / "pipeline_test.joblib"

    joblib.dump(sample_artifact, filepath)

    return filepath


# ============================================================
# 2. TESTS DE LA FONCTION load_pipeline()
# ============================================================

def test_load_pipeline_success(saved_model):
    """Vérifie le chargement correct du modèle."""

    artifact = load_pipeline(saved_model)

    assert isinstance(artifact, dict)
    assert "pipeline" in artifact
    assert "threshold" in artifact
    assert "feature_columns" in artifact

    assert artifact["threshold"] == 0.5


def test_load_pipeline_file_not_found(tmp_path):
    """Vérifie le comportement si le fichier est absent."""

    filepath = tmp_path / "fichier_inexistant.joblib"

    with pytest.raises(FileNotFoundError):
        load_pipeline(filepath)


def test_load_pipeline_invalid_file(tmp_path):
    """Vérifie le comportement avec un fichier invalide."""

    filepath = tmp_path / "fichier_invalide.joblib"

    filepath.write_text(
        "contenu invalide",
        encoding="utf-8"
    )

    with pytest.raises(RuntimeError):
        load_pipeline(filepath)


def test_load_pipeline_invalid_artifact(tmp_path):
    """Vérifie que le contenu doit être un dictionnaire."""

    filepath = tmp_path / "artifact_invalide.joblib"

    joblib.dump(["pipeline", "threshold"], filepath)

    with pytest.raises(TypeError):
        load_pipeline(filepath)


def test_load_pipeline_missing_keys(tmp_path, sample_pipeline):
    """Vérifie la présence de toutes les clés obligatoires."""

    artifact = {
        "pipeline": sample_pipeline,
        "threshold": 0.5
    }

    filepath = tmp_path / "artifact_incomplet.joblib"

    joblib.dump(artifact, filepath)

    with pytest.raises(ValueError, match="Éléments manquants"):
        load_pipeline(filepath)


def test_load_pipeline_feature_columns_not_list(
    tmp_path,
    sample_pipeline
):
    """Vérifie que les colonnes attendues doivent être une liste."""

    artifact = {
        "pipeline": sample_pipeline,
        "threshold": 0.5,
        "feature_columns": "Time, V1, Amount"
    }

    filepath = tmp_path / "colonnes_invalides.joblib"

    joblib.dump(artifact, filepath)

    with pytest.raises(TypeError):
        load_pipeline(filepath)


def test_load_pipeline_non_string_columns(
    tmp_path,
    sample_pipeline
):
    """Vérifie que les noms de colonnes doivent être des chaînes."""

    artifact = {
        "pipeline": sample_pipeline,
        "threshold": 0.5,
        "feature_columns": ["Time", 123, "Amount"]
    }

    filepath = tmp_path / "colonnes_non_string.joblib"

    joblib.dump(artifact, filepath)

    with pytest.raises(TypeError):
        load_pipeline(filepath)


def test_load_pipeline_duplicate_columns(
    tmp_path,
    sample_pipeline
):
    """Vérifie le rejet des colonnes dupliquées."""

    artifact = {
        "pipeline": sample_pipeline,
        "threshold": 0.5,
        "feature_columns": ["Time", "V1", "Time"]
    }

    filepath = tmp_path / "colonnes_dupliquees.joblib"

    joblib.dump(artifact, filepath)

    with pytest.raises(ValueError, match="doublons"):
        load_pipeline(filepath)


@pytest.mark.parametrize(
    "threshold",
    ["0.5", None, True, False, {"value": 0.5}]
)
def test_load_pipeline_invalid_threshold_type(
    tmp_path,
    sample_pipeline,
    threshold
):
    """Vérifie le rejet des seuils non numériques ou booléens."""

    artifact = {
        "pipeline": sample_pipeline,
        "threshold": threshold,
        "feature_columns": ["Time", "V1", "Amount"]
    }

    filepath = tmp_path / "seuil_invalide.joblib"

    joblib.dump(artifact, filepath)

    with pytest.raises(TypeError):
        load_pipeline(filepath)


@pytest.mark.parametrize(
    "threshold",
    [-0.1, 1.1, np.nan, np.inf, -np.inf]
)
def test_load_pipeline_threshold_out_of_range(
    tmp_path,
    sample_pipeline,
    threshold
):
    """Vérifie que le seuil doit être fini et compris entre 0 et 1."""

    artifact = {
        "pipeline": sample_pipeline,
        "threshold": threshold,
        "feature_columns": ["Time", "V1", "Amount"]
    }

    filepath = tmp_path / "seuil_hors_limites.joblib"

    joblib.dump(artifact, filepath)

    with pytest.raises(ValueError):
        load_pipeline(filepath)


def test_load_pipeline_without_predict_proba(tmp_path):
    """Vérifie que le pipeline doit posséder predict_proba."""

    artifact = {
        "pipeline": object(),
        "threshold": 0.5,
        "feature_columns": ["Time", "V1", "Amount"]
    }

    filepath = tmp_path / "pipeline_sans_predict_proba.joblib"

    joblib.dump(artifact, filepath)

    with pytest.raises(TypeError, match="predict_proba"):
        load_pipeline(filepath)


# ============================================================
# 3. TESTS DE LA FONCTION predict_transactions()
# ============================================================

def test_predict_transactions_success(sample_artifact, sample_X):
    """Vérifie que les prédictions sont correctement réalisées."""

    predictions, probabilities = predict_transactions(
        sample_artifact,
        sample_X
    )

    assert len(predictions) == len(sample_X)
    assert len(probabilities) == len(sample_X)

    assert set(np.unique(predictions)).issubset({0, 1})

    assert np.all(np.isfinite(probabilities))
    assert np.all((probabilities >= 0) & (probabilities <= 1))


def test_predict_transactions_output_types(sample_artifact, sample_X):
    """Vérifie les types des résultats."""

    predictions, probabilities = predict_transactions(
        sample_artifact,
        sample_X
    )

    assert isinstance(predictions, np.ndarray)
    assert isinstance(probabilities, np.ndarray)


def test_predict_transactions_threshold_application(
    sample_artifact,
    sample_X
):
    """Vérifie l'application du seuil de décision."""

    predictions, probabilities = predict_transactions(
        sample_artifact,
        sample_X
    )

    expected_predictions = (
        probabilities >= sample_artifact["threshold"]
    ).astype(int)

    np.testing.assert_array_equal(
        predictions,
        expected_predictions
    )


def test_predict_transactions_reordered_columns(
    sample_artifact,
    sample_X
):
    """Vérifie que l'ordre initial des colonnes n'est pas obligatoire."""

    X_reordered = sample_X[
        ["Amount", "Time", "V1"]
    ]

    predictions, probabilities = predict_transactions(
        sample_artifact,
        X_reordered
    )

    assert len(predictions) == len(sample_X)
    assert len(probabilities) == len(sample_X)


def test_predict_transactions_invalid_artifact(sample_X):
    """Vérifie le rejet d'un modèle qui n'est pas un dictionnaire."""

    with pytest.raises(TypeError):
        predict_transactions(None, sample_X)


def test_predict_transactions_missing_keys(sample_X):
    """Vérifie les clés obligatoires du modèle."""

    artifact = {
        "pipeline": None
    }

    with pytest.raises(ValueError):
        predict_transactions(artifact, sample_X)


def test_predict_transactions_invalid_dataframe(sample_artifact):
    """Vérifie que les données doivent être un DataFrame."""

    with pytest.raises(TypeError):
        predict_transactions(sample_artifact, [1, 2, 3])


def test_predict_transactions_empty_dataframe(sample_artifact):
    """Vérifie le rejet d'un DataFrame vide."""

    X_empty = pd.DataFrame(
        columns=["Time", "V1", "Amount"]
    )

    with pytest.raises(ValueError, match="vides"):
        predict_transactions(sample_artifact, X_empty)


def test_predict_transactions_duplicate_columns(
    sample_artifact,
    sample_X
):
    """Vérifie le rejet des colonnes dupliquées."""

    X_duplicate = sample_X.copy()

    X_duplicate.columns = [
        "Time",
        "V1",
        "Time"
    ]

    with pytest.raises(ValueError, match="dupliquées"):
        predict_transactions(sample_artifact, X_duplicate)


def test_predict_transactions_missing_columns(
    sample_artifact,
    sample_X
):
    """Vérifie le rejet des colonnes manquantes."""

    X_missing = sample_X.drop(columns=["V1"])

    with pytest.raises(ValueError, match="Colonnes absentes"):
        predict_transactions(sample_artifact, X_missing)


def test_predict_transactions_extra_columns(
    sample_artifact,
    sample_X
):
    """Vérifie le rejet des colonnes supplémentaires."""

    X_extra = sample_X.copy()

    X_extra["Extra"] = 100

    with pytest.raises(ValueError, match="Colonnes supplémentaires"):
        predict_transactions(sample_artifact, X_extra)


def test_predict_transactions_non_numeric_values(
    sample_artifact,
    sample_X
):
    """Vérifie le rejet des valeurs non numériques."""

    X_invalid = sample_X.copy()

    X_invalid["Amount"] = "valeur_invalide"

    with pytest.raises(TypeError, match="numériques"):
        predict_transactions(sample_artifact, X_invalid)


@pytest.mark.parametrize(
    "invalid_value",
    [np.nan, np.inf, -np.inf]
)
def test_predict_transactions_non_finite_values(
    sample_artifact,
    sample_X,
    invalid_value
):
    """
    Vérifie le rejet des valeurs manquantes ou infinies.

    La conversion en float est nécessaire pour permettre à Pandas
    d'accepter les valeurs inf et -inf dans la colonne Amount.
    """

    X_invalid = sample_X.copy()

    # Correction : convertir la colonne entière en float
    X_invalid["Amount"] = X_invalid["Amount"].astype(float)

    # Injection de la valeur invalide
    X_invalid.loc[0, "Amount"] = invalid_value

    # La fonction de prédiction doit rejeter ces valeurs
    with pytest.raises(ValueError, match="non finies"):
        predict_transactions(sample_artifact, X_invalid)


def test_predict_transactions_invalid_threshold(
    sample_artifact,
    sample_X
):
    """Vérifie que le seuil est contrôlé avant la prédiction."""

    invalid_artifact = sample_artifact.copy()

    invalid_artifact["threshold"] = 2.0

    with pytest.raises(ValueError, match="compris entre 0 et 1"):
        predict_transactions(invalid_artifact, sample_X)


def test_predict_transactions_invalid_threshold_type(
    sample_artifact,
    sample_X
):
    """Vérifie le type du seuil au moment de la prédiction."""

    invalid_artifact = sample_artifact.copy()

    invalid_artifact["threshold"] = "0.5"

    with pytest.raises(TypeError, match="numérique"):
        predict_transactions(invalid_artifact, sample_X)


def test_predict_transactions_invalid_feature_columns(
    sample_artifact,
    sample_X
):
    """Vérifie le format de la liste des colonnes attendues."""

    invalid_artifact = sample_artifact.copy()

    invalid_artifact["feature_columns"] = "Time, V1, Amount"

    with pytest.raises(TypeError):
        predict_transactions(invalid_artifact, sample_X)


def test_predict_transactions_non_finite_probabilities(
    sample_artifact,
    sample_X
):
    """Vérifie que les probabilités produites sont valides."""

    predictions, probabilities = predict_transactions(
        sample_artifact,
        sample_X
    )

    assert np.isfinite(probabilities).all()
    assert (probabilities >= 0).all()
    assert (probabilities <= 1).all()
