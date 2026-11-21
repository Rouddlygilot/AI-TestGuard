import pandas as pd
import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler
from app.preprocessing import (REQUIRED_COLUMNS, validate_data, preprocess_data)

# Création d'un echantillon de jeu de données valide pour les tests
def create_valid_data(n=3):
    data = {
        col: [float(i + 1) for i in range(n)]
        for col in REQUIRED_COLUMNS
    }
    data["Class"] = [0, 1, 0][:n]
    return pd.DataFrame(data)

# TEST 1 : présence de toutes les colonnes requises
def test_required_columns_present():
    df = create_valid_data()

    assert validate_data(df) is True

# TEST 2 : colonne absente
def test_missing_column():
    df = create_valid_data()
    df = df.drop(columns=["Amount"])

    with pytest.raises(ValueError, match="Colonnes absentes"):
        validate_data(df)

# TEST 3 : colonne dupliquée
def test_duplicate_column():
    df = create_valid_data()
    df.columns = list(df.columns[:-1]) + ["Amount"]

    with pytest.raises(ValueError, match="colonnes dupliquées"):
        validate_data(df)

# TEST 4 : présence de valeurs manquantes
def test_missing_values():
    df = create_valid_data()
    df.loc[0, "Amount"] = float("nan")

    with pytest.raises(ValueError, match="valeurs manquantes"):
        validate_data(df)

# TEST 5 : présence de valeurs infinies
def test_infinite_values():
    df = create_valid_data()
    df.loc[0, "Amount"] = float("inf")

    with pytest.raises(ValueError, match="valeurs infinies"):
        validate_data(df)

# TEST 6 : jeu de données vide
def test_empty_data():
    df = create_valid_data().iloc[0:0]

    with pytest.raises(ValueError, match="vide"):
        validate_data(df)

# TEST 7 : type incorrect d'une variable
def test_incorrect_variable_type():
    df = create_valid_data()
    df["Amount"] = df["Amount"].astype(str)

    with pytest.raises(TypeError, match="doit être numérique"):
        validate_data(df)

# TEST 8 : valeurs de cible différentes de 0 et 1
def test_invalid_target_values():
    df = create_valid_data()
    df.loc[0, "Class"] = 2

    with pytest.raises(ValueError, match="uniquement 0 et 1"):
        validate_data(df)

# TEST 9 : traitement d'une seule observation
def test_single_observation():
    df = create_valid_data(n=1)

    X, y = preprocess_data(df)

    assert X.shape == (1, 30)
    assert len(y) == 1

# TEST 10 : cohérence des dimensions après transformation
def test_dimensions_after_transformation():
    df = create_valid_data(n=3)
    X, y = preprocess_data(df)
    assert X.shape[0] == df.shape[0]
    assert X.shape[1] == df.shape[1] - 1
    assert len(y) == df.shape[0]

 # TEST 11 : vérification de la standardisation des variables Time et Amount
def test_standard_scaler():
    data = pd.DataFrame({
        "Amount": [100.0, 200.0, 300.0]
    })

    # Application de la standardisation
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)

    # Vérification des propriétés de la transformation
    assert np.isfinite(scaled).all()
    assert abs(scaled.mean()) < 1e-10
    assert np.isclose(scaled.std(), 1.0)

# TEST 12 : une valeur limite, montant égal à zéro
def test_zero_amount():
    df = create_valid_data(n=3)
    df["Amount"] = [0.0, 100.0, 200.0]
    X, y = preprocess_data(df)
    assert X["Amount"].notna().all()
    assert np.isfinite(X["Amount"]).all()
    assert len(y) == 3


# TEST 13 : montant constant
def test_constant_amount():
    df = create_valid_data(n=3)
    df["Amount"] = [100.0, 100.0, 100.0]
    X, y = preprocess_data(df)
    assert np.allclose(X["Amount"], 0.0)
    assert len(y) == 3

# TEST 14 : vérifier le résultat de la standardisation réelle
def test_preprocessing_scaling_result():
    df = create_valid_data(n=3)
    X, y = preprocess_data(df)
    assert abs(X["Time"].mean()) < 1e-10
    assert abs(X["Amount"].mean()) < 1e-10
    assert np.isclose(X["Time"].std(ddof=0), 1.0)
    assert np.isclose(X["Amount"].std(ddof=0), 1.0)
    assert X.shape == (3, 30)
    assert len(y) == 3
