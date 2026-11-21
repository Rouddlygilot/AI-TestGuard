
import joblib
import numpy as np
import pandas as pd


def load_pipeline(filepath):
    """
    Charge le pipeline de prédiction sauvegardé.
    """

    try:
        artifact = joblib.load(filepath)

    except FileNotFoundError:
        raise FileNotFoundError(
            f"Le fichier du modèle est introuvable : {filepath}"
        )

    except Exception as error:
        raise RuntimeError(
            f"Impossible de charger le modèle : {error}"
        )

    if not isinstance(artifact, dict):
        raise ValueError(
            "Le fichier sauvegardé est invalide."
        )

    if "pipeline" not in artifact or "threshold" not in artifact:
        raise ValueError(
            "Le fichier ne contient pas le pipeline et le seuil."
        )

    if "feature_columns" not in artifact:
        raise ValueError(
            "La liste des colonnes attendues est absente."
        )

    return artifact


def predict_transactions(artifact, X):
    """
    Effectue les prédictions sur les transactions.

    Vérifie :
    - Le type des données.
    - Les données vides.
    - Les colonnes dupliquées.
    - Les colonnes manquantes.
    - Les colonnes supplémentaires.
    - Les types de données.
    - Les valeurs infinies ou non numériques.
    """

    # 1. Vérifier le type des données
    if not isinstance(X, pd.DataFrame):
        raise TypeError(
            "Les données doivent être un DataFrame pandas."
        )

    # 2. Vérifier si les données sont vides
    if X.empty:
        raise ValueError(
            "Les données à prédire sont vides."
        )

    # 3. Vérifier les colonnes dupliquées
    if X.columns.duplicated().any():
        raise ValueError(
            "Les données contiennent des colonnes dupliquées."
        )

    # 4. Récupérer les informations du modèle
    pipeline = artifact["pipeline"]
    threshold = artifact["threshold"]
    expected_columns = artifact["feature_columns"]

    # 5. Vérifier les colonnes manquantes
    missing_columns = [
        col for col in expected_columns
        if col not in X.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Colonnes absentes : {missing_columns}"
        )

    # 6. Vérifier les colonnes supplémentaires
    extra_columns = [
        col for col in X.columns
        if col not in expected_columns
    ]

    if extra_columns:
        raise ValueError(
            f"Colonnes supplémentaires non autorisées : {extra_columns}"
        )

    # 7. Respecter l'ordre des colonnes attendu
    X = X[expected_columns].copy()

    # 8. Vérifier que les variables sont numériques
    try:
        values = X.to_numpy(dtype=float)

    except (TypeError, ValueError) as error:
        raise TypeError(
            "Les variables doivent être numériques."
        ) from error

    # 9. Vérifier les valeurs infinies et non finies
    if not np.isfinite(values).all():
        raise ValueError(
            "Les données contiennent des valeurs non finies."
        )

    # 10. Calculer les probabilités
    probabilities = pipeline.predict_proba(X)[:, 1]

    # 11. Appliquer le seuil de décision
    predictions = (
        probabilities >= threshold
    ).astype(int)

    return predictions, probabilities
