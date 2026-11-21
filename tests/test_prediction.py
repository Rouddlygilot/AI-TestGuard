
import joblib
import numpy as np
import pandas as pd


def load_pipeline(filepath):
    """
    Charge le modèle sauvegardé depuis un fichier.

    Args:
        filepath: Chemin vers le fichier du modèle.

    Returns:
        Un dictionnaire contenant le pipeline, le seuil
        de décision et les colonnes attendues.

    Raises:
        FileNotFoundError: Si le fichier est introuvable.
        RuntimeError: Si le chargement du fichier échoue.
        TypeError: Si le contenu ou ses éléments sont de types incorrects.
        ValueError: Si des éléments obligatoires sont absents
                    ou si le seuil est invalide.
    """
    try:
        artifact = joblib.load(filepath)

    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Le fichier du modèle est introuvable : {filepath}"
        ) from error

    except (OSError, EOFError, ImportError, AttributeError) as error:
        raise RuntimeError(
            f"Impossible de charger le modèle : {error}"
        ) from error

    if not isinstance(artifact, dict):
        raise TypeError(
            "Le fichier sauvegardé doit contenir un dictionnaire."
        )

    required_keys = {"pipeline", "threshold", "feature_columns"}
    missing_keys = required_keys - artifact.keys()

    if missing_keys:
        raise ValueError(
            f"Éléments manquants dans le fichier : {sorted(missing_keys)}"
        )

    expected_columns = artifact["feature_columns"]

    if not isinstance(expected_columns, list):
        raise TypeError(
            "Les colonnes attendues doivent être fournies sous forme de liste."
        )

    if not all(isinstance(col, str) for col in expected_columns):
        raise TypeError(
            "Les noms des colonnes doivent être des chaînes de caractères."
        )

    if len(set(expected_columns)) != len(expected_columns):
        raise ValueError(
            "La liste des colonnes attendues contient des doublons."
        )

    threshold = artifact["threshold"]

    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise TypeError(
            "Le seuil de décision doit être numérique."
        )

    threshold = float(threshold)

    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError(
            "Le seuil de décision doit être compris entre 0 et 1."
        )

    if not hasattr(artifact["pipeline"], "predict_proba"):
        raise TypeError(
            "Le pipeline doit posséder une méthode predict_proba."
        )

    artifact["threshold"] = threshold

    return artifact


def predict_transactions(artifact, X):
    """
    Prédit les transactions frauduleuses.

    Args:
        artifact: Dictionnaire contenant le pipeline et le seuil.
        X: DataFrame pandas contenant les variables explicatives.

    Returns:
        predictions: Classes prédites (0 ou 1).
        probabilities: Probabilités de fraude.

    Raises:
        TypeError: Si les types des données sont incorrects.
        ValueError: Si les données sont invalides.
    """
    if not isinstance(artifact, dict):
        raise TypeError(
            "Le modèle sauvegardé doit être un dictionnaire."
        )

    required_keys = {"pipeline", "threshold", "feature_columns"}

    if not required_keys.issubset(artifact):
        raise ValueError(
            "Le dictionnaire du modèle ne contient pas tous les éléments requis."
        )

    if not isinstance(X, pd.DataFrame):
        raise TypeError(
            "Les données doivent être un DataFrame pandas."
        )

    if X.empty:
        raise ValueError(
            "Les données à prédire sont vides."
        )

    if X.columns.duplicated().any():
        raise ValueError(
            "Les données contiennent des colonnes dupliquées."
        )

    pipeline = artifact["pipeline"]
    threshold = artifact["threshold"]
    expected_columns = artifact["feature_columns"]

    if not isinstance(expected_columns, list):
        raise TypeError(
            "Les colonnes attendues doivent être fournies sous forme de liste."
        )

    if not all(isinstance(col, str) for col in expected_columns):
        raise TypeError(
            "Les noms des colonnes doivent être des chaînes de caractères."
        )

    if len(set(expected_columns)) != len(expected_columns):
        raise ValueError(
            "La liste des colonnes attendues contient des doublons."
        )

    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise TypeError(
            "Le seuil de décision doit être numérique."
        )

    threshold = float(threshold)

    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError(
            "Le seuil de décision doit être compris entre 0 et 1."
        )

    if not hasattr(pipeline, "predict_proba"):
        raise TypeError(
            "Le pipeline doit posséder une méthode predict_proba."
        )

    missing_columns = [
        col for col in expected_columns if col not in X.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Colonnes absentes : {missing_columns}"
        )

    extra_columns = [
        col for col in X.columns if col not in expected_columns
    ]

    if extra_columns:
        raise ValueError(
            f"Colonnes supplémentaires : {extra_columns}"
        )

    X = X[expected_columns].copy()

    try:
        values = X.to_numpy(dtype=float)

    except (TypeError, ValueError) as error:
        raise TypeError(
            "Les variables doivent être numériques."
        ) from error

    if not np.isfinite(values).all():
        raise ValueError(
            "Les données contiennent des valeurs non finies."
        )

    probabilities = pipeline.predict_proba(X)[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    return predictions, probabilities
