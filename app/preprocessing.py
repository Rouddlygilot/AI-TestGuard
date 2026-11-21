import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# Liste des colonnes attendues dans le jeu de données
REQUIRED_COLUMNS = (
    ["Time"]
    + [f"V{i}" for i in range(1, 29)]
    + ["Amount", "Class"]
)


def validate_data(df):
    """
    Vérifie que le jeu de données respecte les conditions attendues.
    """

    # 1. Vérifier que le jeu de données n'est pas vide
    if df.empty:
        raise ValueError("Le jeu de données est vide.")

    # 2. Vérifier les colonnes dupliquées
    if df.columns.duplicated().any():
        raise ValueError("Le jeu de données contient des colonnes dupliquées.")

    # 3. Vérifier la présence de toutes les colonnes requises
    missing_columns = [
        col for col in REQUIRED_COLUMNS if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Colonnes absentes : {missing_columns}")

    # 4. Vérifier les valeurs manquantes
    if df[REQUIRED_COLUMNS].isnull().any().any():
        raise ValueError("Le jeu de données contient des valeurs manquantes.")

    # 5. Vérifier que les variables ont un type numérique
    for col in REQUIRED_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(f"La colonne {col} doit être numérique.")

    # 6. Vérifier les valeurs infinies
    values = df[REQUIRED_COLUMNS].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("Le jeu de données contient des valeurs infinies.")

    # 7. Vérifier les valeurs de la variable cible
    if not df["Class"].isin([0, 1]).all():
        raise ValueError("La variable cible Class doit contenir uniquement 0 et 1.")

    return True


def preprocess_data(df):
    """
    Valide les données et transforme les variables Time et Amount.
    Retourne les variables explicatives X et la cible y.
    """

    validate_data(df)

    X = df.drop(columns=["Class"]).copy()
    y = df["Class"].copy()

    # Standardisation des variables Time et Amount
    scaler = StandardScaler()
    X[["Time", "Amount"]] = scaler.fit_transform(
        X[["Time", "Amount"]]
    )

    return X, y