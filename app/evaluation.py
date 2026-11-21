import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix)

def evaluate_predictions(y_true, predictions, probabilities):
    if len(y_true) == 0:
        raise ValueError("Les données de test sont vides.")
    if not (len(y_true) == len(predictions) == len(probabilities)):
        raise ValueError("Les dimensions des données sont incohérentes.")
    if not np.isfinite(np.asarray(probabilities, dtype=float)).all():
        raise ValueError("Les probabilités contiennent des valeurs non finies.")
    if not np.isin(predictions, [0, 1]).all():
        raise ValueError("Les prédictions doivent être égales à 0 ou 1.")
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    accuracy = accuracy_score(y_true, predictions)
    precision = precision_score(y_true, predictions, zero_division=0)
    recall = recall_score(y_true, predictions, zero_division=0)
    f1 = f1_score(y_true, predictions, zero_division=0)
    if len(np.unique(y_true)) == 2:
        roc_auc = roc_auc_score(y_true, probabilities)
        pr_auc = average_precision_score(y_true, probabilities)
    else:
        roc_auc = np.nan
        pr_auc = np.nan

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    results = {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1-score": f1, "ROC-AUC": roc_auc,
        "PR-AUC (AP)": pr_auc, "FPR": fpr, "FNR": fnr, "TN": tn, "FP": fp, "FN": fn, "TP": tp}
    return results

def generate_report(results):
    report = pd.DataFrame(results.items(), columns=["Métrique", "Valeur"])
    print("\n===== RAPPORT D'ÉVALUATION =====")
    print(report.to_string(index=False))
    return report
