"""
scripts/metrics.py
──────────────────
Shared evaluation helpers used by both train.py and evaluate.py.
Centralises the duplicated accuracy / precision / recall / F1 printing logic.
"""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def print_metrics(
    name: str,
    y_true,
    y_pred,
    *,
    extended: bool = False,
    class_names: list[str] | None = None,
) -> dict:
    """Compute and print standard classification metrics.

    Parameters
    ----------
    name : str
        Label shown in the header (e.g. model or split name).
    y_true : array-like
        Ground-truth labels.
    y_pred : array-like
        Predicted labels.
    extended : bool
        If True, also prints the full classification report.
    class_names : list[str] | None
        Optional target-name list passed to classification_report.

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1
    """
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)

    print(f"\n{'─' * 50}")
    print(f"  [{name}] Evaluation results")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-score : {f1:.4f}")
    print(f"  Confusion matrix:\n{confusion_matrix(y_true, y_pred)}")

    if extended:
        print(f"\n  Classification Report:")
        print(
            classification_report(
                y_true, y_pred,
                target_names=class_names,
                zero_division=0,
            )
        )

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
