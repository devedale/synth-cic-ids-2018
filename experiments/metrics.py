import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

def compute_full_metrics(y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray, class_names: list, t0: float) -> dict:
    """
    Returns accuracy, precision, recall, f1, auc_roc, fpr, fnr.
    """
    import time
    exec_time_s = time.time() - t0
    
    acc = accuracy_score(y_true, y_pred)
    
    # Macro averages for multi-class
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    
    # Binary equivalents (Benign=0 vs Attack=1)
    y_true_bin = (y_true > 0).astype(int)
    y_pred_bin = (y_pred > 0).astype(int)
    
    from sklearn.metrics import confusion_matrix
    cm_bin = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
    
    if cm_bin.shape == (2, 2):
        tn, fp, fn, tp = cm_bin.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    else:
        fpr, fnr = 0.0, 0.0
        
    n_classes = len(class_names)
    if n_classes > 2:
        try:
            auc = roc_auc_score(y_true, probs, multi_class='ovr', labels=range(n_classes)) if len(np.unique(y_true)) > 1 else 0.5
        except:
            auc = 0.5
    else:
        # Binary auc
        if probs.shape[1] > 1:
            auc = roc_auc_score(y_true, probs[:, 1]) if len(np.unique(y_true)) > 1 else 0.5
        else:
            auc = 0.5
            
    return {
        "accuracy": acc,
        "precision": p_macro,
        "recall": r_macro,
        "f1": f1_macro,
        "fpr": fpr,
        "fnr": fnr,
        "auc_roc": auc,
        "exec_time_s": exec_time_s
    }
