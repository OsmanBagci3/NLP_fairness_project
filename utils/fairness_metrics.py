"""
fairness_metrics.py — Fonctions partagées pour l'analyse de fairness.
Utilisé par les notebooks 06, 07, 08 (et au-delà).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve


def compute_fairness_metrics(y_true, y_pred, genders, classes):
    """Calcule DP, EO, EqOdds et accuracy par groupe."""
    groups = np.unique(genders)
    out = {"accuracy": accuracy_score(y_true, y_pred)}
    for g in groups:
        m = genders == g
        out[f"accuracy_{g.lower()}"] = accuracy_score(y_true[m], y_pred[m])

    dp_gaps, eo_gaps, eq_gaps = [], [], []
    for cid in range(len(classes)):
        rates, tprs, fprs = {}, {}, {}
        for g in groups:
            m = genders == g
            rates[g] = np.mean(y_pred[m] == cid)
            pm = m & (y_true == cid)
            nm = m & (y_true != cid)
            tprs[g] = np.mean(y_pred[pm] == cid) if pm.sum() > 0 else 0.0
            fprs[g] = np.mean(y_pred[nm] == cid) if nm.sum() > 0 else 0.0
        if len(groups) == 2:
            g0, g1 = groups[0], groups[1]
            dp_gaps.append(abs(rates[g0] - rates[g1]))
            eo_gaps.append(abs(tprs[g0] - tprs[g1]))
            eq_gaps.append(max(abs(tprs[g0]-tprs[g1]), abs(fprs[g0]-fprs[g1])))

    out["delta_dp"]      = float(np.mean(dp_gaps))
    out["delta_eo"]      = float(np.mean(eo_gaps))
    out["delta_eqodds"]  = float(np.mean(eq_gaps))
    out["mean_gap"]      = float(np.mean([out["delta_dp"], out["delta_eo"], out["delta_eqodds"]]))
    return out


def compute_predictive_parity(y_true, y_pred, genders, classes):
    """Predictive Parity : |PPV_g0 - PPV_g1| par classe."""
    groups = np.unique(genders)
    results = {}
    for cid, cls in enumerate(classes):
        ppvs = {}
        for g in groups:
            m = genders == g
            pred_pos = (y_pred[m] == cid)
            if pred_pos.sum() > 0:
                ppvs[g] = np.mean(y_true[m][pred_pos] == cid)
            else:
                ppvs[g] = float("nan")
        if len(groups) == 2:
            g0, g1 = groups[0], groups[1]
            gap = abs(ppvs.get(g0, 0) - ppvs.get(g1, 0))
            results[cls] = {"ppv_" + g0.lower(): ppvs.get(g0), "ppv_" + g1.lower(): ppvs.get(g1), "gap": gap}
    return results


def bootstrap_ci(y_true, y_pred, genders, classes, metric="delta_eo", n_iter=1000, ci=95):
    """Bootstrap confidence interval pour une métrique de fairness."""
    n = len(y_true)
    samples = []
    for _ in range(n_iter):
        idx = np.random.choice(n, n, replace=True)
        fm = compute_fairness_metrics(y_true[idx], y_pred[idx], genders[idx], classes)
        samples.append(fm[metric])
    lo = np.percentile(samples, (100 - ci) / 2)
    hi = np.percentile(samples, 100 - (100 - ci) / 2)
    return float(np.mean(samples)), lo, hi


def roc_by_group(y_true, y_proba, genders, classes):
    """Courbes ROC OvR par groupe de genre."""
    groups = np.unique(genders)
    results = {}
    for cid, cls in enumerate(classes):
        results[cls] = {}
        for g in groups:
            m = genders == g
            if m.sum() == 0:
                continue
            yt = (y_true[m] == cid).astype(int)
            yp = y_proba[m][:, cid]
            if yt.sum() == 0 or yt.sum() == m.sum():
                continue
            fpr, tpr, _ = roc_curve(yt, yp)
            auc = roc_auc_score(yt, yp)
            results[cls][g] = {"fpr": fpr, "tpr": tpr, "auc": auc}
    return results


def find_fair_thresholds(y_true, y_proba, genders, classes, criterion="dp"):
    """
    Cherche les seuils de décision par groupe minimisant les gaps de fairness.
    criterion: 'dp' (demographic parity) ou 'eo' (equality of opportunity).
    Retourne un dict {class: {group: threshold}}.
    """
    groups = np.unique(genders)
    thresholds = {}
    for cid, cls in enumerate(classes):
        thresholds[cls] = {}
        target_rates = []
        for g in groups:
            m = genders == g
            if criterion == "dp":
                rate = np.mean((y_proba[m][:, cid] > 0.5).astype(int))
            else:  # eo
                pos_m = m & (y_true == cid)
                rate = np.mean(y_proba[pos_m][:, cid]) if pos_m.sum() > 0 else 0.5
            target_rates.append(rate)
        target = np.mean(target_rates)  # taux cible = moyenne des groupes
        for g in groups:
            m = genders == g
            probs = np.sort(y_proba[m][:, cid])
            # Trouver le seuil qui donne un taux proche de target
            idx = int((1 - target) * len(probs))
            idx = max(0, min(len(probs)-1, idx))
            thresholds[cls][g] = float(probs[idx])
    return thresholds
