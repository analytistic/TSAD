import numpy as np
from tqdm import tqdm


"""
--- threshold dependent ---
"""

def metric_prec_recall_curve(labels, score):
    idx = np.argsort(score)[::-1]
    thr = score[idx]    
    tpr = np.cumsum(labels[idx]) / np.sum(labels)
    prec = np.cumsum(labels[idx]) / np.arange(1, len(labels) + 1)
    return prec, tpr, thr

def metric_PointF1(labels, preds):
    prec = np.sum((labels == 1) & (preds == 1)) / (np.sum(preds) + np.finfo(float).eps)
    recall = np.sum((labels == 1) & (preds == 1)) / (np.sum(labels) + np.finfo(float).eps)
    f1 = 2 * prec * recall / (prec + recall + np.finfo(float).eps)
    return f1

















"""
--- threshold independent ---

"""


def _range_convers_new(label):
    '''
    input: arrays of binary values
    output: list of ordered pair [[a0,b0], [a1,b1]... ] of the inputs
    '''
    anomaly_starts = np.where(np.diff(label) == 1)[0] + 1
    anomaly_ends, = np.where(np.diff(label) == -1)
    if len(anomaly_ends):
        if not len(anomaly_starts) or anomaly_ends[0] < anomaly_starts[0]:
            # we started with an anomaly, so the start of the first anomaly is the start of the labels
            anomaly_starts = np.concatenate([[0], anomaly_starts])
    if len(anomaly_starts):
        if not len(anomaly_ends) or anomaly_ends[-1] < anomaly_starts[-1]:
            # we ended on an anomaly, so the end of the last anomaly is the end of the labels
            anomaly_ends = np.concatenate([anomaly_ends, [len(label) - 1]])
    return list(zip(anomaly_starts, anomaly_ends))

def _new_sequence(label, sequence_original, window):
    a = max(sequence_original[0][0] - window // 2, 0)
    sequence_new = []
    for i in range(len(sequence_original) - 1):
        if sequence_original[i][1] + window // 2 < sequence_original[i + 1][0] - window // 2:
            sequence_new.append((a, sequence_original[i][1] + window // 2))
            a = sequence_original[i + 1][0] - window // 2
    sequence_new.append((a, min(sequence_original[len(sequence_original) - 1][1] + window // 2, len(label) - 1)))
    return sequence_new

def _smooth_sequence(label, pos_range, window):
    """
    smooth and extent the bounds of the pos labels range by window
    
    """
    label = label.copy().astype(float)
    length = len(label)
    for range in pos_range:
        s = range[0]
        e = range[1]
        range_right_expand = np.arange(e + 1, min(e + window // 2 + 1, length))
        range_left_expand = np.arange(max(s - window // 2, 0), s)
        label[range_right_expand] += np.sqrt(1 - (range_right_expand - e)/(window))
        label[range_left_expand] += np.sqrt(1 - (s - range_left_expand)/(window))
    label = np.minimum(np.ones(length), label) # cap the label to 1
    return label



def metric_AUC_ROC(labels, score):
    """
    AUC_ROC = 1/2 · Σ (TPRᵢ₋₁ + TPRᵢ) · ΔFPRᵢ
    ΔFPRᵢ = FPRᵢ - FPRᵢ₋₁
    """
    idx = np.argsort(score)[::-1]
    thr = score[idx]    
    tpr = np.cumsum(labels[idx]) / np.sum(labels)
    fpr = np.cumsum(1 - labels[idx]) / np.sum(1 - labels)
    auc_roc = np.trapezoid(tpr, fpr)
    return auc_roc

def metric_AUC_PR(labels, score):
    """
    AUC_PR = 1/2 · Σ (Pᵢ₋₁ + Pᵢ) · ΔTPRᵢ
    ΔTPRᵢ = TPRᵢ - TPRᵢ₋₁
    And most paper use AP to approximate AUC_PR, which is defined as:
    AP = Σ Pᵢ · ΔTPRᵢ
    """
    idx = np.argsort(score)[::-1]
    thr = score[idx]    
    tpr = np.cumsum(labels[idx]) / np.sum(labels)
    prec = np.cumsum(labels[idx]) / np.arange(1, len(labels) + 1)
    # auc_pr = np.trapezoid(
    #     prec, tpr
    # )     
    auc_ap = sum(prec*np.diff(np.concatenate([[0], tpr])))
    return auc_ap   

def metric_VUS(labels, score, window=1, sample=250):
    """
    https://www.vldb.org/pvldb/vol15/p2774-paparrizos.pdf
    Compute Volume Under the  Volume Under the Surface (VUS) for ROC and PR curves under
    a set of buffer regions at the boundary of outlier (Range-AUC-ROC and Range-AUC-PR)
    ## VUS-PR (Equation 18)
    .. math::
        VUS\\text{-}PR = \\frac{1}{4} \\sum_{w=1}^{L} \\sum_{k=1}^{N} \\Delta^{(k,w)} * \\Delta^{w}

    where:
        - :math:`\\Delta^{(k,w)} = \\Delta^{k}_{Pr_{\\ell_w}} * \\Delta^{k}_{Re_{\\ell_w}} + \\Delta^{k}_{Pr_{\\ell_{w-1}}} * \\Delta^{k}_{Re_{\\ell_{w-1}}}`
        - :math:`\\Delta^{k}_{Re_{\\ell_w}} = Recall_{\\ell_w}(Th_k) - Recall_{\\ell_w}(Th_{k-1})`
        - :math:`\\Delta^{k}_{Pr_{\\ell_w}} = Precision_{\\ell_w}(Th_{k-1}) + Precision_{\\ell_w}(Th_k)`
        - :math:`\\Delta^{w} = |\\ell_w - \\ell_{w-1}|` (time window interval)

    ## VUS-ROC (Equation 18)
    .. math::
        VUS\\text{-}ROC = \\frac{1}{4} \\sum_{w=1}^{L} \\sum_{k=1}^{N} \\Delta^{(k,w)} * \\Delta^{w}

    where:
        - :math:`\\Delta^{(k,w)} = \\Delta^{k}_{TPR_{\\ell_w}} * \\Delta^{k}_{FPR_{\\ell_w}} + \\Delta^{k}_{TPR_{\\ell_{w-1}}} * \\Delta^{k}_{FPR_{\\ell_{w-1}}}`
        - :math:`\\Delta^{k}_{TPR_{\\ell_w}} = TPR_{\\ell_w}(Th_k) - TPR_{\\ell_w}(Th_{k-1})`
        - :math:`\\Delta^{k}_{FPR_{\\ell_w}} = FPR_{\\ell_w}(Th_k) - FPR_{\\ell_w}(Th_{k-1})`
        - :math:`\\Delta^{w} = |\\ell_w - \\ell_{w-1}|` (time window interval)
    
    ## Returns
    - tpr_3d: TPR values for each threshold and window
    - fpr_3d: FPR values for each threshold and window
    - prec_3d: Precision values for each threshold and window
    - window_3d: The different window sizes used for smoothing
    - avg_vus_roc: Average VUS-ROC across all windows
    - avg_vus_pr: Average VUS-PR across all windows


    """
    window_3d = np.arange(0, window+1, 1)
    pos = np.sum(labels)
    pos_range = _range_convers_new(labels)
    l = _new_sequence(labels, pos_range, window)
    score_sorted = -np.sort(-score)
    idx = np.argsort(score)[::-1] # descend
    thr = score[idx]
    if sample is not None and len(thr) > sample:
        thr = thr[np.linspace(0, len(score) - 1, sample, dtype=int)]
    else:
        sample = len(thr)
    
    tpr_3d = np.zeros((window + 1, sample + 2))
    fpr_3d = np.zeros((window + 1, sample + 2))
    prec_3d = np.zeros((window + 1, sample + 1))

    auc_3d = np.zeros(window + 1)
    ap_3d = np.zeros(window + 1) # use ap to approximate auc_pr

    tp = np.zeros(sample)
    N_pred = np.zeros(sample)

    for k, i in enumerate(np.linspace(0, len(score) - 1, sample).astype(int)):
        threshold = score_sorted[i]
        pred = score >= threshold
        N_pred[k] = np.sum(pred)

    for window in tqdm(window_3d, total=len(window_3d), desc="Processing windows"):
        range_label = _smooth_sequence(labels, pos_range, window)
        L = _new_sequence(range_label, pos_range, window)

        TF_list = np.zeros((sample + 2, 2))
        Precision_list = np.ones(sample + 1)
        j = 0

        for i in np.linspace(0, len(score) - 1, sample).astype(int):
            threshold = score_sorted[i]
            pred = score >= threshold
            labels = range_label.copy()
            existence = 0

            for seg in L:
                labels[seg[0]:seg[1] + 1] = range_label[seg[0]:seg[1] + 1] * pred[seg[0]:seg[1] + 1]
                if (pred[seg[0]:(seg[1] + 1)] > 0).any():
                    existence += 1
            for seg in pos_range:
                labels[seg[0]:seg[1] + 1] = 1

            TP = 0
            N_labels = 0
            for seg in l:
                TP += np.dot(labels[seg[0]:seg[1] + 1], pred[seg[0]:seg[1] + 1])
                N_labels += np.sum(labels[seg[0]:seg[1] + 1])

            TP += tp[j]
            FP = N_pred[j] - TP

            existence_ratio = existence / len(L)

            P_new = (pos + N_labels) / 2
            recall = min(TP / P_new, 1)

            TPR = recall * existence_ratio
            N_new = len(labels) - P_new
            FPR = FP / N_new

            Precision = TP / N_pred[j]

            j += 1
            TF_list[j] = [TPR, FPR]
            Precision_list[j] = Precision

        TF_list[j + 1] = [1, 1]  # otherwise, range-AUC will stop earlier than (1,1)

        tpr_3d[window] = TF_list[:, 0]
        fpr_3d[window] = TF_list[:, 1]
        prec_3d[window] = Precision_list

        width = TF_list[1:, 1] - TF_list[:-1, 1]
        height = (TF_list[1:, 0] + TF_list[:-1, 0]) / 2
        AUC_range = np.dot(width, height)
        auc_3d[window] = (AUC_range)

        width_PR = TF_list[1:-1, 0] - TF_list[:-2, 0]
        height_PR = Precision_list[1:]

        AP_range = np.dot(width_PR, height_PR)
        ap_3d[window] = AP_range

    return tpr_3d, fpr_3d, prec_3d, window_3d, sum(auc_3d) / len(window_3d), sum(ap_3d) / len(window_3d)



    