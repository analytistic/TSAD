import numpy as np
from .basic_metrics import *

class BasicMetricor():
    def __init__(self,
                 a=1,
                 probability=True,
                 bias='flat',
                 ):
        self.a = a
        self.probability = probability
        self.bias = bias

    def metric_AUC_ROC(self, labels, score):
        return metric_AUC_ROC(labels, score)

    def metric_AUC_PR(self, labels, score):
        return metric_AUC_PR(labels, score)
    
    def metric_VUS(self, labels, score, window=1):
        results = metric_VUS(labels, score, window=window)
        return {
            'VUS_ROC': results[-2],
            'VUS_PR': results[-1],
        }
    
    def metric_PointF1(self, labels, score, preds=None):
        """
        Point F1 score at the best threshold or preds
        """
        if preds is None:
            prec, recall, thr = metric_prec_recall_curve(labels, score)
            f1 = 2 * prec * recall / (prec + recall + np.finfo(float).eps)
            return np.max(f1)
        else:
            f1 = metric_PointF1(labels, preds)
            return f1
        


        

