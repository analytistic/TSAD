
from typing import Dict
from .metricor import BasicMetricor
from .utils import ResultMerger

# def get_metrics(score, labels, slidingWindow=100, pred=None, version='opt', thre=250):
#     metrics = {}

#     '''
#     Threshold Independent
#     '''
#     grader = BasicMetricor()
#     # AUC_ROC, Precision, Recall, PointF1, PointF1PA, Rrecall, ExistenceReward, OverlapReward, Rprecision, RF, Precision_at_k = grader.metric_new(labels, score, pred, plot_ROC=False)
#     AUC_ROC = grader.metric_ROC(labels, score)
#     AUC_PR = grader.metric_PR(labels, score)

#     # R_AUC_ROC, R_AUC_PR, _, _, _ = grader.RangeAUC(labels=labels, score=score, window=slidingWindow, plot_ROC=True)
#     _, _, _, _, _, _,VUS_ROC, VUS_PR = generate_curve(labels.astype(int), score, slidingWindow, version, thre)


#     '''
#     Threshold Dependent
#     if pred is None --> use the oracle threshold
#     '''

#     PointF1 = grader.metric_PointF1(labels, score, preds=pred)
#     PointF1PA = grader.metric_PointF1PA(labels, score, preds=pred)
#     EventF1PA = grader.metric_EventF1PA(labels, score, preds=pred)
#     RF1 = grader.metric_RF1(labels, score, preds=pred)
#     Affiliation_F = grader.metric_Affiliation(labels, score, preds=pred)

#     metrics['AUC-PR'] = AUC_PR
#     metrics['AUC-ROC'] = AUC_ROC
#     metrics['VUS-PR'] = VUS_PR
#     metrics['VUS-ROC'] = VUS_ROC

#     metrics['Standard-F1'] = PointF1
#     metrics['PA-F1'] = PointF1PA
#     metrics['Event-based-F1'] = EventF1PA
#     metrics['R-based-F1'] = RF1
#     metrics['Affiliation-F'] = Affiliation_F
#     return metrics

def get_metrics(score, labels, window=1)-> Dict:
    metrics = BasicMetricor()
    point_metrics = {}
    range_metrics = {}
    point_metrics['AUC-ROC'] = metrics.metric_AUC_ROC(labels, score)
    point_metrics['AUC-PR'] = metrics.metric_AUC_PR(labels, score)
    point_metrics['Point-F1'] = metrics.metric_PointF1(labels, score)
    vus = metrics.metric_VUS(labels, score, window=window)
    range_metrics['VUS-ROC'], range_metrics['VUS-PR'] = vus['VUS_ROC'], vus['VUS_PR']
    return {
        'point_metrics': point_metrics,
        'range_metrics': range_metrics
    }


def get_stream_metrics(*args, **kwargs)-> Dict:
    return {}

__all__ = ['get_metrics', 'get_stream_metrics', 'ResultMerger']