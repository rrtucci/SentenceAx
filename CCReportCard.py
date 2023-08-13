import numpy as np
from CCTree import *

class CCScores:
    """
    formerly metric.Record
    Refs:
    See my book Bayesuvius, chapter "ROC curves

    a=actual value \in {0,1}
    x=predicted value \in {0,1}

    sometimes  people use 0 = F = N and 1 = T = P

    true positive (TP): N_1|1, hit
    true negative (TN): N_0|0, is_correct rejection
    false positive (FP): N_1|0, false alarm, type I error or overestimation
    false negative (FN): N_0|1, miss, type II error or underestimation

    """

    def __init__(self):
        self.N1L1 = 0 # TP
        self.N0L0 = 0  # TN
        self.N1L0 = 0  # FP
        self.N0L1 = 0  # FN

    def accuracy(self):
        total = self.N1L1 + self.N1L0 + self.N0L1 + self.N0L0

        return (self.N1L1 + self.N0L0) / total if total > 0 else np.nan

    def precision(self):
        denom = self.N1L1 + self.N1L0
        return self.N1L1 / denom if denom > 0 else np.nan

    def recall(self):
        denom = self.N1L1 + self.N0L1
        return self.N1L1 / denom if denom > 0 else np.nan

    def f1_score(self):
        prec = self.precision()
        if prec is not np.nan:
            rec = self.recall()
            if rec is not np.nan:
                denom = prec + rec
                if denom > 0:
                    return (2 * prec * rec) / denom
        return np.nan


class CCReportCard:
    """
    formerly metric.Counter
    `student` formerly `criteria`

    """
    def __init__(self, student):
        assert student in ["whole", "outer", "inner", "exact"]
        self.student = student
        self.subject_to_scores = {}

    def reset(self):
        self.subject_to_scores.clear()

    def append(self, pred_ccnodes, true_ccnodes):
        for ccloc in sorted([ccnode.ccloc for ccnode in true_ccnodes]):
            pred_ccnode = CCTree.get_ccnode_from_ccloc(ccloc, pred_ccnodes)
            true_ccnode = CCTree.get_ccnode_from_ccloc(ccloc, true_ccnodes)
            if pred_ccnode is not None and true_ccnode is not None:
                pred_spans = pred_ccnode.spans
                true_spans = true_ccnode.spans
                depth = true_ccnode.depth
                is_correct = False # R.Tucci: I added this line
                if self.student == "whole":
                    is_correct = (pred_spans[0][0] == true_spans[0][0]
                        and pred_spans[-1][1] == true_spans[-1][1])
                elif self.student == "outer":
                    is_correct = (pred_spans[0] == true_spans[0]
                        and pred_spans[-1] == true_spans[-1])
                elif self.student == "inner":
                    pred_pair = pred_ccnode.get_pair(ccloc, check=True)
                    true_pair = true_ccnode.get_pair(ccloc, check=True)
                    is_correct = (pred_pair == true_pair)
                elif self.student == "exact":
                    is_correct = pred_spans == true_spans
                self.subject_to_scores["overall"].N1L1 += 1
                self.subject_to_scores[depth].N1L1 += 1
                if is_correct:
                    self.subject_to_scores["overall"].N1L1_t += 1
                    self.subject_to_scores[depth].N1L1_t += 1
                else:
                    self.subject_to_scores["overall"].N1L1_f += 1
                    self.subject_to_scores[depth].N1L1_f += 1
            if pred_ccnode is not None and true_ccnode is None:
                self.subject_to_scores["overall"].N1L0 += 1
            if pred_ccnode is None and true_ccnode is not None:
                depth = true_ccnode.label
                self.subject_to_scores["overall"].N0L1 += 1
                self.subject_to_scores[depth].N0L1 += 1
            if pred_ccnode is None and true_ccnode is None:
                self.subject_to_scores["overall"].N0L0 += 1

    def overall(self):
        return self.subject_to_scores["overall"]
