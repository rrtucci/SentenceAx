import numpy as np
from CCTree import *


class CCScores:
    """
    similar to Openie6.metric.Record

    This method stores scores N_{x|a} where x\in{0,1} is the measurement and
    a\in{ 0,1} is the actual value.

    Refs:
    See chapter entitled "ROC curves" in my book Bayesuvius,

    a=actual value \in {0,1}
    x=predicted value \in {0,1}

    sometimes  people use 0 = False(F) = Negative(N) and 1 = True(T) =
    Positive(P)

    N_{x|a} for x,a in {0,1}

    true positive (TP): N_1|1, correctly predicted hit
    true negative (TN): N_0|0, correctly predicted miss
    false positive (FP): N_1|0, false alarm, type I error or overestimation
    false negative (FN): N_0|1, miss, type II error or underestimation

    Attributes
    ----------
    N0L0: int
    N0L1: int
    N1L0: int
    N1L1: int
    N1L1_f: int
    N1L1_t: int


    """

    def __init__(self):
        """
        Constructor


        """
        self.N1L1 = 0  # True Positive (TP)
        self.N0L0 = 0  # True Negative (TN)
        self.N1L0 = 0  # False Positive (FP)
        self.N0L1 = 0  # False Negative (FN)
        self.N1L1_t = 0
        self.N1L1_f = 0

    def reset(self):
        """
        This method sets to zero all N_{x|a}.

        Returns
        -------
        None

        """
        self.N1L1 = 0  # True Positive (TP)
        self.N0L0 = 0  # True Negative (TN)
        self.N1L0 = 0  # False Positive (FP)
        self.N0L1 = 0  # False Negative (FN)
        self.N1L1_t = 0
        self.N1L1_f = 0

    def accuracy(self):
        """
        This method returns a function of the N_{x|a}.

        Returns
        -------
        float

        """
        total = self.N1L1 + self.N1L0 + self.N0L1 + self.N0L0

        return (self.N1L1 + self.N0L0) / total if total > 0 else np.nan

    def precision(self):
        """
        This method returns a function of the N_{x|a}.

        Returns
        -------
        float

        """
        denom = self.N1L1 + self.N1L0
        return self.N1L1 / denom if denom > 0 else np.nan

    def recall(self):
        """
        This method returns a function of the N_{x|a}.

        Returns
        -------
        float

        """
        denom = self.N1L1 + self.N0L1
        return self.N1L1 / denom if denom > 0 else np.nan

    def f1_score(self):
        """
        This method returns a function of the N_{x|a}.

        Returns
        -------
        float

        """
        prec = self.precision()
        if prec is not np.nan:
            rec = self.recall()
            if rec is not np.nan:
                denom = prec + rec
                if denom > 0:
                    return (2 * prec * rec) / denom
        return np.nan


class CCScoreManager:
    """
    similar to Openie6.metric.Counter

    CCScores similar to Openie6.metric.Record
    CCScoreManager similar to Openie6.metric.Counter
    `kind` similar to `Openie6.criteria`, both in ["whole", "outer", "inner",
    "exact"]
    
    This class increments and resets the scores keep by class CCScores.

    Attributes
    ----------
    kind: str
    depth_scores: CCScores
    overall_scores: CCScores


    """

    def __init__(self, kind):
        """
        Constructor

        Parameters
        ----------
        kind: str
        """
        assert kind in ["whole", "outer", "inner", "exact"]
        self.kind = kind
        self.overall_scores = CCScores()
        self.depth_scores = CCScores()
        # print("vbgn", self.overall_scores, self.depth_scores)

    def reset(self):
        """
        This method resets overall and depth scores to zero.

        Returns
        -------
        None

        """
        self.overall_scores.reset()
        self.depth_scores.reset()

    def absorb_new_sample(self, pred_ccnodes, true_ccnodes):
        """
        similar to Openie6.metric.Counter.append()

        This method takes as input `pred_ccnodes` (predicted CCNodes) and
        `true_ccnodes` (true CCNodes). Then it changes the scores
        `self.overall_scores` and `self.depth_scores` to reflect the
        proximity of the predictions to the truth.


        Parameters
        ----------
        pred_ccnodes: list[CCNode]
        true_ccnodes: list[CCNode]

        Returns
        -------
        None

        """
        for ccloc in sorted([ccnode.ccloc for ccnode in true_ccnodes]):
            pred_ccnode = CCTree.get_ccnode_from_ccloc(ccloc, pred_ccnodes)
            true_ccnode = CCTree.get_ccnode_from_ccloc(ccloc, true_ccnodes)
            if pred_ccnode and true_ccnode:
                pred_spans = pred_ccnode.spans
                true_spans = true_ccnode.spans
                # depth = true_ccnode.depth
                if self.kind == "whole":
                    is_correct = (pred_spans[0][0] == true_spans[0][0]
                                  and pred_spans[-1][1] == true_spans[-1][1])
                elif self.kind == "outer":
                    is_correct = (pred_spans[0] == true_spans[0]
                                  and pred_spans[-1] == true_spans[-1])
                elif self.kind == "inner":
                    pred_pair = pred_ccnode.get_span_pair(
                        ccloc, allow_None=False)
                    true_pair = true_ccnode.get_span_pair(
                        ccloc, allow_None=False)
                    is_correct = (pred_pair == true_pair)
                elif self.kind == "exact":
                    is_correct = (pred_spans == true_spans)
                else:
                    assert False
                self.overall_scores.N1L1 += 1
                self.depth_scores.N1L1 += 1
                if is_correct:
                    self.overall_scores.N1L1_t += 1
                    self.depth_scores.N1L1_t += 1
                else:
                    self.overall_scores.N1L1_f += 1
                    self.depth_scores.N1L1_f += 1
            if pred_ccnode and not true_ccnode:
                self.overall_scores.N1L0 += 1
            if not pred_ccnode and true_ccnode:
                # depth = true_ccnode.ilabel
                self.overall_scores.N0L1 += 1
                self.depth_scores.N0L1 += 1
            if not pred_ccnode and not true_ccnode:
                self.overall_scores.N0L0 += 1
