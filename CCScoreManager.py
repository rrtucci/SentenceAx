import numpy as np
from CCTree import *

SCORE_KINDS = ["whole", "outer", "inner", "exact"]


class CCScore:
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

    Actually, N1L0=N0L0=0 so we only remember N1L1 and N0L1

    Attributes
    ----------
    N0L1: int
    N1L1: int
    N1L1_f: int
    N1L1_t: int
    kind: str

    """

    def __init__(self, kind="exact"):
        """
        Constructor

        Parameters
        ----------
        kind: str

        """
        self.kind = kind

        self.N1L1 = 0  # True Positive (TP)

        # not used
        # self.N0L0 = 0  # True Negative (TN)
        # self.N1L0 = 0  # False Positive (FP)

        self.N0L1 = 0  # False Negative (FN)

        # N1L1 = N1L1_f + N1L1_t
        self.N1L1_f, self.N1L1_t = 0, 0

    def reset(self):
        """
        This method sets to zero all N_{x|a}.

        Returns
        -------
        None

        """
        self.N1L1 = 0  # True Positive (TP)
        # self.N0L0 = 0  # True Negative (TN)
        # self.N1L0 = 0  # False Positive (FP)
        self.N0L1 = 0  # False Negative (FN)

        self.N1L1_f, self.N1L1_t = 0, 0

    def get_acc_nsam(self):
        """
        This method returns a tuple (acc, nsam) where

        acc= accuracy = probability = N1L1/(N0L1 + N1L1)
        nsam = number of samples

        Returns
        -------
        tuple(float)

        """
        # When N_L0=0, returns N1L1/(N1L1 + N0L1)
        # total = self.N1L1 + self.N1L0 + self.N0L1 + self.N0L0
        #
        # return (self.N1L1 + self.N0L0) / total if total > 0 else None

        denom = self.N1L1 + self.N0L1
        return (self.N1L1 / denom, denom) if denom > 0 else None

    def get_acc_nsam11(self):
        """
        This method returns a tuple (acc, nsam) where

        acc= accuracy = probability = N1L1_t/(N1L1_f + N1L1_t)
        nsam = number of samples


        Returns
        -------
        tuple(float)

        """
        # When N_L0=0, returns N1L1/(N1L1 + N0L1)
        # total = self.N1L1 + self.N1L0 + self.N0L1 + self.N0L0
        #
        # return (self.N1L1 + self.N0L0) / total if total > 0 else None

        denom = self.N1L1_f + self.N1L1_t
        assert denom == self.N1L1
        return (self.N1L1_t / denom, denom) if denom > 0 else None

    # def precision(self):
    #     """
    #     This method returns a function of the N_{x|a}.
    #
    #     Returns
    #     -------
    #     float
    #
    #     """
    #     # When N_L0=0, returns 1
    #     denom = self.N1L1 + self.N1L0
    #     return self.N1L1 / denom if denom > 0 else None
    #
    # def recall(self):
    #     """
    #     This method returns a function of the N_{x|a}.
    #
    #     Returns
    #     -------
    #     float
    #
    #     """
    #     # When N_L0=0, returns N1L1/(N1L1 + N0L1)
    #     denom = self.N1L1 + self.N0L1
    #     return self.N1L1 / denom if denom > 0 else None
    #
    # def f1_score(self):
    #     """
    #     This method returns a function of the N_{x|a}.
    #
    #     Returns
    #     -------
    #     float
    #
    #     """
    #     # When N_L0=0, prec=1, rec= N1L1/(N1L1 + N0L1),
    #     # this returns the 2*(rec||prec) =  2*(1/rec + 1/prec)
    #     score = None
    #     prec = self.precision()
    #     if prec is not None:
    #         rec = self.recall()
    #         if rec is not None:
    #             denom = prec + rec
    #             if denom > 0:
    #                 score= (2 * prec * rec) / denom
    #     return score


class CCScoreManager:
    """
    similar to Openie6.metric.Counter

    CCScore similar to Openie6.metric.Record

    CCScoreManager similar to Openie6.metric.Counter

    `kind` similar to `Openie6.criteria`, both in ["whole", "outer", "inner",
    "exact"]

    This class increments and resets the scores kept by class CCScore.

    Attributes
    ----------
    all_node_ccscore: CCScore
    ccloc_to_ccscore: dict[int, CCScore]
    kind: str


    """

    def __init__(self, kind):
        """
        Constructor

        Parameters
        ----------
        kind: str
        
        """
        assert kind in SCORE_KINDS
        self.kind = kind
        # both self.all_node_ccscore and self.ccloc_to_ccscore
        self.all_node_ccscore = CCScore(kind)
        # self.ccloc_to_ccscore is private
        # to the class. The class increments it and resets it to 0
        self.ccloc_to_ccscore = defaultdict(lambda: CCScore(kind))
        # print("vbgn", self.all_node_ccscore, self.ccloc_to_ccscore[ccloc])

    def reset(self):
        """
        This method calls CCScore.reset() for all CCScore values of the
        dictionary `ccloc_to_ccscore`.

        Returns
        -------
        None

        """
        for ccloc in self.ccloc_to_ccscore.keys():
            self.ccloc_to_ccscore[ccloc].reset()

    def kind_condition_is_t(self,
                            pred_ccnode,
                            true_ccnode):
        """

        Parameters
        ----------
        pred_ccnode: CCNode
        true_ccnode: CCNode

        Returns
        -------
        bool

        """
        # In SentenceAx, ccnode has only 2 spans and 1 ccloc.
        # CCTagsLine can have more than 2 spans and more than 1 ccloc
        # pred_spans = pred_ccnode.spans
        # true_spans = true_ccnode.spans
        pred_span0, pred_span1 = \
            pred_ccnode.span_pair[0], pred_ccnode.span_pair[1]
        true_span0, true_span1 = \
            true_ccnode.span_pair[0], true_ccnode.span_pair[1]
        if self.kind == "whole":
            # success = (pred_spans[0][0] == true_spans[0][0]
            #               and pred_spans[-1][1] == true_spans[-1][1])
            condition_is_t = pred_span0[0] == true_span0[0] and \
                             pred_span1[1] == true_span1[1]
        elif self.kind == "outer":
            # success = (pred_spans[0] == true_spans[0]
            #               and pred_spans[-1] == true_spans[-1])
            condition_is_t = pred_span0 == true_span0 and \
                             pred_span1 == true_span1
        elif self.kind == "inner":
            # pred_pair = pred_ccnode.get_span_pair(
            #     true_ccloc, throw_if_None=True)
            # true_pair = true_ccnode.get_span_pair(
            #     true_ccloc, throw_if_None=True)
            # success = (pred_pair == true_pair)
            condition_is_t = pred_ccnode.ccloc == true_ccnode.ccloc
        elif self.kind == "exact":
            condition_is_t = true_ccnode == pred_ccnode
        else:
            assert False
        return condition_is_t

    def absorb_new_sample(self, pred_ccnodes, true_ccnodes):
        """
        similar to Openie6.metric.Counter.append()

        This method takes as input `pred_ccnodes` (predicted CCNodes) and
        `true_ccnodes` (true CCNodes). Then it changes the scores
        `self.all_node_ccscore` and `self.ccloc_to_ccscore[ ccloc]` to
        reflect the proximity of the predictions to the truth.


        Parameters
        ----------
        pred_ccnodes: list[CCNode]
        true_ccnodes: list[CCNode]

        Returns
        -------
        None

        """
        true_cclocs = sorted([ccnode.ccloc for ccnode in true_ccnodes])
        for true_ccloc in true_cclocs:
            pred_ccnode = \
                CCTree.get_ccnode_from_ccloc(true_ccloc, pred_ccnodes)
            true_ccnode = \
                CCTree.get_ccnode_from_ccloc(true_ccloc, true_ccnodes)
            # true_ccnode will never be None
            if pred_ccnode and not true_ccnode:
                assert False, "true_ccnode will never be None"
                # self.all_node_ccscore.N1L0 += 1
                # self.ccloc_to_ccscore[true_ccloc].N1L0 += 1
            elif not pred_ccnode and true_ccnode:
                self.all_node_ccscore.N0L1 += 1
                self.ccloc_to_ccscore[true_ccloc].N0L1 += 1
            elif not pred_ccnode and not true_ccnode:
                assert False, "true_ccnode will never be None"
                # self.all_node_ccscore.N0L0 += 1
                # self.ccloc_to_ccscore[true_ccloc].N0L0 += 1
            elif pred_ccnode and true_ccnode:
                self.all_node_ccscore.N1L1 += 1
                self.ccloc_to_ccscore[true_ccloc].N1L1 += 1
                condition_is_t = self.kind_condition_is_t(pred_ccnode,
                                                          true_ccnode)

                if condition_is_t:
                    self.all_node_ccscore.N1L1_t += 1
                    self.ccloc_to_ccscore[true_ccloc].N1L1_t += 1
                else:
                    self.all_node_ccscore.N1L1_f += 1
                    self.ccloc_to_ccscore[true_ccloc].N1L1_f += 1
