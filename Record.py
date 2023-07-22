import numpy as np
class Record:
    """
    Refs:
    See my book Bayesuvius, chapter "ROC curves

    a=actual value \in {0,1}
    x=predicted value \in {0,1}

    sometimes  people use 0 = F = N and 1 = T = P

    true positive (TP): N_1|1, hit
    true negative (TN): N_0|0, correct rejection
    false positive (FP): N_1|0, false alarm, type I error or underestimation
    false negative (FN): N_0|1, miss, type II error or overestimation

    FN means N_x=N|a=!x

    """

    def __init__(self):
        self.N1L1 = 0 # TP
        self.N0L0 = 0  # TN
        self.N1L0 = 0  # FP
        self.N0L1 = 0  # FN

    def accuracy(self):
        total = self.NIL1 + self.N1L0 + self.N0L1 + self.N0L0

        return (self.NIL1 + self.N0L0) / total if total > 0 else np.nan

    def precision(self):
        denom = self.NIL1 + self.N1L0
        return self.NIL1 / denom if denom > 0 else np.nan

    def recall(self):
        denom = self.NIL1 + self.N0L1
        return self.NIL1 / denom if denom > 0 else np.nan

    def f1_score(self):
        precision = self.precision
        if precision is not np.nan:
            recall = self.recall
            if recall is not np.nan:
                denom = precision + recall
                if denom > 0:
                    return (2 * precision * recall) / denom
        return np.nan


class Recorder:  # formerly Counter
    def __init__(self, criteria):
        self._criteria = criteria
        self._records = []
        assert criteria in ["whole", "outer", "inner", "exact"]