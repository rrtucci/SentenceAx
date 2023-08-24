from sample_classes import *

class ExOutput:

    def __init__(self, num_samples):
        self.l_sample = []
        for i in range(num_samples):
            self.l_sample.append(ExTagsSample())

class CCOutput:

    def __init__(self, num_samples):
        self.l_sample = []
        for i in range(num_samples):
            self.l_sample.append(CCTagsSample())

class SplitPredOutput:
    def __init__(self, num_samples, max_cc_depth, max_ex_depth):
        self.l_sample = []
        for i in range(num_samples):
            self.l_sample.append(
                SplitPredSample(max_cc_depth, max_ex_depth))
