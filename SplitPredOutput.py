from sample_classes  import *
class SplitPredOutput:
    def __init__(self, num_samples):
        self.l_sample = []
        for i in range(num_samples):
            self.l_sample.append(
                SplitPredSample())