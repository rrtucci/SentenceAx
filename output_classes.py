from sample_classes import *


class MOutput:
    def __init__(self, task, lll_ilabel=None):
        self.task = task
        self.num_samples = None
        self.loss = None
        self.l_sample = None
        self.lll_ilabel = lll_ilabel
        if lll_ilabel:
            self.num_samples = len(lll_ilabel)
            self.set_lll_ilabel(lll_ilabel)

    def set_lll_ilabel(self, lll_ilabel):
        for sample_id, ll_ilabel in enumerate(lll_ilabel):
            self.l_sample[sample_id].set_children(ll_ilabel)

    def set_ll_confi(self, ll_confi):
        max_depth = len(ll_confi[0])
        for sample_id, l_confi in enumerate(ll_confi):
            sam = self.l_sample[sample_id]
            assert sam.max_depth == max_depth
            for depth in range(max_depth):
                sam.l_child[depth].confi = l_confi[depth]

    def write_tags_file(self, path, with_confis=False):
        with_unused_tokens = False
        if self.task == "ex":
            with_unused_tokens = True

        write_samples_file(self.l_sample,
                           path,
                           with_confis=with_confis,
                           with_unused_tokens=with_unused_tokens)

class SplitPredOutput:
    def __init__(self, num_samples):
        self.l_sample = []
        for i in range(num_samples):
            self.l_sample.append(
                SplitPredSample())
