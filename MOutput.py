from sample_classes import *


class MOutput:
    def __init__(self, task):
        self.task = task
        self.num_samples = None
        self.loss = None
        self.l_sample = None
        self.meta_data = None
        
        self.lll_ilabel = None
        self.ll_score=None
        self.l_orig_sent = None
            
    def absorb_l_orig_sent(self, l_orig_sent):
        for k, orig_sent in enumerate(l_orig_sent):
            self.l_sample[k].orig_sent = orig_sent

    def absorb_lll_ilabel(self, lll_ilabel):
        for sample_id, ll_ilabel in enumerate(lll_ilabel):
            self.l_sample[sample_id].absorb_children(ll_ilabel)

    def absorb_ll_score(self, ll_score):
        for sample_id, l_score in enumerate(ll_score):
            self.l_sample[sample_id].absorb_scores(l_score)
            
    def absorb_all_possible(self):
        if self.l_orig_sent:
            self.absorb_l_orig_sent(self.l_orig_sent)
        if self.lll_ilabel:
            self.num_samples = len(self.lll_ilabel)
            self.l_sample = []
            for k in range(self.num_samples):
                self.l_sample.append(Sample(self.task))
            self.absorb_lll_ilabel(self.lll_ilabel)
        if self.ll_score:
            self.num_samples = len(self.ll_score)
            self.absorb_ll_score(self.ll_score)

    def write_tags_file(self, path, with_scores=False):
        with_unused_tokens = False
        if self.task == "ex":
            with_unused_tokens = True

        write_samples_file(self.l_sample,
                           path,
                           with_scores=with_scores,
                           with_unused_tokens=with_unused_tokens)

