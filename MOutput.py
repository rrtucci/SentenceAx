from sample_classes import *


class MOutput:
    def __init__(self, task):
        self.task = task
        self.num_samples = None
        self.l_orig_sent = []
        self.lll_icode = []

        self.ll_score= []
        self.loss = None
            
    # def set_l_orig_sent(self, l_orig_sent):
    #     for k, orig_sent in enumerate(l_orig_sent):
    #         self.l_sample[k].orig_sent = orig_sent
    # def get_l_orig_sent(self):
    #     l_orig_sent = []
    #     for k in range(self.num_samples):
    #         l_orig_sent.append(self.l_sample[k])
    #     return l_orig_sent
    #
    #
    # def set_lll_icode(self, lll_icode):
    #     for sample_id, ll_icode in enumerate(lll_icode):
    #         self.l_sample[sample_id].absorb_children(ll_icode)
    #
    # def get_lll_icode(self):
    #     lll_icode = []
    #     for sample_id in range(self.num_samples):
    #         ll_icode = []
    #         for depth in range(MAX_DEPTH):
    #             for words in range(self.l_sample[sample_id].
    #
    # def absorb_ll_score(self, ll_score):
    #     for sample_id, l_score in enumerate(ll_score):
    #         self.l_sample[sample_id].absorb_scores(l_score)
    #
    # def absorb_all_possible(self):
    #     if self.l_orig_sent:
    #         self.set_l_orig_sent(self.l_orig_sent)
    #     if self.lll_icode:
    #         self.num_samples = len(self.lll_icode)
    #         self.l_sample = []
    #         for k in range(self.num_samples):
    #             self.l_sample.append(Sample(self.task))
    #         self.set_lll_icode(self.lll_icode)
    #     if self.ll_score:
    #         self.num_samples = len(self.ll_score)
    #         self.absorb_ll_score(self.ll_score)

    def write_tags_file(self, path, with_scores=False):
        with_unused_tokens = False
        if self.task == "ex":
            with_unused_tokens = True

        write_samples_file(self.l_sample,
                           path,
                           with_scores=with_scores,
                           with_unused_tokens=with_unused_tokens)

