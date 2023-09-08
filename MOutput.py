import torch
from words_tags_ilabels_translation import *


class MOutput:
    def __init__(self, task):
        self.task = task
        self.num_samples = None
        self.l_orig_sent = []
        self.lll_ilabel = []

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
    # def set_lll_ilabel(self, lll_ilabel):
    #     for sample_id, ll_ilabel in enumerate(lll_ilabel):
    #         self.l_sample[sample_id].absorb_children(ll_ilabel)
    #
    # def get_lll_ilabel(self):
    #     lll_ilabel = []
    #     for sample_id in range(self.num_samples):
    #         ll_ilabel = []
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
    #     if self.lll_ilabel:
    #         self.num_samples = len(self.lll_ilabel)
    #         self.l_sample = []
    #         for k in range(self.num_samples):
    #             self.l_sample.append(Sample(self.task))
    #         self.set_lll_ilabel(self.lll_ilabel)
    #     if self.ll_score:
    #         self.num_samples = len(self.ll_score)
    #         self.absorb_ll_score(self.ll_score)


    def to_cpu(self):
        self.l_orig_sent= self.l_orig_sent.cpu()
        self.ll_score = self.ll_score.cpu()
        self.lll_ilabel = self.lll_ilabel.cpu()

    def get_l_orig_sentL(self):
        return [self.l_orig_sent[k] + UNUSED_TOKENS_STR
                for k in range(len(self.l_orig_sent))]

    def get_lll_word(self,type):
        if type=="ex":
            trans_fun = translate_ilabels_to_words_via_extags
        elif type=="cc":
            trans_fun = translate_ilabels_to_words_via_cctags
        else:
            assert False
        l_orig_sentL = self.get_l_orig_sentL()
        num_samples = len(self.lll_ilabel)
        max_depth = len(self.lll_ilabel[0])
        sent_len = len(self.lll_ilabel[0][0])
        lll_word = []
        for sam in range(num_samples):
            ll_word=[]
            for depth in range(max_depth):
                ll_word.append(
                    trans_fun(self.lll_ilabel[sam][depth], l_orig_sentL))
            lll_word.append(ll_word)
        return lll_word

        