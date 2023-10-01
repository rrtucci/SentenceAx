import torch
from words_tags_ilabels_translation import *


class MOutput:
    def __init__(self):
        self.num_samples = None
        self.l_orig_sent = []
        self.lll_ilabel = []

        self.ll_confi= []
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
    #         for depth in range(NUM_DEPTHS()):
    #             for words in range(self.l_sample[sample_id].
    #
    # def absorb_ll_confi(self, ll_confi):
    #     for sample_id, l_confi in enumerate(ll_confi):
    #         self.l_sample[sample_id].absorb_confis(l_confi)
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
    #     if self.ll_confi:
    #         self.num_samples = len(self.ll_confi)
    #         self.absorb_ll_confi(self.ll_confi)


    def to_cpu(self):
        self.l_orig_sent= self.l_orig_sent.cpu()
        self.ll_confi = self.ll_confi.cpu()
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
        num_depths = len(self.lll_ilabel[0])
        sent_len = len(self.lll_ilabel[0][0])
        lll_word = []
        for sam in range(num_samples):
            ll_word=[]
            for depth in range(num_depths):
                ll_word.append(
                    trans_fun(self.lll_ilabel[sam][depth], l_orig_sentL))
            lll_word.append(ll_word)
        return lll_word

        