from sax_utils import *
from CCTree import *


class SampleChild:
    def __init__(self, tags=None):
        self.tags = tags
        self.score = None
        self.simple_sent = None
        self.depth = None

    def get_tag_str(self):
        return " ".join(self.tags)

    def get_nontrivial_locs(self):
        locs = []
        for loc, tag in enumerate(self.tags):
            if tag != "NONE":
                locs.append(loc)
        return locs


class Sample:

    def __init__(self, task):
        self.task = task
        self.orig_sent = None
        self.l_child = []
        self.max_depth = None

        self.ll_ilabel = []
        self.l_score = []

        self.word_start_locs = [] # shape=(encoding_len,)
        self.ilabels =[] # shape=(encoding_len,)
        
        self.pos_mask = []  # shape=(num_words,)
        self.pos_locs = []  # shape=(,num_words,)
        self.verb_mask = []  # shape=(num_words,)
        self.verb_locs = []  # shape=(num_words,)



    def absorb_ll_ilabel(self, ll_ilabel):
        self.l_child = []
        for depth, l_ilabel in enumerate(ll_ilabel):
            self.l_child.append(SampleChild())
            self.l_child[-1].depth = depth
        self.set_tags(ll_ilabel)

    def absorb_l_score(self, l_score):
        for k, child in enumerate(self.l_child):
            child.score = l_score[k]

    def set_tags(self, ll_ilabel):
        for depth, l_ilabel in enumerate(ll_ilabel):
            child = self.l_child[depth]
            child.tags = []
            for ilabel in l_ilabel:
                if self.task == "ex":
                    child.tags.append(ILABEL_TO_EXTAG[ilabel])
                elif self.task == "cc":
                    child.tags.append(ILABEL_TO_CCTAG[ilabel])
                else:
                    assert False
    def absorb_all_possible(self):
        if self.ll_ilabel:
            self.max_depth = len(self.ll_ilabel)
            self.absorb_ll_ilabel(self.ll_ilabel)
        if self.l_score:
            self.absorb_l_score(self.l_score)


# class SplitPredSample():
#     def __init__(self):
#         self.l_child = []
#         for i in range(MAX_CC_DEPTH):
#             self.l_child.append(CCTagsSample())
#             self.l_child[-1].l_child = []
#             for j in range(MAX_EX_DEPTH):
#                 self.l_child[-1].l_child.append(ExTagsSample())