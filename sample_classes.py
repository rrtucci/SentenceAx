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

        self.ll_icode = []
        self.l_score = []

        self.word_start_locs = [] # shape=(encoding_len,)
        self.icodes =[] # shape=(encoding_len,)
        
        self.pos_mask = []  # shape=(num_words,)
        self.pos_locs = []  # shape=(,num_words,)
        self.verb_mask = []  # shape=(num_words,)
        self.verb_locs = []  # shape=(num_words,)



    def absorb_ll_icode(self, ll_icode):
        self.l_child = []
        for depth, l_icode in enumerate(ll_icode):
            self.l_child.append(SampleChild())
            self.l_child[-1].depth = depth
        self.set_tags(ll_icode)

    def absorb_l_score(self, l_score):
        for k, child in enumerate(self.l_child):
            child.score = l_score[k]

    def set_tags(self, ll_icode):
        for depth, l_icode in enumerate(ll_icode):
            child = self.l_child[depth]
            child.tags = []
            for icode in l_icode:
                if self.task == "ex":
                    child.tags.append(ICODE_TO_EXTAG[icode])
                elif self.task == "cc":
                    child.tags.append(ICODE_TO_CCTAG[icode])
                else:
                    assert False
    def absorb_all_possible(self):
        if self.ll_icode:
            self.max_depth = len(self.ll_icode)
            self.absorb_ll_icode(self.ll_icode)
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
