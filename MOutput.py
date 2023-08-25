from CCTagsSample import *
from SimpSentsSample import *

class ExOutput:
    sam = ex_output[sample_id]
    child = sam.l_child[depth]
        meta_data  sam.orig_sent
        self.lll_prediction = child.tags
        self.ll_score = None child.score
        self.loss = None ex_output.loss
        self.train_loss = ex_output.train_loss
        self.ground_truth = None sam.ground_truth_sample
        self.ll_orig_sent  sam.orig_sent

        self.cc_l_spanned_words = [] get_words(child.simple_sent)
        self.cc_ll_spanned_loc = [] child.get_nontrivial_locs()
        self.cc_l_pred_str = [] child.simple_sent

        self.l_pred_sentL = None
