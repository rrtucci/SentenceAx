from CCTagsSample import *
from SimpSentsSample import *

class ExOutput:
    sam = ex_output[sample_id]
    child = sam.l_child[depth]
        meta_data  sam.orig_sent
        self.lll_ilabel = child.tags
        self.ll_score = None child.score
        self.loss_fun = None ex_output.loss_fun
        self.train_loss = ex_output.train_loss
        self.true_batch_out.lll_ilabel = None sam.ground_truth_sample
        self.ll_orig_sent  sam.orig_sent

        self.cc_ll_spanned_word = [] get_words(child.simple_sent)
        self.cc_ll_spanned_loc = [] child.get_nontrivial_locs()
        self.cc_l_pred_str = [] child.simple_sent

        self.l_pred_sentL = None
