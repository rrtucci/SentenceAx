from copy import deepcopy
from sax_globals import *
import torch


# from torchtext.data.utils import get_tokenizer
# from torchtext.vocab import build_vocab_from_iterator

class SaxDataPadder:

    def __init__(self, pad_ilabel, use_spacy_model):

        self.pad_ilabel = pad_ilabel
        self.use_spacy_model = use_spacy_model

    @staticmethod
    def get_padded_ll_x(unpadded_ll_x, padding_x):
        # padding a 2 dim array

        ll_x = deepcopy(unpadded_ll_x)
        max_dim1 = -1
        for l_x in ll_x:
            if len(l_x) > max_dim1:
                max_dim1 = len(l_x)
        padded_ll_x = []
        for l_x in ll_x:
            padding_len = max_dim1 - len(l_x)
            l_x += [padding_x] * padding_len
            padded_ll_x.append(l_x)
        return torch.tensor(padded_ll_x)

    @staticmethod
    def get_padded_lll_ilabel(unpadded_lll_ilabel):
        lll_ilabel = deepcopy(unpadded_lll_ilabel)
        for sam in range(len(lll_ilabel)):
            pad_depth = MAX_DEPTH - len(lll_ilabel[sam])
            num_words = len(lll_ilabel[sam][0])
            # ilabel = 0 for extag=NONE
            lll_ilabel[sam] = lll_ilabel[sam] + [[0] * num_words] * pad_depth

        max_num_words = -1
        for ll_ilabel in lll_ilabel:
            if (len(ll_ilabel[0]) > max_num_words):
                max_num_words = len(ll_ilabel[0])
        padded_lll_ilabel = []
        for ll_ilabel in lll_ilabel:
            padded_ll_ilabel = []
            for l_ilabel in ll_ilabel:
                padding_len = max_num_words - len(l_ilabel)
                l_ilabel += [-100] * padding_len
                padded_ll_ilabel.append(l_ilabel)
            padded_lll_ilabel.append(padded_ll_ilabel)
        return torch.tensor(padded_lll_ilabel)
        
    # def build_vocab(self, m_input):
    #     """
    #
    #     A vocabulary (vocab) is a function that turns
    #     word lists to int lists
    #
    #     vocab(['here', 'is', 'an', 'example'])
    #     >>> [475, 21, 30, 5297]
    #     """
    #
    #     # tokenizer = get_tokenizer("basic_english")
    #     tokenizer = self.auto_tokenizer
    #     def yield_tokens(m_input):
    #         for example_d in m_input:
    #             orig_sent = example_d["orig_sent"]
    #             yield tokenizer(orig_sent)
    #
    #     vocab = build_vocab_from_iterator(yield_tokens(m_input),
    #                                       specials=["<unk>"])
    #     vocab.set_default_index(vocab["<unk>"])
    #
    #     return vocab

    def get_padded_data_d(self, m_input):
        """
        similar to data.pad_data()



        Parameters
        ----------
        m_input

        Returns
        -------

        """

        # data_in = m_input
        # example_d = {
        #     'sentL_ids': sentL_ids,
        #     'll_label': labels_for_each_ex[:MAX_EX_DEPTH],
        #     'word_starts': word_starts,
        #     'orig_sent': orig_sent,
        #     # if use_spacy_model:
        #     'pos_mask': pos_mask,
        #     'pos_locs': pos_locs,
        #     'verb_mask': verb_mask,
        #     'verb_locs': verb_locs
        # }

        padded_l_os_ilabels = SaxDataPadder.get_padded_ll_x(
            m_input.l_os_ilabels, self.pad_ilabel)

        padded_lll_ilabel =\
            SaxDataPadder.get_padded_lll_ilabel(m_input.lll_ilabel)

        padded_l_os_word_start_locs =\
            SaxDataPadder.get_padded_ll_x(m_input.l_os_word_start_locs,
                                          0)

        padded_data_d = {'l_os_ilabels': padded_l_os_ilabels,
                       'lll_ilabel': padded_lll_ilabel,
                       'l_os_word_start_locs': padded_l_os_word_start_locs,
                       'l_orig_sent': m_input.l_orig_sent}

        if self.use_spacy_model:
            padded_data_d["l_os_pos_mask"]  =\
                SaxDataPadder.get_padded_ll_x(m_input.l_os_pos_mask, 0)
            padded_data_d["l_os_pos_locs"] = \
                SaxDataPadder.get_padded_ll_x(m_input.l_os_pos_locs, 0)
            padded_data_d["l_os_verb_mask"] = \
                SaxDataPadder.get_padded_ll_x(m_input.l_os_verb_mask, 0)
            padded_data_d["l_os_verb_locs"] = \
                SaxDataPadder.get_padded_ll_x(m_input.l_os_verb_locs, 0)

        return padded_data_d
