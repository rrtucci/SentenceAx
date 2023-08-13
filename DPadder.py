from copy import deepcopy
from sax_globals import *
# from torchtext.data.utils import get_tokenizer
# from torchtext.vocab import build_vocab_from_iterator

class DPadder:

    def __init__(self, sent_pad_id, spacy_model):

        self.sent_pad_id = sent_pad_id
        self.spacy_model = spacy_model

    @staticmethod
    def get_padded_list(li_word, pad_id):
        # padding a 1 dim array
        max_word_len = -1
        for word in li_word:
            if len(word) > max_word_len:
                max_word_len = len(word)
        padded_li_word = []
        for word in li_word:
            num_pad_id = max_word_len - len(word)
            padded_word = word.copy() + [pad_id] * num_pad_id
            padded_li_word.append(padded_word)
        return padded_li_word

    @staticmethod
    def get_padded_list_list(ll_word,
                             pad_id0,
                             pad_id1,
                             max_outer_dim):
        # padding a 2 dim array

        max_l_word_len = -1
        for l_word in ll_word:
            if len(l_word) > max_l_word_len:
                max_l_word_len = len(l_word)

        # padding outer dimension
        assert len(ll_word) <= max_outer_dim
        padded_ll_word = deepcopy(ll_word)
        for i in range(len(ll_word), max_outer_dim):
            padded_ll_word.append([pad_id0] * max_l_word_len)
        # padding inner dimension
        for i in range(len(ll_word)):
            padded_ll_word[i] = ll_word[i].copy + [pad_id1] * max_l_word_len
        return padded_ll_word

    # def build_vocab(self, example_ds):
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
    #     def yield_tokens(example_ds):
    #         for example_d in example_ds:
    #             orig_sent = example_d["orig_sent"]
    #             yield tokenizer(orig_sent)
    #
    #     vocab = build_vocab_from_iterator(yield_tokens(example_ds),
    #                                       specials=["<unk>"])
    #     vocab.set_default_index(vocab["<unk>"])
    #
    #     return vocab


    def pad_data(self, l_example_d):
        """
        formerly data.pad_data()



        Parameters
        ----------
        l_example_d

        Returns
        -------

        """


        # data_in = l_example_d
        # example_d = {
        #     'sent_plus_ids': sent_plus_ids,
        #     'l_ilabels': ilabels_for_each_ex[:MAX_EXTRACTION_LENGTH],
        #     'word_starts': word_starts,
        #     'orig_sent': orig_sent,
        #     # if spacy_model:
        #     'pos_mask': pos_mask,
        #     'pos_indices': pos_indices,
        #     'verb_mask': verb_mask,
        #     'verb_indices': verb_indices
        # }

        l_sent_plus_ids = [example_d['sent_plus_ids'] for example_d in
                           l_example_d]
        padded_l_sent_plus_ids = DPadder.get_padded_list(l_sent_plus_ids,
                                                         self.sent_pad_id)

        ll_ilabels = [example_d['l_ilabels'] for example_d in l_example_d]
        padded_ll_ilabels = DPadder.get_padded_list_list(
            ll_ilabels,
            pad_id0=0,
            pad_id1=-100,
            max_outer_dim=MAX_DEPTH)

        l_word_starts = \
            [example_d['word_starts'] for example_d in l_example_d]
        padded_l_word_starts = DPadder.get_padded_list(l_word_starts, 0)


        l_orig_sent = [example_d['orig_sent'] for
                       example_d in l_example_d]

        # padded_l_sent_plus_ids = torch.tensor(padded_l_sent_plus_ids)
        # padded_ll_ilabels = torch.tensor(padded_ll_ilabels)
        # padded_l_word_starts = torch.tensor(padded_l_word_starts)

        padded_data = {'l_sent_plus_ids': padded_l_sent_plus_ids,
                       'll_ilabels': padded_ll_ilabels,
                       'l_word_starts': padded_l_word_starts,
                       'l_orig_sent': l_orig_sent}

        if self.spacy_model:
            names = ["pos_mask", "pos_indices", "verb_mask", "verb_indices"]
            for i in range(len(names)):
                l = [example_d[names[i]] for example_d in
                     l_example_d]
                padded_l = DPadder.get_padded_list(l, pad_id=0)
                # padded_data[names[i]] = torch.tensor(padded_l)

        # input data=l_example_d was a list of dictionaries
        # padded_data is a dictionary
        return padded_data
