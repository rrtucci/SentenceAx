from copy import deepcopy
from Params import *
import torch
from collections import OrderedDict
from MInput import *


# from torchtext.data.utils import get_tokenizer
# from torchtext.vocab import build_vocab_from_iterator

class SaxDataPadder:
    """

    Attributes
    ----------
    m_input: MInput
    num_samples: int
    pad_ilabel: int
    padded_data_d: dict[str, torch.tensor]
    use_spacy_model: bool
    """

    def __init__(self, m_input, pad_ilabel, use_spacy_model):
        """

        Parameters
        ----------
        m_input: MInput
        pad_ilabel: int
        use_spacy_model: bool
        """

        self.m_input = m_input
        self.pad_ilabel = pad_ilabel
        self.use_spacy_model = use_spacy_model
        self.padded_data_d = None
        self.set_padded_data_d()
        self.num_samples = len(self.m_input.l_orig_sent)

    @staticmethod
    def get_padded_ll_x(unpadded_ll_x, ipad=0):
        """

        Parameters
        ----------
        unpadded_ll_x: list[list[torch.tensor]]
        ipad: int

        Returns
        -------
        torch.tensor

        """
        # padding a 2 dim array

        ll_x = deepcopy(unpadded_ll_x)
        max_dim1 = -1
        for l_x in ll_x:
            if len(l_x) > max_dim1:
                max_dim1 = len(l_x)
        padded_ll_x = []
        for l_x in ll_x:
            padding_len = max_dim1 - len(l_x)
            l_x += [ipad] * padding_len
            padded_ll_x.append(l_x)
        # for i in range(len(padded_ll_x)):
        #     print(i,len(padded_ll_x[i]), padded_ll_x)

        return torch.tensor(padded_ll_x)

    @staticmethod
    def get_padded_lll_ilabel(unpadded_lll_ilabel, ipad=0):
        """

        Parameters
        ----------
        unpadded_lll_ilabel: list[list[list[int]]]
        ipad: int

        Returns
        -------
        torch.tensor
        """
        lll_ilabel = deepcopy(unpadded_lll_ilabel)

        for sam in range(len(lll_ilabel)):
            pad_depth = MAX_EX_DEPTH - len(lll_ilabel[sam])
            if pad_depth > 0:
                num_words = len(lll_ilabel[sam][ipad])
                # ilabel = 0 for extag=NONE
                lll_ilabel[sam] = lll_ilabel[sam] + [
                    [ipad] * num_words] * pad_depth
            elif pad_depth == 0:
                pass
            else:
                rg = range(MAX_EX_DEPTH, len(lll_ilabel[sam]))
                # must delete last extraction first
                for depth in reversed(rg):
                    print("deleting this extraction because over max: " + \
                          f"(sample, depth)={sam}, {depth}")
                    del lll_ilabel[sam][depth]

        max_num_words = -1
        for ll_ilabel in lll_ilabel:
            if len(ll_ilabel[ipad]) > max_num_words:
                max_num_words = len(ll_ilabel[ipad])
        for ll_ilabel in lll_ilabel:
            for l_ilabel in ll_ilabel:
                padding_len = max_num_words - len(l_ilabel)
                l_ilabel += [ipad] * padding_len

        # for sam in range(len(lll_ilabel)):
        #     print(sam, len(lll_ilabel[sam]), len(lll_ilabel[sam][0]))
        return torch.tensor(lll_ilabel)

    # def build_vocab(self, self.m_input):
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
    #     def yield_tokens(self.m_input):
    #         for example_d in self.m_input:
    #             orig_sent = example_d["orig_sent"]
    #             yield tokenizer(orig_sent)
    #
    #     vocab = build_vocab_from_iterator(yield_tokens(self.m_input),
    #                                       specials=["<unk>"])
    #     vocab.set_default_index(vocab["<unk>"])
    #
    #     return vocab

    def set_padded_data_d(self):
        """
        similar to Openie6.data.pad_data()

        Returns
        -------
        dict[str, torch.tensor]

        """

        # data_in = self.m_input
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

        padded_l_osent_ilabels = SaxDataPadder. \
            get_padded_ll_x(self.m_input.l_osent_ilabels, self.pad_ilabel)

        padded_lll_ilabel = SaxDataPadder. \
            get_padded_lll_ilabel(self.m_input.lll_ilabel)

        padded_l_osent_wstart_locs = SaxDataPadder. \
            get_padded_ll_x(self.m_input.l_osent_wstart_locs, 0)

        padded_data_d = OrderedDict(
            {'l_osent_ilabels': padded_l_osent_ilabels,
             'lll_ilabel': padded_lll_ilabel,
             'l_osent_wstart_locs': padded_l_osent_wstart_locs})

        if self.use_spacy_model:
            padded_data_d["l_osent_pos_mask"] = SaxDataPadder. \
                get_padded_ll_x(self.m_input.l_osent_pos_mask, 0)
            padded_data_d["l_osent_pos_locs"] = SaxDataPadder. \
                get_padded_ll_x(self.m_input.l_osent_pos_locs, 0)
            padded_data_d["l_osent_verb_mask"] = SaxDataPadder. \
                get_padded_ll_x(self.m_input.l_osent_verb_mask, 0)
            padded_data_d["l_osent_verb_locs"] = SaxDataPadder. \
                get_padded_ll_x(self.m_input.l_osent_verb_locs, 0)

        self.padded_data_d = padded_data_d

    def print_padded_data_d_shapes(self):
        """

        Returns
        -------
        None

        """
        print("num_samples=", self.num_samples)
        for key, value in self.padded_data_d.items():
            print(f"{key}.shape: ", value.shape)


if __name__ == "__main__":
    def main():
        in_fp = "testing_files/extags_test.txt"
        model_str = "bert-base-uncased"
        auto = AutoTokenizer.from_pretrained(
            model_str,
            do_lower_case=True,
            use_fast=True,
            data_dir=CACHE_DIR,
            add_special_tokens=False,
            additional_special_tokens=UNUSED_TOKENS)
        use_spacy_model = True
        m_input = MInput(in_fp,
                         auto,
                         use_spacy_model)
        # full encoding is [101, 0, 102], 101=BOS_ILABEL, 102=EOS_ILABEL
        pad_ilabel = auto.encode(auto.pad_token)[1]
        # print("pad_token, pad_ilabel=", auto.pad_token, pad_ilabel)
        padder = SaxDataPadder(m_input, pad_ilabel, use_spacy_model)
        padder.print_padded_data_d_shapes()


    main()
