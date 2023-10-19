from copy import deepcopy
from Params import *
import torch
from collections import OrderedDict
from MInput import *


# from torchtext.data.utils import tags_in_fp_tokenizer
# from torchtext.vocab import build_vocab_from_iterator

class SaxDataPadder:
    """

    Attributes
    ----------
    m_in: MInput
    num_samples: int
    pad_icode: int
    padded_data_d: dict[str, torch.Tensor]
    use_spacy_model: bool
    """

    def __init__(self, m_in, pad_icode, use_spacy_model):
        """

        Parameters
        ----------
        m_in: MInput
        pad_icode: int
        use_spacy_model: bool
        """

        self.m_in = m_in
        self.pad_icode = pad_icode
        assert pad_icode == 0
        self.use_spacy_model = use_spacy_model
        self.padded_data_d = None
        self.set_padded_data_d()
        self.num_samples = len(self.m_in.l_orig_sent)

    @staticmethod
    def tags_in_fp_padded_ll_x(unpadded_ll_x, ipad1=0):
        """
        The number at the end of `ipad` refers to the dimension. The
        dimensions here are called 0, 1 (1 is the innermost). -100 is a good
        ipad for the innermost dimension because it is ignored by loss
        functions like cross entropy. -100 is a good ipad for the innermost
        dimension because it is ignored by loss functions like cross entropy.


        Parameters
        ----------
        unpadded_ll_x: list[list[torch.Tensor]]
        ipad: int

        Returns
        -------
        torch.Tensor

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
            l_x += [ipad1] * padding_len
            padded_ll_x.append(l_x)
        # for i in range(len(padded_ll_x)):
        #     print(i,len(padded_ll_x[i]), padded_ll_x)

        return torch.Tensor(padded_ll_x)

    @staticmethod
    def tags_in_fp_padded_lll_ilabel(unpadded_lll_ilabel, ipad1=0, ipad2=0):
        """
        The number at the end of `ipad` refers to the dimension. The
        dimensions here are called 0, 1, 2 (2 is the innermost).

        Parameters
        ----------
        unpadded_lll_ilabel: list[list[list[int]]]
        ipad: int

        Returns
        -------
        torch.Tensor
        """
        lll_ilabel = deepcopy(unpadded_lll_ilabel)
        if not lll_ilabel[0]:
            return torch.tensor(lll_ilabel)

        for sam in range(len(lll_ilabel)):
            pad_depth = EX_NUM_DEPTHS - len(lll_ilabel[sam])
            if pad_depth > 0:
                # print("lmjki", lll_ilabel[sam])
                num_words = len(lll_ilabel[sam][0])
                # ilabel = 0 for extag=NONE
                lll_ilabel[sam] = lll_ilabel[sam] + [
                    [ipad1] * num_words] * pad_depth
            elif pad_depth == 0:
                pass
            else:
                rg = range(EX_NUM_DEPTHS, len(lll_ilabel[sam]))
                # must delete last extraction first
                for depth in reversed(rg):
                    del lll_ilabel[sam][depth]
                print("SaxDataPadder deleting these extraction because "
                        "over max depth: sample= " + str(sam) +
                      ", depth=" + str(rg))

        max_num_words = -1
        for ll_ilabel in lll_ilabel:
            if len(ll_ilabel[0]) > max_num_words:
                max_num_words = len(ll_ilabel[0])
        for ll_ilabel in lll_ilabel:
            for l_ilabel in ll_ilabel:
                padding_len = max_num_words - len(l_ilabel)
                l_ilabel += [ipad2] * padding_len

        # for sam in range(len(lll_ilabel)):
        #     print(sam, len(lll_ilabel[sam]), len(lll_ilabel[sam][0]))
        return torch.Tensor(lll_ilabel)

    # def build_vocab(self, self.m_in):
    #     """
    #
    #     A vocabulary (vocab) is a function that turns
    #     word lists to int lists
    #
    #     vocab(['here', 'is', 'an', 'example'])
    #     >>> [475, 21, 30, 5297]
    #     """
    #
    #     # tokenizer = tags_in_fp_tokenizer("basic_english")
    #     tokenizer = self.auto_tokenizer
    #     def yield_tokens(self.m_in):
    #         for example_d in self.m_in:
    #             orig_sent = example_d["orig_sent"]
    #             yield tokenizer(orig_sent)
    #
    #     vocab = build_vocab_from_iterator(yield_tokens(self.m_in),
    #                                       specials=["<unk>"])
    #     vocab.set_default_index(vocab["<unk>"])
    #
    #     return vocab

    def set_padded_data_d(self):
        """
        similar to Openie6.data.pad_data()

        Returns
        -------
        dict[str, torch.Tensor]

        """

        # data_in = self.m_in
        # example_d = {
        #     'sentL_ids': sentL_ids,
        #     'll_label': labels_for_each_ex[:EX_NUM_DEPTHS],
        #     'word_starts': word_starts,
        #     'orig_sent': orig_sent,
        #     # if use_spacy_model:
        #     'pos_bools': pos_bools,
        #     'pos_locs': pos_locs,
        #     'verb_bools': verb_bools,
        #     'verb_locs': verb_locs
        # }

        padded_ll_osent_icode = SaxDataPadder. \
            tags_in_fp_padded_ll_x(self.m_in.ll_osent_icode)

        padded_lll_ilabel = SaxDataPadder. \
            tags_in_fp_padded_lll_ilabel(self.m_in.lll_ilabel)

        padded_ll_osent_wstart_loc = SaxDataPadder. \
            get_padded_ll_x(self.m_in.ll_osent_wstart_loc)

        padded_data_d = OrderedDict(
            {'ll_osent_icode': padded_ll_osent_icode,
             'lll_ilabel': padded_lll_ilabel,
             'll_osent_wstart_loc': padded_ll_osent_wstart_loc})

        if self.use_spacy_model:
            padded_data_d["ll_osent_pos_bool"] = SaxDataPadder. \
                get_padded_ll_x(self.m_in.ll_osent_pos_bool)
            padded_data_d["ll_osent_pos_loc"] = SaxDataPadder. \
                get_padded_ll_x(self.m_in.ll_osent_pos_loc)
            padded_data_d["ll_osent_verb_bool"] = SaxDataPadder. \
                get_padded_ll_x(self.m_in.ll_osent_verb_bool)
            padded_data_d["ll_osent_verb_loc"] = SaxDataPadder. \
                get_padded_ll_x(self.m_in.ll_osent_verb_loc)

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
    def main(task, in_fp):
        model_str = "bert-base-uncased"
        auto = AutoTokenizer.from_pretrained(
            model_str,
            do_lower_case=True,
            use_fast=True,
            data_dir=CACHE_DIR,
            add_special_tokens=False,
            additional_special_tokens=UNUSED_TOKENS)
        use_spacy_model = True
        m_in = MInput(task,
                      in_fp,
                      auto,
                      use_spacy_model)
        # full encoding is [101, 0, 102], 101=BOS_ICODE, 102=EOS_ICODE
        pad_icode = auto.encode(auto.pad_token)[1]
        # print("pad_token, pad_icode=", auto.pad_token, pad_icode)
        padder = SaxDataPadder(m_in, pad_icode, use_spacy_model)
        padder.print_padded_data_d_shapes()


    main(task="ex",
         in_fp="tests/extags_test.txt")
    main(task="ex",
         in_fp="predictions/small_pred.txt")
