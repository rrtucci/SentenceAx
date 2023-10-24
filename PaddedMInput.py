from copy import deepcopy
from Params import *
import torch
from collections import OrderedDict
from MInput import *


# from torchtext.data.utils import get_tokenizer
# from torchtext.vocab import build_vocab_from_iterator

class PaddedMInput(MInput):
    """
    data processing chain
    (optional allen_fp->)tags_in_fp->MInput->PaddedMInput->SaxDataSet
    ->SaxDataLoaderTool

    Attributes
    ----------
    l_orig_sent: list[str]
    ll_osent_icode: torch.Tensor
    ll_osent_pos_bool: torch.Tensor
    ll_osent_pos_loc: torch.Tensor
    ll_osent_verb_bool: torch.Tensor
    ll_osent_verb_loc:  torch.Tensor
    ll_osent_wstart_loc: torch.Tensor
    lll_ilabel: torch.Tensor
    m_in: MInput
    num_samples: int
    pad_icode: int
    x_d: OrderedDict
    y_d: dict[str, torch.Tensor]
    """

    def __init__(self, m_in):
        """


        Parameters
        ----------
        m_in: MInput
        """
        MInput.__init__(self,
                        m_in.params,
                        m_in.tags_in_fp,
                        m_in.auto_tokenizer,
                        read=False,
                        verbose=m_in.verbose)

        self.m_in = m_in
        self.pad_icode = self.m_in.auto_tokenizer.encode(
            self.auto_tokenizer.pad_token)[1]
        assert self.pad_icode == 0
        self.num_samples = len(self.m_in.l_orig_sent)
        self.set_padded_data()
        self.l_orig_sent = m_in.l_orig_sent

        # call this after self.set_padded_data() does its type changes
        self.y_d = {"lll_ilabel": self.lll_ilabel}
        self.x_d = OrderedDict({
            "ll_osent_icode": self.ll_osent_icode,
            "ll_osent_wstart_loc": self.ll_osent_wstart_loc,
        })
        if USE_SPACY_MODEL:
            self.x_d["ll_osent_pos_bool"] = self.ll_osent_pos_bool
            self.x_d["ll_osent_pos_loc"] = self.ll_osent_pos_loc
            self.x_d["ll_osent_verb_bool"] = self.ll_osent_verb_bool
            self.x_d["ll_osent_verb_loc"] = self.ll_osent_verb_loc

    @staticmethod
    def get_padded_ll_x(unpadded_ll_x, ipad1=0):
        """
        The number at the end of `ipad` refers to the dimension. The
        dimensions here are called 0, 1 (1 is the innermost). -100 is a good
        ipad for the innermost dimension because it is ignored by loss
        functions like cross entropy. -100 is a good ipad for the innermost
        dimension because it is ignored by loss functions like cross entropy.


        Parameters
        ----------
        unpadded_ll_x: list[list[torch.Tensor]]
        ipad1: int

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

        return torch.Tensor(padded_ll_x).long()

    @staticmethod
    def get_padded_lll_ilabel(unpadded_lll_ilabel,
                              ipad1=0, ipad2=-100):
        """
        The number at the end of `ipad` refers to the dimension. The
        dimensions here are called 0, 1, 2 (2 is the innermost).

        Both 0 and -100 are recognized as pad icode. Openie6 uses both to
        distinguish between dim1 and dim2 padding.

        -100 is necessary to distinguish padding from ilabel=0

        Parameters
        ----------
        unpadded_lll_ilabel: list[list[list[int]]]
        ipad1: int
        ipad2: int

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
            else: # pad_depth < 0
                rg = range(EX_NUM_DEPTHS, len(lll_ilabel[sam]))
                # must delete last extraction first
                for depth in reversed(rg):
                    del lll_ilabel[sam][depth]
                print("PaddedMInput omitting extractions: sample= " + str(
                    sam) + ", depths=" + str(list(rg)))

        max_num_words = -1

        for ll_ilabel in lll_ilabel:
            for l_ilabel in ll_ilabel:
                # print("mnjk", len(l_ilabel))
                if len(l_ilabel) > max_num_words:
                    max_num_words = len(l_ilabel)
        # print("vvvv-max_num_words=", max_num_words)
        for ll_ilabel in lll_ilabel:
            for l_ilabel in ll_ilabel:
                padding_len = max_num_words - len(l_ilabel)
                l_ilabel += [ipad2] * padding_len

        # for sam in range(len(lll_ilabel)):
        #     print(sam, len(lll_ilabel[sam]), len(lll_ilabel[sam][0]))
        return torch.Tensor(lll_ilabel).long()

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
    #     # tokenizer = get_tokenizer("basic_english")
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

    def set_padded_data(self):
        """
        similar to Openie6.data.pad_data()

        This method changes the type of some m_in attributes but not their
        names. For example, lll_ilabel goes from type list[list[list[int]]] to
        type torch.tensor.

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
        #     # if USE_SPACY_MODEL:
        #     'pos_bools': pos_bools,
        #     'pos_locs': pos_locs,
        #     'verb_bools': verb_bools,
        #     'verb_locs': verb_locs
        # }

        self.ll_osent_icode = PaddedMInput. \
            get_padded_ll_x(self.m_in.ll_osent_icode)

        self.lll_ilabel = PaddedMInput. \
            get_padded_lll_ilabel(self.m_in.lll_ilabel)

        self.ll_osent_wstart_loc = PaddedMInput. \
            get_padded_ll_x(self.m_in.ll_osent_wstart_loc)

        if USE_SPACY_MODEL:
            self.ll_osent_pos_bool = PaddedMInput. \
                get_padded_ll_x(self.m_in.ll_osent_pos_bool)
            self.ll_osent_pos_loc = PaddedMInput. \
                get_padded_ll_x(self.m_in.ll_osent_pos_loc)
            self.ll_osent_verb_bool = PaddedMInput. \
                get_padded_ll_x(self.m_in.ll_osent_verb_bool)
            self.ll_osent_verb_loc = PaddedMInput. \
                get_padded_ll_x(self.m_in.ll_osent_verb_loc)

    def print_padded_data_shapes(self):
        """

        Returns
        -------
        None

        """
        print("num_samples=", self.num_samples)
        print("x_d:")
        for key, value in self.x_d.items():
            # print("lmhb", key, type(value))
            print(f"{key}.shape: ", value.shape)
        print("y_d:")
        for key, value in self.y_d.items():
            # print("lmhb", key, type(value))
            print(f"{key}.shape: ", value.shape)


if __name__ == "__main__":
    def main(in_fp):
        params = Params(1)  # 1, task="ex", mode="train_test"
        model_str = "bert-base-uncased"
        do_lower_case = ('uncased' in model_str)
        auto = AutoTokenizer.from_pretrained(
            model_str,
            do_lower_case=do_lower_case,
            use_fast=True,
            data_dir=CACHE_DIR,
            add_special_tokens=False,
            additional_special_tokens=UNUSED_TOKENS)

        m_in = MInput(params,
                      in_fp,
                      auto)
        padded_m_in = PaddedMInput(m_in)
        padded_m_in.print_padded_data_shapes()
        li1 = get_words(padded_m_in.l_orig_sent[0])
        li2 = padded_m_in.lll_ilabel[0][0]
        print([(k, li1[k]) for k in range(len(li1))])
        print([(k, li2[k]) for k in range(len(li2))])


    # main(in_fp="tests/extags_test.txt")
    # main(in_fp="predictions/small_pred.txt")
    main(in_fp="input_data/carb-data/test.txt")
