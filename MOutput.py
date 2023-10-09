import torch
from words_tags_ilabels_translation import *
from MInput import *


class MOutput:
    """

    Attributes
    ----------
    auto_tokenizer: AutoTokenizer
    in_cpu: bool
    lll_pred_ex_confi: torch.Tensor
    lll_osent_icode: torch.Tensor
    llll_pred_ex_ilabel: torch.Tensor]
    batch_loss: torch.Tensor
    task: str
    """

    def __init__(self,
                 l_orig_sent,
                 lll_ex_ilabel,
                 lll_pred_ex_ilabel,
                 ll_pred_ex_confi,
                 batch_loss):
        """
        The inputs to the constractor are torch.Tensor and in gpu. They get
        converted to lists and cpu before being stored as attributes of the
        class.

         stands for ten-sor

        Parameters
        ----------
        l_orig_sent: list[str]
        lll_ex_ilabel: torch.Tensor
        lll_pred_ex_ilabel: torch.Tensor
        ll_pred_ex_confi: torch.Tensor
        batch_loss: torch.Tensor
        task: str
        auto_tokenizer: AutoTokenizer
        
        """
        self.l_orig_sent = l_orig_sent
        self.lll_ex_ilabel = lll_ex_ilabel
        self.lll_pred_ex_ilabel = lll_pred_ex_ilabel
        self.ll_pred_ex_confi = ll_pred_ex_confi
        self.batch_loss = batch_loss

        self.in_cpu = False

    def move_to_cpu(self):
        """

        Returns
        -------
        None

        """
        assert not self.in_cpu
        self.in_cpu = True

        self.lll_ex_ilabel = self.lll_ex_ilabel.cpu()
        self.lll_pred_ex_ilabel = self.lll_pred_ex_ilabel.cpu()
        self.ll_pred_ex_confi = self.ll_pred_ex_confi.cpu()
        self.batch_loss = self.batch_loss.cpu()

        self.ll_pred_ex_confi = (self.ll_pred_ex_confi * 100).round() / 100

    # def get_l_orig_sent(self):
    #     """
    #
    #     Returns
    #     -------
    #     list[str]
    #
    #     """
    #     l_orig_sent2 = \
    #         MInput.decode_ll_icode(Li(self.ll_osent_icode),
    #                                self.auto_tokenizer)
    #     if self.task == "ex":
    #         return undoL(l_orig_sent2)
    #     else:
    #         return l_orig_sent2
    #
    # def get_lll_word(self):
    #     if self.task == "ex":
    #         translator = translate_ilabels_to_words_via_extags
    #     elif self.task == "cc":
    #         translator = translate_ilabels_to_words_via_cctags
    #     else:
    #         assert False
    #     l_orig_sentL = redoL(self.get_l_orig_sent())
    #     num_samples = len(self.llll_pred_ex_ilabel)
    #     num_depths = len(self.llll_pred_ex_ilabel[0])
    #     # sent_len = len(self.lll_ex_ilabel[0][0])
    #     lll_word = []
    #     for sam in range(num_samples):
    #         ll_word = []
    #         for depth in range(num_depths):
    #             ll_word.append(
    #                 translator(self.llll_pred_ex_ilabel[sam][depth],
    #                            l_orig_sentL[sam]))
    #         lll_word.append(ll_word)
    #     return lll_word

