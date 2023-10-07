import torch
from words_tags_ilabels_translation import *
from MInput import *


class MOutput:
    """

    Attributes
    ----------
    auto_tokenizer: AutoTokenizer
    in_cpu: bool
    lll_confi: torch.Tensor
    lll_osent_icode: torch.Tensor
    llll_ex_icode: torch.Tensor]
    batch_loss: torch.Tensor
    task: str
    """

    def __init__(self,
                 lll_osent_icode,
                 llll_ex_icode,
                 llll_true_ex_icode,
                 lll_confi,
                 batch_loss,
                 task,
                 auto_tokenizer):
        """
        The inputs to the constractor are torch.Tensor and in gpu. They get
        converted to lists and cpu before being stored as attributes of the
        class.

         stands for ten-sor

        Parameters
        ----------
        ll_osent_icode: torch.Tensor
        llll_ex_icode: torch.Tensor
        llll_true_ex_icode: torch.Tensor
        lll_confi: torch.Tensor
        batch_loss: torch.Tensor
        task: str
        auto_tokenizer: AutoTokenizer
        
        """
        self.ll_osent_icode = ll_osent_icode
        self.llll_ex_icode = llll_ex_icode
        self.llll_true_ex_icode = llll_true_ex_icode
        self.lll_confi = lll_confi
        self.batch_loss = batch_loss

        self.in_cpu = False
        self.task = task
        self.auto_tokenizer = auto_tokenizer

    def move_to_cpu(self):
        """

        Returns
        -------
        None

        """
        assert not self.in_cpu
        self.in_cpu = True

        self.ll_osent_icode = self.ll_osent_icode.cpu()
        self.llll_ex_icode = self.llll_ex_icode.cpu()
        self.llll_true_ex_icode = self.llll_true_ex_icode.cpu()
        self.lll_confi = self.lll_confi.cpu()
        self.batch_loss = self.batch_loss.cpu()

        self.lll_confi = (self.lll_confi * 100).round() / 100

    def get_l_orig_sent(self):
        """
        
        Returns
        -------
        list[str]

        """
        l_orig_sent2 = \
            MInput.decode_ll_icode(Li(self.ll_osent_icode),
                                   self.auto_tokenizer)
        if self.task == "ex":
            return undoL(l_orig_sent2)
        else:
            return l_orig_sent2

    def get_lll_word(self):
        if self.task == "ex":
            translator = translate_ilabels_to_words_via_extags
        elif self.task == "cc":
            translator = translate_ilabels_to_words_via_cctags
        else:
            assert False
        l_orig_sentL = redoL(self.get_l_orig_sent())
        num_samples = len(self.llll_ex_icode)
        num_depths = len(self.llll_ex_icode[0])
        # sent_len = len(self.lll_ex_ilabel[0][0])
        lll_word = []
        for sam in range(num_samples):
            ll_word = []
            for depth in range(num_depths):
                ll_word.append(
                    translator(self.llll_ex_icode[sam][depth],
                               l_orig_sentL[sam]))
            lll_word.append(ll_word)
        return lll_word
